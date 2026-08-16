"""CUDA model backends: the real implementations behind the pipeline interfaces.

None of this runs on Apple Silicon. It is written to run on Colab (see
`colab/glasses3d_colab.ipynb`) and validated there by that notebook's
"Phase 1 check" cell rather than by the local test suite — the local tests
exercise `MockReconstructor` through exactly the same interface.

Reconstruction backend of choice is MapAnything: a 1B-parameter feed-forward
model that regresses *metric* 3D geometry and optionally conditions on known
intrinsics and poses. Metric matters — monocular reconstruction is otherwise
scale-ambiguous, and a room that is geometrically right but arbitrarily sized
breaks VR placement and physical measurement alike.

Two weight variants exist and the difference is legal, not technical:

    facebook/map-anything           CC-BY-NC 4.0  — research only
    facebook/map-anything-apache    Apache 2.0    — commercial-safe

This module defaults to the Apache one. Opting into the NC weights should be a
deliberate act, not something inherited from a copied snippet.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

APACHE_MODEL = "facebook/map-anything-apache"
NC_MODEL = "facebook/map-anything"


@dataclass
class ReconstructionResult:
    """Geometry recovered from a set of views."""

    points: np.ndarray  # (N,3) world-space, metres
    colors: np.ndarray  # (N,3) uint8
    confidence: np.ndarray  # (N,)
    poses: np.ndarray  # (V,4,4) camera-to-world
    intrinsics: np.ndarray  # (V,3,3)
    view_count: int = 0

    def filtered(self, min_conf: float = 0.5, max_points: int = 2_000_000,
                 seed: int = 0) -> "ReconstructionResult":
        """Drop low-confidence points, then subsample to a budget.

        Both steps are load-bearing. Feed-forward models emit a prediction for
        *every* pixel including sky, motion blur, and untextured wall, and the
        low-confidence tail is a halo of floaters that dominates the visual
        impression of the result. Separately, N views at 720x1280 is millions of
        points per view — enough to exhaust VRAM during export and to stall
        every downstream viewer.
        """
        keep = self.confidence >= min_conf
        pts, cols, conf = self.points[keep], self.colors[keep], self.confidence[keep]
        if len(pts) > max_points:
            # Uniform random, not stride-based: strided subsampling of a
            # row-major pixel grid biases toward whichever views came first.
            idx = np.random.RandomState(seed).choice(len(pts), max_points, replace=False)
            pts, cols, conf = pts[idx], cols[idx], conf[idx]
        return ReconstructionResult(pts, cols, conf, self.poses, self.intrinsics,
                                    self.view_count)

    def summary(self) -> str:
        if len(self.points) == 0:
            return "empty reconstruction"
        lo, hi = self.points.min(axis=0), self.points.max(axis=0)
        return ("%d points from %d views | extent %.2f x %.2f x %.2f m | "
                "confidence %.2f-%.2f"
                % (len(self.points), self.view_count,
                   hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2],
                   float(self.confidence.min()), float(self.confidence.max())))


class Reconstructor(abc.ABC):
    @abc.abstractmethod
    def reconstruct(self, images: Sequence[np.ndarray],
                    intrinsics: Optional[np.ndarray] = None,
                    poses: Optional[np.ndarray] = None) -> ReconstructionResult:
        ...


class MapAnythingReconstructor(Reconstructor):
    """MapAnything backend.

        pip install -e .  # from a clone of facebookresearch/map-anything

    Priors are optional but valuable: passing the intrinsics from M2 calibration
    and the poses from SLAM removes two things the model would otherwise have to
    infer, and it is strictly more accurate when it does not have to guess.
    """

    def __init__(self, model_id: str = APACHE_MODEL, device: str = "cuda",
                 memory_efficient: bool = True, amp_dtype: str = "bf16"):
        self.model_id = model_id
        self.device = device
        self.memory_efficient = memory_efficient
        self.amp_dtype = amp_dtype
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model
        import torch  # noqa: F401  (imported for its side effect of failing loudly)
        from mapanything.models import MapAnything

        if self.model_id == NC_MODEL:
            print("[backends] NOTE: using CC-BY-NC weights — research use only. "
                  "Switch to %s before shipping anything." % APACHE_MODEL)
        self._model = MapAnything.from_pretrained(self.model_id).to(self.device)
        self._model.eval()
        return self._model

    def reconstruct(self, images, intrinsics=None, poses=None) -> ReconstructionResult:
        import torch

        model = self.load()
        views = self._build_views(images, intrinsics, poses)

        with torch.no_grad():
            preds = model.infer(views, use_amp=True, amp_dtype=self.amp_dtype,
                                memory_efficient_inference=self.memory_efficient)

        pts, cols, confs = [], [], []
        out_poses, out_K = [], []
        for image, pred in zip(images, preds):
            p = _to_numpy(pred["pts3d"]).reshape(-1, 3)
            c = _to_numpy(pred["conf"]).reshape(-1)
            rgb = image.reshape(-1, 3)[:, ::-1]  # BGR (OpenCV) -> RGB
            if len(rgb) != len(p):
                # Model output is at its own working resolution, which need not
                # match the source frame. Bail loudly rather than zipping
                # mismatched arrays into silently wrong colours.
                raise ValueError(
                    "pixel count mismatch: image has %d, prediction has %d — "
                    "resize the source frames to the model's input resolution"
                    % (len(rgb), len(p)))
            pts.append(p)
            cols.append(rgb)
            confs.append(c)
            if "camera_poses" in pred:
                out_poses.append(_to_numpy(pred["camera_poses"]).reshape(4, 4))
            if "intrinsics" in pred:
                out_K.append(_to_numpy(pred["intrinsics"]).reshape(3, 3))

        return ReconstructionResult(
            points=np.concatenate(pts) if pts else np.zeros((0, 3)),
            colors=np.concatenate(cols).astype(np.uint8) if cols else np.zeros((0, 3), np.uint8),
            confidence=np.concatenate(confs) if confs else np.zeros(0),
            poses=np.array(out_poses) if out_poses else np.zeros((0, 4, 4)),
            intrinsics=np.array(out_K) if out_K else np.zeros((0, 3, 3)),
            view_count=len(images),
        )

    def _build_views(self, images, intrinsics, poses) -> List[dict]:
        import torch

        views = []
        for i, img in enumerate(images):
            rgb = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, 0..1
            view = {"img": torch.from_numpy(rgb).permute(2, 0, 1)[None].to(self.device)}
            if intrinsics is not None:
                K = intrinsics[i] if intrinsics.ndim == 3 else intrinsics
                view["intrinsics"] = torch.from_numpy(
                    np.asarray(K, np.float32))[None].to(self.device)
            if poses is not None:
                view["camera_poses"] = torch.from_numpy(
                    np.asarray(poses[i], np.float32))[None].to(self.device)
            views.append(view)
        return views


class MockReconstructor(Reconstructor):
    """Deterministic synthetic geometry, so the driver is testable with no GPU."""

    def __init__(self, points_per_view: int = 500, seed: int = 0):
        self.points_per_view = points_per_view
        self.seed = seed

    def reconstruct(self, images, intrinsics=None, poses=None) -> ReconstructionResult:
        rng = np.random.RandomState(self.seed)
        n = self.points_per_view * max(len(images), 1)
        return ReconstructionResult(
            points=rng.normal(0, 1.0, (n, 3)),
            colors=rng.randint(0, 256, (n, 3)).astype(np.uint8),
            confidence=rng.uniform(0, 1, n),
            poses=np.tile(np.eye(4), (len(images), 1, 1)),
            intrinsics=np.tile(np.eye(3), (len(images), 1, 1)),
            view_count=len(images),
        )


def _to_numpy(x):
    return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


# Colab compute-unit burn rates, CU/hour. Published figures vary between sources
# (A100 is quoted anywhere from 5.4 to 15), so treat these as planning estimates
# and check your actual balance in the Colab UI. Pro includes 100 CU/month.
CU_PER_HOUR = {
    "T4": 1.19,
    "L4": 2.6,
    "A100-40": 5.40,
    "A100-80": 7.52,
    "RTX PRO 6000": 9.0,
}
COLAB_PRO_MONTHLY_CU = 100.0


def _cu_key(name: str, vram_gb: float) -> str:
    key = name.upper()
    if "A100" in key:
        return "A100-80" if vram_gb > 60 else "A100-40"
    if "L4" in key:
        return "L4"
    if "T4" in key:
        return "T4"
    if "RTX PRO 6000" in key or "H100" in key:
        return "RTX PRO 6000"
    return "L4"  # unknown premium GPU — assume mid-tier for planning


def estimate_cost(hours: float, gpu: Optional[dict] = None) -> dict:
    """What a job of this length costs against a Colab Pro allowance.

    Worth checking before a long run. The allowance is smaller than it looks:
    100 CU is about 84 hours of T4 but only ~18 hours of A100 40GB, so the
    premium GPUs are a budget to spend deliberately rather than a default.
    """
    gpu = gpu or describe_gpu()
    if not gpu.get("available"):
        return {"error": gpu.get("reason", "no GPU")}
    key = _cu_key(gpu["name"], gpu["vram_gb"])
    rate = CU_PER_HOUR[key]
    cu = rate * hours
    return {
        "gpu": gpu["name"],
        "class": key,
        "cu_per_hour": rate,
        "hours": hours,
        "compute_units": round(cu, 2),
        "percent_of_monthly_pro": round(100.0 * cu / COLAB_PRO_MONTHLY_CU, 1),
        "hours_available_on_full_allowance": round(COLAB_PRO_MONTHLY_CU / rate, 1),
    }


def budget_advice(gpu: Optional[dict] = None) -> List[str]:
    """Concrete ways not to waste a Colab Pro allowance on this pipeline."""
    gpu = gpu or describe_gpu()
    if not gpu.get("available"):
        return ["No GPU attached — nothing is being billed."]
    key = _cu_key(gpu["name"], gpu["vram_gb"])
    tips = []
    if key.startswith("A100") or key == "RTX PRO 6000":
        tips.append(
            "You are on a premium GPU at %.2f CU/hr — about %.0f hours from a full "
            "100 CU allowance. Everything below costs real budget."
            % (CU_PER_HOUR[key], COLAB_PRO_MONTHLY_CU / CU_PER_HOUR[key]))
        tips.append(
            "Do pip installs, model downloads, calibration and any mock-backend "
            "runs on a T4 instead. The first-run install alone is ~10 minutes; "
            "that is ~0.9 CU on an A100 versus ~0.2 on a T4, every session.")
        tips.append(
            "Cache the repo and weights to Drive so premium sessions skip setup "
            "entirely and spend their time on inference.")
        tips.append(
            "Disconnect the runtime the moment a job finishes. Idle premium "
            "sessions bill at the same rate as busy ones.")
    else:
        tips.append(
            "You are on %s at %.2f CU/hr — roughly %.0f hours from a full "
            "allowance. Cheap enough to develop on freely."
            % (key, CU_PER_HOUR[key], COLAB_PRO_MONTHLY_CU / CU_PER_HOUR[key]))
        tips.append(
            "Use this runtime for setup, calibration and mock runs, then switch "
            "to A100 only for the reconstruction itself.")
    if not gpu.get("live_viable"):
        tips.append(
            "This GPU is below the ~8 fps live threshold. Spending premium units "
            "on live mode here would buy a slideshow — use offline mode.")
    return tips


def describe_gpu() -> dict:
    """What GPU did Colab actually hand us, and what does that imply?

    Colab allocates opportunistically — the same notebook can get a T4 on one
    run and an L4 on the next — so capability has to be discovered at runtime
    rather than assumed. The fps estimates below are anchored to MASt3R-SLAM's
    published 15 FPS on an RTX 4090 and scaled by rough relative throughput;
    they are order-of-magnitude guidance for choosing a mode, not benchmarks.
    """
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "no CUDA device (Runtime > Change runtime type)"}

    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    key = name.upper()
    # Relative to an RTX 4090 at 1.0.
    if "A100" in key:
        rel, tier = 0.65, "good"
    elif "RTX PRO 6000" in key or "G4" in key or "H100" in key:
        rel, tier = 1.1, "excellent"
    elif "L4" in key:
        rel, tier = 0.30, "usable"
    elif "T4" in key:
        rel, tier = 0.16, "offline only"
    else:
        rel, tier = 0.30, "unknown"

    slam_fps = 15.0 * rel
    return {
        "available": True,
        "name": name,
        "vram_gb": round(vram_gb, 1),
        "relative_to_4090": rel,
        "tier": tier,
        "expected_slam_fps": round(slam_fps, 1),
        "live_viable": slam_fps >= 8.0,
        "note": ("Live tracking needs roughly 8+ fps to feel live. "
                 "Below that, use offline mode — same pipeline, better results."),
    }
