"""Tests the offline reconstruction driver with a mock backend.

The point is to validate everything that is *not* the model, so that running on
Colab introduces exactly one new variable. Keyframe selection especially: it
decides what the reconstructor ever sees, and a selector that quietly admits
blurred frames degrades the output in a way that looks like a model problem.

    python3 tools/test_reconstruct.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "server"))

from backends import (APACHE_MODEL, MockReconstructor, NC_MODEL,  # noqa: E402
                      ReconstructionResult, describe_gpu)
from export import read_ply_header  # noqa: E402
from reconstruct import pick_keyframes  # noqa: E402
from tracker import sharpness  # noqa: E402

results = []


def check(label, condition):
    results.append((label, bool(condition)))


rng = np.random.RandomState(0)


def textured(seed=0):
    t = np.random.RandomState(seed).randint(0, 255, (240, 320, 3)).astype(np.uint8)
    return cv2.GaussianBlur(t, (5, 5), 1)


# --- keyframe selection -----------------------------------------------------
# 60 frames where every 5th is heavily blurred.
frames, blurred_idx = [], set()
for i in range(60):
    f = textured(i)
    if i % 5 == 0:
        f = cv2.GaussianBlur(f, (31, 31), 12)
        blurred_idx.add(i)
    frames.append(f)

picks = pick_keyframes(frames, target=12)
check("selection returns roughly the requested count", 8 <= len(picks) <= 12)
overlap = blurred_idx.intersection(picks)
check("blurred frames are excluded from selection", not overlap)
check("selections are unique and ordered", picks == sorted(set(picks)))

# Temporal spread: picks should cover the clip, not cluster at the start.
span = (max(picks) - min(picks)) / float(len(frames) - 1)
check("selections span most of the clip", span > 0.7)
gaps = np.diff(picks)
check("selections are not bunched together", gaps.min() >= 2)

# Fewer frames than the target must still work.
few = pick_keyframes(frames[:6], target=12)
check("handles fewer frames than requested", 0 < len(few) <= 6)
check("handles a single frame", len(pick_keyframes([textured(1)], target=12)) == 1)

# --- ReconstructionResult filtering ----------------------------------------
n = 5000
res = ReconstructionResult(
    points=rng.normal(0, 1, (n, 3)),
    colors=rng.randint(0, 256, (n, 3)).astype(np.uint8),
    confidence=np.linspace(0, 1, n),
    poses=np.zeros((3, 4, 4)), intrinsics=np.zeros((3, 3, 3)), view_count=3)

f = res.filtered(min_conf=0.8)
check("confidence filter drops the low tail", len(f.points) < n)
check("confidence filter keeps only what it should", f.confidence.min() >= 0.8)
check("filter keeps arrays aligned",
      len(f.points) == len(f.colors) == len(f.confidence))

cap = res.filtered(min_conf=0.0, max_points=100)
check("point budget is enforced", len(cap.points) == 100)
check("subsample keeps arrays aligned", len(cap.colors) == 100)
# Uniform sampling, not strided — a strided pick of a sorted-confidence array
# would preserve the full range but bias which views survive.
check("subsample spans the whole confidence range",
      cap.confidence.max() - cap.confidence.min() > 0.5)

# Feed-forward models emit NaN/Inf for pixels they cannot resolve. A single Inf
# is not one bad point - it makes every downstream bounding-box computation
# infinite, so the viewer frames an infinite scene and renders nothing.
nf = ReconstructionResult(
    points=np.array([[0., 0., 0.], [1., 1., 1.], [np.nan, 1., 2.],
                     [np.inf, 0., 0.], [-np.inf, 0., 0.]]),
    colors=np.zeros((5, 3), np.uint8), confidence=np.ones(5),
    poses=np.zeros((1, 4, 4)), intrinsics=np.zeros((1, 3, 3)), view_count=1).filtered(min_conf=0.0)
check("non-finite points are dropped", len(nf.points) == 2)
check("every survivor is finite", np.isfinite(nf.points).all())
check("arrays stay aligned after dropping non-finite",
      len(nf.points) == len(nf.colors) == len(nf.confidence))
check("extent is finite once they are gone", "inf" not in nf.summary())

check("empty reconstruction summarises safely",
      "empty" in ReconstructionResult(np.zeros((0, 3)), np.zeros((0, 3)),
                                      np.zeros(0), np.zeros((0, 4, 4)),
                                      np.zeros((0, 3, 3))).summary())

# --- backend contract -------------------------------------------------------
mock = MockReconstructor(points_per_view=100)
out = mock.reconstruct([textured(i) for i in range(4)])
check("mock backend honours the Reconstructor interface", isinstance(out, ReconstructionResult))
check("mock backend reports its view count", out.view_count == 4)
check("mock backend scales points with views", len(out.points) == 400)

check("Apache weights are the default", APACHE_MODEL.endswith("-apache"))
check("the NC model id is distinct", NC_MODEL != APACHE_MODEL)

gpu = describe_gpu()
check("GPU probe never raises", isinstance(gpu, dict) and "available" in gpu)
check("GPU probe explains itself when unavailable",
      gpu["available"] or bool(gpu.get("reason")))

# --- Colab Pro compute-unit budgeting --------------------------------------
from backends import (COLAB_PRO_MONTHLY_CU, CU_PER_HOUR, _cu_key,  # noqa: E402
                      budget_advice, estimate_cost)

check("A100 80GB is distinguished from 40GB by VRAM",
      _cu_key("NVIDIA A100-SXM4-80GB", 80.0) == "A100-80"
      and _cu_key("NVIDIA A100-SXM4-40GB", 40.0) == "A100-40")
check("T4 is classified", _cu_key("Tesla T4", 15.0) == "T4")
check("unknown premium GPUs fall back to a mid tier",
      _cu_key("Some Future GPU", 48.0) in CU_PER_HOUR)
check("premium GPUs cost more per hour than a T4",
      CU_PER_HOUR["A100-40"] > CU_PER_HOUR["L4"] > CU_PER_HOUR["T4"])

a100 = {"available": True, "name": "NVIDIA A100-SXM4-40GB", "vram_gb": 40.0,
        "live_viable": True}
cost = estimate_cost(2.0, a100)
check("cost scales with hours", abs(cost["compute_units"] - 2 * 5.40) < 1e-6)
check("cost is expressed against the Pro allowance",
      abs(cost["percent_of_monthly_pro"] - 10.8) < 0.1)
check("allowance hours are reported",
      abs(cost["hours_available_on_full_allowance"]
          - COLAB_PRO_MONTHLY_CU / 5.40) < 0.1)
check("a full A100 allowance is under 20 hours",
      cost["hours_available_on_full_allowance"] < 20)

t4 = {"available": True, "name": "Tesla T4", "vram_gb": 15.0, "live_viable": False}
check("a T4 hour costs far less than an A100 hour",
      estimate_cost(1.0, t4)["compute_units"] < cost["compute_units"] / 4)
check("cost estimate degrades gracefully with no GPU",
      "error" in estimate_cost(1.0, {"available": False, "reason": "no CUDA"}))

adv_premium = " ".join(budget_advice(a100))
check("premium advice warns about setup on an expensive GPU", "T4" in adv_premium)
check("premium advice mentions disconnecting", "isconnect" in adv_premium)
adv_cheap = " ".join(budget_advice(t4))
check("cheap-GPU advice suggests switching up for the real run", "A100" in adv_cheap)
check("advice warns when the GPU is below the live threshold",
      "live" in adv_cheap.lower())
check("advice is safe with no GPU", len(budget_advice({"available": False})) >= 1)

# --- full CLI run -----------------------------------------------------------
tmp = tempfile.mkdtemp(prefix="glasses3d-recon-")
video = os.path.join(tmp, "clip.mp4")
vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 24, (320, 240))
for i in range(48):
    vw.write(textured(i))
vw.release()

out_dir = os.path.join(tmp, "out")
proc = subprocess.run(
    [sys.executable, os.path.join(ROOT, "server", "reconstruct.py"),
     "--video", video, "--out", out_dir, "--backend", "mock",
     "--views", "8", "--width", "0"],
    capture_output=True, text=True, timeout=180)
check("CLI exits cleanly", proc.returncode == 0)
for name in ("points.ply", "scene.ply", "scene.json", "meta.json"):
    check("CLI wrote %s" % name, os.path.exists(os.path.join(out_dir, name)))

if os.path.exists(os.path.join(out_dir, "scene.ply")):
    h = read_ply_header(os.path.join(out_dir, "scene.ply"))
    check("CLI output is a valid 3DGS PLY", "rot_3" in h["properties"] and h["count"] > 0)

failed = 0
for label, ok in results:
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    failed += 0 if ok else 1
print("\n%d passed, %d failed" % (len(results) - failed, failed))
raise SystemExit(1 if failed else 0)
