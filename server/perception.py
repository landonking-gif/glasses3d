"""Object perception: what is in the scene, where it is, and how it is oriented.

Two stages, deliberately decoupled because they have very different costs:

    DETECT   SAM 3 — open-vocabulary. Prompt it with a noun phrase ("pencil",
             "coffee mug") and it returns a mask and a persistent track ID for
             every instance, in images and across video frames.

    POSE     FoundationPose — model-free 6DoF pose estimation and tracking for
             novel objects, from a handful of reference views. No CAD model
             needed, which matters because you will not have CAD for your own
             pencil.

The rate split is the point. Detection is expensive and only needs to run often
enough to notice objects entering or leaving the scene; pose tracking is cheap
and needs to run continuously so motion looks smooth. SAM 3's persistent track
IDs are what make this safe — they carry object identity between detections, so
the pose tracker never loses which object it is following.

Neither model runs on Apple Silicon. `MockDetector` / `MockPoseEstimator`
exercise every code path here without CUDA, so the wiring and the world-model
gate can be validated before renting a GPU.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from worldmodel import ChangeEvent, WorldModel


@dataclass
class Detection:
    """One detected instance in one frame."""

    track_id: int  # stable across frames — this is the identity anchor
    label: str
    confidence: float
    bbox: tuple  # (x, y, w, h) in pixels
    mask: Optional[np.ndarray] = None  # bool, HxW

    @property
    def area(self) -> int:
        if self.mask is not None:
            return int(self.mask.sum())
        return int(self.bbox[2] * self.bbox[3])


class ObjectDetector(abc.ABC):
    """Open-vocabulary detector with persistent track IDs."""

    @abc.abstractmethod
    def detect(self, image: np.ndarray, prompts: Sequence[str]) -> List[Detection]:
        ...


class PoseEstimator(abc.ABC):
    """6DoF pose for a detected instance."""

    @abc.abstractmethod
    def register(self, track_id: int, image: np.ndarray, det: Detection) -> bool:
        """Bank reference views so a novel object can be tracked without CAD."""

    @abc.abstractmethod
    def estimate(self, track_id: int, image: np.ndarray,
                 det: Detection) -> Optional[tuple]:
        """Return (4x4 object-to-camera pose, confidence), or None."""


class SAM3Detector(ObjectDetector):
    """SAM 3 adapter. Requires CUDA — see README 'Hardware'.

    Deliberately thin. The value is in the contract, not the wrapper: whatever
    open-vocabulary detector ships next slots in behind the same interface, and
    nothing downstream changes.
    """

    def __init__(self, checkpoint: str = "facebook/sam3", device: str = "cuda"):
        self.checkpoint = checkpoint
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            raise NotImplementedError(
                "SAM3Detector needs a CUDA box. Install per "
                "huggingface.co/facebook/sam3 and wire it here; the interface "
                "above is what the rest of the pipeline depends on."
            )
        return self._model

    def detect(self, image: np.ndarray, prompts: Sequence[str]) -> List[Detection]:
        model = self._load()
        raise NotImplementedError  # pragma: no cover


class FoundationPoseEstimator(PoseEstimator):
    """FoundationPose adapter. Requires CUDA.

    Model-free mode: register a few reference views of the object, then track.
    NVIDIA states the model is cleared for commercial use, unlike the MASt3R
    checkpoints used for camera tracking — worth keeping straight if this ships.
    """

    def __init__(self, device: str = "cuda", min_reference_views: int = 4):
        self.device = device
        self.min_reference_views = min_reference_views
        self._refs: Dict[int, List[np.ndarray]] = {}

    def register(self, track_id: int, image: np.ndarray, det: Detection) -> bool:
        refs = self._refs.setdefault(track_id, [])
        if det.mask is not None:
            refs.append(image)
        return len(refs) >= self.min_reference_views

    def estimate(self, track_id: int, image: np.ndarray, det: Detection):
        raise NotImplementedError(
            "FoundationPoseEstimator needs a CUDA box. See "
            "github.com/NVlabs/FoundationPose."
        )


# --- mocks: exercise the full pipeline with no GPU -------------------------


class MockDetector(ObjectDetector):
    """Replays a scripted detection sequence, one entry per frame."""

    def __init__(self, script: Sequence[Sequence[Detection]]):
        self.script = list(script)
        self.calls = 0

    def detect(self, image: np.ndarray, prompts: Sequence[str]) -> List[Detection]:
        out = self.script[self.calls] if self.calls < len(self.script) else []
        self.calls += 1
        return [d for d in out if not prompts or d.label in prompts]


class MockPoseEstimator(PoseEstimator):
    """Returns scripted poses, and models the registration requirement."""

    def __init__(self, poses: Dict[int, Sequence[tuple]], min_reference_views: int = 2):
        self.poses = {k: list(v) for k, v in poses.items()}
        self.min_reference_views = min_reference_views
        self._refs: Dict[int, int] = {}
        self._idx: Dict[int, int] = {}

    def register(self, track_id: int, image: np.ndarray, det: Detection) -> bool:
        self._refs[track_id] = self._refs.get(track_id, 0) + 1
        return self._refs[track_id] >= self.min_reference_views

    def estimate(self, track_id: int, image: np.ndarray, det: Detection):
        if self._refs.get(track_id, 0) < self.min_reference_views:
            return None
        seq = self.poses.get(track_id)
        if not seq:
            return None
        i = min(self._idx.get(track_id, 0), len(seq) - 1)
        self._idx[track_id] = i + 1
        return seq[i]


# --- pipeline --------------------------------------------------------------


@dataclass
class PerceptionConfig:
    prompts: Sequence[str] = field(default_factory=lambda: ["pencil", "mug", "book"])
    detect_every: int = 15  # frames between full detections
    min_detection_confidence: float = 0.5
    min_area_px: int = 400  # reject specks; they track terribly


class PerceptionPipeline:
    """Wires detection and pose estimation into world-model updates.

    Emits `ChangeEvent`s — the same stream a viewer consumes to move geometry.
    """

    def __init__(self, detector: ObjectDetector, poser: PoseEstimator,
                 world: WorldModel, config: Optional[PerceptionConfig] = None):
        self.detector = detector
        self.poser = poser
        self.world = world
        self.config = config or PerceptionConfig()
        self._tracked: Dict[int, str] = {}  # SAM track_id -> world object id
        self._last_dets: List[Detection] = []
        self.detections_run = 0

    def _obj_id(self, det: Detection) -> str:
        return "%s-%d" % (det.label.replace(" ", "_"), det.track_id)

    def process(self, image: np.ndarray, seq: int) -> List[ChangeEvent]:
        cfg = self.config
        events: List[ChangeEvent] = []

        # Detection is expensive; pose tracking is not. Re-detect periodically to
        # catch objects entering or leaving, and reuse the last detections in
        # between — persistent track IDs make that identity-safe.
        if seq % cfg.detect_every == 0 or not self._last_dets:
            self._last_dets = [
                d for d in self.detector.detect(image, cfg.prompts)
                if d.confidence >= cfg.min_detection_confidence
                and d.area >= cfg.min_area_px
            ]
            self.detections_run += 1

        for det in self._last_dets:
            obj_id = self._obj_id(det)

            # A novel object must be seen from several angles before its pose is
            # meaningful. Registering and estimating in the same frame yields a
            # pose fit to one view, which looks right from the camera and is
            # wrong in 3D.
            ready = self.poser.register(det.track_id, image, det)
            if not ready:
                continue

            result = self.poser.estimate(det.track_id, image, det)
            if result is None:
                continue
            pose, confidence = result

            if det.track_id not in self._tracked:
                added = self.world.add_object(obj_id, det.label,
                                              "geo/%s" % obj_id, pose, seq)
                if added is None:
                    # Refused — deformable label, or already registered under a
                    # previous track. Record the mapping anyway so the next
                    # frame does not retry the same rejected add forever.
                    self._tracked[det.track_id] = obj_id
                    continue
                self._tracked[det.track_id] = obj_id
                events.append(added)
                continue

            moved = self.world.observe(obj_id, pose, confidence, seq)
            if moved is not None:
                events.append(moved)

        events.extend(self.world.sweep(seq))
        return events
