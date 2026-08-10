"""M3: keyframe selection between the live tracker and the async densifier.

This is the seam that keeps the pipeline real-time. MASt3R-SLAM runs at ~15 FPS
on an RTX 4090 and must not block; densification (WorldMirror / MapAnything) is
a heavy forward pass that must not run per-frame. Keyframe selection is the
valve between them, and it is the single most consequential tuning knob in the
system.

The MASt3R-SLAM binding itself lands once a CUDA box is available (see
README "Hardware"). Everything here is device-independent and testable now.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class Pose:
    """Camera-to-world transform."""

    R: np.ndarray  # 3x3 rotation
    t: np.ndarray  # 3, translation, metres

    @classmethod
    def identity(cls) -> "Pose":
        return cls(R=np.eye(3), t=np.zeros(3))

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> "Pose":
        return cls(R=np.array(T[:3, :3]), t=np.array(T[:3, 3]))

    def translation_to(self, other: "Pose") -> float:
        """Metres between two camera centres."""
        return float(np.linalg.norm(self.t - other.t))

    def rotation_to(self, other: "Pose") -> float:
        """Geodesic angle between two orientations, in degrees.

        Uses the trace identity: for a relative rotation Rrel, the rotation
        angle is arccos((trace(Rrel) - 1) / 2). Clamped because floating-point
        drift can push the argument just outside [-1, 1] and make arccos NaN —
        which would silently poison every downstream comparison.
        """
        rel = self.R.T @ other.R
        cos_theta = (np.trace(rel) - 1.0) / 2.0
        return float(math.degrees(math.acos(max(-1.0, min(1.0, cos_theta)))))


@dataclass
class Keyframe:
    seq: int
    pose: Pose
    image: np.ndarray
    sharpness: float
    confidence: float


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian — higher is sharper.

    Head motion produces markedly more motion blur than handheld camera work,
    and a blurred keyframe poisons densification more than a missing one does:
    the densifier will happily fit geometry to smeared pixels and produce
    confident, wrong surfaces. Scale is resolution- and content-dependent, so
    calibrate the threshold against your own captures rather than trusting an
    absolute number.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# YOUR CALL — see README "Your contribution"
# ---------------------------------------------------------------------------

# Starting points, not answers. Tune against a real capture.
TRANSLATION_M = 0.15   # loose ~0.30 | tight ~0.10
ROTATION_DEG = 10.0    # loose ~15.0 | tight ~5.0
MIN_SHARPNESS = 60.0
MIN_CONFIDENCE = 0.4


def select_keyframe(current_pose: Pose,
                    last_keyframe: Optional[Keyframe],
                    tracking_confidence: float,
                    current_sharpness: float) -> bool:
    """Decide whether the current frame becomes a keyframe.

    Return True to send this frame downstream to the densifier.

    The trade-off, concretely:

      Loose thresholds (0.30 m / 15 deg) -> sparse keyframes, densifier keeps up
        easily, but fast head turns leave holes in the reconstruction because
        you moved past a region without ever banking a view of it.

      Tight thresholds (0.10 m / 5 deg) -> dense coverage and better geometry,
        but the densifier saturates. When it saturates the queue backs up, and a
        backed-up queue means the reconstruction lags where you are actually
        looking.

    Two more considerations specific to head-mounted capture:

      - Blur. `current_sharpness` (variance of Laplacian) is low on motion-blurred
        frames. A sharp frame slightly below the distance threshold usually beats
        a blurry one just above it.
      - Confidence. When SLAM tracking degrades, `tracking_confidence` drops and
        the pose itself is unreliable — admitting a keyframe with a bad pose
        misplaces real geometry, which is harder to detect later than a gap.

    The very first frame must always be admitted (`last_keyframe is None`),
    otherwise the reconstruction never starts.

    Roughly 5-10 lines.
    """
    # TODO(you): implement. Delete the line below when you do.
    raise NotImplementedError(
        "select_keyframe is yours to write — see server/tracker.py and the "
        "README section 'Your contribution'."
    )


# ---------------------------------------------------------------------------


@dataclass
class KeyframeBuffer:
    """Accumulates keyframes for the densifier, with a bounded capacity.

    Capacity is bounded because densification is slower than selection. When the
    buffer fills, the least useful keyframe is evicted rather than the oldest:
    dropping by age would preferentially discard the start of the capture, which
    is usually where the reconstruction is best anchored.
    """

    capacity: int = 256
    frames: List[Keyframe] = field(default_factory=list)
    evicted: int = 0
    considered: int = 0

    def maybe_add(self, seq: int, pose: Pose, image: np.ndarray,
                  confidence: float) -> Optional[Keyframe]:
        self.considered += 1
        sharp = sharpness(image)
        last = self.frames[-1] if self.frames else None
        if not select_keyframe(pose, last, confidence, sharp):
            return None
        kf = Keyframe(seq=seq, pose=pose, image=image,
                      sharpness=sharp, confidence=confidence)
        self.frames.append(kf)
        if len(self.frames) > self.capacity:
            self._evict()
        return kf

    def _evict(self) -> None:
        # Never evict the first frame — it anchors the world origin.
        scores = [(kf.sharpness * kf.confidence, i)
                  for i, kf in enumerate(self.frames) if i != 0]
        _, worst = min(scores)
        self.frames.pop(worst)
        self.evicted += 1

    @property
    def selection_rate(self) -> float:
        if not self.considered:
            return 0.0
        return (len(self.frames) + self.evicted) / float(self.considered)
