"""Camera intrinsics and runtime undistortion (M2).

The glasses use a 12MP ultrawide lens with strong barrel distortion. Feeding
distorted frames to a SLAM system produces warped geometry and drifting scale,
so this is the highest quality-per-effort step in the whole pipeline.

Two models are supported:

    "fisheye"  cv2.fisheye — correct for the ultrawide, use this
    "pinhole"  cv2 standard — fallback if fisheye calibration fails to converge

Intrinsics are stored per *stream resolution*. The DAT quality ladder can hand
you 720x1280, 504x896, or 360x640, and a K matrix calibrated at one resolution
is wrong at another. `Undistorter` rescales rather than silently misapplying.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import cv2
import numpy as np

DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calib", "intrinsics.json"
)


@dataclass
class Intrinsics:
    model: str  # "fisheye" | "pinhole"
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list  # 4 coeffs for fisheye, 5 for pinhole
    rms: float = 0.0  # reprojection error from calibration, in pixels

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def D(self) -> np.ndarray:
        return np.array(self.dist, dtype=np.float64).reshape(-1, 1)

    def scaled_to(self, width: int, height: int) -> "Intrinsics":
        """Rescale intrinsics to a different stream resolution.

        Valid because the DAT ladder changes resolution by resampling the same
        sensor crop — the field of view is unchanged, so focal length and
        principal point scale linearly and distortion coefficients (which are
        defined in normalized coordinates) carry over untouched.
        """
        if (width, height) == (self.width, self.height):
            return self
        sx = width / float(self.width)
        sy = height / float(self.height)
        return Intrinsics(
            model=self.model, width=width, height=height,
            fx=self.fx * sx, fy=self.fy * sy,
            cx=self.cx * sx, cy=self.cy * sy,
            dist=list(self.dist), rms=self.rms,
        )

    def save(self, path: str = DEFAULT_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> "Intrinsics":
        with open(path) as fh:
            return cls(**json.load(fh))


class Undistorter:
    """Applies undistortion with cached remap tables.

    Building the remap tables costs milliseconds; applying them costs
    microseconds. At 24fps that difference decides whether undistortion fits in
    the frame budget, so tables are cached per resolution and only rebuilt when
    the incoming resolution actually changes.
    """

    def __init__(self, intr: Intrinsics, balance: float = 0.0):
        self.base = intr
        self.balance = balance
        self._maps: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._for_size: Optional[Tuple[int, int]] = None
        self.new_K: Optional[np.ndarray] = None

    def _build(self, width: int, height: int) -> None:
        intr = self.base.scaled_to(width, height)
        size = (width, height)
        if intr.model == "fisheye":
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                intr.K, intr.D, size, np.eye(3), balance=self.balance
            )
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(
                intr.K, intr.D, np.eye(3), new_K, size, cv2.CV_16SC2
            )
        else:
            new_K, _ = cv2.getOptimalNewCameraMatrix(
                intr.K, intr.D, size, self.balance, size
            )
            m1, m2 = cv2.initUndistortRectifyMap(
                intr.K, intr.D, np.eye(3), new_K, size, cv2.CV_16SC2
            )
        self._maps = (m1, m2)
        self._for_size = size
        self.new_K = new_K

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if self._for_size != (w, h):
            self._build(w, h)
            print("[calib] built undistort maps for %dx%d (model=%s)"
                  % (w, h, self.base.model))
        m1, m2 = self._maps  # type: ignore[misc]
        return cv2.remap(image, m1, m2, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)


def identity_for(width: int, height: int) -> Intrinsics:
    """Placeholder intrinsics for before calibration exists.

    Assumes a ~90 degree horizontal FOV and zero distortion. Good enough to get
    M1 running end to end; it will produce visibly warped geometry in M3, which
    is exactly the signal that M2 is not optional.
    """
    f = width / (2.0 * np.tan(np.deg2rad(90.0) / 2.0))
    return Intrinsics(model="pinhole", width=width, height=height,
                      fx=f, fy=f, cx=width / 2.0, cy=height / 2.0,
                      dist=[0.0] * 5, rms=-1.0)
