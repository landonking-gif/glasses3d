"""Checks the device-independent parts of the tracker (M3).

Pose math and the blur metric run anywhere; the MASt3R-SLAM binding needs CUDA.
The keyframe-buffer tests stay skipped until `select_keyframe` is implemented —
see README "Your contribution".

    python3 tools/test_tracker.py
"""

from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "server"))

from tracker import Keyframe, KeyframeBuffer, Pose, sharpness  # noqa: E402

results = []


def check(label, condition):
    results.append((label, bool(condition)))


def rot(axis, degrees):
    v = np.array(axis, dtype=np.float64)
    v = v / np.linalg.norm(v) * math.radians(degrees)
    R, _ = cv2.Rodrigues(v)
    return R


# --- Pose ------------------------------------------------------------------
a = Pose.identity()
b = Pose(R=np.eye(3), t=np.array([0.3, 0.4, 0.0]))
check("translation_to measures euclidean distance", abs(a.translation_to(b) - 0.5) < 1e-9)
check("translation is symmetric", abs(a.translation_to(b) - b.translation_to(a)) < 1e-9)

for deg in (5.0, 30.0, 90.0, 179.0):
    c = Pose(R=rot([0, 0, 1], deg), t=np.zeros(3))
    check("rotation_to recovers %.0f deg" % deg, abs(a.rotation_to(c) - deg) < 1e-6)

# Rotation about a tilted axis, to catch anything that only works on-axis.
d = Pose(R=rot([1, 1, 0], 42.0), t=np.zeros(3))
check("rotation_to handles off-axis rotation", abs(a.rotation_to(d) - 42.0) < 1e-6)

# The clamp matters: floating-point drift can push the arccos argument past 1
# and yield NaN, which would silently poison every downstream comparison.
drifted = np.eye(3) * (1.0 + 1e-12)
check("rotation_to survives non-orthonormal drift",
      not math.isnan(Pose(R=drifted, t=np.zeros(3)).rotation_to(a)))

check("from_matrix reads a 4x4 transform",
      np.allclose(Pose.from_matrix(np.block([
          [rot([0, 1, 0], 20.0), np.array([[1.0], [2.0], [3.0]])],
          [np.zeros((1, 3)), np.ones((1, 1))]])).t, [1.0, 2.0, 3.0]))

# --- sharpness -------------------------------------------------------------
rng = np.random.RandomState(0)
sharp = rng.randint(0, 255, (240, 240, 3)).astype(np.uint8)
blurred = cv2.GaussianBlur(sharp, (21, 21), 8)
check("sharpness ranks sharp above blurred", sharpness(sharp) > sharpness(blurred) * 2)
check("sharpness of flat image is ~0", sharpness(np.zeros((64, 64, 3), np.uint8)) < 1e-6)

# --- select_keyframe -------------------------------------------------------
from tracker import (MIN_CONFIDENCE, MIN_SHARPNESS, ROTATION_DEG,  # noqa: E402
                     TRANSLATION_M, Keyframe as KF, select_keyframe)

GOOD = MIN_SHARPNESS * 2  # comfortably sharp


def kf_at(pose, sharpness=None):
    return KF(seq=0, pose=pose, image=sharp,
              sharpness=GOOD if sharpness is None else sharpness, confidence=1.0)


origin = kf_at(Pose.identity())
check("first frame is always admitted", select_keyframe(Pose.identity(), None, 1.0, GOOD))
check("first frame admitted even when blurred and unconfident",
      select_keyframe(Pose.identity(), None, 0.0, 0.0))

near = Pose(R=np.eye(3), t=np.array([TRANSLATION_M * 0.2, 0, 0]))
far = Pose(R=np.eye(3), t=np.array([TRANSLATION_M * 3.0, 0, 0]))
check("a small move is rejected", not select_keyframe(near, origin, 1.0, GOOD))
check("a large move is admitted", select_keyframe(far, origin, 1.0, GOOD))

turned = Pose(R=rot([0, 1, 0], ROTATION_DEG * 3.0), t=np.zeros(3))
check("rotation alone can admit a keyframe", select_keyframe(turned, origin, 1.0, GOOD))
small_turn = Pose(R=rot([0, 1, 0], ROTATION_DEG * 0.2), t=np.zeros(3))
check("a small rotation is rejected", not select_keyframe(small_turn, origin, 1.0, GOOD))

# Rejection gates must dominate distance — a big move seen badly is still bad.
check("low confidence rejects even a large move",
      not select_keyframe(far, origin, MIN_CONFIDENCE * 0.5, GOOD))
check("blur rejects even a large move",
      not select_keyframe(far, origin, 1.0, MIN_SHARPNESS * 0.5))

# The sharpness discount: a crisp frame should clear a threshold that an
# otherwise identical marginal frame does not.
borderline = Pose(R=np.eye(3), t=np.array([TRANSLATION_M * 0.75, 0, 0]))
check("very sharp frames earn a threshold discount",
      select_keyframe(borderline, origin, 1.0, MIN_SHARPNESS * 100))
check("marginally sharp frames do not get the discount",
      not select_keyframe(borderline, origin, 1.0, MIN_SHARPNESS * 1.01))
# ...but the discount is clamped, so it cannot collapse to always-true.
tiny = Pose(R=np.eye(3), t=np.array([TRANSLATION_M * 0.5, 0, 0]))
check("the discount is clamped at 0.6",
      not select_keyframe(tiny, origin, 1.0, MIN_SHARPNESS * 1e6))

# --- KeyframeBuffer --------------------------------------------------------
buf = KeyframeBuffer(capacity=4)
first = buf.maybe_add(0, Pose.identity(), sharp, 1.0)
check("buffer admits the first frame", first is not None)
check("buffer rejects a near-identical second frame",
      buf.maybe_add(1, Pose.identity(), sharp, 1.0) is None)
check("buffer admits a frame after real movement",
      buf.maybe_add(2, Pose(R=np.eye(3), t=np.array([1.0, 0, 0])), sharp, 1.0) is not None)
check("buffer tracks how many frames it considered", buf.considered == 3)
check("selection rate is a sensible fraction", 0.0 < buf.selection_rate <= 1.0)

# Capacity is enforced, and eviction never drops the origin anchor.
big = KeyframeBuffer(capacity=3)
for i in range(12):
    big.maybe_add(i, Pose(R=np.eye(3), t=np.array([i * 1.0, 0, 0])), sharp, 1.0)
check("buffer respects its capacity", len(big.frames) <= 3)
check("buffer evicted something", big.evicted > 0)
check("origin frame survives eviction", big.frames[0].seq == 0)

# Eviction is independent of the stub, so it can be checked directly.
buf.frames = [Keyframe(seq=i, pose=Pose.identity(), image=sharp,
                       sharpness=float(10 * (i + 1)), confidence=1.0)
              for i in range(5)]
buf._evict()
check("eviction never drops the origin-anchoring first frame",
      buf.frames[0].seq == 0)
check("eviction drops the lowest-scoring frame", 1 not in [k.seq for k in buf.frames])

# --- report ----------------------------------------------------------------
failed = 0
for label, ok in results:
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    failed += 0 if ok else 1
print("\n%d passed, %d failed" % (len(results) - failed, failed))
raise SystemExit(1 if failed else 0)
