"""End-to-end test of the dynamic pipeline with mocked models.

Simulates the actual scenario: a pencil sits on a desk, then gets moved. Asserts
the 3D world learns about it, ignores the noise, follows the move, and settles.
Runs anywhere — the CUDA-only models are stubbed behind their interfaces.

    python3 tools/test_perception.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "server"))

from perception import (Detection, MockDetector, MockPoseEstimator,  # noqa: E402
                        PerceptionConfig, PerceptionPipeline)
from worldmodel import MOVING, SETTLED, WorldModel  # noqa: E402

results = []


def check(label, condition):
    results.append((label, bool(condition)))


def T(z, x=0.0):
    M = np.eye(4)
    M[:3, 3] = [x, 0.0, z]
    return M


def det(track_id=1, label="pencil", conf=0.9, area=40):
    mask = np.ones((area, area), bool)
    return Detection(track_id=track_id, label=label, confidence=conf,
                     bbox=(10, 10, area, area), mask=mask)


IMG = np.zeros((64, 64, 3), np.uint8)
N = 90

# The pencil holds still for 30 frames, is moved over the next 20, then rests.
poses = []
for i in range(N):
    if i < 30:
        z = 0.80 + (0.0012 * math.sin(i))          # pose jitter only
    elif i < 50:
        z = 0.80 + 0.010 * (i - 29)                # decisive move
    else:
        z = 0.80 + 0.010 * 21 + 0.0012 * math.sin(i)  # at rest again
    poses.append((T(z), 0.92))

pipeline = PerceptionPipeline(
    detector=MockDetector([[det()] for _ in range(N)]),
    poser=MockPoseEstimator({1: poses}, min_reference_views=2),
    world=WorldModel(),
    config=PerceptionConfig(prompts=["pencil"], detect_every=15, min_area_px=100),
)

events = []
for seq in range(N):
    events.extend(pipeline.process(IMG, seq))

kinds = [e.kind for e in events]
check("object was added to the world", "added" in kinds)
check("added exactly once", kinds.count("added") == 1)
check("motion was detected", "moving" in kinds)
check("object settled after motion stopped", "settled" in kinds)
check("settle happens after the move", kinds.index("moving") < kinds.index("settled"))

world = pipeline.world
obj = world.objects["pencil-1"]
check("final state is SETTLED", obj.state == SETTLED)
check("exactly one move was counted", obj.moved_count == 1)
check("final pose reflects the real displacement",
      abs(obj.settled_pose.t[2] - 1.01) < 0.02)

# Detection must be rate-limited; pose tracking must not be.
check("detection ran far less often than pose tracking",
      pipeline.detections_run <= (N // 15) + 1)
check("pose tracking ran every frame", obj.observations >= N - 5)

# No events at all during the still period beyond the initial add.
still = [e for e in events if e.seq < 28 and e.kind != "added"]
check("no spurious events while the object is still", not still)

# --- registration gate ------------------------------------------------------
p2 = PerceptionPipeline(
    detector=MockDetector([[det()] for _ in range(6)]),
    poser=MockPoseEstimator({1: [(T(0.8), 0.9)] * 6}, min_reference_views=5),
    world=WorldModel(),
    config=PerceptionConfig(prompts=["pencil"], detect_every=15, min_area_px=100),
)
early = [e for seq in range(4) for e in p2.process(IMG, seq)]
check("no pose is trusted before enough reference views", not early)
later = [e for seq in range(4, 6) for e in p2.process(IMG, seq)]
check("object registers once enough views are banked",
      any(e.kind == "added" for e in later))

# --- filters ----------------------------------------------------------------
p3 = PerceptionPipeline(
    detector=MockDetector([[det(conf=0.2)] for _ in range(4)]),
    poser=MockPoseEstimator({1: [(T(0.8), 0.9)] * 4}, min_reference_views=1),
    world=WorldModel(),
    config=PerceptionConfig(prompts=["pencil"], detect_every=15, min_area_px=100),
)
check("low-confidence detections are filtered out",
      not [e for seq in range(4) for e in p3.process(IMG, seq)])

p4 = PerceptionPipeline(
    detector=MockDetector([[det(area=5)] for _ in range(4)]),
    poser=MockPoseEstimator({1: [(T(0.8), 0.9)] * 4}, min_reference_views=1),
    world=WorldModel(),
    config=PerceptionConfig(prompts=["pencil"], detect_every=15, min_area_px=400),
)
check("tiny detections are filtered out",
      not [e for seq in range(4) for e in p4.process(IMG, seq)])

# --- deformables must not be retried forever --------------------------------
p5 = PerceptionPipeline(
    detector=MockDetector([[det(track_id=7, label="cat")] for _ in range(10)]),
    poser=MockPoseEstimator({7: [(T(0.8), 0.9)] * 10}, min_reference_views=1),
    world=WorldModel(),
    config=PerceptionConfig(prompts=["cat"], detect_every=15, min_area_px=100),
)
cat_events = [e for seq in range(10) for e in p5.process(IMG, seq)]
check("deformable object never enters the world", not cat_events)
check("deformable rejection is not retried every frame",
      len(p5.world.rejected_deformable) == 1)

failed = 0
for label, ok in results:
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    failed += 0 if ok else 1
print("\n%d passed, %d failed" % (len(results) - failed, failed))
raise SystemExit(1 if failed else 0)
