"""Tests the dynamic-scene gate: does the 3D world change when — and only when —
the real world does?

The failure modes this guards against are the ones that make a live 3D scene
feel broken: a room that shimmers from pose noise, objects that flicker between
moving and still at the threshold, and geometry that teleports because a
low-confidence estimate was believed.

    python3 tools/test_worldmodel.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "server"))

from worldmodel import MISSING, MOVING, SETTLED, Pose6D, WorldModel  # noqa: E402

results = []


def check(label, condition):
    results.append((label, bool(condition)))


def T(x=0.0, y=0.0, z=0.0, deg=0.0, axis=(0, 0, 1)):
    """Build a 4x4 from a translation and an axis-angle rotation."""
    th = math.radians(deg)
    a = np.array(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [x, y, z]
    return M


def fresh(**kw):
    w = WorldModel(**kw)
    w.add_object("pencil-1", "pencil", "geo/pencil", T(0.0, 0.0, 0.8))
    return w


# --- an object at rest must not move the world -----------------------------
w = fresh()
rng = np.random.RandomState(3)
events = []
for seq in range(1, 60):
    jitter = rng.normal(0, 0.0015, 3)  # ~1.5 mm of pose noise, realistic
    e = w.observe("pencil-1", T(jitter[0], jitter[1], 0.8 + jitter[2]), 0.9, seq)
    if e:
        events.append(e)
check("pose noise on a still object emits no events", not events)
check("still object stays SETTLED", w.objects["pencil-1"].state == SETTLED)

# --- a real move must register ---------------------------------------------
w = fresh()
moving = [w.observe("pencil-1", T(0.0, 0.0, 0.8 + 0.01 * i), 0.9, i)
          for i in range(1, 12)]
moving = [e for e in moving if e]
check("a decisive move emits events", len(moving) > 0)
check("first move event is 'moving'", moving[0].kind == "moving")
check("object enters MOVING state", w.objects["pencil-1"].state == MOVING)

# --- and must settle when it stops -----------------------------------------
rest = T(0.0, 0.0, 0.95)
settled = None
for seq in range(12, 30):
    e = w.observe("pencil-1", rest, 0.9, seq)
    if e and e.kind == "settled":
        settled = e
        break
check("object settles once motion stops", settled is not None)
check("settle reports total displacement", settled and abs(settled.translation - 0.15) < 1e-6)
check("object returns to SETTLED", w.objects["pencil-1"].state == SETTLED)
check("move counted exactly once", w.objects["pencil-1"].moved_count == 1)

# --- rotation alone counts as motion ---------------------------------------
w = fresh()
rot = [w.observe("pencil-1", T(0, 0, 0.8, deg=4.0 * i), 0.9, i) for i in range(1, 8)]
check("pure rotation triggers a move", any(e and e.kind == "moving" for e in rot))

# --- no flapping at the boundary -------------------------------------------
# Hovering right at the move-on threshold must not toggle state repeatedly.
w = fresh(move_on_m=0.02)
flaps = 0
prev = SETTLED
for seq in range(1, 80):
    wobble = 0.0205 if seq % 2 else 0.0195  # straddles the threshold
    w.observe("pencil-1", T(0, 0, 0.8 + wobble), 0.9, seq)
    now = w.objects["pencil-1"].state
    if now != prev:
        flaps += 1
    prev = now
check("no flapping when hovering at the threshold", flaps <= 1)

# --- low-confidence poses are ignored --------------------------------------
w = fresh()
low = [w.observe("pencil-1", T(0, 0, 0.8 + 0.05 * i), 0.10, i) for i in range(1, 10)]
check("low-confidence observations are dropped", not any(low))
check("low-confidence leaves pose untouched",
      abs(w.objects["pencil-1"].live_pose.t[2] - 0.8) < 1e-9)

# --- deformables are refused, not tracked badly ----------------------------
w = WorldModel()
check("deformable label is refused", w.add_object("c", "cat", "geo/cat", T()) is None)
check("refusal is recorded", w.rejected_deformable == ["cat"])
check("rigid label is accepted", w.add_object("m", "mug", "geo/mug", T()) is not None)

# --- looking away must not delete the world --------------------------------
w = fresh(missing_after=10)
w.observe("pencil-1", T(0, 0, 0.8), 0.9, 1)
gone = w.sweep(seq=40)
check("unobserved object is marked missing", len(gone) == 1 and gone[0].kind == "missing")
check("missing object is NOT deleted", "pencil-1" in w.objects)
check("missing object keeps its last settled pose",
      abs(w.objects["pencil-1"].settled_pose.t[2] - 0.8) < 1e-9)
back = w.observe("pencil-1", T(0, 0, 0.8), 0.9, 41)
check("re-observation emits 'returned'", back is not None and back.kind == "returned")
check("returned object is SETTLED again", w.objects["pencil-1"].state == SETTLED)

# --- non-finite poses must be refused, not absorbed -------------------------
# NaN is worse than a missing pose and silently so: every comparison against it
# is False, so the object would never register as moving and never settle - it
# would freeze in place with nothing logged anywhere.
w = WorldModel()
nan_pose = T(); nan_pose[0, 3] = float("nan")
inf_pose = T(); inf_pose[2, 3] = float("inf")
check("add_object refuses a NaN pose", w.add_object("m", "mug", "g", nan_pose) is None)
check("nothing was registered from it", len(w) == 0)
w.add_object("m", "mug", "g", T(0, 0, 0.8))
check("observe refuses a NaN pose", w.observe("m", nan_pose, 0.9, 1) is None)
check("observe refuses an Inf pose", w.observe("m", inf_pose, 0.9, 2) is None)
check("the stored pose survived the refusals",
      np.isfinite(w.objects["m"].live_pose.t).all())

# --- Pose6D round-trip ------------------------------------------------------
m = T(0.1, -0.2, 0.3, deg=33.0, axis=(1, 1, 0))
check("Pose6D round-trips a 4x4", np.allclose(Pose6D.from_matrix(m).to_matrix(), m))
d, r = Pose6D.from_matrix(T()).distance_to(Pose6D.from_matrix(m))
check("distance_to recovers translation", abs(d - np.linalg.norm([0.1, -0.2, 0.3])) < 1e-9)
check("distance_to recovers rotation", abs(r - 33.0) < 1e-6)

# --- scene graph ------------------------------------------------------------
w = fresh()
w.background_key = "geo/room"
g = w.scene_graph()
check("scene graph names the background", g["background"] == "geo/room")
check("scene graph carries a flat 16-value pose", len(g["objects"][0]["pose"]) == 16)
check("scene graph reports state", g["objects"][0]["state"] == SETTLED)

# --- report -----------------------------------------------------------------
failed = 0
for label, ok in results:
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    failed += 0 if ok else 1
print("\n%d passed, %d failed" % (len(results) - failed, failed))
raise SystemExit(1 if failed else 0)
