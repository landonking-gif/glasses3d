"""Dynamic scene state: a static background plus rigidly-tracked objects.

The requirement is that moving a pencil in the real world moves it in the 3D
world. The naive reading is "reconstruct continuously" — full 4D reconstruction,
where geometry itself is a function of time. That is enormously expensive,
needs multi-view coverage a single head-mounted camera never has, and spends all
its effort re-deriving the 95% of the room that did not change.

The decomposition used here instead:

    world(t) = static background  +  sum over objects of ( geometry_i @ T_i(t) )

Reconstruct the room once. Segment and track objects on top of it. When the
pencil moves, update one 4x4 matrix — the geometry is unchanged, because a
pencil is rigid. This turns a reconstruction problem into a pose-tracking
problem, which runs in real time and degrades gracefully.

The cost of the assumption: rigid objects only. A pencil, mug, book, or chair
works. Deforming things — a towel, a cat, your hands — are outside this model
and must be excluded rather than tracked badly. `DEFORMABLE_HINT` lists common
labels worth refusing.

This module is pure state and policy: no CUDA, no models, fully testable. The
perception that feeds it lives in `perception.py`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# Labels whose motion the rigid-transform model cannot represent. Tracking these
# produces confident nonsense — better to leave them in the static background or
# mask them out entirely.
DEFORMABLE_HINT = frozenset({
    "person", "hand", "hands", "arm", "cat", "dog", "towel", "cloth", "shirt",
    "curtain", "plant", "hair", "rope", "bag", "blanket",
})


def rotation_angle(A: np.ndarray, B: np.ndarray) -> float:
    """Geodesic angle in degrees between two rotation matrices."""
    rel = A.T @ B
    cos_theta = (np.trace(rel) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))


@dataclass
class Pose6D:
    """Object-to-world rigid transform."""

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> "Pose6D":
        return cls(R=np.array(T[:3, :3], dtype=np.float64),
                   t=np.array(T[:3, 3], dtype=np.float64))

    def to_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def distance_to(self, other: "Pose6D") -> Tuple[float, float]:
        """(metres, degrees) separating two poses."""
        return (float(np.linalg.norm(self.t - other.t)),
                rotation_angle(self.R, other.R))


# State machine. An object is SETTLED (at rest, world geometry is authoritative),
# MOVING (actively being displaced, stream pose updates), or MISSING (not
# observed recently — see WorldObject.state for why that is not the same as gone).
SETTLED, MOVING, MISSING = "settled", "moving", "missing"


@dataclass
class ChangeEvent:
    """Emitted when the 3D world needs to change. This is the viewer's input."""

    kind: str  # "added" | "moving" | "settled" | "missing" | "returned"
    obj_id: str
    label: str
    pose: np.ndarray  # 4x4
    seq: int
    ts: float
    translation: float = 0.0  # metres moved since the last settled pose
    rotation: float = 0.0  # degrees

    def to_dict(self) -> dict:
        return {"kind": self.kind, "obj_id": self.obj_id, "label": self.label,
                "pose": [round(v, 6) for v in self.pose.ravel().tolist()],
                "seq": self.seq, "ts": self.ts,
                "translation_m": round(self.translation, 4),
                "rotation_deg": round(self.rotation, 2)}


@dataclass
class WorldObject:
    obj_id: str
    label: str
    geometry_key: str
    settled_pose: Pose6D  # last pose we committed to the world
    live_pose: Pose6D  # most recent observation
    state: str = SETTLED
    last_seen_seq: int = 0
    observations: int = 0
    moved_count: int = 0
    _above: int = 0  # consecutive observations past the move-on threshold
    _below: int = 0  # consecutive observations under the move-off threshold


class WorldModel:
    """Holds scene state and decides when the 3D world should actually change.

    The interesting part is not storage, it is the *gate*. Raw 6DoF pose
    estimates jitter by millimetres and fractions of a degree every frame even
    for a motionless object. Piping that straight into the viewer makes the
    entire room shimmer, which reads as broken. Thresholding on a single value
    instead makes objects flicker between moving and still whenever the estimate
    sits near the boundary.

    So this uses a Schmitt trigger: a high threshold to *start* believing motion
    and a lower one to stop, each requiring several consecutive confirmations.
    Motion has to be decisive to register, and has to genuinely stop to clear.

        move_on_m / move_on_deg    displacement from the settled pose that
                                   begins a move, sustained for confirm_on frames
        move_off_m / move_off_deg  frame-to-frame delta small enough to count as
                                   at rest, sustained for confirm_off frames

    Defaults are tuned for desk-scale objects seen from ~1 m. A pencil nudged
    2 cm should register; breathing-level pose noise should not.
    """

    def __init__(self,
                 move_on_m: float = 0.02, move_on_deg: float = 6.0,
                 move_off_m: float = 0.005, move_off_deg: float = 2.0,
                 confirm_on: int = 2, confirm_off: int = 4,
                 missing_after: int = 45,
                 min_confidence: float = 0.35):
        self.move_on_m = move_on_m
        self.move_on_deg = move_on_deg
        self.move_off_m = move_off_m
        self.move_off_deg = move_off_deg
        self.confirm_on = confirm_on
        self.confirm_off = confirm_off
        self.missing_after = missing_after
        self.min_confidence = min_confidence
        self.objects: Dict[str, WorldObject] = {}
        self.background_key: Optional[str] = None
        self.rejected_deformable: List[str] = []

    # -- ingestion ---------------------------------------------------------

    def add_object(self, obj_id: str, label: str, geometry_key: str,
                   pose: np.ndarray, seq: int = 0) -> Optional[ChangeEvent]:
        """Register a newly reconstructed object. Refuses deformables."""
        if label.lower() in DEFORMABLE_HINT:
            # A rigid transform cannot represent this. Tracking it anyway would
            # produce a confident, wrong pose — worse than not tracking it.
            self.rejected_deformable.append(label)
            return None
        if obj_id in self.objects:
            return None
        p = Pose6D.from_matrix(pose)
        self.objects[obj_id] = WorldObject(
            obj_id=obj_id, label=label, geometry_key=geometry_key,
            settled_pose=p, live_pose=p, last_seen_seq=seq, observations=1,
        )
        return ChangeEvent("added", obj_id, label, p.to_matrix(), seq, time.time())

    def observe(self, obj_id: str, pose: np.ndarray, confidence: float,
                seq: int) -> Optional[ChangeEvent]:
        """Feed one 6DoF observation. Returns an event only if the world changed."""
        obj = self.objects.get(obj_id)
        if obj is None:
            return None

        # A low-confidence pose is worse than no pose: it moves real geometry to
        # a wrong place, and a misplaced object is much harder to notice later
        # than one that simply did not update.
        if confidence < self.min_confidence:
            return None

        new = Pose6D.from_matrix(pose)
        prev_live = obj.live_pose
        obj.live_pose = new
        obj.last_seen_seq = seq
        obj.observations += 1

        returned = None
        if obj.state == MISSING:
            obj.state = SETTLED
            obj._above = obj._below = 0
            returned = ChangeEvent("returned", obj_id, obj.label,
                                   new.to_matrix(), seq, time.time())

        d_settled, r_settled = obj.settled_pose.distance_to(new)
        d_step, r_step = prev_live.distance_to(new)

        if obj.state == SETTLED:
            if d_settled > self.move_on_m or r_settled > self.move_on_deg:
                obj._above += 1
                if obj._above >= self.confirm_on:
                    obj.state = MOVING
                    obj._above = obj._below = 0
                    obj.moved_count += 1
                    return ChangeEvent("moving", obj_id, obj.label,
                                       new.to_matrix(), seq, time.time(),
                                       d_settled, r_settled)
            else:
                obj._above = 0
            return returned

        # MOVING: stream updates, and watch for the object coming to rest.
        if d_step < self.move_off_m and r_step < self.move_off_deg:
            obj._below += 1
            if obj._below >= self.confirm_off:
                obj.state = SETTLED
                obj._below = 0
                d_total, r_total = obj.settled_pose.distance_to(new)
                obj.settled_pose = new
                return ChangeEvent("settled", obj_id, obj.label,
                                   new.to_matrix(), seq, time.time(),
                                   d_total, r_total)
        else:
            obj._below = 0
        return ChangeEvent("moving", obj_id, obj.label, new.to_matrix(),
                           seq, time.time(), d_settled, r_settled)

    def sweep(self, seq: int) -> List[ChangeEvent]:
        """Flag objects not seen recently.

        Deliberately *not* removed from the world. A head-mounted camera looks
        away constantly, and the overwhelmingly likely reason an object stopped
        being observed is that you turned your head — not that it vanished.
        Deleting it would make the room flicker as you glance around. It stays
        at its last settled pose, marked stale so the viewer can dim it.
        """
        events = []
        for obj in self.objects.values():
            if obj.state != MISSING and seq - obj.last_seen_seq > self.missing_after:
                obj.state = MISSING
                events.append(ChangeEvent("missing", obj.obj_id, obj.label,
                                          obj.settled_pose.to_matrix(), seq,
                                          time.time()))
        return events

    # -- output ------------------------------------------------------------

    def scene_graph(self) -> dict:
        """Serialisable world state — what a viewer loads or re-syncs against."""
        return {
            "background": self.background_key,
            "objects": [
                {
                    "id": o.obj_id,
                    "label": o.label,
                    "geometry": o.geometry_key,
                    "pose": [round(v, 6) for v in
                             (o.live_pose if o.state == MOVING
                              else o.settled_pose).to_matrix().ravel().tolist()],
                    "state": o.state,
                    "observations": o.observations,
                    "moves": o.moved_count,
                }
                for o in self.objects.values()
            ],
        }

    def active(self) -> Iterator[WorldObject]:
        return (o for o in self.objects.values() if o.state != MISSING)

    def __len__(self) -> int:
        return len(self.objects)
