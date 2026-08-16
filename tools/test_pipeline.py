"""Tests the live loop (server/pipeline.py) with mocked models.

Each stage was tested in isolation already. What is only testable here is how
they behave *together*: that the slow stages never stall the fast ones, and that
a failure in one does not take the loop down.

    python3 tools/test_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "server"))

from backends import MockReconstructor, ReconstructionResult  # noqa: E402
from export import read_ply_header  # noqa: E402
from pipeline import LivePipeline, MockCameraTracker, PipelineConfig  # noqa: E402
from sources import MockSource  # noqa: E402
from tracker import Pose  # noqa: E402
from worldmodel import WorldModel  # noqa: E402

results = []
check = lambda label, cond: results.append((label, bool(cond)))  # noqa: E731

N = 60
TMP = tempfile.mkdtemp(prefix="glasses3d-pipe-")
IMG = [np.random.RandomState(i).randint(0, 255, (64, 64, 3)).astype(np.uint8)
       for i in range(N)]
# ~1 m/s at 24fps = 0.04 m/frame. Using a sprint-speed trajectory here makes
# every frame clear the keyframe threshold and quietly invalidates the test.
WALK = [Pose(R=np.eye(3), t=np.array([i * 0.04, 0.0, 0.0])) for i in range(N)]


def build(name, recon=None, tracker=None, world=None, perception=None, **cfg):
    return LivePipeline(
        tracker or MockCameraTracker(WALK),
        recon or MockReconstructor(points_per_view=80),
        world or WorldModel(), perception=perception,
        config=PipelineConfig(out_dir=os.path.join(TMP, name), **cfg))


def drain(p, n=N):
    for i in range(n):
        p.step(IMG[i], i)
    p.close()
    return p


# --- the loop produces a world ---------------------------------------------
p = drain(build("basic", densify_every_n_keyframes=3, min_keyframes_to_densify=2))
p.export()
out = os.path.join(TMP, "basic")
check("every frame processed", p.stats.frames == N)
check("all tracked, none lost", p.stats.tracked == N and p.stats.lost == 0)
check("keyframes are a subset at walking pace", 0 < p.stats.keyframes < N)
check("export is throttled, not per frame", p.stats.exports < N)
check("a background was produced", p.background is not None and len(p.background))
check("scene.ply is valid 3DGS",
      read_ply_header(os.path.join(out, "scene.ply"))["count"] > 0)
check("points.ply carries colour the viewer reads",
      read_ply_header(os.path.join(out, "points.ply"))["properties"]
      == ["x", "y", "z", "red", "green", "blue"])
check("memory and disk agree on the background",
      json.load(open(os.path.join(out, "scene.json")))["background"]
      == p.world.scene_graph()["background"] == "scene.ply")


# --- a slow densifier must not stall the loop ------------------------------
class Slow(MockReconstructor):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0

    def reconstruct(self, images, intrinsics=None, poses=None):
        self.calls += 1
        time.sleep(0.25)
        return super().reconstruct(images)


slow = Slow(points_per_view=40)
t0 = time.time()
p2 = drain(build("slow", recon=slow, densify_every_n_keyframes=1,
                 min_keyframes_to_densify=2))
elapsed = time.time() - t0
# Synchronous densification would cost roughly keyframes x 0.25s here.
check("loop never blocked on densification (%.2fs)" % elapsed, elapsed < 2.0)
check("overlapping passes were dropped, not queued", p2.stats.densify_dropped > 0)
check("dropped passes never executed", slow.calls == p2.stats.densify_runs)
check("a background still arrived despite drops", p2.background is not None)


# --- failures must not take the loop down ----------------------------------
class Broken(MockReconstructor):
    def reconstruct(self, images, intrinsics=None, poses=None):
        raise RuntimeError("CUDA out of memory")


p3 = drain(build("broken", recon=Broken(), densify_every_n_keyframes=2,
                 min_keyframes_to_densify=2))
check("loop survives a failing densifier", p3.stats.frames == N)
check("failures are counted", p3.stats.densify_failed > 0)
check("no background from a broken densifier", p3.background is None)


class Empty(MockReconstructor):
    def reconstruct(self, images, intrinsics=None, poses=None):
        return ReconstructionResult(np.zeros((0, 3)), np.zeros((0, 3), np.uint8),
                                    np.zeros(0), np.zeros((0, 4, 4)),
                                    np.zeros((0, 3, 3)), view_count=len(images))


p4 = drain(build("empty", recon=Empty(), densify_every_n_keyframes=2,
                 min_keyframes_to_densify=2), n=20)
check("an empty reconstruction is handled", p4.background is None)
check("export still writes the graph", "graph" in p4.export())
check("no scene.ply when there is nothing to write",
      not os.path.exists(os.path.join(TMP, "empty", "scene.ply")))
check("disk agrees there is no background",
      json.load(open(os.path.join(TMP, "empty", "scene.json")))["background"] is None)


# --- lost tracking ----------------------------------------------------------
# Everything downstream is anchored to camera pose, so a frame without one must
# contribute nothing rather than anchoring geometry to a fictional location.
p5 = drain(build("lost", tracker=MockCameraTracker(WALK, lose_between=(10, 30)),
                 densify_every_n_keyframes=3, min_keyframes_to_densify=2))
check("lost frames are counted", p5.stats.lost == 20)
check("tracked + lost accounts for every frame",
      p5.stats.tracked + p5.stats.lost == N)
check("the loop recovers when tracking returns", p5.stats.tracked > 30)
check("no keyframe admitted without a pose",
      all(not (10 <= kf.seq < 30) for kf in p5.keyframes.frames))


# --- async driving ----------------------------------------------------------
async def _drive():
    return await build("source", densify_every_n_keyframes=2,
                       min_keyframes_to_densify=2).run(
        MockSource(count=24, fps=200, width=64, height=64), verbose=False)


stats = asyncio.run(_drive())
check("run() drives the loop from a FrameSource", stats.frames == 24)
check("run() exports on the way out", stats.exports > 0)
check("stats summary is populated", "frames" in stats.summary())

p6 = build("idem", densify_every_n_keyframes=99)
p6.step(IMG[0], 0)
p6.close()
p6.close()
check("close() is idempotent", True)

failed = 0
for label, ok in results:
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    failed += 0 if ok else 1
print("\n%d passed, %d failed" % (len(results) - failed, failed))
raise SystemExit(1 if failed else 0)
