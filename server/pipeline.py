"""The live loop: frames in, a continuously-updating 3D world out.

Every stage before this was built and tested in isolation. This is what runs
them together:

    frames -> undistort -> camera tracking -> keyframe gate ---> densify
                                |                                   |
                                +------> perception -> world model <-+
                                                          |
                                                     scene.ply + scene.json

The organising constraint is that these stages have wildly different costs and
must not wait on each other:

    per frame     camera tracking, object pose tracking     (must keep up)
    every ~15     object detection                          (expensive)
    occasional    densification                             (very expensive)
    throttled     writing artifacts to disk                 (I/O)

Densification therefore runs on a worker thread and the loop never blocks on
it. If a pass is still running when the next is due, the new one is **dropped
rather than queued** — a backlog would produce geometry describing where the
camera was a minute ago, and stale geometry is worse than less geometry. Same
reasoning as the phone relay dropping frames instead of buffering them.

Runs end-to-end with mocks (no GPU). Swapping in the CUDA backends changes
which objects are constructed, not the shape of the loop.
"""

from __future__ import annotations

import abc
import asyncio
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from backends import ReconstructionResult, Reconstructor
from export import Splats, write_point_cloud, write_scene_graph, write_splats
from perception import PerceptionPipeline
from tracker import KeyframeBuffer, Pose
from worldmodel import ChangeEvent, WorldModel


class CameraTracker(abc.ABC):
    """Per-frame camera pose. MASt3R-SLAM sits behind this on a CUDA box."""

    @abc.abstractmethod
    def track(self, image: np.ndarray, seq: int) -> Tuple[Optional[Pose], float]:
        """Return (pose, confidence). A None pose means tracking is lost."""


class MASt3RSLAMTracker(CameraTracker):
    """MASt3R-SLAM adapter. CUDA only.

    Licence note: the MASt3R checkpoints are CC BY-NC-ND — non-commercial. This
    is the only stage in the pipeline carrying that restriction, so it is the
    one to replace before shipping anything.
    """

    def __init__(self, config: str = "config/base.yaml", device: str = "cuda"):
        self.config = config
        self.device = device

    def track(self, image, seq):
        raise NotImplementedError(
            "MASt3RSLAMTracker needs a CUDA box — see github.com/rmurai0610/MASt3R-SLAM. "
            "The interface above is what the pipeline depends on."
        )


class MockCameraTracker(CameraTracker):
    """Replays a scripted trajectory so the loop is testable without a GPU."""

    def __init__(self, poses: Sequence[Pose], confidence: float = 0.9,
                 lose_between: Tuple[int, int] = (-1, -1)):
        self.poses = list(poses)
        self.confidence = confidence
        self.lose_between = lose_between   # [lo, hi) frames where tracking fails

    def track(self, image, seq):
        lo, hi = self.lose_between
        if lo <= seq < hi:
            return None, 0.0
        return self.poses[min(seq, len(self.poses) - 1)], self.confidence


@dataclass
class PipelineConfig:
    densify_every_n_keyframes: int = 8
    min_keyframes_to_densify: int = 4
    export_every_seconds: float = 5.0
    out_dir: str = "out"
    splat_radius: float = 0.012
    max_points: int = 800_000
    min_confidence: float = 0.5
    write_artifacts: bool = True


@dataclass
class PipelineStats:
    frames: int = 0
    tracked: int = 0
    lost: int = 0
    keyframes: int = 0
    densify_runs: int = 0
    densify_dropped: int = 0
    densify_failed: int = 0
    exports: int = 0
    events: List[ChangeEvent] = field(default_factory=list)

    def summary(self) -> str:
        return ("%d frames | %d tracked, %d lost | %d keyframes | "
                "%d densify (%d dropped, %d failed) | %d exports | %d world events"
                % (self.frames, self.tracked, self.lost, self.keyframes,
                   self.densify_runs, self.densify_dropped, self.densify_failed,
                   self.exports, len(self.events)))


class LivePipeline:
    """Runs the stages together without letting the slow ones stall the fast ones."""

    def __init__(self, tracker: CameraTracker, reconstructor: Reconstructor,
                 world: WorldModel, perception: Optional[PerceptionPipeline] = None,
                 config: Optional[PipelineConfig] = None):
        self.tracker = tracker
        self.reconstructor = reconstructor
        self.world = world
        self.perception = perception
        self.config = config or PipelineConfig()
        self.keyframes = KeyframeBuffer()
        self.stats = PipelineStats()
        self.background: Optional[Splats] = None

        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="densify")
        self._densify_future: Optional[Future] = None
        self._since_densify = 0
        self._last_export = 0.0
        self._closed = False

    # -- per frame ---------------------------------------------------------

    def step(self, image: np.ndarray, seq: int) -> List[ChangeEvent]:
        """Process one frame. Returns whatever world changes it caused."""
        self.stats.frames += 1
        events: List[ChangeEvent] = []

        pose, confidence = self.tracker.track(image, seq)
        if pose is None:
            # Tracking lost. Everything downstream is anchored to camera pose,
            # so continuing would place real geometry at a fictional location —
            # worse than a gap, and far harder to spot afterwards.
            self.stats.lost += 1
            self._collect_densify_result()
            return events
        self.stats.tracked += 1

        if self.keyframes.maybe_add(seq, pose, image, confidence) is not None:
            self.stats.keyframes += 1
            self._since_densify += 1
            self._maybe_densify()

        if self.perception is not None:
            events.extend(self.perception.process(image, seq))

        self._collect_densify_result()
        self._maybe_export()
        self.stats.events.extend(events)
        return events

    # -- densification, off the hot path -----------------------------------

    def _maybe_densify(self) -> None:
        cfg = self.config
        if self._since_densify < cfg.densify_every_n_keyframes:
            return
        if len(self.keyframes.frames) < cfg.min_keyframes_to_densify:
            return

        if self._densify_future is not None and not self._densify_future.done():
            # Busy. Drop this pass rather than queue it.
            self.stats.densify_dropped += 1
            self._since_densify = 0
            return

        # Snapshot the image references before handing them to the worker. The
        # buffer evicts entries as it fills, so iterating it from another thread
        # would race with the main loop; the images themselves are never
        # mutated, so copying the references is enough.
        views = [kf.image for kf in self.keyframes.frames]
        self._since_densify = 0
        self.stats.densify_runs += 1
        self._densify_future = self._pool.submit(self._densify, views)

    def _densify(self, views: Sequence[np.ndarray]) -> Optional[Splats]:
        cfg = self.config
        result: ReconstructionResult = self.reconstructor.reconstruct(views)
        result = result.filtered(min_conf=cfg.min_confidence,
                                 max_points=cfg.max_points)
        if len(result.points) == 0:
            return None
        return Splats.from_points(result.points, result.colors,
                                  radius=cfg.splat_radius)

    def _collect_densify_result(self) -> None:
        """Adopt a finished densify pass, if there is one. Never blocks."""
        fut = self._densify_future
        if fut is None or not fut.done():
            return
        self._densify_future = None
        try:
            splats = fut.result()
        except Exception as exc:                        # noqa: BLE001
            # A failed densify must not kill the loop: camera tracking and
            # object updates stay useful, and the next pass may well succeed.
            self.stats.densify_failed += 1
            print("[pipeline] densify failed: %s" % exc)
            return
        if splats is not None:
            self.background = splats
            # Record it on the world itself rather than patching the dict on its
            # way to disk. Otherwise the file says one thing and an in-memory
            # `world.scene_graph()` says another - two sources of truth for the
            # same fact, and only one of them ever right.
            self.world.background_key = "scene.ply"

    # -- artifacts ---------------------------------------------------------

    def _maybe_export(self) -> None:
        if not self.config.write_artifacts:
            return
        now = time.time()
        if now - self._last_export < self.config.export_every_seconds:
            return
        self._last_export = now
        self.export()

    def export(self) -> dict:
        """Write the current world. Safe to call at any time."""
        cfg = self.config
        os.makedirs(cfg.out_dir, exist_ok=True)
        written = {}
        if self.background is not None and len(self.background):
            written["scene"] = write_splats(
                os.path.join(cfg.out_dir, "scene.ply"), self.background)
            # The point cloud is what the walkthrough viewer and Blender read,
            # so write it too rather than leaving only the splat file.
            written["points"] = write_point_cloud(
                os.path.join(cfg.out_dir, "points.ply"),
                self.background.means, _sh_to_rgb(self.background.sh_dc))
        written["graph"] = write_scene_graph(
            os.path.join(cfg.out_dir, "scene.json"), self.world.scene_graph())
        self.stats.exports += 1
        return written

    def close(self) -> None:
        """Drain the worker and adopt any last result. Idempotent."""
        if self._closed:
            return
        self._closed = True
        fut = self._densify_future
        if fut is not None:
            # Wait here specifically: shutdown is the one place blocking is
            # correct, since discarding an in-flight pass would throw away work
            # already paid for.
            try:
                fut.result(timeout=120)
            except Exception:                            # noqa: BLE001
                pass
        self._collect_densify_result()
        self._pool.shutdown(wait=True)

    # -- driving -----------------------------------------------------------

    async def run(self, source, undistort=None, max_frames: int = 0,
                  verbose: bool = True) -> PipelineStats:
        """Drive the loop from a FrameSource until it ends or the cap is hit."""
        try:
            async for frame in source.frames():
                image = undistort(frame.image) if undistort is not None else frame.image
                for event in self.step(image, frame.header.seq):
                    if verbose:
                        print("[world] %-8s %-12s %s" % (event.kind, event.label,
                                                         event.obj_id))
                if max_frames and self.stats.frames >= max_frames:
                    break
        finally:
            await source.close()
            self.close()
            if self.config.write_artifacts:
                self.export()
        return self.stats


def _sh_to_rgb(sh_dc: np.ndarray) -> np.ndarray:
    """Invert the SH degree-0 encoding back to 0..255 RGB."""
    from export import SH_C0
    return np.clip((0.5 + SH_C0 * sh_dc) * 255.0, 0, 255).astype(np.uint8)
