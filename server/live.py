"""Run the live loop as a command.

    # no hardware, no GPU - proves the whole path
    python3 server/live.py --source mock --backend mock

    # glasses streaming through the phone relay, real models
    python3 server/live.py --source ws --backend mapanything --track mast3r

    # a recorded clip, replayed at native speed
    python3 server/live.py --source file --path clip.mp4 --realtime

Writes `scene.ply`, `points.ply` and `scene.json` into --out as it goes, so the
walkthrough viewer can be pointed at the output folder while this is still
running. Ctrl-C exits cleanly and writes a final export.

`pipeline.py` holds the loop itself; this file only builds the objects and hands
them over. Choosing mock or real backends is the only difference between the
two commands above.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backends import APACHE_MODEL, MockReconstructor, describe_gpu  # noqa: E402
from calibration import DEFAULT_PATH, Intrinsics, Undistorter  # noqa: E402
from perception import (MockDetector, MockPoseEstimator, PerceptionConfig,  # noqa: E402
                        PerceptionPipeline)
from pipeline import (LivePipeline, MASt3RSLAMTracker, MockCameraTracker,  # noqa: E402
                      PipelineConfig)
from sources import MockSource, VideoFileSource, WebSocketSource  # noqa: E402
from tracker import Pose  # noqa: E402
from worldmodel import WorldModel  # noqa: E402


def build_source(args):
    if args.source == "ws":
        return WebSocketSource(host=args.host, port=args.port)
    if args.source == "file":
        if not args.path:
            raise SystemExit("--source file requires --path")
        return VideoFileSource(args.path, realtime=args.realtime)
    return MockSource(count=args.frames or 240, width=args.width, height=args.height)


def build_tracker(args):
    if args.track == "mast3r":
        return MASt3RSLAMTracker()
    # A straight-line walk at roughly 1 m/s. Enough baseline to exercise the
    # keyframe gate; obviously not a real trajectory.
    n = max(args.frames or 240, 1)
    return MockCameraTracker([Pose(R=np.eye(3), t=np.array([i * 0.04, 0.0, 0.0]))
                              for i in range(n)])


def build_reconstructor(args):
    if args.backend == "mock":
        return MockReconstructor(points_per_view=400)
    from backends import MapAnythingReconstructor
    return MapAnythingReconstructor(model_id=args.model)


def build_perception(args, world):
    if not args.detect:
        return None
    prompts = [p.strip() for p in args.detect.split(",") if p.strip()]
    if args.backend == "mock":
        # Nothing to detect in synthetic frames; keep the stage wired but inert
        # rather than pretending it found something.
        return PerceptionPipeline(MockDetector([]), MockPoseEstimator({}), world,
                                  PerceptionConfig(prompts=prompts))
    from perception import FoundationPoseEstimator, SAM3Detector
    return PerceptionPipeline(SAM3Detector(), FoundationPoseEstimator(), world,
                              PerceptionConfig(prompts=prompts,
                                               detect_every=args.detect_every))


async def main_async(args) -> int:
    real = args.backend != "mock" or args.track == "mast3r"
    if real:
        gpu = describe_gpu()
        if not gpu.get("available"):
            print("[gpu] %s" % gpu.get("reason"))
            print("      Real backends need CUDA. Use --backend mock to test the path.")
            return 2
        print("[gpu] %s | %.1f GB | est. SLAM %.1f fps%s"
              % (gpu["name"], gpu["vram_gb"], gpu["expected_slam_fps"],
                 "" if gpu["live_viable"] else "  <- below the ~8 fps live threshold"))

    undistort = None
    if os.path.exists(args.calib) and not args.no_undistort:
        intr = Intrinsics.load(args.calib)
        undistort = Undistorter(intr)
        print("[calib] %s intrinsics, RMS %.3f px" % (intr.model, intr.rms))
    else:
        print("[calib] no intrinsics at %s - geometry will be warped" % args.calib)

    world = WorldModel()
    pipeline = LivePipeline(
        tracker=build_tracker(args),
        reconstructor=build_reconstructor(args),
        world=world,
        perception=build_perception(args, world),
        config=PipelineConfig(out_dir=args.out,
                              densify_every_n_keyframes=args.densify_every,
                              export_every_seconds=args.export_every,
                              max_points=args.max_points))

    source = build_source(args)
    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    print("[live] writing to %s - point the viewer at it while this runs" % args.out)
    print("[live] Ctrl-C to stop\n")
    t0 = time.time()

    # Race the frame iterator against the stop event. Checking a flag inside
    # `async for` only runs once a frame has already arrived, so an idle source
    # would ignore Ctrl-C entirely - the same bug that had to be fixed in
    # ingest.py.
    frames = source.frames().__aiter__()
    stop_task = asyncio.ensure_future(stop.wait())
    try:
        while True:
            next_task = asyncio.ensure_future(frames.__anext__())
            done, _ = await asyncio.wait({next_task, stop_task},
                                         return_when=asyncio.FIRST_COMPLETED)
            if stop_task in done:
                next_task.cancel()
                break
            try:
                frame = next_task.result()
            except StopAsyncIteration:
                break

            image = undistort(frame.image) if undistort is not None else frame.image
            for event in pipeline.step(image, frame.header.seq):
                print("[world] %-8s %-12s %s" % (event.kind, event.label, event.obj_id))

            if pipeline.stats.frames % args.report_every == 0:
                elapsed = max(time.time() - t0, 1e-6)
                print("[live] %5.1f fps | %s"
                      % (pipeline.stats.frames / elapsed, pipeline.stats.summary()))
    finally:
        stop_task.cancel()
        await source.close()
        pipeline.close()
        written = pipeline.export()
        print("\n=== final ===")
        print(pipeline.stats.summary())
        for k, v in written.items():
            print("  %-7s %s" % (k, v))
        if pipeline.background is None:
            print("\nNo geometry was produced. Too few keyframes, or every "
                  "densify pass failed - check the log above.")
        else:
            print("\nWalk through it: open viewer/walkthrough.html and drop "
                  "%s on it." % os.path.join(args.out, "points.ply"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["ws", "file", "mock"], default="mock")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--path", help="video path for --source file")
    ap.add_argument("--realtime", action="store_true", help="replay files at native fps")
    ap.add_argument("--frames", type=int, default=0, help="cap; 0 runs until the source ends")
    ap.add_argument("--width", type=int, default=720, help="mock source width")
    ap.add_argument("--height", type=int, default=1280, help="mock source height")

    ap.add_argument("--backend", choices=["mock", "mapanything"], default="mock")
    ap.add_argument("--model", default=APACHE_MODEL)
    ap.add_argument("--track", choices=["mock", "mast3r"], default="mock")
    ap.add_argument("--detect", default="", help='comma-separated prompts, e.g. "pencil,mug"')
    ap.add_argument("--detect-every", type=int, default=15)

    ap.add_argument("--out", default="out")
    ap.add_argument("--calib", default=DEFAULT_PATH)
    ap.add_argument("--no-undistort", action="store_true")
    ap.add_argument("--densify-every", type=int, default=8)
    ap.add_argument("--export-every", type=float, default=5.0)
    ap.add_argument("--max-points", type=int, default=800_000)
    ap.add_argument("--report-every", type=int, default=60)
    args = ap.parse_args()

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
