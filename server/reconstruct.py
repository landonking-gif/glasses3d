"""Offline reconstruction driver: video or frames in, scene.ply out.

    python3 server/reconstruct.py --video clip.mp4 --out out/ --backend mock
    python3 server/reconstruct.py --video clip.mp4 --out out/ --backend mapanything

This is the path that actually works on a Colab T4. Live tracking needs roughly
8+ fps to feel live and a T4 delivers a fraction of that, but offline
reconstruction does not care how long it takes — and it consumes the 3K recorded
clips, which carry about nine times the pixels of the 720p live stream. Better
input, no latency budget, no tunnel.

Reuses `select_keyframe` from the live tracker rather than reimplementing frame
selection, so both paths make the same decisions about what is worth
reconstructing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Sequence

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backends import (APACHE_MODEL, MockReconstructor, NC_MODEL,  # noqa: E402
                      Reconstructor, ReconstructionResult, describe_gpu)
from calibration import Intrinsics, Undistorter, DEFAULT_PATH  # noqa: E402
from export import Splats, write_point_cloud, write_splats, write_scene_graph  # noqa: E402
from tracker import KeyframeBuffer, Pose, sharpness  # noqa: E402


def load_frames(path: str, max_frames: int = 0, stride: int = 1) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("cannot open %s" % path)
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames


def pick_keyframes(frames: Sequence[np.ndarray], target: int = 32,
                   min_sharpness_pct: float = 35.0) -> List[int]:
    """Choose which frames to reconstruct from.

    Without camera poses (this runs before any SLAM) `select_keyframe`'s
    pose-delta logic has nothing to compare, so selection falls back to two
    signals that need no geometry:

      1. Sharpness — reject the blurriest frames outright. Head-mounted footage
         has a lot of them, and a blurred view contributes confident, wrong
         geometry rather than merely nothing.
      2. Even temporal spread over what survives — a feed-forward reconstructor
         wants baseline between views. Thirty consecutive frames from one second
         of standing still carry almost no more information than one frame.
    """
    scores = np.array([sharpness(f) for f in frames])
    if len(frames) <= target:
        floor = np.percentile(scores, min_sharpness_pct) if len(scores) else 0.0
        return [i for i in range(len(frames)) if scores[i] >= floor] or list(range(len(frames)))

    floor = np.percentile(scores, min_sharpness_pct)
    usable = [i for i in range(len(frames)) if scores[i] >= floor]
    if len(usable) <= target:
        return usable

    # Spread `target` picks evenly across the usable frames, taking the sharpest
    # frame within each bucket rather than the bucket's first frame.
    picks, buckets = [], np.array_split(np.array(usable), target)
    for bucket in buckets:
        if len(bucket):
            picks.append(int(bucket[np.argmax(scores[bucket])]))
    return sorted(set(picks))


def build_backend(name: str, model_id: str) -> Reconstructor:
    if name == "mock":
        return MockReconstructor()
    if name == "mapanything":
        from backends import MapAnythingReconstructor
        return MapAnythingReconstructor(model_id=model_id)
    raise SystemExit("unknown backend: %s" % name)


def run(args) -> int:
    t_start = time.time()
    gpu = describe_gpu()
    if args.backend != "mock":
        if not gpu.get("available"):
            print("[gpu] %s" % gpu.get("reason"))
            print("      Use --backend mock to exercise the pipeline without a GPU.")
            return 2
        print("[gpu] %s | %.1f GB | tier: %s | est. SLAM %.1f fps"
              % (gpu["name"], gpu["vram_gb"], gpu["tier"], gpu["expected_slam_fps"]))

    frames = load_frames(args.video, args.max_frames, args.stride)
    if not frames:
        raise SystemExit("no frames decoded from %s" % args.video)
    print("[input] %d frames at %dx%d" % (len(frames), frames[0].shape[1], frames[0].shape[0]))

    intr: Optional[Intrinsics] = None
    if os.path.exists(args.calib) and not args.no_undistort:
        intr = Intrinsics.load(args.calib)
        undistort = Undistorter(intr)
        frames = [undistort(f) for f in frames]
        print("[calib] undistorted with %s intrinsics (RMS %.3f px)" % (intr.model, intr.rms))
    else:
        print("[calib] no intrinsics at %s — reconstructing DISTORTED frames.\n"
              "        Geometry will be warped. Run server/calibrate.py first." % args.calib)

    idx = pick_keyframes(frames, target=args.views)
    views = [frames[i] for i in idx]
    print("[select] %d keyframes of %d frames" % (len(views), len(frames)))

    if args.width:
        # The model works at its own resolution; resizing here keeps the pixel
        # counts of image and prediction aligned so colours map correctly.
        h = int(round(views[0].shape[0] * args.width / views[0].shape[1]))
        h -= h % 14  # most ViT backbones need dimensions divisible by the patch size
        views = [cv2.resize(v, (args.width, h)) for v in views]
        print("[resize] views at %dx%d" % (args.width, h))

    K = None
    if intr is not None:
        scaled = intr.scaled_to(views[0].shape[1], views[0].shape[0])
        K = np.tile(scaled.K.astype(np.float32), (len(views), 1, 1))

    backend = build_backend(args.backend, args.model)
    print("[recon] running %s on %d views..." % (args.backend, len(views)))
    t0 = time.time()
    result: ReconstructionResult = backend.reconstruct(views, intrinsics=K)
    print("[recon] %.1fs — %s" % (time.time() - t0, result.summary()))

    result = result.filtered(min_conf=args.min_conf, max_points=args.max_points)
    print("[filter] %s" % result.summary())
    if len(result.points) == 0:
        print("[filter] everything was filtered out — lower --min-conf")
        return 1

    os.makedirs(args.out, exist_ok=True)
    cloud = write_point_cloud(os.path.join(args.out, "points.ply"),
                              result.points, result.colors)
    splats = Splats.from_points(result.points, result.colors,
                                radius=args.splat_radius)
    scene = write_splats(os.path.join(args.out, "scene.ply"), splats)
    graph = write_scene_graph(os.path.join(args.out, "scene.json"),
                              {"background": "scene.ply", "objects": []})
    meta = {
        "source": os.path.basename(args.video),
        "frames_decoded": len(frames),
        "views_used": len(views),
        "points": int(len(result.points)),
        "backend": args.backend,
        "model": args.model if args.backend != "mock" else None,
        "calibrated": intr is not None,
        "gpu": gpu.get("name"),
        "seconds": round(time.time() - t_start, 1),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\nwrote:\n  %s\n  %s\n  %s" % (cloud, scene, graph))
    print("\n%d splats in %.1fs. Open scene.ply at superspl.at/editor to check it."
          % (len(splats), meta["seconds"]))
    if len(splats) > 400_000:
        print("NOTE: %d splats exceeds the ~400k Quest budget for 72fps. "
              "Lower --max-points before targeting VR." % len(splats))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--backend", choices=["mock", "mapanything"], default="mapanything")
    ap.add_argument("--model", default=APACHE_MODEL,
                    help="Apache-2.0 weights by default; %s is CC-BY-NC" % NC_MODEL)
    ap.add_argument("--views", type=int, default=32, help="keyframes to reconstruct from")
    ap.add_argument("--stride", type=int, default=1, help="decode every Nth frame")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--width", type=int, default=518, help="0 to keep native size")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--max-points", type=int, default=1_500_000)
    ap.add_argument("--splat-radius", type=float, default=0.012)
    ap.add_argument("--calib", default=DEFAULT_PATH)
    ap.add_argument("--no-undistort", action="store_true")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
