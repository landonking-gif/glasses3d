"""M2: produce calib/intrinsics.json from checkerboard captures.

Usage:
    # 1. Print a checkerboard. Default expects 9x6 *inner corners*
    #    (a 10x7 square board). Measure a square and pass --square-mm.
    # 2. Record ~30s with the glasses, moving the board to the frame corners —
    #    ultrawide distortion is strongest at the edges, so a board that never
    #    leaves the centre yields coefficients that fit nothing.
    # 3. Extract frames and calibrate:

    python3 server/calibrate.py --video calib/board.mp4 --square-mm 25
    python3 server/calibrate.py --images 'calib/shots/*.jpg' --square-mm 25

Exit criterion for M2: RMS reprojection error < 0.5 px, and straight real-world
edges render straight in the undistorted preview (--preview).
"""

from __future__ import annotations

import argparse
import glob as globlib
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import Intrinsics, Undistorter, DEFAULT_PATH  # noqa: E402

SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)


def find_corners(images, pattern: Tuple[int, int], square_mm: float, stride: int):
    """Detect checkerboard corners, returning object/image point pairs."""
    objp = np.zeros((1, pattern[0] * pattern[1], 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp *= (square_mm / 1000.0)  # metres — keeps translation units metric

    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    size = None
    kept = 0

    for idx, image in enumerate(images):
        if idx % stride:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = gray.shape[::-1]
        elif gray.shape[::-1] != size:
            print("  skip frame %d: resolution changed %s -> %s"
                  % (idx, size, gray.shape[::-1]))
            continue
        ok, corners = cv2.findChessboardCorners(
            gray, pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK,
        )
        if not ok:
            continue
        cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), SUBPIX)
        obj_points.append(objp)
        img_points.append(corners.reshape(1, -1, 2).astype(np.float64))
        kept += 1
        print("  frame %d: corners found (%d usable)" % (idx, kept))

    return obj_points, img_points, size


def _fisheye_once(obj_points, img_points, size, flags):
    n = len(obj_points)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rms, K, D, _, _ = cv2.fisheye.calibrate(
        obj_points, img_points, size, K, D,
        [np.zeros((1, 1, 3), np.float64) for _ in range(n)],
        [np.zeros((1, 1, 3), np.float64) for _ in range(n)],
        flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
    )
    return rms, K, D.ravel().tolist()


def calibrate_fisheye(obj_points, img_points, size):
    """Fisheye calibration, retried without CALIB_RECOMPUTE_EXTRINSIC on failure.

    RECOMPUTE_EXTRINSIC generally improves the fit, but OpenCV's fisheye
    initialiser aborts with `fabs(norm_u1) > 0` when any single view yields a
    degenerate homography — one near-frontal or poorly-spread board is enough to
    kill the whole run. Rather than lose the calibration to one bad view, drop
    the flag and continue; the fit is slightly worse but the RMS check still
    gates quality.
    """
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
    try:
        return _fisheye_once(obj_points, img_points, size, flags)
    except cv2.error as exc:
        if "norm_u1" not in str(exc):
            raise
        print("  note: RECOMPUTE_EXTRINSIC hit a degenerate view; retrying without it")
        return _fisheye_once(obj_points, img_points, size,
                             cv2.fisheye.CALIB_FIX_SKEW)


def calibrate_pinhole(obj_points, img_points, size):
    obj = [o.reshape(-1, 3).astype(np.float32) for o in obj_points]
    img = [i.reshape(-1, 1, 2).astype(np.float32) for i in img_points]
    rms, K, D, _, _ = cv2.calibrateCamera(obj, img, size, None, None)
    return rms, K, D.ravel().tolist()


def load_images(args) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit("cannot open video: %s" % args.video)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
    if args.images:
        for path in sorted(globlib.glob(args.images)):
            img = cv2.imread(path)
            if img is not None:
                frames.append(img)
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="path to a checkerboard recording")
    ap.add_argument("--images", help="glob of checkerboard stills")
    ap.add_argument("--pattern", default="9x6", help="inner corners, WxH (default 9x6)")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--stride", type=int, default=5,
                    help="use every Nth frame; consecutive video frames are near-duplicates")
    ap.add_argument("--model", choices=["fisheye", "pinhole", "auto"], default="auto")
    ap.add_argument("--out", default=DEFAULT_PATH)
    ap.add_argument("--preview", action="store_true", help="show an undistorted sample")
    args = ap.parse_args()

    if not args.video and not args.images:
        ap.error("pass --video or --images")

    pw, ph = (int(v) for v in args.pattern.lower().split("x"))
    frames = load_images(args)
    if not frames:
        raise SystemExit("no frames loaded")
    print("loaded %d frames; detecting %dx%d corners..." % (len(frames), pw, ph))

    obj_points, img_points, size = find_corners(frames, (pw, ph), args.square_mm, args.stride)
    if len(obj_points) < 10:
        raise SystemExit(
            "only %d usable views — need >= 10 (ideally 20+ with the board near "
            "the frame corners). Improve lighting or slow the capture." % len(obj_points)
        )
    print("calibrating on %d views at %dx%d..." % (len(obj_points), size[0], size[1]))

    model = args.model
    if model in ("fisheye", "auto"):
        try:
            rms, K, dist = calibrate_fisheye(obj_points, img_points, size)
            model = "fisheye"
        except cv2.error as exc:
            if args.model == "fisheye":
                raise SystemExit("fisheye calibration failed: %s" % exc)
            print("fisheye failed (%s); falling back to pinhole" % exc.err.strip())
            rms, K, dist = calibrate_pinhole(obj_points, img_points, size)
            model = "pinhole"
    else:
        rms, K, dist = calibrate_pinhole(obj_points, img_points, size)

    intr = Intrinsics(model=model, width=size[0], height=size[1],
                      fx=float(K[0, 0]), fy=float(K[1, 1]),
                      cx=float(K[0, 2]), cy=float(K[1, 2]),
                      dist=[float(d) for d in dist], rms=float(rms))
    intr.save(args.out)

    print("\nmodel  : %s" % model)
    print("fx, fy : %.2f, %.2f" % (intr.fx, intr.fy))
    print("cx, cy : %.2f, %.2f" % (intr.cx, intr.cy))
    print("dist   : %s" % ", ".join("%.5f" % d for d in intr.dist))
    print("RMS    : %.4f px  %s" % (rms, "PASS" if rms < 0.5 else "FAIL (want < 0.5)"))
    print("wrote  : %s" % args.out)

    if args.preview:
        undistort = Undistorter(intr)
        sample = frames[len(frames) // 2]
        cv2.imshow("original", sample)
        cv2.imshow("undistorted", undistort(sample))
        print("\nCheck that straight real-world edges are straight. Any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0 if rms < 0.5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
