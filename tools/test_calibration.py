"""Validates the M2 calibration machinery against synthetic ground truth.

Renders checkerboard views through a *known* fisheye model, runs the real
calibration code over them, and checks the recovered intrinsics match. This
proves the pipeline works before anyone prints a board and spends twenty minutes
waving it at a pair of glasses — if calibration is broken, you want to find out
here, not from a subtly warped reconstruction three milestones later.

    python3 tools/test_calibration.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "server"))

from calibration import Intrinsics, Undistorter  # noqa: E402
from calibrate import find_corners, calibrate_fisheye  # noqa: E402

W, H = 720, 1280
PATTERN = (9, 6)          # inner corners
SQUARE_PX = 90            # board rendering scale
SQUARE_MM = 25.0

# Ground truth we will try to recover. Strong barrel distortion, as an ultrawide
# lens produces.
#
# The focal length is not arbitrary. OpenCV's fisheye model is equidistant:
# theta_d = theta * (1 + k1*theta^2 + ...). With negative k1 that polynomial is
# only monotonic up to theta ~ 2.4 rad and peaks at theta_d ~ 1.64. A frame
# corner sits at radius sqrt(360^2 + 640^2) / fx, so fx = 300 would demand
# theta_d = 2.45 — outside the model's representable range. The corners would
# then be undefined and the synthetic "ground truth" would not be a fisheye
# image at all. fx = 560 puts the corner at 1.31 rad (~150 deg diagonal FOV),
# comfortably inside the invertible region and realistic for an ultrawide.
GT_K = np.array([[560.0, 0.0, W / 2.0],
                 [0.0, 560.0, H / 2.0],
                 [0.0, 0.0, 1.0]])
GT_D = np.array([[-0.055], [0.012], [-0.003], [0.0005]])


def board_image() -> np.ndarray:
    """White canvas with a checkerboard; margin matters — findChessboardCorners
    needs a quiet border around the pattern to locate the outer corners."""
    cols, rows = PATTERN[0] + 1, PATTERN[1] + 1
    margin = SQUARE_PX
    img = np.full((rows * SQUARE_PX + 2 * margin,
                   cols * SQUARE_PX + 2 * margin), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y, x = margin + r * SQUARE_PX, margin + c * SQUARE_PX
                img[y:y + SQUARE_PX, x:x + SQUARE_PX] = 0
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def distortion_maps() -> tuple:
    """Maps taking distorted-image pixels back to ideal-pinhole pixels.

    cv2 gives us undistorted->distorted directly; we need the inverse to *apply*
    distortion to a synthetic ideal render. Built by undistorting every pixel
    coordinate once, then reprojecting through the ideal camera.
    """
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
    norm = cv2.fisheye.undistortPoints(pts, GT_K, GT_D).reshape(-1, 2)
    src_x = (norm[:, 0] * GT_K[0, 0] + GT_K[0, 2]).reshape(H, W).astype(np.float32)
    src_y = (norm[:, 1] * GT_K[1, 1] + GT_K[1, 2]).reshape(H, W).astype(np.float32)
    return src_x, src_y


def render_views(n: int = 30) -> list:
    """Views of the board at genuine rigid 3D poses, pushed through GT distortion.

    The poses must be real rigid transforms, not arbitrary 2D perspective warps.
    An arbitrary projective transform of the board does not correspond to any
    rigid plane pose — it shears the board — and calibration, which assumes a
    rigid square target, cannot fit that. It converges to nonsense with a large
    reprojection error, which looks exactly like a broken calibrator.

    So: place the board in 3D, project its corners through the ideal pinhole
    camera, and derive the warp from those projected corners.
    """
    board = board_image()
    bh_px, bw_px = board.shape[:2]
    src = np.float32([[0, 0], [bw_px, 0], [bw_px, bh_px], [0, bh_px]])

    # Board extent in metres, including the quiet margin.
    m_per_px = SQUARE_MM / 1000.0 / SQUARE_PX
    bw_m, bh_m = bw_px * m_per_px, bh_px * m_per_px
    corners_3d = np.float64([[0, 0, 0], [bw_m, 0, 0], [bw_m, bh_m, 0], [0, bh_m, 0]])

    map_x, map_y = distortion_maps()
    rng = np.random.RandomState(7)
    views = []

    for i in range(n):
        # Tilt the board and spread it across the frame — including the corners,
        # where ultrawide distortion is strongest and most informative.
        rvec = np.float64([rng.uniform(-0.45, 0.45),
                           rng.uniform(-0.45, 0.45),
                           rng.uniform(-0.25, 0.25)])
        R, _ = cv2.Rodrigues(rvec)
        z = rng.uniform(0.42, 0.72)
        # Aim the board centre at a target pixel by back-projecting through K.
        u = W * (0.20 + 0.60 * ((i % 5) / 4.0)) + rng.uniform(-30, 30)
        v = H * (0.15 + 0.70 * ((i // 5) / max(1, (n // 5) - 1))) + rng.uniform(-40, 40)
        centre = np.float64([(u - GT_K[0, 2]) / GT_K[0, 0] * z,
                             (v - GT_K[1, 2]) / GT_K[1, 1] * z, z])
        t = centre - R @ np.float64([bw_m / 2.0, bh_m / 2.0, 0.0])

        cam = (R @ corners_3d.T).T + t
        if np.any(cam[:, 2] < 0.05):  # behind or too near the camera
            continue
        proj = cam[:, :2] / cam[:, 2:3]
        dst = np.float32(proj * np.float64([GT_K[0, 0], GT_K[1, 1]])
                         + np.float64([GT_K[0, 2], GT_K[1, 2]]))

        ideal = np.full((H, W, 3), 255, np.uint8)
        Hm = cv2.getPerspectiveTransform(src, dst)
        cv2.warpPerspective(board, Hm, (W, H), dst=ideal,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255))
        views.append(cv2.remap(ideal, map_x, map_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255)))
    return views


def main() -> int:
    print("rendering synthetic fisheye checkerboard views...")
    views = render_views()

    obj_points, img_points, size = find_corners(views, PATTERN, SQUARE_MM, stride=1)
    print("\ndetected the board in %d / %d views" % (len(obj_points), len(views)))
    if len(obj_points) < 10:
        print("FAIL: too few detections to calibrate")
        return 1

    rms, K, dist = calibrate_fisheye(obj_points, img_points, size)
    intr = Intrinsics(model="fisheye", width=size[0], height=size[1],
                      fx=float(K[0, 0]), fy=float(K[1, 1]),
                      cx=float(K[0, 2]), cy=float(K[1, 2]),
                      dist=[float(d) for d in dist], rms=float(rms))

    fx_err = abs(intr.fx - GT_K[0, 0]) / GT_K[0, 0] * 100
    cx_err = abs(intr.cx - GT_K[0, 2])
    cy_err = abs(intr.cy - GT_K[1, 2])

    print("\n            recovered      truth")
    print("fx     %12.2f %10.2f   (%.2f%% off)" % (intr.fx, GT_K[0, 0], fx_err))
    print("fy     %12.2f %10.2f" % (intr.fy, GT_K[1, 1]))
    print("cx     %12.2f %10.2f   (%.1f px off)" % (intr.cx, GT_K[0, 2], cx_err))
    print("cy     %12.2f %10.2f   (%.1f px off)" % (intr.cy, GT_K[1, 2], cy_err))
    print("k1     %12.5f %10.5f" % (intr.dist[0], GT_D[0, 0]))
    print("RMS    %12.4f px" % rms)

    # Undistortion must also round-trip: straight lines back to straight.
    und = Undistorter(intr)
    out = und(views[len(views) // 2])
    ok_shape = out.shape == views[0].shape

    checks = [
        ("board detected in >= 10 views", len(obj_points) >= 10),
        ("RMS < 0.5 px", rms < 0.5),
        ("focal length within 5%", fx_err < 5.0),
        ("principal point within 25 px", cx_err < 25 and cy_err < 25),
        ("undistort preserves frame size", ok_shape),
    ]
    print()
    failed = 0
    for label, passed in checks:
        print("  %s  %s" % ("PASS" if passed else "FAIL", label))
        failed += 0 if passed else 1
    print("\n%s" % ("all checks passed" if not failed else "%d check(s) failed" % failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
