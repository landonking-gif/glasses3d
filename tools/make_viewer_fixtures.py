"""Writes deterministic PLY fixtures for tools/test_viewer.mjs.

Sentinel values at the first and last vertex, so a stride or endianness error in
the JS parser surfaces as an exact mismatch rather than as a cloud that merely
looks a bit wrong.

    python3 tools/make_viewer_fixtures.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "server"))

from export import SH_C0, Splats, write_point_cloud, write_splats  # noqa: E402

OUT = os.path.join(ROOT, "tools", "fixtures")
N = 1000


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(11)

    xyz = rng.uniform(-5, 5, (N, 3)).astype(np.float32)
    rgb = rng.randint(0, 256, (N, 3)).astype(np.uint8)
    # Sentinels the JS test asserts on exactly.
    xyz[0] = [1.5, -2.25, 3.75]; rgb[0] = [255, 128, 7]
    xyz[-1] = [9.5, 0.5, -0.25]; rgb[-1] = [3, 200, 99]
    write_point_cloud(os.path.join(OUT, "cloud.ply"), xyz, rgb)

    # Splat fixture: same positions, colour via SH degree 0. Vertex 0 is pure
    # red and vertex 1 is mid-grey, so a decode that forgets the SH constant
    # returns grey for both and the test catches it.
    splat_rgb = rgb.astype(np.float32) / 255.0
    splat_rgb[0] = [1.0, 0.0, 0.0]
    splat_rgb[1] = [0.5, 0.5, 0.5]
    splats = Splats.from_points(xyz, (splat_rgb * 255).astype(np.uint8))
    write_splats(os.path.join(OUT, "splat.ply"), splats)

    with open(os.path.join(OUT, "ascii.ply"), "w") as fh:
        fh.write("ply\nformat ascii 1.0\nelement vertex 3\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n"
                 "1 2 3 255 0 0\n4 5 6 10 20 30\n7 8 9 0 0 255\n")

    for name in ("cloud.ply", "splat.ply", "ascii.ply"):
        p = os.path.join(OUT, name)
        print("  %-11s %7d bytes" % (name, os.path.getsize(p)))
    print("\nSH_C0 = %.10f (the constant the viewer must undo)" % SH_C0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
