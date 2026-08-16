"""Verifies the export artifacts are actually well-formed.

A malformed PLY typically loads without complaint and renders as garbage, so
these checks assert on the bytes and on the maths — particularly that
transforming a splat rotates its orientation, not just its position.

    python3 tools/test_export.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "server"))

from export import (SH_C0, Splats, bake_scene, read_ply_header,  # noqa: E402
                    write_point_cloud, write_splats, write_scene_graph,
                    _mat_to_quat, _quat_mul)

results = []


def check(label, condition):
    results.append((label, bool(condition)))


def rotz(deg):
    th = math.radians(deg)
    return np.array([[math.cos(th), -math.sin(th), 0],
                     [math.sin(th), math.cos(th), 0],
                     [0, 0, 1.0]])


def T(R=None, t=(0, 0, 0)):
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    M[:3, 3] = t
    return M


tmp = tempfile.mkdtemp(prefix="glasses3d-")
rng = np.random.RandomState(0)
xyz = rng.normal(0, 1, (500, 3))
rgb = rng.randint(0, 256, (500, 3)).astype(np.uint8)

# --- point cloud ------------------------------------------------------------
p = write_point_cloud(os.path.join(tmp, "points.ply"), xyz, rgb)
h = read_ply_header(p)
check("point cloud is binary little endian", h["format"] == "binary_little_endian")
check("point cloud vertex count is right", h["count"] == 500)
check("point cloud has xyz+rgb", h["properties"] == ["x", "y", "z", "red", "green", "blue"])
expected = h["data_offset"] + 500 * (3 * 4 + 3)
check("point cloud byte size matches header", os.path.getsize(p) == expected)

p2 = write_point_cloud(os.path.join(tmp, "plain.ply"), xyz)
check("point cloud without colour omits rgb",
      read_ply_header(p2)["properties"] == ["x", "y", "z"])

# --- splats -----------------------------------------------------------------
splats = Splats.from_points(xyz, rgb, radius=0.02, alpha=0.8)
sp = write_splats(os.path.join(tmp, "scene.ply"), splats)
hs = read_ply_header(sp)
check("splat count is right", hs["count"] == 500)
check("splat layout matches the 3DGS convention",
      hs["properties"] == ["x", "y", "z", "nx", "ny", "nz",
                           "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                           "scale_0", "scale_1", "scale_2",
                           "rot_0", "rot_1", "rot_2", "rot_3"])
check("splat byte size matches header (17 float32 per splat)",
      os.path.getsize(sp) == hs["data_offset"] + 500 * 17 * 4)

# Storage conventions — these are the ones that silently render as fog.
check("scales are stored in log space", abs(splats.scales[0, 0] - np.log(0.02)) < 1e-5)
check("opacity is stored as a logit",
      abs(1.0 / (1.0 + np.exp(-splats.opacities[0])) - 0.8) < 1e-4)
white = Splats.from_points(np.zeros((1, 3)), np.array([[255, 255, 255]]))
check("colour is encoded as SH degree 0",
      abs(white.sh_dc[0, 0] - (0.5 / SH_C0)) < 1e-4)

# --- transforms -------------------------------------------------------------
one = Splats.from_points(np.array([[1.0, 0.0, 0.0]]), np.array([[255, 0, 0]]))
moved = one.transformed(T(t=(0, 0, 2)))
check("translation moves the mean", np.allclose(moved.means[0], [1, 0, 2], atol=1e-5))
check("translation leaves orientation alone",
      np.allclose(moved.rots[0], [1, 0, 0, 0], atol=1e-6))

turned = one.transformed(T(R=rotz(90)))
check("rotation moves the mean", np.allclose(turned.means[0], [0, 1, 0], atol=1e-5))
# This is the bug worth guarding: rotating an object must rotate each Gaussian's
# ellipsoid too, or geometry comes out smeared rather than obviously wrong.
check("rotation also rotates the orientation quaternion",
      not np.allclose(turned.rots[0], [1, 0, 0, 0], atol=1e-3))
check("rotated quaternion matches the rotation",
      np.allclose(np.abs(turned.rots[0]), np.abs(_mat_to_quat(rotz(90))), atol=1e-5))

check("scales survive a rigid transform", np.allclose(turned.scales, one.scales))
check("opacity survives a rigid transform", np.allclose(turned.opacities, one.opacities))

# Quaternion helpers, including the near-180-degree branch that the naive
# trace-only formula loses precision on.
for deg in (0.0, 45.0, 90.0, 179.0, 180.0):
    q = _mat_to_quat(rotz(deg))
    check("quaternion is unit length at %.0f deg" % deg, abs(np.linalg.norm(q) - 1.0) < 1e-6)
ident = _quat_mul(np.array([1.0, 0, 0, 0]), np.array([[0.5, 0.5, 0.5, 0.5]]))
check("quaternion identity multiply is a no-op",
      np.allclose(ident[0], [0.5, 0.5, 0.5, 0.5], atol=1e-9))

# Composing two 90-degree turns must equal one 180-degree turn.
twice = one.transformed(T(R=rotz(90))).transformed(T(R=rotz(90)))
once = one.transformed(T(R=rotz(180)))
check("composed rotations agree with the single equivalent",
      np.allclose(twice.means[0], once.means[0], atol=1e-5))

# --- baking -----------------------------------------------------------------
bg = Splats.from_points(rng.normal(0, 1, (100, 3)), rng.randint(0, 256, (100, 3)))
obj = Splats.from_points(np.zeros((10, 3)), np.full((10, 3), 200))
baked = bake_scene(bg, [(obj, T(t=(5, 0, 0)))])
check("baked cloud contains everything", len(baked) == 110)
check("baked object sits at its posed location",
      np.allclose(baked.means[100], [5, 0, 0], atol=1e-5))
check("baking an empty scene is safe", len(bake_scene(None, [])) == 0)
check("baking with no background works", len(bake_scene(None, [(obj, T())])) == 10)

# --- degenerate clouds ------------------------------------------------------
# An empty cloud is reachable: bake_scene with nothing to bake, and a
# reconstruction where every point failed the confidence filter both produce
# one. numpy's .max() raises on a zero-size array rather than returning a
# default, so this used to crash at write time.
empty = Splats.from_points(np.zeros((0, 3)), np.zeros((0, 3)))
ep = write_splats(os.path.join(tmp, "empty.ply"), empty)
check("empty splat cloud writes a valid 0-vertex PLY", read_ply_header(ep)["count"] == 0)
check("empty cloud has consistent array shapes",
      len(empty) == 0 and empty.scales.shape == (0, 3) and empty.rots.shape == (0, 4))

# --- scene graph ------------------------------------------------------------
g = write_scene_graph(os.path.join(tmp, "scene.json"), {"background": "geo/room", "objects": []})
check("scene graph file is written", os.path.getsize(g) > 0)

failed = 0
for label, ok in results:
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    failed += 0 if ok else 1
print("\n%d passed, %d failed" % (len(results) - failed, failed))
print("artifacts in %s" % tmp)
raise SystemExit(1 if failed else 0)
