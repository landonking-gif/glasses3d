# glasses3d

Live 3D reconstruction from the Meta Ray-Ban glasses camera stream, where the 3D
world **tracks real-world change** — move a pencil, and it moves in the scene.
Exports to Blender, the web, game engines, and VR.

Full plan and research: `~/.claude/plans/look-into-tencents-new-purring-dawn.md`

## How the dynamic part works

The naive reading of "the 3D world updates when reality does" is continuous 4D
reconstruction, where geometry itself is a function of time. That is enormously
expensive, needs multi-view coverage a single head-mounted camera never has, and
spends all its effort re-deriving the 95% of the room that did not change.

The decomposition here instead:

```
world(t) = static background + Σ ( geometry_i @ T_i(t) )
```

Reconstruct the room **once**. Detect and track *objects* on top of it. When the
pencil moves, update one 4×4 matrix — the geometry is unchanged, because a pencil
is rigid. A reconstruction problem becomes a pose-tracking problem, which runs in
real time and degrades gracefully.

```
glasses ──▶ phone (DAT SDK) ──WebSocket──▶ ingest ──▶ undistort
                                                          │
                        ┌─────────────────────────────────┤
                        ▼                                 ▼
              MASt3R-SLAM (camera pose)          SAM 3 (what + where + track ID)
                        │                                 │
                        │ keyframes                       ▼
                        ▼                        FoundationPose (6DoF per object)
              WorldMirror / MapAnything                   │
                        │ background splats               ▼
                        └──────────────▶ WorldModel ◀─────┘
                                              │ ChangeEvents
                                              ▼
                          scene.ply + scene.json ──▶ Blender / web / Unity / VR
```

**Cost of the assumption:** rigid objects only. Pencil, mug, book, chair — fine.
Deforming things (towel, cat, your hands) cannot be represented by a rigid
transform, so `worldmodel.DEFORMABLE_HINT` refuses them rather than tracking them
badly. Confident-but-wrong geometry is worse than absent geometry.

### Yes, object detection is required

It is not an optimisation — it is the mechanism. Without segmentation there is no
"the pencil" to move, only undifferentiated splats. The pairing:

| Stage | Model | Why |
|---|---|---|
| Detect + segment + track | [SAM 3](https://huggingface.co/facebook/sam3) | Open-vocabulary — prompt it `"pencil"`. Returns masks **and persistent track IDs** across frames, which is what carries object identity between expensive detections |
| 6DoF pose | [FoundationPose](https://github.com/NVlabs/FoundationPose) (NVIDIA) | Model-free: tracks novel objects from a few reference views, no CAD. Cleared for commercial use, unlike the MASt3R camera-tracking weights |

Detection runs every ~15 frames; pose tracking runs every frame. That split is
why it is real-time, and the track IDs are what make it identity-safe.

### MotionBricks is the wrong tool here

[MotionBricks](https://nvlabs.github.io/motionbricks/) is real and impressive —
350,000 motion clips, ~2 ms latency, shipped into GR00T Whole-Body Control at
SIGGRAPH 2026. But it *generates* motion for humanoid robots and game characters.
It synthesises plausible movement; it does not perceive actual movement. Your
problem is the inverse: measuring where a real pencil went. Nothing in
MotionBricks helps with that. (If you later want to *populate* the world with
moving characters, it becomes relevant.)

Also from NVIDIA at SIGGRAPH 2026 and genuinely relevant later: **ArtiFixer**
completes incomplete 3DGS reconstructions — useful for filling regions the camera
never saw, with the same caveat as Voyager: plausible fills, not measurements.

## Status

| Milestone | State |
|---|---|
| M1 — frame transport | server **done, tested**; iOS relay **written, uncompiled** |
| M2 — calibration + undistortion | **done**, validated against synthetic ground truth |
| M3 — camera tracking | `select_keyframe` is **yours**; MASt3R-SLAM binding needs CUDA |
| M4 — densification | adapters defined; WorldMirror/MapAnything need CUDA |
| M4b — **dynamic objects** | **world model + pipeline done and tested**; SAM 3 / FoundationPose need CUDA |
| M5 — export | **done, tested** — 3DGS PLY + scene graph |

88 tests pass locally. Everything model-dependent is written behind an interface
with a working mock, so the logic is verified even though the models are not.

## Your contribution

`server/tracker.py` still has one function deliberately unimplemented:

```python
def select_keyframe(current_pose, last_keyframe, tracking_confidence, current_sharpness) -> bool
```

The valve between the real-time tracker and the async densifier, and the single
most consequential tuning knob. Loose (0.30 m / 15°) → densifier keeps up, but
fast head turns leave holes. Tight (0.10 m / 5°) → better coverage, but the
densifier saturates and the reconstruction lags where you are looking. Motion
blur is worse head-mounted than handheld, and a keyframe admitted at low
tracking confidence misplaces real geometry. Always admit the first frame.

Everything around it is written and tested. ~5–10 lines. `tools/test_tracker.py`
currently asserts the stub is *unimplemented* — flip that check when you fill it in.

## Layout

```
server/protocol.py     wire format: [len][JSON header][JPEG]
server/sources.py      frame sources: WebSocket | video file | mock
server/ingest.py       receive, undistort, report health
server/calibration.py  Intrinsics + cached-remap Undistorter
server/calibrate.py    checkerboard captures -> calib/intrinsics.json
server/tracker.py      camera pose math, blur metric, keyframe buffer
server/worldmodel.py   dynamic scene state + the motion gate
server/perception.py   SAM 3 / FoundationPose adapters + mocks + pipeline
server/export.py       3DGS PLY, point cloud, scene graph, scene baking
ios/Glasses3DRelay.swift   phone relay (drop into VisionClaw CameraAccess)
tools/                 mock sender + tests, all runnable without a GPU
```

## Quick start (no hardware)

```bash
python3 -m pip install -r requirements.txt
```

```bash
python3 server/ingest.py --source mock
```

Full transport path — server in one terminal, simulated phone in another:

```bash
python3 -u server/ingest.py --source ws
```

```bash
python3 tools/mock_sender.py --ladder
```

## Tests

```bash
for t in tracker worldmodel perception export calibration; do python3 tools/test_$t.py; done
```

`test_calibration.py` takes a couple of minutes — it runs genuine corner
detection on 30 synthetic frames and checks the real calibrator recovers a known
fisheye model (focal within 5%, RMS < 0.5 px).

## iOS relay

`ios/Glasses3DRelay.swift` is written against the **actual** DAT SDK API as used
in `~/king-ai/apps/visionclaw/samples/CameraAccess` — note this differs from
Meta's published docs, which say `StreamConfiguration` where the SDK has
`StreamSessionConfig(videoCodec:resolution:frameRate:)`.

It has **not been compiled** — it needs the MWDAT modules and an Xcode target.
Drop it into the VisionClaw CameraAccess target, or delete its session management
and call `relay.send(image:)` from the existing `videoFramePublisher` listener
next to `webrtcSessionVM?.pushVideoFrame`.

It implements backpressure (drops frames when >3 are in flight — a backlog makes
every subsequent frame later still, which corrupts pose association worse than a
drop does) and optional periodic full-resolution photo capture, tagged
`jpeg-hires` so the server routes it as a high-detail keyframe.

## Hardware

Tracking, densification, SAM 3, and FoundationPose all need **CUDA**. This
machine is an M1 Air with 8 GB RAM, so M3/M4 are cloud-only (RunPod, Lambda).
M1, M2, M5 and the entire dynamic-scene logic run locally, which is why they are
finished first.

## Sharp edges

- **The stream is portrait**, 720×1280. Easy to forget until geometry comes out
  rotated. `MockSource` renders portrait on purpose.
- **`--preview` is not a benchmark.** `cv2.imshow` costs ~95 ms/call on Apple
  Silicon and must run on the main thread, so it cannot be threaded off the hot
  path. It throttles ingest from 24 fps to ~18. The summary says so when it's on.
- **The DAT quality ladder is silent.** Per-frame header dimensions turn a
  bandwidth degradation into a log line instead of a mystery quality regression.
- **Latency across two clocks is not directly measurable.** `LatencyMeter`
  reports latency *above the session best case* — a lower bound, labelled as one.
- **MASt3R checkpoints are CC BY-NC-ND — non-commercial.** Swap to
  `facebook/map-anything-apache` if this ships. FoundationPose is fine commercially.
- **Developer Preview**: build and test, but not distribute, until Meta opens
  publishing.
- **Battery**: design for 2–5 minute sessions, not continuous scanning.
