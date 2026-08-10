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
| M1 — frame transport | server **done, tested**; iOS relay **compiles against DAT SDK 0.4.0** |
| M2 — calibration + undistortion | **done**, validated against synthetic ground truth |
| M3 — keyframe selection | **done, tested**; MASt3R-SLAM binding still needs CUDA |
| M4 — densification | **offline path done, tested end-to-end** (mock backend); MapAnything needs CUDA |
| M4b — dynamic objects | **world model + pipeline done and tested**; SAM 3 / FoundationPose need CUDA |
| M5 — export | **done, tested** — 3DGS PLY + scene graph |
| Colab | **notebook validated**; models unrun |

133 tests pass locally. Everything model-dependent sits behind an interface with
a working mock, so the surrounding logic is verified even where the models are not.

**Still genuinely unverified:** no neural model in this repo has ever executed.
MapAnything, SAM 3, FoundationPose and MASt3R-SLAM are all written against their
documented APIs and exercised only through mocks. First real Colab run is where
that gets tested.

## Running it on Colab

`colab/glasses3d_colab.ipynb` — open in Colab, set Runtime → GPU, work down.

The notebook measures the GPU it was assigned before you commit to a mode, because
Colab allocates opportunistically and the same notebook gets a T4 one run and an
L4 the next:

| GPU | vs RTX 4090 | Est. SLAM fps | Verdict |
|---|---|---|---|
| T4 (free tier) | ~0.16× | ~2.4 | **offline only** |
| L4 (Pro) | ~0.30× | ~4.5 | marginal |
| A100 40/80GB | ~0.65× | ~9.8 | live viable |
| RTX PRO 6000 "G4" | ~1.1× | ~16 | comfortable |

Live tracking needs roughly 8 fps to feel live, so **free-tier Colab cannot do
live** — and that is fine, because offline is the better path anyway: recorded
clips are 3K/60, about 9× the pixels of the 720p stream, with no latency budget
to fight.

The other live-mode obstacle is structural: **Colab has no public inbound
address**, so the phone cannot reach it. The notebook opens a `cloudflared`
quick tunnel and prints a `wss://` URL for `Glasses3DRelay.swift`, but that adds
100–300 ms on top of the glasses→phone hop, and the URL changes every restart.
For real live work, a persistent GPU box with a stable address beats Colab.

Offline, from a shell:

```bash
python3 server/reconstruct.py --video clip.mp4 --out out/ --backend mapanything --views 32
```

Swap `--backend mock` to exercise the whole path with no GPU at all.

## Licensing, since it is easy to get wrong

`backends.py` defaults to **`facebook/map-anything-apache`** (Apache-2.0). The
otherwise-identical `facebook/map-anything` is **CC-BY-NC** — research only.
Opting into the NC weights should be deliberate, and the backend prints a warning
if you do. FoundationPose is cleared for commercial use; **MASt3R's checkpoints
are CC BY-NC-ND**, so the live camera tracker is the one piece that would need
replacing before this could ship.

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
server/backends.py     CUDA model adapters (MapAnything) + GPU capability probe
server/reconstruct.py  offline driver: video in, scene.ply out
colab/                 Colab notebook (offline + live modes)
ios/Glasses3DRelay.swift   phone relay, compiles against DAT SDK 0.4.0
ios/verify-build.sh        proves it compiles, without touching VisionClaw
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
for t in tracker worldmodel perception export reconstruct calibration; do python3 tools/test_$t.py; done
```

`test_calibration.py` takes a couple of minutes — it runs genuine corner
detection on 30 synthetic frames and checks the real calibrator recovers a known
fisheye model (focal within 5%, RMS < 0.5 px).

## iOS relay

`ios/Glasses3DRelay.swift` is written against the **actual** DAT SDK API as used
in `~/king-ai/apps/visionclaw/samples/CameraAccess` — this differs from Meta's
published docs, which say `StreamConfiguration` where the SDK actually has
`StreamSessionConfig(videoCodec:resolution:frameRate:)`.

**It compiles.** `ios/verify-build.sh` builds it in an isolated SPM package
pinned to `meta-wearables-dat-ios` 0.4.0 — the same version CameraAccess resolves
— so a failure there is a problem with the relay, not with your app:

```bash
./ios/verify-build.sh
```

Verified under both Swift 5.0 (matching the CameraAccess target) and Swift 6.0
strict concurrency.

To use it, drop the file into the VisionClaw CameraAccess target — or delete its
session management and call `relay.send(image:)` from the existing
`videoFramePublisher` listener, right next to `webrtcSessionVM?.pushVideoFrame`.

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
