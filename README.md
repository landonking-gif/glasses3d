# glasses3d

Live 3D reconstruction from the Meta Ray-Ban glasses camera stream, exportable to
Blender, the web, game engines, and VR.

Full plan and research: `~/.claude/plans/look-into-tencents-new-purring-dawn.md`

## Status

| Milestone | State |
|---|---|
| M1 — frame transport (phone → server) | server done and tested; **iOS app not written** |
| M2 — calibration + undistortion | done, validated against synthetic ground truth |
| M3 — live tracking | scaffolded; **needs your `select_keyframe`** + a CUDA box |
| M4 — densification and export | not started |
| M5 — Blender / web / engine / VR export | not started |

## Your contribution

`server/tracker.py` has one function left deliberately unimplemented:

```python
def select_keyframe(current_pose, last_keyframe, tracking_confidence, current_sharpness) -> bool
```

This is the valve between the real-time tracker and the async densifier, and it
is the single most consequential tuning knob in the system. It is left to you
because the trade-off is genuine and depends on how *you* move your head:

- **Loose** (0.30 m / 15°) — densifier keeps up easily, but fast head turns leave
  holes where you moved past a region without banking a view of it.
- **Tight** (0.10 m / 5°) — better coverage and geometry, but the densifier
  saturates, the queue backs up, and the reconstruction lags where you're looking.

Two head-mounted-specific factors: motion blur is worse than handheld (a sharp
frame just under the distance threshold usually beats a blurry one just over it),
and admitting a keyframe while `tracking_confidence` is low places real geometry
in the wrong spot — harder to spot later than a gap. Always admit the first
frame, or the reconstruction never starts.

Everything around it — pose math, blur metric, bounded buffer with
lowest-score-first eviction — is written and tested. Roughly 5–10 lines.

`tools/test_tracker.py` currently asserts the stub is *unimplemented*; flip that
check once you fill it in.

## Layout

```
server/protocol.py     wire format: [len][JSON header][JPEG]
server/sources.py      pluggable frame sources: WebSocket | video file | mock
server/ingest.py       M1/M2 entry point — receive, undistort, report health
server/calibration.py  Intrinsics + cached-remap Undistorter
server/calibrate.py    M2 CLI: checkerboard captures -> calib/intrinsics.json
server/tracker.py      M3 pose math, blur metric, keyframe buffer
tools/mock_sender.py   stands in for the phone app, simulates the DAT quality ladder
tools/test_*.py        self-contained validation, no hardware needed
```

## Quick start (no hardware)

```bash
python3 -m pip install -r requirements.txt
```

Mock frames straight through the server:

```bash
python3 server/ingest.py --source mock --preview
```

Full transport path — server in one terminal, simulated phone in another:

```bash
python3 -u server/ingest.py --source ws --preview
```

```bash
python3 tools/mock_sender.py --ladder
```

`--ladder` makes the sender step 720×1280 → 504×896 → 360×640 mid-stream,
reproducing the DAT bandwidth ladder. The server should log each change.

## Tests

```bash
python3 tools/test_calibration.py && python3 tools/test_tracker.py
```

`test_calibration.py` renders checkerboards through a *known* fisheye model and
checks the real calibration code recovers it (focal within 5%, RMS < 0.5 px). It
takes a couple of minutes — it is doing genuine corner detection on 30 frames.

## Calibration (M2)

Print a checkerboard — the default expects **9×6 inner corners** (a 10×7 square
board). Record ~30s with the glasses, working the board into the **frame corners**,
where ultrawide distortion is strongest and most informative. A board that never
leaves the centre produces coefficients that fit nothing.

```bash
python3 server/calibrate.py --video calib/board.mp4 --square-mm 25 --preview
```

Passes at RMS < 0.5 px. Until `calib/intrinsics.json` exists, `ingest.py` runs
without undistortion and says so.

## Hardware

Tracking and densification need **CUDA** — MASt3R-SLAM's 15 FPS figure is on an
RTX 4090. Apple Silicon cannot run the CUDA rasterizers, so M3/M4 are cloud-only
from a Mac (RunPod or Lambda). M1 and M2 run fine anywhere, which is why they are
finished first.

## Sharp edges

- **The stream is portrait**, 720×1280. Easy to forget until geometry comes out
  rotated. `MockSource` renders portrait on purpose.
- **The DAT quality ladder is silent.** It drops resolution, then frame rate
  (never below 15fps), under bandwidth pressure. The per-frame header carries
  dimensions so this shows up in the log instead of as a mystery quality
  regression three milestones later.
- **Latency across two clocks is not directly measurable.** `LatencyMeter`
  reports latency *above the session's best case*, which is the number that
  matters for jitter — but it is a lower bound on absolute latency, not absolute
  latency.
- **MASt3R checkpoints are CC BY-NC-ND — non-commercial.** If this ever ships,
  swap to the Apache-2.0 `facebook/map-anything-apache` weights. Decide early.
- **Developer Preview**: you can build and test, but not distribute, until Meta
  opens publishing.
- **Battery**: design for 2–5 minute sessions, not continuous scanning.
