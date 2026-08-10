"""M1/M2: receive glasses frames, undistort, report health.

    python3 server/ingest.py --source mock --preview
    python3 server/ingest.py --source ws --port 8765 --preview
    python3 server/ingest.py --source file --path clip.mp4

Exit criterion for M1: live frames render, with measured end-to-end latency
(target p50 < 200 ms).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time
from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import Intrinsics, Undistorter, identity_for, DEFAULT_PATH  # noqa: E402
from sources import Frame, FrameSource, MockSource, VideoFileSource, WebSocketSource  # noqa: E402


class LatencyMeter:
    """Estimates transport latency across two unsynchronised clocks.

    The phone stamps `ts_send` on its clock; the server stamps `ts_recv` on its
    own. Their difference conflates real latency with clock offset, and the two
    are not separable from one-way timings alone.

    What is recoverable: the *minimum* observed difference over a session is the
    best case the link achieved, so treating it as the offset makes every other
    sample read as "latency above the floor". That is the number that matters
    for diagnosing jitter and buffering — but it is a lower bound on absolute
    latency, not absolute latency. Reported as such.
    """

    def __init__(self, window: int = 600):
        self.samples: Deque[float] = deque(maxlen=window)
        self.offset: Optional[float] = None

    def add(self, frame: Frame) -> float:
        raw = frame.ts_recv - frame.header.ts_send
        if self.offset is None or raw < self.offset:
            self.offset = raw
        excess = raw - self.offset
        self.samples.append(excess)
        return excess

    def percentiles(self):
        if not self.samples:
            return 0.0, 0.0
        arr = np.fromiter(self.samples, dtype=np.float64)
        return float(np.percentile(arr, 50)), float(np.percentile(arr, 99))


class IngestStats:
    """Tracks throughput, gaps, and — critically — delivered resolution.

    The DAT quality ladder silently steps resolution down, then frame rate (never
    below 15fps), when bandwidth tightens. Logging every resolution change is
    what keeps a bandwidth problem from being misdiagnosed as a reconstruction
    bug three milestones later.
    """

    def __init__(self, report_every: float = 5.0):
        self.report_every = report_every
        self.latency = LatencyMeter()
        self.count = 0
        self.gaps = 0
        self.last_seq: Optional[int] = None
        self.last_res: Optional[tuple] = None
        self.res_changes = 0
        self.t0 = time.time()
        self._last_report = self.t0
        self._since_report = 0

    def update(self, frame: Frame) -> None:
        if self.count == 0:
            # Start the clock at the first frame, not at process launch. The
            # server may idle for many seconds waiting for the phone to connect,
            # and folding that wait into the first window reports a framerate
            # far below the real one.
            self.t0 = time.time()
            self._last_report = self.t0
        self.count += 1
        self._since_report += 1
        self.latency.add(frame)

        seq = frame.header.seq
        if self.last_seq is not None and seq != self.last_seq + 1:
            missing = seq - self.last_seq - 1
            if missing > 0:
                self.gaps += missing
        self.last_seq = seq

        res = (frame.image.shape[1], frame.image.shape[0])
        if res != self.last_res:
            if self.last_res is not None:
                self.res_changes += 1
                print("[ingest] RESOLUTION CHANGED %dx%d -> %dx%d at seq=%d "
                      "(bandwidth ladder, not a bug in your code)"
                      % (self.last_res[0], self.last_res[1], res[0], res[1], seq))
            else:
                print("[ingest] stream resolution %dx%d" % res)
            self.last_res = res

    def maybe_report(self) -> None:
        now = time.time()
        elapsed = now - self._last_report
        if elapsed < self.report_every:
            return
        p50, p99 = self.latency.percentiles()
        fps = self._since_report / elapsed
        print("[ingest] %5.1f fps | frames %5d | dropped-seq %3d | res-changes %d "
              "| latency+ p50 %5.1f ms p99 %5.1f ms"
              % (fps, self.count, self.gaps, self.res_changes, p50 * 1e3, p99 * 1e3))
        self._last_report = now
        self._since_report = 0

    def summary(self) -> None:
        p50, p99 = self.latency.percentiles()
        dur = max(time.time() - self.t0, 1e-6)
        print("\n=== ingest summary ===")
        print("frames        : %d in %.1fs (%.1f fps avg)" % (self.count, dur, self.count / dur))
        print("dropped seq   : %d" % self.gaps)
        print("res changes   : %d" % self.res_changes)
        print("latency above floor : p50 %.1f ms | p99 %.1f ms" % (p50 * 1e3, p99 * 1e3))
        if self.latency.offset is not None:
            print("(clock offset + best-case latency was %.1f ms; absolute "
                  "latency is that floor plus the figures above)"
                  % (self.latency.offset * 1e3))


def build_source(args) -> FrameSource:
    if args.source == "ws":
        return WebSocketSource(host=args.host, port=args.port)
    if args.source == "file":
        if not args.path:
            raise SystemExit("--source file requires --path")
        return VideoFileSource(args.path, realtime=args.realtime)
    return MockSource(count=args.count)


def load_intrinsics(args) -> Optional[Intrinsics]:
    if args.no_undistort:
        return None
    path = args.calib
    if not os.path.exists(path):
        print("[calib] no %s yet — running WITHOUT undistortion.\n"
              "        Geometry will be warped until M2 is done. See "
              "server/calibrate.py" % path)
        return None
    intr = Intrinsics.load(path)
    print("[calib] loaded %s intrinsics (RMS %.3f px, calibrated at %dx%d)"
          % (intr.model, intr.rms, intr.width, intr.height))
    return intr


async def run(args) -> None:
    source = build_source(args)
    intr = load_intrinsics(args)
    undistort = Undistorter(intr) if intr is not None else None
    stats = IngestStats()
    stop = asyncio.Event()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    try:
        async for frame in source.frames():
            if stop.is_set():
                break
            stats.update(frame)
            image = undistort(frame.image) if undistort is not None else frame.image

            if args.preview:
                shown = image
                if shown.shape[0] > 900:  # portrait 1280 does not fit most screens
                    scale = 900.0 / shown.shape[0]
                    shown = cv2.resize(shown, None, fx=scale, fy=scale)
                cv2.imshow("glasses3d ingest", shown)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            stats.maybe_report()
    finally:
        await source.close()
        if args.preview:
            cv2.destroyAllWindows()
        stats.summary()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["ws", "file", "mock"], default="mock")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--path", help="video path for --source file")
    ap.add_argument("--realtime", action="store_true", help="replay files at native fps")
    ap.add_argument("--count", type=int, default=240, help="mock frame count")
    ap.add_argument("--calib", default=DEFAULT_PATH)
    ap.add_argument("--no-undistort", action="store_true")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
