"""Stands in for the phone relay app, so the server is testable with no hardware.

Speaks the same wire protocol the Swift app will, including the DAT quality
ladder — pass --ladder to have it step 720x1280 -> 504x896 -> 360x640 mid-stream
and confirm the server notices and logs the change.

    python3 server/ingest.py --source ws --preview        # terminal 1
    python3 tools/mock_sender.py --ladder                 # terminal 2
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))
from protocol import FrameHeader, encode  # noqa: E402

# The three tiers the Wearables Device Access Toolkit exposes.
LADDER = [(720, 1280), (504, 896), (360, 640)]


def render(seq: int, width: int, height: int) -> np.ndarray:
    image = np.zeros((height, width, 3), np.uint8)
    offset = (seq * 5) % 100
    for y in range(-100, height, 100):
        for x in range(-100, width, 100):
            if ((x // 100) + (y // 100)) % 2 == 0:
                cv2.rectangle(image, (x + offset, y), (x + offset + 99, y + 99),
                              (190, 190, 190), -1)
    cv2.putText(image, "seq %d" % seq, (16, 48), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 140, 255), 2)
    cv2.putText(image, "%dx%d" % (width, height), (16, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 120), 2)
    return image


async def main_async(args) -> None:
    import websockets

    uri = "ws://%s:%d" % (args.host, args.port)
    print("connecting to %s ..." % uri)
    async with websockets.connect(uri, max_size=8 * 1024 * 1024) as ws:
        print("connected; streaming %d frames at %.0f fps" % (args.count, args.fps))
        period = 1.0 / args.fps
        tier = 0
        next_at = time.perf_counter()

        for seq in range(args.count):
            if args.ladder and seq and seq % args.ladder_every == 0:
                tier = min(tier + 1, len(LADDER) - 1)
                print("  -> stepping down to %dx%d" % LADDER[tier])
            width, height = LADDER[tier]

            image = render(seq, width, height)
            ts_capture = time.time()
            ok, buf = cv2.imencode(".jpg", image,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
            if not ok:
                continue
            header = FrameHeader(seq=seq, ts_capture=ts_capture, ts_send=time.time(),
                                 width=width, height=height, fmt="jpeg",
                                 session=args.session)
            await ws.send(encode(header, buf.tobytes()))

            next_at += period
            sleep = next_at - time.perf_counter()
            if sleep > 0:
                await asyncio.sleep(sleep)
            else:
                next_at = time.perf_counter()
        print("done")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--fps", type=float, default=24.0, help="DAT allows 2, 7, 15, 24, 30")
    ap.add_argument("--count", type=int, default=480)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--session", default="mock-session")
    ap.add_argument("--ladder", action="store_true", help="simulate bandwidth degradation")
    ap.add_argument("--ladder-every", type=int, default=100)
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args))
    except (KeyboardInterrupt, OSError) as exc:
        print("stopped: %s" % exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
