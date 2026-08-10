"""Pluggable frame sources.

Everything downstream (undistortion, tracking, densification) consumes `Frame`
objects and never learns where they came from. That boundary is what lets the
offline path — recorded 3K clips, ~9x the pixels of the live stream — reuse the
entire pipeline if live fidelity disappoints.

Sources:
    WebSocketSource  live glasses stream, relayed by the phone app
    VideoFileSource  recorded clip on disk (offline high-quality path)
    MockSource       synthetic frames, no hardware required
"""

from __future__ import annotations

import abc
import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import cv2
import numpy as np

from protocol import FrameHeader, decode


@dataclass
class Frame:
    header: FrameHeader
    image: np.ndarray  # BGR, uint8
    ts_recv: float  # seconds, server clock, at decode completion

    @property
    def shape(self) -> tuple:
        return self.image.shape[:2]


class FrameSource(abc.ABC):
    """Async iterable of Frames."""

    @abc.abstractmethod
    def frames(self) -> AsyncIterator[Frame]:
        ...

    async def close(self) -> None:
        return None


class WebSocketSource(FrameSource):
    """Accepts one phone connection and yields decoded frames.

    Single-client by design. Two glasses streaming into one tracker would need
    multi-session handling that does not exist yet, so a second connection is
    rejected rather than silently interleaved into the same reconstruction.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765, queue_size: int = 8):
        self.host = host
        self.port = port
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._server = None
        self._connected = False
        self.dropped = 0

    async def _handler(self, websocket):
        if self._connected:
            await websocket.close(code=1013, reason="another client is connected")
            return
        self._connected = True
        peer = getattr(websocket, "remote_address", ("?", 0))
        print("[ws] phone connected from %s:%s" % (peer[0], peer[1]))
        try:
            async for message in websocket:
                if isinstance(message, str):
                    continue  # control channel, unused for now
                try:
                    header, payload = decode(message)
                except ValueError as exc:
                    print("[ws] malformed frame: %s" % exc)
                    continue
                buf = np.frombuffer(payload, dtype=np.uint8)
                image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if image is None:
                    print("[ws] undecodable payload for seq=%d" % header.seq)
                    continue
                frame = Frame(header=header, image=image, ts_recv=time.time())
                try:
                    self._queue.put_nowait(frame)
                except asyncio.QueueFull:
                    # Drop the oldest, keep the newest. A live tracker wants the
                    # freshest view of the world; a stale backlog is worse than
                    # a gap, because it makes pose estimates lag reality.
                    _ = self._queue.get_nowait()
                    self._queue.put_nowait(frame)
                    self.dropped += 1
        finally:
            self._connected = False
            print("[ws] phone disconnected")

    async def frames(self) -> AsyncIterator[Frame]:
        import websockets

        self._server = await websockets.serve(
            self._handler, self.host, self.port, max_size=8 * 1024 * 1024
        )
        print("[ws] listening on ws://%s:%d" % (self.host, self.port))
        while True:
            yield await self._queue.get()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()


class VideoFileSource(FrameSource):
    """Replays a recorded clip. Used for the offline high-quality path."""

    def __init__(self, path: str, realtime: bool = False):
        self.path = path
        self.realtime = realtime
        self._cap: Optional[cv2.VideoCapture] = None

    async def frames(self) -> AsyncIterator[Frame]:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError("cannot open video: %s" % self.path)
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 24.0
        period = 1.0 / fps
        seq = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            now = time.time()
            h, w = image.shape[:2]
            yield Frame(
                header=FrameHeader(
                    seq=seq, ts_capture=now, ts_send=now, width=w, height=h,
                    fmt="raw", session=self.path,
                ),
                image=image,
                ts_recv=now,
            )
            seq += 1
            if self.realtime:
                await asyncio.sleep(period)
            else:
                await asyncio.sleep(0)

    async def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


class MockSource(FrameSource):
    """Synthetic frames so the whole server is testable with no glasses.

    Renders a moving checkerboard at the DAT `.high` resolution (720x1280
    portrait) so orientation bugs surface immediately — the live stream is
    portrait, which is easy to forget until geometry comes out rotated.
    """

    def __init__(self, count: int = 120, fps: float = 24.0,
                 width: int = 720, height: int = 1280):
        self.count = count
        self.fps = fps
        self.width = width
        self.height = height

    async def frames(self) -> AsyncIterator[Frame]:
        period = 1.0 / self.fps
        # Deadline-based pacing, not a fixed sleep after each yield. A real
        # camera emits on its own schedule regardless of how long the consumer
        # takes; a fixed post-yield sleep instead *adds* consumer time to every
        # frame interval, so a slow consumer would silently make the mock source
        # look like a slow camera.
        next_at = time.perf_counter()
        for seq in range(self.count):
            image = np.zeros((self.height, self.width, 3), np.uint8)
            offset = (seq * 7) % 80
            for y in range(-80, self.height, 80):
                for x in range(-80, self.width, 80):
                    if ((x // 80) + (y // 80)) % 2 == 0:
                        cv2.rectangle(image, (x + offset, y + offset),
                                      (x + offset + 79, y + offset + 79),
                                      (200, 200, 200), -1)
            cv2.putText(image, "seq %d" % seq, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 128, 255), 3)
            now = time.time()
            yield Frame(
                header=FrameHeader(seq=seq, ts_capture=now, ts_send=now,
                                   width=self.width, height=self.height,
                                   fmt="raw", session="mock"),
                image=image,
                ts_recv=now,
            )
            next_at += period
            slack = next_at - time.perf_counter()
            if slack > 0:
                await asyncio.sleep(slack)
            else:
                # Consumer is slower than the nominal rate. Resync rather than
                # accumulating debt and then bursting to catch up.
                next_at = time.perf_counter()
                await asyncio.sleep(0)
