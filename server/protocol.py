"""Wire format between the phone relay app and the ingest server.

One frame = one binary WebSocket message:

    [4 bytes big-endian uint32: header length N][N bytes UTF-8 JSON header][JPEG payload]

The header travels with every frame rather than being negotiated once at
connection time. That is deliberate: the Wearables Device Access Toolkit runs an
automatic quality ladder that silently drops resolution, then frame rate, when
bandwidth tightens. If resolution were negotiated once, a bandwidth-induced
quality drop would be indistinguishable from a reconstruction bug. Per-frame
dimensions make the degradation visible in the logs.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

HEADER_LEN_STRUCT = struct.Struct(">I")
MAX_HEADER_BYTES = 8192


@dataclass
class FrameHeader:
    """Metadata the phone attaches to each frame.

    `seq` and `ts_capture` cannot be reconstructed downstream — once a frame is
    decoded it is just pixels — so the phone must stamp them at capture time.
    Pose association in the tracker keys off `seq`.
    """

    seq: int
    ts_capture: float  # seconds, phone clock, at frame arrival from glasses
    ts_send: float  # seconds, phone clock, immediately before socket write
    width: int
    height: int
    fmt: str = "jpeg"
    session: str = "default"

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, raw: bytes) -> "FrameHeader":
        data = json.loads(raw.decode("utf-8"))
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


def encode(header: FrameHeader, payload: bytes) -> bytes:
    head = header.to_bytes()
    if len(head) > MAX_HEADER_BYTES:
        raise ValueError("header too large: %d bytes" % len(head))
    return HEADER_LEN_STRUCT.pack(len(head)) + head + payload


def decode(message: bytes) -> Tuple[FrameHeader, bytes]:
    if len(message) < HEADER_LEN_STRUCT.size:
        raise ValueError("message shorter than length prefix")
    (head_len,) = HEADER_LEN_STRUCT.unpack_from(message, 0)
    if head_len > MAX_HEADER_BYTES:
        raise ValueError("implausible header length %d — desync?" % head_len)
    start = HEADER_LEN_STRUCT.size
    end = start + head_len
    if len(message) < end:
        raise ValueError("truncated header")
    return FrameHeader.from_bytes(message[start:end]), message[end:]
