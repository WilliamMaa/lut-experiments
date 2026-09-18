#!/usr/bin/env python3
"""zmq wire protocol between scheduler (ROUTER) and workers (DEALER).

Multipart message: [header_json (utf-8)] [payload_bytes (optional)].
payload carries a serialized KV object (icn_proto.kvcodec.dumps) and is
present iff header["has_payload"] is true. All control fields are JSON;
payloads never touch the JSON header so big tensors stay binary.
"""

import json

import zmq

ENC = "utf-8"


def send(sock, header: dict, payload: bytes | None = None, ident=None):
    if payload is not None:
        header = dict(header, has_payload=True)
    frames = []
    if ident is not None:
        frames.append(ident)
    frames.append(json.dumps(header).encode(ENC))
    frames.append(payload if payload is not None else b"")
    sock.send_multipart(frames)


def recv(sock):
    """Returns (ident, header, payload|None). ident is None on non-ROUTER."""
    frames = sock.recv_multipart()
    ident = frames[0] if len(frames) == 3 else None
    off = 1 if ident is not None else 0
    header = json.loads(frames[off].decode(ENC))
    payload = frames[off + 1] if header.get("has_payload") else None
    return ident, header, payload


def router(ctx, addr):
    s = ctx.socket(zmq.ROUTER)
    s.bind(addr)
    return s


def dealer(ctx, addr, identity=None):
    s = ctx.socket(zmq.DEALER)
    if identity is not None:
        s.setsockopt(zmq.IDENTITY, identity.encode("utf-8"))
    s.connect(addr)
    return s
