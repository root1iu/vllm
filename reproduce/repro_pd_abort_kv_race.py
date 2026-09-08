#!/usr/bin/env python3
"""
Abort PD requests while their remote KV transfer is still in flight.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import socket
import struct
import time
from collections import Counter
from dataclasses import dataclass
from typing import NamedTuple
from urllib.parse import urlsplit

CRLF = "\r\n"
BODY_UNIT = " kv-transfer-stability-test"
TOKENS_PER_UNIT = 8
MAX_TOKENS = 16
TIMEOUT = 120.0


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int
    path: str


def parse_endpoint(url: str) -> Endpoint:
    parsed = urlsplit(url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("--url must be an http:// URL")
    return Endpoint(parsed.hostname, parsed.port or 80, parsed.path or "/")


def build_request(
    endpoint: Endpoint, model: str, prompt_tokens: int, request_id: str
) -> bytes:
    """Serialize one streaming chat completion request, headers and body.

    Args:
        request_id: Sent as ``X-Request-Id`` and used as the unique prompt
            prefix that keeps the request off every prefix cache.

    Returns:
        The complete HTTP/1.1 request, ready to write to the socket.
    """
    prompt = f"{request_id}-{time.time_ns()}" + BODY_UNIT * max(
        1, prompt_tokens // TOKENS_PER_UNIT
    )
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "stream": True,
        },
        separators=(",", ":"),
    ).encode()

    host = endpoint.host if endpoint.port == 80 else f"{endpoint.host}:{endpoint.port}"
    headers = {
        "Host": host,
        "Content-Type": "application/json",
        "Content-Length": str(len(payload)),
        "Connection": "close",
        "X-Request-Id": request_id,
    }
    head = f"POST {endpoint.path} HTTP/1.1{CRLF}"
    head += "".join(f"{name}: {value}{CRLF}" for name, value in headers.items())
    return (head + CRLF).encode() + payload


async def read_status(reader: asyncio.StreamReader) -> int:
    line = await reader.readline()
    parts = line.decode(errors="replace").split()
    if len(parts) < 2 or not parts[1].isdigit():
        raise ConnectionError(f"malformed status line: {line[:120]!r}")
    return int(parts[1])


async def read_body(reader: asyncio.StreamReader) -> str:
    """Return the response body, to report why a request was rejected."""
    raw = await reader.read(4096)
    return raw.decode(errors="replace").split(CRLF * 2, 1)[-1].strip()


async def wait_first_token(reader: asyncio.StreamReader) -> None:
    """Return once the first SSE ``data:`` chunk of the response arrives."""
    buffer = b""
    in_body = False
    while True:
        chunk = await reader.read(4096)
        if not chunk:
            raise ConnectionError("closed before the first token")
        buffer += chunk
        if not in_body:
            head, separator, buffer = buffer.partition(b"\r\n\r\n")
            if not separator:
                buffer = head
                continue
            in_body = True
        if b"data:" in buffer:
            return
        buffer = buffer[-16:]


def reset(writer: asyncio.StreamWriter) -> None:
    """Close with a TCP RST so the disconnect reaches the ASGI server at once."""
    raw_socket = writer.get_extra_info("socket")
    if raw_socket is not None:
        raw_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
        )
    writer.close()


class Result(NamedTuple):
    kind: str
    ttft: float | None = None
    detail: str = ""


async def stream_request(
    args: argparse.Namespace,
    endpoint: Endpoint,
    request_id: str,
    abort_after: float | None = None,
) -> Result:
    """Send one request and race a reset against its first token.

    Args:
        abort_after: Seconds to wait after the request is accepted before
            resetting the connection. ``None`` waits for the first token
            instead, which is how the probe measures the TTFT.

    Returns:
        ``in_window`` if the reset landed before the first token, ``late`` if
        the first token won (carrying the measured TTFT), or ``http_<code>``
        with the response body if the request was rejected.
    """
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(endpoint.host, endpoint.port), timeout=TIMEOUT
    )
    try:
        writer.write(
            build_request(endpoint, args.model, args.prompt_tokens, request_id)
        )
        await writer.drain()
        # The status line proves the body was read in full and the request
        # reached the engine; resetting any earlier would just discard it.
        status = await asyncio.wait_for(read_status(reader), timeout=TIMEOUT)
        if status != 200:
            return Result(f"http_{status}", detail=await read_body(reader))
        accepted = time.monotonic()
        try:
            await asyncio.wait_for(
                wait_first_token(reader), timeout=abort_after or TIMEOUT
            )
        except asyncio.TimeoutError:
            if abort_after is None:
                raise
            return Result("in_window")
        return Result("late", ttft=time.monotonic() - accepted)
    finally:
        reset(writer)


async def run(args: argparse.Namespace) -> int:
    endpoint = parse_endpoint(args.url)
    if not 0 < args.abort_lo <= args.abort_hi:
        raise ValueError("require 0 < --abort-lo <= --abort-hi")

    try:
        probe = await stream_request(args, endpoint, "abort-race-probe")
    except (OSError, asyncio.TimeoutError) as exc:
        print(f"probe failed: {exc}")
        return 1
    if probe.ttft is None:
        print(f"probe failed: {probe.kind} {probe.detail}")
        return 1
    low, high = args.abort_lo * probe.ttft, args.abort_hi * probe.ttft
    print(
        f"ttft={probe.ttft * 1000:.0f}ms "
        f"abort_window={low * 1000:.0f}-{high * 1000:.0f}ms"
    )

    request_ids = iter(f"abort-race-{index}" for index in range(args.requests))
    tally: Counter[str] = Counter()

    async def worker() -> None:
        for request_id in request_ids:
            delay = random.uniform(low, high)
            try:
                result = await stream_request(args, endpoint, request_id, delay)
            except (OSError, asyncio.TimeoutError) as exc:
                result = Result("error", detail=str(exc))
            tally[result.kind] += 1
            line = f"{request_id} delay={delay * 1000:.0f}ms {result.kind}"
            print(f"{line} {result.detail}".rstrip())

    await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    print("summary: " + " ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    return 0 if tally["in_window"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="/v1/chat/completions URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=24000,
        help="approximate prompt length; must stay under max_model_len",
    )
    parser.add_argument(
        "--abort-lo",
        type=float,
        default=0.5,
        help="earliest reset, as a fraction of the measured TTFT",
    )
    parser.add_argument(
        "--abort-hi",
        type=float,
        default=2.0,
        help="latest reset, as a fraction of the measured TTFT",
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
