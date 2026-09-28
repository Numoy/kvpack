"""Tunnelling HTTP requests to an ASGI app over any async-generator channel.

In production the kvpack inference servers run inside Modal GPU containers with
no public URL. The cloud API still talks to them with an ordinary `httpx` client:
`BridgeTransport` turns each request into a call of a remote generator
(`Inference.handle`), and `call_asgi` runs the request against the kvpack app on
the other side, streaming the response back chunk by chunk.

    API container                                 GPU container
    httpx.AsyncClient ── BridgeTransport ──RPC──▶ call_asgi(kvpack app)
                      ◀── status, then chunks ───

Both halves are plain Python, so they're tested locally without Modal.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

# Yields {"status": int, "headers": [[name, value], ...]} first, then body chunks (bytes).
Channel = Callable[[str, str, list[list[str]], bytes], AsyncIterator[Any]]


async def call_asgi(app, method: str, path: str, headers: list[list[str]], body: bytes) -> AsyncIterator[Any]:
    """Run one HTTP request against an ASGI app, yielding the status line then body chunks."""
    raw_path, _, query = path.partition("?")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": raw_path,
        "raw_path": raw_path.encode(),
        "query_string": query.encode(),
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
        "server": ("bridge", 80),
        "client": ("bridge", 0),
    }
    messages: asyncio.Queue = asyncio.Queue()
    request_sent = False
    finished = asyncio.Event()

    async def receive() -> dict:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await finished.wait()  # only returns once we're done, i.e. the client "disconnected"
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        await messages.put(message)

    async def run() -> None:
        try:
            await app(scope, receive, send)
        finally:
            await messages.put(None)

    task = asyncio.create_task(run())
    try:
        while True:
            message = await messages.get()
            if message is None:
                break
            if message["type"] == "http.response.start":
                headers_out = [[k.decode(), v.decode()] for k, v in message.get("headers", [])]
                yield {"status": message["status"], "headers": headers_out}
            elif message["type"] == "http.response.body":
                if message.get("body"):
                    yield message["body"]
                if not message.get("more_body"):
                    break
    finally:
        finished.set()  # tells a streaming response that the client went away, if it's still running
        await task


class BridgeTransport(httpx.AsyncBaseTransport):
    """An httpx transport that sends requests through a `Channel` instead of the network."""

    def __init__(self, channel: Channel):
        self.channel = channel

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        path = request.url.raw_path.decode()
        headers = [[k, v] for k, v in request.headers.items()]
        stream = aiter(self.channel(request.method, path, headers, body))
        head = await anext(stream)
        return httpx.Response(head["status"], headers=head["headers"], stream=_BodyStream(stream))


class _BodyStream(httpx.AsyncByteStream):
    def __init__(self, stream: AsyncIterator[Any]):
        self.stream = stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self.stream:
            yield chunk

    async def aclose(self) -> None:
        close = getattr(self.stream, "aclose", None)
        if close:
            await close()
