"""Reaching the kvpack servers that answer chat requests.

Every base model gets its own `kvpack serve`, watching that model's cartridge folder.
The cloud API forwards chat requests to it with an ordinary httpx client:

* locally, the server runs in this process behind an in-memory ASGI transport,
  protected by an internal key users never see;
* on Modal, it runs in a GPU container with no public URL, reached through
  Modal RPC by `bridge.BridgeTransport`.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable

import httpx
from kvpack.server import create_app
from kvpack.store import CartridgeStore

from .bridge import BridgeTransport, Channel
from .storage import Storage


class Upstreams:
    """Maps a base model to an HTTP client for its kvpack server."""

    def client(self, base_model: str) -> httpx.AsyncClient:
        raise NotImplementedError

    async def aclose(self) -> None:
        pass


class LocalUpstreams(Upstreams):
    """kvpack servers in this process, created on first use."""

    def __init__(self, storage: Storage, load_model: Callable, max_output_tokens: int = 4096):
        self.storage, self.load_model, self.max_output_tokens = storage, load_model, max_output_tokens
        self.internal_key = secrets.token_urlsafe(24)
        self._clients: dict[str, httpx.AsyncClient] = {}

    def client(self, base_model: str) -> httpx.AsyncClient:
        if base_model not in self._clients:
            model, _, chat_format = self.load_model(base_model)
            directory = self.storage.cartridge_dir(base_model)
            directory.mkdir(parents=True, exist_ok=True)
            store = CartridgeStore(model.config.name_or_path, device=model.device, directory=directory)
            app = create_app(
                model, chat_format, store, api_keys=[self.internal_key], max_output_tokens=self.max_output_tokens
            )
            self._clients[base_model] = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://upstream",
                headers={"Authorization": f"Bearer {self.internal_key}"},
                timeout=600,
            )
        return self._clients[base_model]

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()


class BridgeUpstreams(Upstreams):
    """kvpack servers reached through a `bridge.Channel` (Modal RPC in production)."""

    def __init__(self, channels: dict[str, Channel]):
        self.channels = channels
        self._clients: dict[str, httpx.AsyncClient] = {}

    def client(self, base_model: str) -> httpx.AsyncClient:
        if base_model not in self._clients:
            self._clients[base_model] = httpx.AsyncClient(
                transport=BridgeTransport(self.channels[base_model]),
                base_url="http://inference",
                timeout=httpx.Timeout(900, connect=900),  # a cold start loads the model onto a GPU
            )
        return self._clients[base_model]

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
