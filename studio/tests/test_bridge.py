"""The Modal RPC path, exercised locally: API -> BridgeTransport -> call_asgi -> kvpack server."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from kvpack import Cartridge
from kvpack.server import create_app as create_kvpack_app

from kvpack_studio.bridge import BridgeTransport, call_asgi


@pytest.fixture
def upstream(tiny):
    model, _, chat_format = tiny
    cartridge = Cartridge.from_text(model, chat_format, "The bell is Old Gerda. " * 10, num_tokens=16)
    return create_kvpack_app(model, chat_format, {"kestrel": cartridge})


def channel_for(app):
    async def channel(method, path, headers, body):
        async for item in call_asgi(app, method, path, headers, body):
            yield item

    return channel


async def test_json_request_through_the_bridge(upstream):
    async with httpx.AsyncClient(transport=BridgeTransport(channel_for(upstream)), base_url="http://x") as client:
        models = await client.get("/v1/models")
        assert models.status_code == 200
        assert [m["id"] for m in models.json()["data"]] == ["kestrel", "tiny-qwen3"]

        body = {"model": "kestrel", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 3}
        r = await client.post("/v1/chat/completions", json=body)
        assert r.status_code == 200 and r.json()["object"] == "chat.completion"

        missing = await client.post("/v1/chat/completions", json={**body, "model": "nope"})
        assert missing.status_code == 404


async def test_streaming_through_the_bridge(upstream):
    body = {"model": "kestrel", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 3, "stream": True}
    async with httpx.AsyncClient(transport=BridgeTransport(channel_for(upstream)), base_url="http://x") as client:
        async with client.stream("POST", "/v1/chat/completions", json=body) as r:
            assert r.headers["content-type"].startswith("text/event-stream")
            events = [line.removeprefix("data: ") async for line in r.aiter_lines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    assert json.loads(events[0])["choices"][0]["delta"] == {"role": "assistant"}


def test_studio_api_over_the_bridge(settings, tiny, tmp_path):
    """The full Studio API with inference behind the bridge, as it runs on Modal."""
    from conftest import StubGenerator, signup, upload, wait_until_done
    from kvpack.store import CartridgeStore

    from kvpack_studio.api import create_app
    from kvpack_studio.builds import LocalBuildRunner
    from kvpack_studio.db import Database
    from kvpack_studio.inference import BridgeUpstreams
    from kvpack_studio.storage import Storage

    model, _, chat_format = tiny
    db = Database(settings.database_url)
    db.create_tables()
    storage = Storage(settings.storage_dir)
    runner = LocalBuildRunner(
        db, lambda name: tiny, lambda m, cf: StubGenerator(cf), isolation="thread", synth_batch_size=4,
        train_batch_size=4,
    )  # fmt: skip
    directory = storage.cartridge_dir("tiny-qwen3")
    directory.mkdir(parents=True, exist_ok=True)
    inference_app = create_kvpack_app(model, chat_format, CartridgeStore("tiny-qwen3", directory=directory))
    app = create_app(settings, db, storage, runner, BridgeUpstreams({"tiny-qwen3": channel_for(inference_app)}))

    with TestClient(app) as client:
        headers = signup(client)
        ready = wait_until_done(client, headers, upload(client, headers)["id"])
        body = {"model": ready["id"], "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 3}
        r = client.post("/v1/chat/completions", headers=headers, json=body)
        assert r.status_code == 200, r.text
        with client.stream("POST", "/v1/chat/completions", headers=headers, json={**body, "stream": True}) as s:
            lines = [line for line in s.iter_lines() if line.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        assert client.get("/v1/usage", headers=headers).json()["chat_requests"] == 2
    runner.shutdown()
