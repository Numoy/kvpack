import json
import threading

import httpx
import pytest
from conftest import DOCUMENT
from fastapi.testclient import TestClient

from kvpack import Cartridge, OpenAIGenerator, complete
from kvpack.server import _Gate, _stream, create_app


@pytest.fixture(scope="module")
def cartridge(model, chat_format):
    return Cartridge.from_text(model, chat_format, DOCUMENT, num_tokens=16)


@pytest.fixture(scope="module")
def client(model, chat_format, cartridge):
    return TestClient(create_app(model, chat_format, {"kestrel": cartridge}, max_output_tokens=64))


def ask(model: str, **extra) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Who is the Station Lead?"}],
        "max_tokens": 5,
        "temperature": 0,
        **extra,
    }


# ------------------------------------------------------------------ basics


def test_models_lists_cartridges_and_base_model(client):
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == ["kestrel", "tiny-qwen3"]


def test_chat_completion(client):
    r = client.post("/v1/chat/completions", json=ask("kestrel"))
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(body["choices"][0]["message"]["content"], str)
    assert body["usage"]["completion_tokens"] <= 5
    assert body["choices"][0]["finish_reason"] in ("stop", "length")


def test_base_model_without_cartridge(client):
    assert client.post("/v1/chat/completions", json=ask("tiny-qwen3")).status_code == 200


def test_content_as_list_of_parts(client):
    messages = [{"role": "user", "content": [{"type": "text", "text": "Who is the Station Lead?"}]}]
    assert client.post("/v1/chat/completions", json=ask("kestrel", messages=messages)).status_code == 200


def test_unknown_openai_fields_are_ignored(client):
    r = client.post("/v1/chat/completions", json=ask("kestrel", presence_penalty=0.1, user="abc"))
    assert r.status_code == 200


def test_streaming(client):
    with client.stream("POST", "/v1/chat/completions", json=ask("kestrel", stream=True)) as r:
        events = [line.removeprefix("data: ") for line in r.iter_lines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")


# ------------------------------------------------------------------ errors are OpenAI-shaped


def error_of(r) -> dict:
    return r.json()["error"]


def test_unknown_model(client):
    r = client.post("/v1/chat/completions", json=ask("nope"))
    assert r.status_code == 404
    assert error_of(r)["code"] == "model_not_found"


def test_max_tokens_over_limit(client):
    r = client.post("/v1/chat/completions", json=ask("kestrel", max_tokens=65))
    assert r.status_code == 400
    assert "at most 64" in error_of(r)["message"]


def test_n_must_be_one(client):
    assert client.post("/v1/chat/completions", json=ask("kestrel", n=2)).status_code == 400


def test_empty_messages(client):
    assert client.post("/v1/chat/completions", json=ask("kestrel", messages=[])).status_code == 400


def test_malformed_body(client):
    r = client.post("/v1/chat/completions", json={"model": "kestrel"})
    assert r.status_code == 400
    assert error_of(r)["type"] == "invalid_request_error"


def test_context_length_exceeded(model, chat_format, cartridge, monkeypatch):
    monkeypatch.setattr(model.config, "max_position_embeddings", 40)
    client = TestClient(create_app(model, chat_format, {"kestrel": cartridge}))
    r = client.post("/v1/chat/completions", json=ask("kestrel", max_tokens=20))
    assert r.status_code == 400
    assert error_of(r)["code"] == "context_length_exceeded"


# ------------------------------------------------------------------ auth


def test_api_keys(model, chat_format, cartridge):
    client = TestClient(create_app(model, chat_format, {"kestrel": cartridge}, api_keys=["secret"]))
    assert client.get("/health").status_code == 200  # health stays open for load balancers
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    r = client.post("/v1/chat/completions", json=ask("kestrel"), headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200


# ------------------------------------------------------------------ scheduling


def test_gate_rejects_when_queue_is_full():
    gate = _Gate(max_queue=1)
    gate.enter()  # running
    gate.enter()  # waiting
    with pytest.raises(Exception) as err:
        gate.enter()
    assert err.value.status == 429
    gate.leave()
    gate.enter()  # room again


def test_client_disconnect_cancels_generation_and_frees_the_model(model, chat_format, cartridge):
    gate = _Gate(max_queue=4)
    gate.enter()
    options = dict(cartridge=cartridge, max_new_tokens=200, temperature=0.0, top_p=1.0)
    meta = dict(id="x", created=0, model="kestrel")
    messages = [{"role": "user", "content": "Tell me everything."}]
    stream = _stream(model, chat_format, messages, options, gate, meta, 5, 0.0)

    next(stream)  # role chunk: generation is running
    stream.close()  # the client disconnects

    assert gate.in_flight == 0
    assert gate.gpu.acquire(blocking=False)  # the model is free again
    gate.gpu.release()


def test_cancel_stops_generation_early(model, chat_format, cartridge):
    cancel = threading.Event()
    cancel.set()
    messages = [{"role": "user", "content": "Hi"}]
    result = complete(model, chat_format, messages, cartridge=cartridge, max_new_tokens=50, cancel=cancel)
    assert result.finish_reason == "cancelled"
    assert len(result.token_ids) < 50


# ------------------------------------------------------------------ remote generation


def test_openai_generator_talks_to_an_openai_compatible_server(model, chat_format):
    """Self-study data can come from any OpenAI-compatible server. Use our own."""
    app = create_app(model, chat_format, {})
    generator = OpenAIGenerator(
        chat_format, "http://test/v1", "tiny-qwen3", transport=httpx.ASGITransport(app=app), concurrency=2
    )
    outputs = generator.generate(
        [("Some context", [{"role": "user", "content": "Hi"}])] * 3, max_new_tokens=4, temperature=0
    )
    assert len(outputs) == 3
    assert all(isinstance(ids, list) and ids for ids in outputs)
    assert outputs[0] == outputs[1] == outputs[2]  # greedy decoding is deterministic


# ------------------------------------------------------------------ the official OpenAI SDK


def test_official_openai_sdk(model, chat_format, cartridge):
    import openai

    app = create_app(model, chat_format, {"kestrel": cartridge}, api_keys=["secret"])
    sdk = openai.OpenAI(base_url="http://testserver/v1", api_key="secret", http_client=TestClient(app))

    assert [m.id for m in sdk.models.list()] == ["kestrel", "tiny-qwen3"]

    reply = sdk.chat.completions.create(
        model="kestrel", messages=[{"role": "user", "content": "Hi"}], max_tokens=4, temperature=0
    )
    assert reply.choices[0].message.role == "assistant"
    assert reply.usage.completion_tokens <= 4

    stream = sdk.chat.completions.create(
        model="kestrel", messages=[{"role": "user", "content": "Hi"}], max_tokens=4, temperature=0, stream=True
    )
    streamed = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
    assert streamed.strip() == reply.choices[0].message.content  # same greedy answer either way

    with pytest.raises(openai.NotFoundError):
        sdk.chat.completions.create(model="nope", messages=[{"role": "user", "content": "Hi"}])
    with pytest.raises(openai.AuthenticationError):
        openai.OpenAI(base_url="http://testserver/v1", api_key="bad", http_client=TestClient(app)).models.list()


def test_stream_include_usage(client):
    body = ask("kestrel", stream=True, stream_options={"include_usage": True})
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        events = [line.removeprefix("data: ") for line in r.iter_lines() if line.startswith("data: ")]
    usage_chunk = json.loads(events[-2])
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["completion_tokens"] <= 5
    assert usage_chunk["usage"]["total_tokens"] == (
        usage_chunk["usage"]["prompt_tokens"] + usage_chunk["usage"]["completion_tokens"]
    )
