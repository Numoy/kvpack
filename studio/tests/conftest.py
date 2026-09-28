"""kvpack Studio end to end, on a laptop CPU.

A tiny random Qwen3 model stands in for the real base model and a stub generator
replaces self-study generation, so a full build takes a second or two.
"""

import time

import pytest
import torch
from fastapi.testclient import TestClient
from kvpack import ChatFormat
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from kvpack_studio.app import create_local_app
from kvpack_studio.config import Settings

BASE_MODEL = "tiny-qwen3"
INVITE = "beta-invite"
MANUAL = b"# Kestrel Station\n\nThe Station Lead is Oskar Lindqvist-Mbeki. The bell is called Old Gerda.\n" * 20


@pytest.fixture(scope="session")
def tiny():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
    )  # fmt: skip
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    model.config.name_or_path = BASE_MODEL
    return model, tokenizer, ChatFormat.from_tokenizer(tokenizer)


class StubGenerator:
    def __init__(self, chat_format):
        self.tokenizer, self.end, self.calls = chat_format.tokenizer, chat_format.end_of_turn_id, 0

    def generate(self, requests, max_new_tokens, temperature):
        self.calls += 1
        text = "Who is the Station Lead?" if self.calls % 2 else "Oskar Lindqvist-Mbeki."
        return [self.tokenizer.encode(text, add_special_tokens=False) + [self.end] for _ in requests]


@pytest.fixture
def settings(tmp_path):
    return Settings(
        database_url=f"sqlite:///{tmp_path}/studio.db",
        storage_dir=tmp_path / "data",
        base_models=[BASE_MODEL],
        invite_codes=[INVITE],
        max_cartridges_per_account=3,
        max_upload_mb=1,
    )


@pytest.fixture
def app(settings, tiny):
    def load_model(name):
        assert name == BASE_MODEL
        return tiny

    app = create_local_app(
        settings,
        load_model=load_model,
        make_generator=lambda model, chat_format: StubGenerator(chat_format),
        build_options={"synth_batch_size": 4, "train_batch_size": 4},
        build_isolation="thread",  # the tiny model lives in this process
        sync_check_seconds=None,
    )
    yield app
    app.state.runner.shutdown()


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def signup(client, email="ana@example.com") -> dict:
    r = client.post("/v1/signup", json={"email": email, "invite_code": INVITE})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['api_key']}"}


def upload(client, headers, files=None, **form) -> dict:
    files = files or [("files", ("manual.md", MANUAL, "text/markdown"))]
    data = {"tokens": "16", "samples": "8", **form}
    r = client.post("/v1/cartridges/upload", headers=headers, files=files, data=data)
    assert r.status_code == 202, r.text
    return r.json()


def wait_until_done(client, headers, cartridge_id, timeout=120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = client.get(f"/v1/cartridges/{cartridge_id}", headers=headers).json()
        if record["status"] in ("ready", "failed"):
            return record
        time.sleep(0.2)
    raise TimeoutError(f"{cartridge_id} still {record['status']}")
