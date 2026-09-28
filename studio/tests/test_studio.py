import json

import openai
import pytest
from conftest import BASE_MODEL, INVITE, signup, upload, wait_until_done
from fastapi.testclient import TestClient
from kvpack import Cartridge

# ------------------------------------------------------------------ the whole journey


def test_signup_upload_build_chat_download_delete(client, tmp_path):
    headers = signup(client)

    queued = upload(client, headers, name="kestrel")
    assert queued["status"] == "queued" and queued["name"] == "kestrel"

    ready = wait_until_done(client, headers, queued["id"])
    assert ready["status"] == "ready", ready["error"]
    assert ready["progress"] == 1.0
    assert ready["corpus_tokens"] > 0
    assert set(ready["metrics"]) == {"no_context", "before_training", "after_training", "held_out_conversations"}
    assert ready["metrics"]["held_out_conversations"] == 1  # 5% of 8, but never zero

    models = [m["id"] for m in client.get("/v1/models", headers=headers).json()["data"]]
    assert models == [ready["id"], BASE_MODEL]

    # Chat by id and by name.
    for model in (ready["id"], "kestrel"):
        body = {"model": model, "messages": [{"role": "user", "content": "Who leads?"}], "max_tokens": 4}
        r = client.post("/v1/chat/completions", headers=headers, json=body)
        assert r.status_code == 200, r.text
        assert r.json()["model"] == model

    usage = client.get("/v1/usage", headers=headers).json()
    assert usage["builds"] == 1 and usage["chat_requests"] == 2 and usage["completion_tokens"] > 0

    # Download and load it: customers can self-host what they build.
    r = client.get(f"/v1/cartridges/{ready['id']}/download", headers=headers)
    assert r.status_code == 200
    (tmp_path / "dl.safetensors").write_bytes(r.content)
    assert Cartridge.load(tmp_path / "dl.safetensors").num_tokens == ready["num_tokens"]

    assert client.delete(f"/v1/cartridges/{ready['id']}", headers=headers).json()["deleted"]
    assert client.get(f"/v1/cartridges/{ready['id']}", headers=headers).status_code == 404
    body = {"model": ready["id"], "messages": [{"role": "user", "content": "Hi"}]}
    assert client.post("/v1/chat/completions", headers=headers, json=body).status_code == 404


def test_streaming_is_metered_and_hides_usage_unless_asked(client):
    headers = signup(client)
    ready = wait_until_done(client, headers, upload(client, headers)["id"])
    body = {"model": ready["id"], "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4, "stream": True}

    def events(extra):
        with client.stream("POST", "/v1/chat/completions", headers=headers, json={**body, **extra}) as r:
            assert r.status_code == 200
            return [line.removeprefix("data: ") for line in r.iter_lines() if line.startswith("data: ")]

    plain = events({})
    assert plain[-1] == "[DONE]"
    assert not any(json.loads(e).get("usage") for e in plain[:-1])

    with_usage = events({"stream_options": {"include_usage": True}})
    assert json.loads(with_usage[-2])["usage"]["completion_tokens"] > 0

    assert client.get("/v1/usage", headers=headers).json()["chat_requests"] == 2  # both were metered


def test_official_openai_sdk(client):
    headers = signup(client)
    ready = wait_until_done(client, headers, upload(client, headers, name="kestrel")["id"])
    sdk = openai.OpenAI(
        base_url="http://testserver/v1", api_key=headers["Authorization"].removeprefix("Bearer "), http_client=client
    )
    reply = sdk.chat.completions.create(model="kestrel", messages=[{"role": "user", "content": "Hi"}], max_tokens=4)
    assert reply.choices[0].message.role == "assistant"
    assert ready["id"] in [m.id for m in sdk.models.list()]


# ------------------------------------------------------------------ isolation between accounts


def test_accounts_cannot_see_each_others_cartridges(client):
    ana, ben = signup(client, "ana@example.com"), signup(client, "ben@example.com")
    ready = wait_until_done(client, ana, upload(client, ana, name="secret-plans")["id"])
    cid = ready["id"]

    assert client.get("/v1/cartridges", headers=ben).json()["data"] == []
    assert client.get(f"/v1/cartridges/{cid}", headers=ben).status_code == 404
    assert client.get(f"/v1/cartridges/{cid}/download", headers=ben).status_code == 404
    assert client.delete(f"/v1/cartridges/{cid}", headers=ben).status_code == 404
    for model in (cid, "secret-plans"):
        body = {"model": model, "messages": [{"role": "user", "content": "Hi"}]}
        assert client.post("/v1/chat/completions", headers=ben, json=body).status_code == 404
    assert cid not in [m["id"] for m in client.get("/v1/models", headers=ben).json()["data"]]


def test_upstream_errors_never_reveal_other_cartridges(client, app):
    ana, ben = signup(client, "ana@example.com"), signup(client, "ben@example.com")
    ana_id = wait_until_done(client, ana, upload(client, ana)["id"])["id"]
    ben_id = wait_until_done(client, ben, upload(client, ben)["id"])["id"]
    # Ben's cartridge file vanishes while the database still says it's ready.
    settings = app.state.settings
    (settings.storage_dir / "cartridges" / BASE_MODEL / f"{ben_id}.safetensors").unlink()
    body = {"model": ben_id, "messages": [{"role": "user", "content": "Hi"}]}
    r = client.post("/v1/chat/completions", headers=ben, json=body)
    assert r.status_code == 404
    assert ana_id not in r.text


# ------------------------------------------------------------------ auth


def test_auth(client):
    assert client.get("/v1/me").status_code == 401
    assert client.get("/v1/me", headers={"Authorization": "Bearer kvp_wrong"}).status_code == 401
    headers = signup(client)
    assert client.get("/v1/me", headers=headers).json()["email"] == "ana@example.com"


def test_first_account_needs_no_invite_but_later_ones_do(client):
    assert client.get("/v1/config").json()["first_account"] is True
    r = client.post("/v1/signup", json={"email": "owner@example.com"})
    assert r.status_code == 201  # whoever sets up the server
    assert client.get("/v1/config").json()["first_account"] is False
    assert client.post("/v1/signup", json={"email": "x@example.com", "invite_code": "nope"}).status_code == 403
    signup(client, "x@example.com")
    r = client.post("/v1/signup", json={"email": "X@example.com", "invite_code": INVITE})
    assert r.status_code == 409  # emails are case-insensitive


def test_keys_can_be_created_and_revoked(client):
    headers = signup(client)
    new = client.post("/v1/keys", headers=headers, json={"name": "ci"}).json()
    new_headers = {"Authorization": f"Bearer {new['api_key']}"}
    assert client.get("/v1/me", headers=new_headers).status_code == 200
    assert [k["name"] for k in client.get("/v1/keys", headers=headers).json()["data"]] == ["default", "ci"]
    assert client.delete(f"/v1/keys/{new['id']}", headers=headers).status_code == 200
    assert client.get("/v1/me", headers=new_headers).status_code == 401


# ------------------------------------------------------------------ validation and limits


def test_rejects_unsupported_files(client):
    headers = signup(client)
    files = [("files", ("photo.png", b"\x89PNG", "image/png"))]
    r = client.post("/v1/cartridges/upload", headers=headers, files=files)
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_file_type"


def test_rejects_unknown_base_model_and_bad_sizes(client):
    headers = signup(client)
    files = [("files", ("a.md", b"hello", "text/markdown"))]
    for data in ({"base_model": "gpt-4"}, {"tokens": "1"}, {"samples": "999999"}):
        assert client.post("/v1/cartridges/upload", headers=headers, files=files, data=data).status_code == 400


def test_upload_size_limit(client):
    headers = signup(client)
    big = [("files", ("big.txt", b"x" * (2 * 2**20), "text/plain"))]
    r = client.post("/v1/cartridges/upload", headers=headers, files=big)
    assert r.status_code == 413


def test_cartridge_limit_per_account(client):
    headers = signup(client)
    for _ in range(3):
        upload(client, headers)
    files = [("files", ("a.md", b"hello", "text/markdown"))]
    r = client.post("/v1/cartridges/upload", headers=headers, files=files)
    assert r.status_code == 403 and r.json()["error"]["code"] == "cartridge_limit_reached"


def test_chat_with_a_cartridge_that_is_not_ready(client, app):
    from kvpack_studio.db import CartridgeRecord

    headers = signup(client)
    account_id = client.get("/v1/me", headers=headers).json()["id"]
    record = app.state.db.add_cartridge(
        CartridgeRecord(account_id=account_id, name="wip", base_model=BASE_MODEL, num_tokens=16, num_samples=8,
                        status="building", progress=0.42)
    )  # fmt: skip
    body = {"model": record.id, "messages": [{"role": "user", "content": "Hi"}]}
    r = client.post("/v1/chat/completions", headers=headers, json=body)
    assert r.status_code == 409 and "42% done" in r.json()["error"]["message"]


def test_documents_without_text_fail_with_a_clear_message(client):
    headers = signup(client)
    files = [("files", ("empty.txt", b"   \n", "text/plain"))]
    record = upload(client, headers, files=files)
    done = wait_until_done(client, headers, record["id"])
    assert done["status"] == "failed"
    assert done["error"] == "The sources contain no readable text."


def test_uploaded_file_names_cannot_escape_the_storage_folder(client, app, tmp_path):
    headers = signup(client)
    files = [("files", ("../../../evil.md", b"# hi\n\nsome text", "text/markdown"))]
    record = upload(client, headers, files=files)
    stored = list((app.state.settings.storage_dir / "documents" / record["id"]).iterdir())
    assert [p.name for p in stored] == ["evil.md"]
    assert not (tmp_path / "evil.md").exists()


def test_public_config(client):
    config = client.get("/v1/config").json()
    assert config["base_models"] == [BASE_MODEL]
    assert config["signup_open"] is True
    assert ".md" in config["accepted_file_types"]


@pytest.mark.parametrize("path", ["/", "/index.html"])
def test_web_app_is_served(client, path):
    r = client.get(path)
    assert r.status_code == 200 and "kvpack" in r.text


def test_new_client_sees_same_database(app):
    # Sanity: the TestClient context manager runs the lifespan without errors.
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok"}


# ------------------------------------------------------------------ importing self-built cartridges


def test_import_a_cartridge_built_with_the_cli(client, tiny, tmp_path):
    model, _, chat_format = tiny
    cartridge = Cartridge.from_text(model, chat_format, "Some handbook text. " * 20, num_tokens=24)
    cartridge.metadata.update(name="handbook", corpus_tokens=500, eval_after={"agreement": 0.9, "loss": 0.5})
    path = cartridge.save(tmp_path / "handbook.safetensors")

    headers = signup(client)
    files = {"file": ("handbook.safetensors", path.read_bytes(), "application/octet-stream")}
    r = client.post("/v1/cartridges/import", headers=headers, files=files)
    assert r.status_code == 201, r.text
    record = r.json()
    assert (record["status"], record["name"], record["num_tokens"]) == ("ready", "handbook", 24)
    assert record["metrics"] == {"after_training": {"agreement": 0.9, "loss": 0.5}}

    body = {"model": "handbook", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 3}
    assert client.post("/v1/chat/completions", headers=headers, json=body).status_code == 200


def test_import_rejects_garbage_and_other_models(client, tiny, tmp_path):
    headers = signup(client)
    junk = {"file": ("x.safetensors", b"definitely not a cartridge", "application/octet-stream")}
    assert client.post("/v1/cartridges/import", headers=headers, files=junk).status_code == 400

    model, _, chat_format = tiny
    cartridge = Cartridge.from_text(model, chat_format, "text " * 50, num_tokens=16)
    cartridge.metadata["model"] = "some/other-model"
    path = cartridge.save(tmp_path / "other.safetensors")
    files = {"file": ("other.safetensors", path.read_bytes(), "application/octet-stream")}
    r = client.post("/v1/cartridges/import", headers=headers, files=files)
    assert r.status_code == 400 and "some/other-model" in r.json()["error"]["message"]


def test_interrupted_builds_are_requeued_on_startup(settings, tiny):
    from conftest import StubGenerator

    from kvpack_studio.app import create_local_app
    from kvpack_studio.db import CartridgeRecord, Database
    from kvpack_studio.storage import Storage

    # A build that was running when the server died.
    db = Database(settings.database_url)
    db.create_tables()
    account, key = db.create_account("ana@example.com")
    record = db.add_cartridge(
        CartridgeRecord(account_id=account.id, name="k", base_model=BASE_MODEL, num_tokens=16, num_samples=8,
                        status="building", progress=0.3)
    )  # fmt: skip
    Storage(settings.storage_dir).save_upload(record.id, "manual.md", b"# Manual\n\nThe bell is Old Gerda. " * 30)

    app = create_local_app(
        settings, load_model=lambda name: tiny, build_isolation="thread",
        make_generator=lambda model, chat_format: StubGenerator(chat_format),
        build_options={"synth_batch_size": 4, "train_batch_size": 4},
    )  # fmt: skip
    with TestClient(app) as client:
        done = wait_until_done(client, {"Authorization": f"Bearer {key}"}, record.id)
    app.state.runner.shutdown()
    assert done["status"] == "ready"


def test_async_build_runners_are_awaited(settings, tiny):
    """Modal's runner submits builds asynchronously; the API must await it."""
    from kvpack_studio.api import create_app
    from kvpack_studio.db import Database
    from kvpack_studio.inference import LocalUpstreams
    from kvpack_studio.storage import Storage

    class AsyncRunner:
        def __init__(self):
            self.specs = []

        async def submit(self, spec):
            self.specs.append(spec)

    db = Database(settings.database_url)
    db.create_tables()
    storage = Storage(settings.storage_dir)
    runner = AsyncRunner()
    app = create_app(settings, db, storage, runner, LocalUpstreams(storage, lambda name: tiny))
    with TestClient(app) as client:
        record = upload(client, signup(client))
    assert [spec.cartridge_id for spec in runner.specs] == [record["id"]]
    assert runner.specs[0].documents_dir.endswith(record["id"])
