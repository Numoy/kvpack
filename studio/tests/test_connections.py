"""Connected sources: building from them, syncing, and keeping the server safe."""

import functools
import http.server
import threading
import time

import pytest
from conftest import BASE_MODEL, signup, wait_until_done
from fastapi.testclient import TestClient

from kvpack_studio.sources import InvalidSource, is_public_url, validate


@pytest.fixture
def docs_site(tmp_path):
    """A small local website, served over HTTP."""
    root = tmp_path / "site"
    (root / "guide").mkdir(parents=True)
    (root / "guide" / "index.html").write_text(
        '<h1>Kestrel guide</h1><p>The bell is called Old Gerda.</p><a href="power.html">Power</a>' * 5
    )
    (root / "guide" / "power.html").write_text("<p>The geothermal loop delivers 410 kilowatts.</p>" * 5)

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{server.server_port}/guide/index.html"
    server.shutdown()


@pytest.fixture
def open_settings(settings, tmp_path):
    """Settings for a trusted single-user setup: local URLs and a folder root allowed."""
    settings.allow_private_urls = True
    settings.folder_roots = [tmp_path.resolve()]
    return settings


def create(client, headers, sources, **extra):
    body = {"name": "kestrel", "sources": sources, "tokens": 16, "samples": 8, **extra}
    r = client.post("/v1/cartridges", headers=headers, json=body)
    assert r.status_code == 202, r.text
    return r.json()


# ------------------------------------------------------------------ building from sources


def test_build_from_a_website(open_settings, make_client, docs_site):
    _, url = docs_site
    with make_client(open_settings) as client:
        headers = signup(client)
        record = create(client, headers, [{"type": "web", "url": url}])
        done = wait_until_done(client, headers, record["id"])
    assert done["status"] == "ready", done["error"]
    assert done["version"] == 1 and done["servable"]
    assert done["sources"] == [{"type": "web", "url": url, "max_pages": 200}]


def test_build_from_a_server_folder(open_settings, make_client, tmp_path):
    folder = tmp_path / "handbook"
    folder.mkdir()
    (folder / "manual.md").write_text("# Manual\n\nThe bell is Old Gerda. " * 20)
    with make_client(open_settings) as client:
        headers = signup(client)
        done = wait_until_done(
            client, headers, create(client, headers, [{"type": "folder", "path": str(folder)}])["id"]
        )
    assert done["status"] == "ready", done["error"]


# ------------------------------------------------------------------ sync


def test_sync_only_rebuilds_when_sources_change(open_settings, make_client, docs_site):
    root, url = docs_site
    with make_client(open_settings) as client:
        headers = signup(client)
        cid = create(client, headers, [{"type": "web", "url": url}])["id"]
        first = wait_until_done(client, headers, cid)
        assert first["version"] == 1

        # Nothing changed: the sync finishes without a new version.
        assert client.post(f"/v1/cartridges/{cid}/sync", headers=headers).status_code == 202
        same = wait_until_done(client, headers, cid)
        assert same["version"] == 1 and same["last_synced_at"] >= first["last_synced_at"]
        usage = client.get("/v1/usage", headers=headers).json()
        assert usage["builds"] == 1  # the no-op sync didn't count as a build

        # The website changed: a new version is built.
        (root / "guide" / "power.html").write_text("<p>The geothermal loop now delivers 500 kilowatts.</p>" * 5)
        client.post(f"/v1/cartridges/{cid}/sync", headers=headers)
        changed = wait_until_done(client, headers, cid)
        assert changed["version"] == 2

        # force=true rebuilds even without changes.
        client.post(f"/v1/cartridges/{cid}/sync?force=true", headers=headers)
        assert wait_until_done(client, headers, cid)["version"] == 3


def test_previous_version_keeps_serving_when_an_update_fails(open_settings, make_client, docs_site):
    root, url = docs_site
    with make_client(open_settings) as client:
        headers = signup(client)
        cid = create(client, headers, [{"type": "web", "url": url}])["id"]
        wait_until_done(client, headers, cid)
        for page in (root / "guide").iterdir():  # the site goes empty
            page.write_text("")
        client.post(f"/v1/cartridges/{cid}/sync", headers=headers)
        failed = wait_until_done(client, headers, cid)
        assert failed["status"] == "failed" and failed["version"] == 1
        body = {"model": cid, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 3}
        assert client.post("/v1/chat/completions", headers=headers, json=body).status_code == 200


def test_automatic_sync_is_scheduled(open_settings, make_client, docs_site):
    from datetime import timedelta

    from kvpack_studio.db import now

    root, url = docs_site
    with make_client(open_settings, sync_check_seconds=0.2) as client:
        headers = signup(client)
        cid = create(client, headers, [{"type": "web", "url": url}], sync_every_hours=24)["id"]
        wait_until_done(client, headers, cid)
        (root / "guide" / "power.html").write_text("<p>Now 600 kilowatts.</p>" * 5)
        # pretend the last sync was two days ago
        client.app.state.db.update_cartridge(cid, last_synced_at=now() - timedelta(days=2))
        deadline = time.time() + 60
        while time.time() < deadline:
            record = client.get(f"/v1/cartridges/{cid}", headers=headers).json()
            if record["version"] == 2 and record["status"] == "ready":
                break
            time.sleep(0.2)
        assert record["version"] == 2


def test_sync_settings_can_be_changed(open_settings, make_client, docs_site):
    _, url = docs_site
    with make_client(open_settings) as client:
        headers = signup(client)
        cid = create(client, headers, [{"type": "web", "url": url}])["id"]
        r = client.patch(f"/v1/cartridges/{cid}", headers=headers, json={"sync_every_hours": 6, "name": "guide"})
        assert (r.json()["sync_every_hours"], r.json()["name"]) == (6, "guide")
        r = client.patch(f"/v1/cartridges/{cid}", headers=headers, json={"sync_every_hours": 0})
        assert r.json()["sync_every_hours"] is None
        assert (
            client.patch(f"/v1/cartridges/{cid}", headers=headers, json={"sync_every_hours": 9999}).status_code == 400
        )


def test_uploads_can_be_resynced_but_not_auto_synced(client):
    headers = signup(client)
    files = [("files", ("manual.md", b"# Manual\n\nThe bell is Old Gerda. " * 20, "text/markdown"))]
    r = client.post("/v1/cartridges/upload", headers=headers, files=files, data={"tokens": "16", "samples": "8"})
    cid = r.json()["id"]
    assert r.json()["sources"] == [{"type": "upload", "files": ["manual.md"]}]
    wait_until_done(client, headers, cid)
    assert client.patch(f"/v1/cartridges/{cid}", headers=headers, json={"sync_every_hours": 24}).status_code == 400


# ------------------------------------------------------------------ security


def test_private_addresses_are_rejected_by_default(client):
    headers = signup(client)
    for url in ("http://127.0.0.1:8080/", "http://169.254.169.254/latest/meta-data/", "http://localhost/"):
        r = client.post("/v1/cartridges", headers=headers, json={"name": "x", "sources": [{"type": "web", "url": url}]})
        assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_source", url


def test_folder_sources_are_off_by_default_and_confined(settings, tmp_path):
    with pytest.raises(InvalidSource, match="turned off"):
        validate({"type": "folder", "path": str(tmp_path)}, settings)
    settings.folder_roots = [(tmp_path / "allowed").resolve()]
    (tmp_path / "allowed").mkdir()
    with pytest.raises(InvalidSource, match="must be inside"):
        validate({"type": "folder", "path": "/etc"}, settings)
    with pytest.raises(InvalidSource, match="must be inside"):
        validate({"type": "folder", "path": str(tmp_path / "allowed" / ".." / "..")}, settings)
    assert validate({"type": "folder", "path": str(tmp_path / "allowed")}, settings)["type"] == "folder"


def test_git_sources_need_https_and_allowlisted_tokens(settings):
    settings.allow_private_urls = True  # skip DNS in this test
    with pytest.raises(InvalidSource, match="https"):
        validate({"type": "git", "url": "ssh://git@github.com/a/b"}, settings)
    with pytest.raises(InvalidSource, match="isn't allowed"):
        validate({"type": "git", "url": "https://github.com/a/b", "token_env": "KVPACK_STUDIO_DATABASE_URL"}, settings)
    settings.git_token_envs = ["DOCS_TOKEN"]
    clean = validate({"type": "git", "url": "https://github.com/a/b", "token_env": "DOCS_TOKEN"}, settings)
    assert clean == {"type": "git", "url": "https://github.com/a/b", "token_env": "DOCS_TOKEN"}


def test_is_public_url():
    assert not is_public_url("http://127.0.0.1/")
    assert not is_public_url("http://10.0.0.5/")
    assert not is_public_url("http://[::1]/")
    assert not is_public_url("file:///etc/passwd")
    assert not is_public_url("http://no-such-host.invalid/")


def test_unknown_source_types_are_rejected(settings):
    with pytest.raises(InvalidSource):
        validate({"type": "ftp", "url": "ftp://x"}, settings)
    with pytest.raises(InvalidSource):
        validate("https://example.com", settings)


def test_config_describes_available_sources(client):
    assert client.get("/v1/config").json()["sources"] == {"folder": False, "folder_roots": [], "git_token_envs": []}


def test_chat_on_base_model_name_works(client):
    headers = signup(client)
    body = {"model": BASE_MODEL, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 2}
    assert client.post("/v1/chat/completions", headers=headers, json=body).status_code == 200


@pytest.fixture
def make_client(tiny):
    from conftest import StubGenerator

    from kvpack_studio.app import create_local_app

    apps = []

    def _make(settings, sync_check_seconds=None):
        app = create_local_app(
            settings, load_model=lambda name: tiny, build_isolation="thread", sync_check_seconds=sync_check_seconds,
            make_generator=lambda model, chat_format: StubGenerator(chat_format),
            build_options={"synth_batch_size": 4, "train_batch_size": 4},
        )  # fmt: skip
        apps.append(app)
        return TestClient(app)

    yield _make
    for app in apps:
        if getattr(app.state, "stop_sync", None):
            app.state.stop_sync.set()
        app.state.runner.shutdown()
