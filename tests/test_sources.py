import functools
import http.server
import subprocess
import threading

import pytest

from kvpack.sources import (
    FolderSource,
    GitSource,
    Snapshot,
    SourceError,
    WebSource,
    html_to_text,
    source_from_config,
    source_from_uri,
)

# ------------------------------------------------------------------ folders


def test_folder_reads_text_and_skips_junk(tmp_path):
    (tmp_path / "guide.md").write_text("# Guide\n\nHello")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("junk")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("junk")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")
    (tmp_path / "page.html").write_text("<html><script>x()</script><h1>Title</h1><p>Body text</p></html>")

    docs = {d.id: d.text for d in FolderSource(tmp_path).fetch().documents}
    assert sorted(docs) == ["guide.md", "page.html", "src/app.py"]
    assert docs["page.html"] == "# Title\n\nBody text"  # HTML files are converted to text


def test_folder_include_and_exclude(tmp_path):
    for name in ("a.md", "b.md", "c.txt"):
        (tmp_path / name).write_text(name)
    source = FolderSource(tmp_path, include=["*.md"], exclude=["b.*"])
    assert [d.id for d in source.fetch().documents] == ["a.md"]


def test_fingerprint_tracks_content(tmp_path):
    (tmp_path / "a.md").write_text("one")
    first = FolderSource(tmp_path).fetch().fingerprint
    assert FolderSource(tmp_path).fetch().fingerprint == first
    (tmp_path / "a.md").write_text("two")
    assert FolderSource(tmp_path).fetch().fingerprint != first


def test_corpus_headers(tmp_path):
    with pytest.raises(ValueError):
        Snapshot([]).corpus()
    (tmp_path / "a.md").write_text("Alpha")
    assert FolderSource(tmp_path).fetch().corpus() == "Alpha"
    (tmp_path / "b.md").write_text("Beta")
    assert FolderSource(tmp_path).fetch().corpus() == "## File: a.md\n\nAlpha\n\n## File: b.md\n\nBeta"


def test_config_roundtrip(tmp_path):
    source = FolderSource(tmp_path, include=["*.md"])
    again = source_from_config(source.to_config())
    assert isinstance(again, FolderSource) and again.include == ["*.md"]


# ------------------------------------------------------------------ git


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    (path / "docs").mkdir()
    (path / "docs" / "guide.md").write_text("The bell is Old Gerda.")
    (path / "README.md").write_text("Readme")

    def git(*args):
        subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("-c", "user.email=a@b.c", "-c", "user.name=t", "add", ".")
    git("-c", "user.email=a@b.c", "-c", "user.name=t", "commit", "-q", "-m", "init")
    return path


def test_git_clone(repo):
    snapshot = GitSource(str(repo)).fetch()
    assert sorted(d.id for d in snapshot.documents) == ["README.md", "docs/guide.md"]
    assert len(snapshot.sources[0]["commit"]) == 40


def test_git_subdir(repo):
    snapshot = GitSource(str(repo), subdir="docs").fetch()
    assert [d.id for d in snapshot.documents] == ["guide.md"]


def test_git_errors_are_readable_and_hide_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("KVPACK_TEST_TOKEN", "s3cret")
    source = GitSource(str(tmp_path / "missing"), token_env="KVPACK_TEST_TOKEN")
    with pytest.raises(SourceError) as err:
        source.fetch()
    assert "Couldn't clone" in str(err.value)
    assert "s3cret" not in str(err.value)
    assert "s3cret" not in str(source.to_config())  # only the variable's name is stored


# ------------------------------------------------------------------ web


@pytest.fixture
def site(tmp_path):
    pages = {
        "guide/index.html": '<h1>Guide</h1><a href="setup.html">Setup</a> <a href="/blog/post.html">Blog</a>'
        '<a href="secret.html">Secret</a><nav>Menu</nav>',
        "guide/setup.html": "<h2>Setup</h2><p>Install it.</p>",
        "guide/secret.html": "<p>Hidden</p>",
        "guide/from-sitemap.html": "<p>Only in the sitemap</p>",
        "blog/post.html": "<p>Out of scope</p>",
        "robots.txt": "User-agent: *\nDisallow: /guide/secret.html\n",
    }
    for name, body in pages.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(tmp_path)))
    base = f"http://127.0.0.1:{server.server_port}"
    (tmp_path / "sitemap.xml").write_text(
        f'<?xml version="1.0"?><urlset><url><loc>{base}/guide/from-sitemap.html</loc></url></urlset>'
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield base
    server.shutdown()


def test_web_crawl_stays_in_scope_and_respects_robots(site):
    snapshot = WebSource(f"{site}/guide/index.html", delay=0).fetch()
    ids = sorted(d.id.removeprefix(site) for d in snapshot.documents)
    assert ids == ["/guide/from-sitemap.html", "/guide/index.html", "/guide/setup.html"]
    texts = {d.id.removeprefix(site): d.text for d in snapshot.documents}
    assert "Menu" not in texts["/guide/index.html"]  # navigation is dropped
    assert texts["/guide/setup.html"] == "## Setup\n\nInstall it."


def test_web_max_pages(site):
    assert len(WebSource(f"{site}/guide/index.html", max_pages=1, delay=0).fetch().documents) == 1


def test_web_nothing_found():
    with pytest.raises(SourceError):
        WebSource("http://127.0.0.1:9/nothing/", delay=0).fetch()


# ------------------------------------------------------------------ helpers


def test_html_to_text_and_links():
    text, links = html_to_text(
        '<html><head><style>p{}</style></head><body><h1>A</h1><p>b <a href="/x">c</a></p></body></html>',
        base_url="https://site.dev/docs/",
    )
    assert text == "# A\n\nb c"
    assert links == ["https://site.dev/x"]


@pytest.mark.parametrize(
    "uri,kind",
    [
        ("https://github.com/acme/handbook", GitSource),
        ("https://gitlab.com/acme/handbook", GitSource),
        ("git+https://git.acme.dev/docs", GitSource),
        ("https://git.acme.dev/docs.git", GitSource),
        ("https://docs.acme.dev/guide/", WebSource),
        ("https://github.com/acme/handbook/wiki/Home", WebSource),
    ],
)
def test_source_from_uri(uri, kind):
    assert isinstance(source_from_uri(uri), kind)


def test_source_from_uri_folder_and_missing(tmp_path):
    assert isinstance(source_from_uri(str(tmp_path)), FolderSource)
    with pytest.raises(SourceError):
        source_from_uri(str(tmp_path / "nope"))


def test_web_allow_url_blocks_pages_and_redirects(site, tmp_path):
    blocked = []

    def allow(url):
        ok = "setup" not in url
        if not ok:
            blocked.append(url)
        return ok

    snapshot = WebSource(f"{site}/guide/index.html", delay=0, allow_url=allow).fetch()
    assert not any("setup" in d.id for d in snapshot.documents)
    assert blocked
