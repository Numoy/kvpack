"""Where a cartridge's knowledge comes from: folders, Git repositories and websites.

A *source* fetches documents and returns a `Snapshot`: the documents plus a
fingerprint of their content. Cartridges remember their sources, so kvpack can
check later whether anything changed and rebuild only when it did.

    source_from_uri("./handbook")                           # a folder or a single file
    source_from_uri("https://github.com/acme/handbook")     # a Git repository
    source_from_uri("git+https://git.acme.dev/docs.git")    # any Git URL
    source_from_uri("https://docs.acme.dev/guide/")         # a website: pages under this path

Adding a connector (Notion, Google Drive, Confluence, ...) means subclassing
`Source` with `fetch()`, `to_config()` and `from_config()`, and registering it
in `SOURCE_TYPES`.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, ClassVar

TEXT_SUFFIXES = {
    ".txt", ".md", ".mdx", ".rst", ".tex", ".html", ".htm", ".csv", ".json", ".jsonl", ".yaml", ".yml",
    ".toml", ".xml", ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt", ".c", ".h",
    ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".sh", ".sql", ".ini", ".cfg",
}  # fmt: skip

# Folders that hold dependencies or build output, not knowledge.
SKIP_DIRS = {"node_modules", "__pycache__", "venv", ".venv", "dist", "build", "target", "site-packages"}

USER_AGENT = "kvpack (+https://github.com/Numoy/kvpack)"


class SourceError(Exception):
    """A source couldn't be fetched; the message is meant for the user."""


# --------------------------------------------------------------------------- documents


@dataclass
class Document:
    id: str  # a stable name: a relative path or a URL
    text: str


@dataclass
class Snapshot:
    """The documents a source returned at one point in time."""

    documents: list[Document]
    sources: list[dict[str, Any]] = field(default_factory=list)  # `Source.to_config()` of each source

    @property
    def fingerprint(self) -> str:
        """Changes whenever any document is added, removed or edited."""
        digest = hashlib.sha256()
        for doc in sorted(self.documents, key=lambda d: d.id):
            digest.update(doc.id.encode() + b"\0" + hashlib.sha256(doc.text.encode()).digest())
        return digest.hexdigest()

    def corpus(self) -> str:
        """All documents as one text, each preceded by a `## File: <id>` header."""
        docs = [d for d in self.documents if d.text.strip()]
        if not docs:
            raise ValueError("The sources contain no readable text.")
        if len(docs) == 1:
            return docs[0].text.strip()
        return "\n\n".join(f"## File: {d.id}\n\n{d.text.strip()}" for d in docs)

    @classmethod
    def merge(cls, snapshots: list[Snapshot]) -> Snapshot:
        return cls(
            documents=[d for s in snapshots for d in s.documents],
            sources=[c for s in snapshots for c in s.sources],
        )


# --------------------------------------------------------------------------- sources


class Source(ABC):
    type: ClassVar[str]

    @abstractmethod
    def fetch(self) -> Snapshot: ...

    @abstractmethod
    def to_config(self) -> dict[str, Any]:
        """A JSON-serializable description, without secrets, to rebuild this source later."""

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> Source: ...

    def describe(self) -> str:
        return f"{self.type}: {next(iter(v for k, v in self.to_config().items() if k != 'type'), '')}"


class FolderSource(Source):
    """Every readable text file in a folder (or a single file)."""

    type = "folder"

    def __init__(self, path: str | Path, include: list[str] | None = None, exclude: list[str] | None = None):
        self.path = Path(path)
        self.include, self.exclude = include or [], exclude or []

    def fetch(self) -> Snapshot:
        return Snapshot(self._documents(self.path), [self.to_config()])

    def _documents(self, root: Path, id_prefix: str = "") -> list[Document]:
        if root.is_file():
            text = read_file(root)
            return [Document(id_prefix + root.name, text)] if text else []
        if not root.is_dir():
            raise SourceError(f"{root} doesn't exist.")
        documents = []
        for f in sorted(root.rglob("*")):
            relative = f.relative_to(root)
            if not f.is_file() or any(p.startswith(".") or p in SKIP_DIRS for p in relative.parts):
                continue
            name = relative.as_posix()
            if self.include and not any(fnmatch.fnmatch(name, g) for g in self.include):
                continue
            if any(fnmatch.fnmatch(name, g) for g in self.exclude):
                continue
            text = read_file(f)
            if text and text.strip():
                documents.append(Document(id_prefix + name, text))
        return documents

    def to_config(self) -> dict[str, Any]:
        config = {"type": self.type, "path": str(self.path.resolve())}
        if self.include:
            config["include"] = self.include
        if self.exclude:
            config["exclude"] = self.exclude
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FolderSource:
        return cls(config["path"], config.get("include"), config.get("exclude"))


class GitSource(Source):
    """A Git repository, cloned fresh (shallow) on every fetch.

    For private repositories, put an access token in an environment variable and
    pass its *name* as `token_env`; the token itself is never stored.
    """

    type = "git"

    def __init__(
        self,
        url: str,
        branch: str | None = None,
        subdir: str | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        token_env: str | None = None,
    ):
        self.url, self.branch, self.subdir = url, branch, subdir
        self.include, self.exclude, self.token_env = include, exclude, token_env

    def fetch(self) -> Snapshot:
        with tempfile.TemporaryDirectory(prefix="kvpack-git-") as tmp:
            command = ["git", "clone", "--depth", "1", "--quiet"]
            if self.branch:
                command += ["--branch", self.branch]
            env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # fail instead of asking for a password
            try:
                subprocess.run(
                    [*command, self._authenticated_url(), tmp], check=True, capture_output=True, env=env, timeout=600
                )
            except FileNotFoundError:
                raise SourceError("Git isn't installed on this machine.") from None
            except subprocess.CalledProcessError as e:
                last_line = (e.stderr.decode(errors="replace").strip().splitlines() or ["git failed"])[-1]
                raise SourceError(f"Couldn't clone {self.url}: {self._redact(last_line)}") from None
            commit = subprocess.run(
                ["git", "-C", tmp, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            root = Path(tmp) / self.subdir if self.subdir else Path(tmp)
            folder = FolderSource(root, self.include, self.exclude)
            documents = folder._documents(root)
        return Snapshot(documents, [{**self.to_config(), "commit": commit}])

    def _authenticated_url(self) -> str:
        token = os.environ.get(self.token_env or "", "")
        if not token or not self.url.startswith("https://"):
            return self.url
        return self.url.replace("https://", f"https://x-access-token:{token}@", 1)

    def _redact(self, text: str) -> str:
        token = os.environ.get(self.token_env or "", "")
        return text.replace(token, "***") if token else text

    def to_config(self) -> dict[str, Any]:
        config = {"type": self.type, "url": self.url}
        for key in ("branch", "subdir", "include", "exclude", "token_env"):
            if getattr(self, key):
                config[key] = getattr(self, key)
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GitSource:
        return cls(
            config["url"],
            config.get("branch"),
            config.get("subdir"),
            config.get("include"),
            config.get("exclude"),
            config.get("token_env"),
        )


class WebSource(Source):
    """Pages of a website: the start page and every page linked below its path.

    Starting at https://docs.acme.dev/guide/ fetches /guide/... pages but not /blog/.
    Uses the site's sitemap.xml when there is one, respects robots.txt, and waits
    `delay` seconds between requests.
    """

    type = "web"

    def __init__(
        self,
        url: str,
        max_pages: int = 200,
        delay: float = 0.2,
        allow_url: Callable[[str], bool] | None = None,
    ):
        """`allow_url` can veto any request, including redirect targets (servers use it to
        keep crawls away from internal addresses)."""
        self.url, self.max_pages, self.delay = url, max_pages, delay
        self.allow_url = allow_url

    def fetch(self) -> Snapshot:
        start = urllib.parse.urlsplit(self.url)
        scope = start.path if start.path.endswith("/") else start.path.rsplit("/", 1)[0] + "/"
        robots = urllib.robotparser.RobotFileParser()
        robots_txt = self._get(f"{start.scheme}://{start.netloc}/robots.txt", any_type=True)
        if robots_txt:
            robots.parse(robots_txt[1].splitlines())
        else:
            robots = None

        def allowed(url: str) -> bool:
            parts = urllib.parse.urlsplit(url)
            return (
                parts.scheme in ("http", "https")
                and parts.netloc == start.netloc
                and parts.path.startswith(scope)
                and (robots is None or robots.can_fetch(USER_AGENT, url))
            )

        queue = [self.url] + [u for u in self._sitemap_urls(start) if allowed(u)]
        seen, documents = set(), []
        while queue and len(documents) < self.max_pages:
            url = _normalize(queue.pop(0))
            if url in seen or not allowed(url):
                continue
            seen.add(url)
            page = self._get(url)
            if page is None:
                continue
            content_type, body = page
            if "html" in content_type:
                text, links = html_to_text(body, base_url=url)
                queue += [link for link in links if _normalize(link) not in seen]
            else:
                text = body
            if text.strip():
                documents.append(Document(url, text))
            time.sleep(self.delay)
        if not documents:
            raise SourceError(f"No readable pages found at {self.url}.")
        return Snapshot(documents, [self.to_config()])

    def _get(self, url: str, any_type: bool = False) -> tuple[str, str] | None:
        """(content type, body) of a page, or None if it failed or isn't text."""
        if self.allow_url and not self.allow_url(url):
            return None
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        opener = urllib.request.build_opener(_CheckedRedirects(self.allow_url))
        try:
            with opener.open(request, timeout=30) as response:
                content_type = response.headers.get("content-type", "")
                if not any_type and not any(t in content_type for t in ("html", "text/plain", "markdown")):
                    return None
                raw = response.read(5 * 2**20)  # skip the tail of absurdly large pages
                charset = response.headers.get_content_charset() or "utf-8"
                return content_type, raw.decode(charset, errors="replace")
        except Exception:
            return None  # a broken link shouldn't fail the whole crawl

    def _sitemap_urls(self, start: urllib.parse.SplitResult) -> list[str]:
        page = self._get(f"{start.scheme}://{start.netloc}/sitemap.xml", any_type=True)
        if page is None or "<urlset" not in page[1]:
            return []
        return re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", page[1])

    def to_config(self) -> dict[str, Any]:
        return {"type": self.type, "url": self.url, "max_pages": self.max_pages}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> WebSource:
        return cls(config["url"], config.get("max_pages", 200))


SOURCE_TYPES: dict[str, type[Source]] = {s.type: s for s in (FolderSource, GitSource, WebSource)}


# --------------------------------------------------------------------------- helpers


def source_from_uri(uri: str) -> Source:
    """Guess the source type from what the user typed."""
    if uri.startswith("git+"):
        return GitSource(uri.removeprefix("git+"))
    if re.match(r"https?://", uri):
        parts = urllib.parse.urlsplit(uri)
        path_parts = [p for p in parts.path.split("/") if p]
        is_repo_host = parts.netloc in ("github.com", "gitlab.com", "bitbucket.org", "codeberg.org")
        if uri.endswith(".git") or (is_repo_host and len(path_parts) == 2):
            return GitSource(uri)
        return WebSource(uri)
    if Path(uri).exists():
        return FolderSource(uri)
    raise SourceError(f"{uri!r} isn't a file, folder, Git repository or web page.")


def source_from_config(config: dict[str, Any]) -> Source:
    try:
        return SOURCE_TYPES[config["type"]].from_config(config)
    except KeyError:
        raise SourceError(f"Unknown source type {config.get('type')!r}.") from None


def fetch_all(sources: list[Source]) -> Snapshot:
    return Snapshot.merge([s.fetch() for s in sources])


def read_file(path: Path) -> str | None:
    """The text of a document file, or None if it isn't a text format we read."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError("Reading PDFs needs pypdf: pip install 'kvpack[pdf]'") from e
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix in (".html", ".htm"):
        text, _ = html_to_text(path.read_text(encoding="utf-8", errors="replace"))
        return text
    if suffix in TEXT_SUFFIXES or suffix == "":
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return None
    return None


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, allow_url: Callable[[str], bool] | None):
        self.allow_url = allow_url

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if self.allow_url and not self.allow_url(newurl):
            raise urllib.error.URLError(f"redirect to a disallowed address: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _normalize(url: str) -> str:
    return urllib.parse.urldefrag(url)[0]


class _HTMLText(HTMLParser):
    """Visible text of an HTML page (headings kept as Markdown), plus its links."""

    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "button"}
    BLOCK = {"p", "div", "section", "article", "main", "li", "tr", "br", "pre", "blockquote", "table", "ul", "ol"}

    def __init__(self, base_url: str | None):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.parts: list[str] = []
        self.links: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in self.BLOCK:
            self.parts.append("\n")
        if tag == "a" and self.base_url:
            href = dict(attrs).get("href")
            if href and not href.startswith(("mailto:", "javascript:", "tel:")):
                self.links.append(urllib.parse.urljoin(self.base_url, href))

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif re.fullmatch(r"h[1-6]", tag) or tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str, base_url: str | None = None) -> tuple[str, list[str]]:
    parser = _HTMLText(base_url)
    parser.feed(html)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, parser.links
