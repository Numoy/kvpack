"""Validating the sources users connect, and building them safely.

Studio is multi-user, so a source is untrusted input:

* folders must lie inside `folder_roots`, which an admin sets (off by default);
* websites and Git remotes must be public addresses unless `allow_private_urls` is on,
  checked again on every request and redirect during a crawl;
* Git tokens are referred to by environment variable name, from an admin-set allowlist.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.parse
from pathlib import Path
from typing import Any

from kvpack.sources import FolderSource, GitSource, Source, WebSource

from .config import Settings

UPLOAD = "upload"


class InvalidSource(ValueError):
    """Shown to the user as-is."""


def is_public_url(url: str) -> bool:
    """True if the URL is http(s) and its host resolves only to public addresses."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    except OSError:
        return False
    return all(ipaddress.ip_address(info[4][0]).is_global for info in infos)


def validate(config: Any, settings: Settings) -> dict[str, Any]:
    """Check a source a user wants to connect. Returns the normalized config."""
    if not isinstance(config, dict) or not isinstance(config.get("type"), str):
        raise InvalidSource("Each source needs a type: git, web or folder.")
    kind = config["type"]

    if kind == "git":
        url = str(config.get("url", "")).strip()
        if not url.startswith("https://"):
            raise InvalidSource("Git sources need an https:// URL.")
        if not settings.allow_private_urls and not is_public_url(url):
            raise InvalidSource(f"{url} isn't a public address.")
        token_env = config.get("token_env")
        if token_env and token_env not in settings.git_token_envs:
            raise InvalidSource(f"The token variable {token_env!r} isn't allowed on this server.")
        clean = {"type": "git", "url": url}
        for key in ("branch", "subdir"):
            if config.get(key):
                clean[key] = str(config[key]).strip("/ ")
        if token_env:
            clean["token_env"] = token_env
        return clean

    if kind == "web":
        url = str(config.get("url", "")).strip()
        if not url.startswith(("http://", "https://")):
            raise InvalidSource("Website sources need an http(s):// URL.")
        if not settings.allow_private_urls and not is_public_url(url):
            raise InvalidSource(f"{url} isn't a public address.")
        max_pages = int(config.get("max_pages") or 200)
        return {"type": "web", "url": url, "max_pages": max(1, min(max_pages, 2000))}

    if kind == "folder":
        if not settings.folder_roots:
            raise InvalidSource("Folder sources are turned off on this server.")
        path = Path(str(config.get("path", ""))).expanduser().resolve()
        if not any(path == root or root in path.parents for root in settings.folder_roots):
            roots = ", ".join(str(r) for r in settings.folder_roots)
            raise InvalidSource(f"Folders must be inside {roots}.")
        if not path.exists():
            raise InvalidSource(f"{path} doesn't exist.")
        return {"type": "folder", "path": str(path)}

    raise InvalidSource(f"Unknown source type {kind!r}. Use git, web or folder.")


def make_source(config: dict[str, Any], documents_dir: str, allow_private_urls: bool) -> Source:
    """Turn a validated config into a kvpack source."""
    kind = config["type"]
    if kind == UPLOAD:
        return FolderSource(documents_dir)
    if kind == "folder":
        return FolderSource(config["path"])
    if kind == "git":
        return GitSource(config["url"], config.get("branch"), config.get("subdir"), token_env=config.get("token_env"))
    if kind == "web":
        guard = None if allow_private_urls else is_public_url
        return WebSource(config["url"], max_pages=config.get("max_pages", 200), allow_url=guard)
    raise InvalidSource(f"Unknown source type {kind!r}.")


def describe(config: dict[str, Any]) -> str:
    kind = config.get("type")
    if kind == UPLOAD:
        files = config.get("files") or []
        return f"{len(files)} uploaded file{'s' if len(files) != 1 else ''}"
    if kind == "git":
        return config["url"] + (f" ({config['branch']})" if config.get("branch") else "")
    return config.get("url") or config.get("path") or kind
