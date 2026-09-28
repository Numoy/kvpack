"""Settings, read from environment variables (all prefixed KVPACK_STUDIO_)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _list(name: str, default: str = "") -> list[str]:
    return [v.strip() for v in os.environ.get(name, default).split(",") if v.strip()]


@dataclass
class Settings:
    database_url: str = "sqlite:///kvpack-studio.db"
    storage_dir: Path = Path("kvpack-studio-data")
    # Base models users can build cartridges for. The first one is the default.
    base_models: list[str] = field(default_factory=lambda: ["Qwen/Qwen3-4B"])
    # The first account can always sign up. After that, POST /v1/signup needs one of these
    # codes; with none configured, only an admin can add accounts (`kvpack-studio create-account`).
    invite_codes: list[str] = field(default_factory=list)

    # Folder sources read the server's disk, so they're off unless an admin names the
    # folders users may build from.
    folder_roots: list[Path] = field(default_factory=list)
    # Website sources may not reach private or loopback addresses unless this is on
    # (e.g. for an intranet wiki on a trusted single-user setup).
    allow_private_urls: bool = False
    # Environment variables holding Git access tokens that sources may refer to by name.
    git_token_envs: list[str] = field(default_factory=list)

    # Limits per build and per account.
    max_upload_mb: int = 50
    max_files: int = 200
    max_corpus_tokens: int = 250_000
    default_cartridge_tokens: int = 2048
    max_cartridge_tokens: int = 8192
    default_samples: int = 1024
    max_samples: int = 8192
    max_cartridges_per_account: int = 25

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        defaults = cls()
        return cls(
            database_url=env("KVPACK_STUDIO_DATABASE_URL", defaults.database_url),
            storage_dir=Path(env("KVPACK_STUDIO_STORAGE_DIR", str(defaults.storage_dir))),
            base_models=_list("KVPACK_STUDIO_BASE_MODELS") or defaults.base_models,
            invite_codes=_list("KVPACK_STUDIO_INVITE_CODES"),
            folder_roots=[Path(p).resolve() for p in _list("KVPACK_STUDIO_FOLDER_ROOTS")],
            allow_private_urls=env("KVPACK_STUDIO_ALLOW_PRIVATE_URLS", "").lower() in ("1", "true", "yes"),
            git_token_envs=_list("KVPACK_STUDIO_GIT_TOKEN_ENVS"),
            max_upload_mb=int(env("KVPACK_STUDIO_MAX_UPLOAD_MB", defaults.max_upload_mb)),
            max_corpus_tokens=int(env("KVPACK_STUDIO_MAX_CORPUS_TOKENS", defaults.max_corpus_tokens)),
            max_samples=int(env("KVPACK_STUDIO_MAX_SAMPLES", defaults.max_samples)),
            max_cartridges_per_account=int(
                env("KVPACK_STUDIO_MAX_CARTRIDGES_PER_ACCOUNT", defaults.max_cartridges_per_account)
            ),
        )
