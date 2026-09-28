"""Where uploaded documents and built cartridges live on disk.

Locally this is a plain folder. On Modal it's a Volume mounted into every container,
so the build workers write cartridges that the inference servers can read.

    <root>/documents/<cartridge id>/<uploaded files>
    <root>/cartridges/<base model slug>/<cartridge id>.safetensors

Each base model has its own cartridge folder, which is exactly what a
`kvpack serve <folder>` for that model watches.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


def model_slug(base_model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "--", base_model)


def safe_filename(name: str) -> str:
    """Keep only the final path component and harmless characters."""
    name = Path(name.replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    return name[:120] or "document.txt"


class Storage:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def documents_dir(self, cartridge_id: str) -> Path:
        return self.root / "documents" / cartridge_id

    def cartridge_dir(self, base_model: str) -> Path:
        return self.root / "cartridges" / model_slug(base_model)

    def cartridge_path(self, base_model: str, cartridge_id: str) -> Path:
        return self.cartridge_dir(base_model) / f"{cartridge_id}.safetensors"

    def save_upload(self, cartridge_id: str, filename: str, data: bytes) -> Path:
        directory = self.documents_dir(cartridge_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / safe_filename(filename)
        stem, suffix, n = path.stem, path.suffix, 1
        while path.exists():  # two uploads with the same name
            path = directory / f"{stem}-{n}{suffix}"
            n += 1
        path.write_bytes(data)
        return path

    def sync(self) -> None:
        """See writes made by other machines (a no-op on a local disk)."""

    def persist(self) -> None:
        """Make this machine's writes visible to others (a no-op on a local disk)."""

    def delete(self, cartridge_id: str, base_model: str) -> None:
        shutil.rmtree(self.documents_dir(cartridge_id), ignore_errors=True)
        self.cartridge_path(base_model, cartridge_id).unlink(missing_ok=True)
