"""Finding, loading and caching the cartridges a server offers.

A store serves cartridges from explicit files and/or a directory, each under its
file name (`docs.safetensors` is served as model "docs"). Only file
headers are read up front; tensors are loaded onto the device the first time a
cartridge is used and kept in a least-recently-used cache, so a server can offer
far more cartridges than fit in GPU memory. New files dropped into the directory
show up without a restart.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path

import torch

from .cartridge import Cartridge, CartridgeInfo, peek

log = logging.getLogger("kvpack")


class CartridgeNotFound(KeyError):
    pass


class CartridgeStore:
    def __init__(
        self,
        model_name: str,
        device: torch.device | str = "cpu",
        files: list[str | Path] = (),
        directory: str | Path | None = None,
        max_loaded: int = 8,
    ):
        self.model_name = model_name
        self.device = device
        self.files = [Path(f) for f in files]
        self.directory = Path(directory) if directory else None
        self.max_loaded = max_loaded
        self._lock = threading.Lock()
        self._infos: dict[str, CartridgeInfo] = {}
        self._mtimes: dict[Path, float] = {}
        self._loaded: OrderedDict[str, Cartridge] = OrderedDict()
        self._pinned: dict[str, Cartridge] = {}
        self.refresh()

    @classmethod
    def from_cartridges(cls, model_name: str, cartridges: dict[str, Cartridge], device="cpu") -> CartridgeStore:
        """A store over cartridges that are already in memory (handy for tests and notebooks)."""
        store = cls(model_name, device)
        for name, cartridge in cartridges.items():
            store._pinned[name] = cartridge.to(device)
            store._infos[name] = CartridgeInfo(
                path=Path(f"<memory:{name}>"),
                name=name,
                model=cartridge.metadata.get("model"),
                num_tokens=cartridge.num_tokens,
                metadata=cartridge.metadata,
            )
        return store

    # ------------------------------------------------------------------ discovery

    def refresh(self) -> None:
        """Re-scan files and the directory. Cheap: only changed files are re-read."""
        paths = list(self.files)
        if self.directory:
            paths += sorted(self.directory.glob("*.safetensors"))
        with self._lock:
            seen: set[Path] = set()
            for path in paths:
                try:
                    mtime = path.stat().st_mtime
                except FileNotFoundError:
                    continue
                seen.add(path)
                if self._mtimes.get(path) == mtime:
                    continue
                try:
                    info = peek(path)
                except Exception as e:  # a half-written or foreign file shouldn't take the server down
                    log.warning("Skipping %s: %s", path, e)
                    continue
                if info.model and info.model != self.model_name:
                    log.warning("Skipping %s: built for %s, server runs %s", path, info.model, self.model_name)
                    continue
                info.name = path.stem  # served under its file name, which is unique within a folder
                owner = self._infos.get(info.name)
                if owner is not None and owner.path != path:
                    log.warning("Skipping %s: the name %r is already taken by %s", path, info.name, owner.path)
                    continue
                self._mtimes[path] = mtime
                self._loaded.pop(info.name, None)  # the file changed: reload on next use
                self._infos[info.name] = info
            # forget cartridges whose files disappeared
            for name, info in list(self._infos.items()):
                if name not in self._pinned and info.path not in seen:
                    del self._infos[name]
                    self._loaded.pop(name, None)

    def list(self) -> list[CartridgeInfo]:
        self.refresh()
        with self._lock:
            return list(self._infos.values())

    def info(self, name: str) -> CartridgeInfo:
        with self._lock:
            if name not in self._infos:
                raise CartridgeNotFound(name)
            return self._infos[name]

    # ------------------------------------------------------------------ loading

    def get(self, name: str) -> Cartridge:
        if name in self._pinned:
            return self._pinned[name]
        with self._lock:
            if name in self._loaded:
                self._loaded.move_to_end(name)
                return self._loaded[name]
        if name not in self._infos:
            self.refresh()
        info = self.info(name)
        cartridge = Cartridge.load(info.path, device=self.device)
        with self._lock:
            self._loaded[name] = cartridge
            while len(self._loaded) > self.max_loaded:
                evicted, _ = self._loaded.popitem(last=False)
                log.info("Unloaded cartridge %s", evicted)
        log.info("Loaded cartridge %s (%d tokens)", name, cartridge.num_tokens)
        return cartridge
