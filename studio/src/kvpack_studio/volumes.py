"""Storage on a shared volume (Modal Volumes in production).

Files written in one container become visible in others only after the writer
commits and the reader reloads. Reloading has two sharp edges this module handles:
it fails while files are open, and the volume looks empty while a reload runs.
So every reload and every file read happen under one lock.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

from kvpack import Cartridge
from kvpack.store import CartridgeStore

from .storage import Storage

log = logging.getLogger("kvpack_studio")


class Volume(Protocol):
    def reload(self) -> None: ...
    def commit(self) -> None: ...


class VolumeStorage(Storage):
    def __init__(self, root, volume: Volume):
        super().__init__(root)
        self.volume = volume
        self.lock = threading.RLock()

    def sync(self) -> None:
        with self.lock:
            try:
                self.volume.reload()
            except Exception as e:  # e.g. a download still has a file open; it'll retry next time
                log.warning("Volume reload skipped: %s", e)

    def persist(self) -> None:
        with self.lock:
            self.volume.commit()


class VolumeCartridgeStore(CartridgeStore):
    """A cartridge store that reloads the volume (at most every `reload_every` seconds)."""

    def __init__(self, *args, volume: Volume, reload_every: float = 5.0, **kwargs):
        self.volume = volume
        self.reload_every = reload_every
        self._volume_lock = threading.RLock()
        self._last_reload = 0.0
        super().__init__(*args, **kwargs)

    def refresh(self) -> None:
        with self._volume_lock:
            if time.monotonic() - self._last_reload >= self.reload_every:
                try:
                    self.volume.reload()
                    self._last_reload = time.monotonic()
                except Exception as e:
                    log.warning("Volume reload skipped: %s", e)
            super().refresh()

    def get(self, name: str) -> Cartridge:
        with self._volume_lock:  # never read a file while a reload is in progress
            return super().get(name)
