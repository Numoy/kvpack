import threading
import time

from kvpack import Cartridge

from kvpack_studio.volumes import VolumeCartridgeStore, VolumeStorage


class FakeVolume:
    """Records calls; `reload` can be made slow to test locking."""

    def __init__(self, reload_seconds=0.0):
        self.reloads = self.commits = 0
        self.reload_seconds = reload_seconds
        self.reloading = False

    def reload(self):
        self.reloading = True
        time.sleep(self.reload_seconds)
        self.reloads += 1
        self.reloading = False

    def commit(self):
        self.commits += 1


def test_storage_commits_and_reloads(tmp_path):
    volume = FakeVolume()
    storage = VolumeStorage(tmp_path, volume)
    storage.persist()
    storage.sync()
    assert (volume.commits, volume.reloads) == (1, 1)


def test_storage_sync_survives_a_busy_volume(tmp_path):
    class Busy(FakeVolume):
        def reload(self):
            raise RuntimeError("volume busy")

    VolumeStorage(tmp_path, Busy()).sync()  # logs and carries on


def test_store_reloads_at_most_every_interval(tiny, tmp_path):
    volume = FakeVolume()
    store = VolumeCartridgeStore("tiny-qwen3", directory=tmp_path, volume=volume, reload_every=60)
    for _ in range(5):
        store.list()
    assert volume.reloads == 1


def test_files_are_never_read_during_a_reload(tiny, tmp_path):
    model, _, chat_format = tiny
    Cartridge.from_text(model, chat_format, "text " * 40, num_tokens=16).save(tmp_path / "a.safetensors")
    volume = FakeVolume(reload_seconds=0.2)
    store = VolumeCartridgeStore("tiny-qwen3", directory=tmp_path, volume=volume, reload_every=0)

    original_load = Cartridge.load
    overlaps = []

    def checking_load(*args, **kwargs):
        overlaps.append(volume.reloading)
        return original_load(*args, **kwargs)

    Cartridge.load = checking_load
    try:
        threads = [threading.Thread(target=store.refresh) for _ in range(3)]
        threads += [threading.Thread(target=store.get, args=("a",)) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        Cartridge.load = original_load
    assert overlaps and not any(overlaps)
