import os

import pytest
from conftest import DOCUMENT

from kvpack import Cartridge
from kvpack.cartridge import peek
from kvpack.store import CartridgeNotFound, CartridgeStore


@pytest.fixture
def make(model, chat_format, tmp_path):
    def _make(name: str, model_name: str = "tiny-qwen3", tokens: int = 12):
        cartridge = Cartridge.from_text(model, chat_format, DOCUMENT, num_tokens=tokens)
        cartridge.metadata.update(name=name, model=model_name)
        return cartridge.save(tmp_path / f"{name}.safetensors")

    return _make


def test_peek_reads_header_only(make):
    info = peek(make("alpha", tokens=14))
    assert (info.name, info.model, info.num_tokens) == ("alpha", "tiny-qwen3", 14)


def test_directory_is_scanned_and_new_files_appear(make, tmp_path):
    make("alpha")
    store = CartridgeStore("tiny-qwen3", directory=tmp_path)
    assert [i.name for i in store.list()] == ["alpha"]

    make("beta")
    assert sorted(i.name for i in store.list()) == ["alpha", "beta"]


def test_deleted_files_disappear(make, tmp_path):
    path = make("alpha")
    store = CartridgeStore("tiny-qwen3", directory=tmp_path)
    os.remove(path)
    assert store.list() == []
    with pytest.raises(CartridgeNotFound):
        store.get("alpha")


def test_cartridges_for_other_models_are_skipped(make, tmp_path):
    make("alpha")
    make("other", model_name="some/other-model")
    store = CartridgeStore("tiny-qwen3", directory=tmp_path)
    assert [i.name for i in store.list()] == ["alpha"]


def test_non_cartridge_files_are_skipped(tmp_path):
    (tmp_path / "junk.safetensors").write_bytes(b"not a safetensors file")
    assert CartridgeStore("tiny-qwen3", directory=tmp_path).list() == []


def test_lru_keeps_at_most_max_loaded(make, tmp_path):
    for name in ("a", "b", "c"):
        make(name)
    store = CartridgeStore("tiny-qwen3", directory=tmp_path, max_loaded=2)
    first = store.get("a")
    assert store.get("a") is first  # cached
    store.get("b")
    store.get("c")  # evicts "a", the least recently used
    assert list(store._loaded) == ["b", "c"]
    assert store.get("a") is not first  # reloaded from disk


def test_rewritten_file_is_reloaded(make, tmp_path):
    path = make("alpha", tokens=12)
    store = CartridgeStore("tiny-qwen3", directory=tmp_path)
    assert store.get("alpha").num_tokens == 12
    make("alpha", tokens=10)
    os.utime(path, (1, 1))  # make sure the mtime changes even on coarse filesystems
    store.refresh()
    assert store.get("alpha").num_tokens == 10


def test_cartridges_are_named_after_their_file(make, tmp_path):
    path = make("alpha")
    path.rename(tmp_path / "handbook-v2.safetensors")
    store = CartridgeStore("tiny-qwen3", directory=tmp_path)
    assert [i.name for i in store.list()] == ["handbook-v2"]


def test_duplicate_names_are_skipped(make, tmp_path):
    first = make("alpha")
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    second = make("beta").rename(other_dir / "alpha.safetensors")
    store = CartridgeStore("tiny-qwen3", files=[first, second])
    infos = store.list()
    assert [(i.name, i.path) for i in infos] == [("alpha", first)]
