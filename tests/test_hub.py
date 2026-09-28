from conftest import DOCUMENT

from kvpack import Cartridge, hub


def saved(model, chat_format, tmp_path):
    cartridge = Cartridge.from_text(model, chat_format, DOCUMENT, num_tokens=20)
    cartridge.metadata.update(
        name="kestrel", corpus_tokens=200, num_samples=384, epochs=4,
        eval_after={"agreement": 0.872, "loss": 0.67}, eval_no_context={"agreement": 0.637, "loss": 1.87},
    )  # fmt: skip
    return cartridge.save(tmp_path / "kestrel.safetensors")


def test_model_card(model, chat_format, tmp_path):
    card = hub.model_card(saved(model, chat_format, tmp_path), "someone/kestrel")
    assert card.startswith("---\nlibrary_name: kvpack\nbase_model: tiny-qwen3\n")
    assert "10.0x compression" in card
    assert "87.2% (vs 63.7% with no context)" in card
    assert "kvpack pull someone/kestrel" in card


def test_push_uploads_cartridge_and_card(model, chat_format, tmp_path, monkeypatch):
    calls = []

    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, repo_id, private, exist_ok):
            calls.append(("create", repo_id, private))
            return f"https://huggingface.co/{repo_id}"

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, commit_message):
            calls.append(("upload", path_in_repo))

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    url = hub.push(saved(model, chat_format, tmp_path), "someone/kestrel", private=True)
    assert url == "https://huggingface.co/someone/kestrel"
    assert calls == [("create", "someone/kestrel", True), ("upload", "cartridge.safetensors"), ("upload", "README.md")]


def test_pull_and_load(model, chat_format, tmp_path, monkeypatch):
    path = saved(model, chat_format, tmp_path)
    monkeypatch.setattr(hub, "hf_hub_download", lambda repo_id, filename, revision, token: str(path))
    cartridge = hub.load_from_hub("someone/kestrel")
    assert cartridge.num_tokens == 20
