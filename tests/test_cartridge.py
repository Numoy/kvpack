import pytest
import torch
from conftest import DOCUMENT, make_tiny_model

from kvpack import Cartridge, UnsupportedModelError
from kvpack.models import next_token_logits


def test_untrained_cartridge_equals_document_in_context(model, chat_format):
    """The core invariant: a cartridge initialized from text must behave exactly like
    having that text in the system prompt. If this holds, positions, masks and the
    chat-template split are all wired correctly."""
    cartridge = Cartridge.from_text(model, chat_format, DOCUMENT, num_tokens=48)
    question = chat_format.conversation_ids([{"role": "user", "content": "Who is the Station Lead?"}])

    with_context = chat_format.system_ids(DOCUMENT, max_tokens=48) + question
    expected = model(torch.tensor([with_context])).logits[0, -len(question) :]

    mask = torch.ones(1, cartridge.num_tokens + len(question), dtype=torch.long)
    positions = torch.tensor([[0, i] for i in range(len(question))])
    actual = next_token_logits(model, torch.tensor([question]), mask, positions, cartridge.to_cache(model))

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_from_text_respects_token_budget(model, chat_format):
    cartridge = Cartridge.from_text(model, chat_format, DOCUMENT, num_tokens=32)
    assert cartridge.num_tokens == 32
    assert cartridge.num_frozen_tokens == 1
    layers, heads, tokens, head_dim = cartridge.keys.shape
    assert (layers, heads, tokens, head_dim) == (2, 2, 32, 16)


def test_batched_cache_repeats_the_cartridge(model, chat_format):
    cartridge = Cartridge.from_text(model, chat_format, DOCUMENT, num_tokens=16)
    cache = cartridge.to_cache(model, batch_size=3)
    assert cache.get_seq_length() == 16
    assert cache.layers[0].keys.shape[0] == 3


def test_save_and_load_roundtrip(model, chat_format, tmp_path):
    cartridge = Cartridge.from_text(model, chat_format, DOCUMENT, num_tokens=24, num_frozen_tokens=2)
    cartridge.metadata["name"] = "kestrel"
    path = cartridge.save(tmp_path / "kestrel.safetensors")

    loaded = Cartridge.load(path)
    assert loaded.num_tokens == 24
    assert loaded.num_frozen_tokens == 2
    assert loaded.metadata["name"] == "kestrel"
    assert loaded.metadata["model"] == "tiny-qwen3"
    # Stored in bfloat16, so compare at bfloat16 precision.
    torch.testing.assert_close(loaded.keys, cartridge.keys.to(torch.bfloat16).float())


def test_size_bytes(model, chat_format):
    cartridge = Cartridge.from_text(model, chat_format, DOCUMENT, num_tokens=10)
    # keys + values, 2 layers, 2 heads, 10 tokens, 16 dims, 2 bytes each
    assert cartridge.size_bytes() == 2 * 2 * 2 * 10 * 16 * 2


def test_sliding_window_models_are_rejected(tokenizer, chat_format):
    model = make_tiny_model(
        len(tokenizer),
        use_sliding_window=True,
        sliding_window=8,
        max_window_layers=1,
        layer_types=["full_attention", "sliding_attention"],
    )
    with pytest.raises(UnsupportedModelError):
        Cartridge.from_text(model, chat_format, DOCUMENT, num_tokens=16)
