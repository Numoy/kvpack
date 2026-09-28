"""Shared fixtures: a tiny, randomly initialized Qwen3 model with the real Qwen3 tokenizer.

The model is far too small to say anything sensible, but it exercises exactly
the same code paths as a real one, and the whole suite runs on a laptop CPU.
"""

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from kvpack import ChatFormat

TOKENIZER = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.fixture(scope="session")
def chat_format(tokenizer):
    return ChatFormat.from_tokenizer(tokenizer)


def make_tiny_model(vocab_size: int, **overrides) -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=4096,
        **overrides,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    model.config.name_or_path = "tiny-qwen3"
    return model


@pytest.fixture(scope="session")
def model(tokenizer):
    return make_tiny_model(len(tokenizer))


DOCUMENT = (
    "Kestrel Station is a research outpost 340 metres beneath the Vardo Ice Shelf. "
    "It is powered by a geothermal loop delivering 410 kilowatts. "
    "The Station Lead is Oskar Lindqvist-Mbeki and the Drill Master is Fergus MacAulay. "
) * 4
