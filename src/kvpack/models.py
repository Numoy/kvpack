"""Loading models and running them with a cartridge in front."""

from __future__ import annotations

import functools
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, PreTrainedModel, PreTrainedTokenizerBase

from .cartridge import check_model_supported
from .chat_format import ChatFormat


def pick_device(device: str | None = None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, dtype: str | None = None) -> torch.dtype:
    if dtype and dtype != "auto":
        return getattr(torch, dtype)
    return torch.float32 if device.type == "cpu" else torch.bfloat16


def load_model(
    name_or_path: str, device: str | None = None, dtype: str | None = None
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, ChatFormat]:
    """Load a frozen model, its tokenizer and its chat format."""
    torch_device = pick_device(device)
    model = AutoModelForCausalLM.from_pretrained(name_or_path, dtype=pick_dtype(torch_device, dtype))
    model.to(torch_device).eval().requires_grad_(False)
    check_model_supported(model)

    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, ChatFormat.from_tokenizer(tokenizer)


def hidden_states_at(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    positions: torch.Tensor,
    past_key_values: DynamicCache | None = None,
) -> torch.Tensor:
    """Run the transformer body and return the final hidden states at `positions`.

    Args:
        input_ids: [batch, seq] token ids (right-padded).
        attention_mask: [batch, cached + seq] mask covering the cache *and* input_ids.
        positions: [n, 2] (batch index, sequence index) pairs.

    Returns:
        [n, hidden_size] hidden states. Apply `model.lm_head` to get logits.
    """
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=past_key_values is not None,
    )
    return outputs.last_hidden_state[positions[:, 0], positions[:, 1]]


def next_token_logits(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    positions: torch.Tensor,
    past_key_values: DynamicCache | None = None,
) -> torch.Tensor:
    """Float32 logits at `positions` only (see `hidden_states_at` for the arguments).

    Computing logits for every position would allocate `seq_len x vocab_size`
    floats per sequence (~600 MB for Qwen at 1k tokens), so we only project the
    positions we need through the output layer.
    """
    return model.lm_head(hidden_states_at(model, input_ids, attention_mask, positions, past_key_values)).float()


@contextmanager
def layer_checkpointing(model: PreTrainedModel) -> Iterator[None]:
    """Recompute each decoder layer during the backward pass instead of storing its activations.

    Training back-propagates through the whole frozen model to reach the cartridge,
    and storing every layer's activations quickly dominates memory. With this on,
    only each layer's input is kept: memory drops by roughly the number of layers,
    at the cost of about one extra forward pass.

    (Hugging Face's built-in gradient checkpointing drops `past_key_values`, which
    would silently discard the cartridge, so we wrap the layers ourselves. The cache
    must be read-only, see `Cartridge.to_cache`, so a re-run layer sees the same inputs.)
    """
    layers = model.model.layers
    originals = [layer.__dict__.get("forward") for layer in layers]
    for layer in layers:
        layer.forward = functools.partial(checkpoint, layer.forward, use_reentrant=False)
    try:
        yield
    finally:
        for layer, original in zip(layers, originals, strict=True):
            if original is None:
                del layer.forward
            else:
                layer.forward = original
