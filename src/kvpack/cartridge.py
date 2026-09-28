"""The Cartridge: a small, trainable KV cache that stands in for a large document.

When a transformer reads a prompt it stores a key and a value vector for every
token in every layer (the "KV cache"). Later tokens only ever look at the prompt
through that cache. A cartridge is a KV cache that we *optimize directly* with
gradient descent, so that a few thousand cached "virtual tokens" can carry the
knowledge of a much longer document.

Shape convention used throughout this file:

    keys, values: [num_layers, num_kv_heads, num_tokens, head_dim]
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import DynamicCache, PreTrainedModel
from transformers.cache_utils import DynamicLayer

from .chat_format import ChatFormat, UnsupportedModelError

FORMAT_VERSION = 1


class Cartridge(nn.Module):
    """A trainable KV cache.

    The first `num_frozen_tokens` positions are kept fixed during training. The
    first token of a sequence acts as an "attention sink" that many heads rely on,
    and the Cartridges paper found that freezing it keeps training stable.
    """

    def __init__(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        num_frozen_tokens: int = 1,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__()
        if keys.shape != values.shape or keys.dim() != 4:
            raise ValueError("keys and values must both have shape [layers, kv_heads, tokens, head_dim]")
        if not 0 <= num_frozen_tokens <= keys.shape[2]:
            raise ValueError("num_frozen_tokens must be between 0 and the number of tokens")

        f = num_frozen_tokens
        # Trainable parameters are stored in float32 for stable optimization; they
        # are cast to the model's dtype whenever the cache is built.
        self.register_buffer("frozen_keys", keys[:, :, :f].detach().float().clone())
        self.register_buffer("frozen_values", values[:, :, :f].detach().float().clone())
        self.trainable_keys = nn.Parameter(keys[:, :, f:].detach().float().clone())
        self.trainable_values = nn.Parameter(values[:, :, f:].detach().float().clone())
        self.metadata: dict[str, Any] = dict(metadata or {})

    # ------------------------------------------------------------------ properties

    @property
    def keys(self) -> torch.Tensor:
        return torch.cat([self.frozen_keys, self.trainable_keys], dim=2)

    @property
    def values(self) -> torch.Tensor:
        return torch.cat([self.frozen_values, self.trainable_values], dim=2)

    @property
    def num_tokens(self) -> int:
        return self.frozen_keys.shape[2] + self.trainable_keys.shape[2]

    @property
    def num_frozen_tokens(self) -> int:
        return self.frozen_keys.shape[2]

    @property
    def num_layers(self) -> int:
        return self.trainable_keys.shape[0]

    def size_bytes(self, dtype: torch.dtype = torch.bfloat16) -> int:
        """Memory the cartridge occupies at inference time in `dtype`."""
        return 2 * self.trainable_keys[:, :, :1].numel() * self.num_tokens * dtype.itemsize

    # ------------------------------------------------------------------ creation

    @classmethod
    @torch.no_grad()
    def from_text(
        cls,
        model: PreTrainedModel,
        chat_format: ChatFormat,
        text: str,
        num_tokens: int,
        num_frozen_tokens: int = 1,
    ) -> Cartridge:
        """Initialize a cartridge with the real KV cache of the start of `text`.

        The text is placed in the system prompt and truncated to `num_tokens` tokens.
        This is the initialization the Cartridges paper found works best: training
        starts from a cache that already "reads" like the document.
        """
        check_model_supported(model)
        input_ids = chat_format.system_ids(text, max_tokens=num_tokens)
        cache = DynamicCache(config=model.config)
        model(input_ids=torch.tensor([input_ids], device=model.device), past_key_values=cache, use_cache=True)

        keys = torch.stack([layer.keys[0] for layer in cache.layers])  # [layers, heads, tokens, dim]
        values = torch.stack([layer.values[0] for layer in cache.layers])
        return cls(
            keys,
            values,
            num_frozen_tokens=num_frozen_tokens,
            metadata={
                "model": model.config.name_or_path,
                "init_text_tokens": len(input_ids),
            },
        )

    # ------------------------------------------------------------------ usage

    def to_cache(self, model: PreTrainedModel, batch_size: int = 1, read_only: bool = False) -> DynamicCache:
        """Build a fresh Hugging Face cache pre-filled with this cartridge.

        The returned cache is differentiable with respect to the cartridge's
        parameters, so it can be used for training as well as generation.

        By default the model appends new tokens to the cache as it goes, which is
        what generation needs, so build a new cache for every generation. With
        `read_only=True` the layers attend to [cartridge + new tokens] without storing
        anything. Training uses that, because it makes re-running a layer (as
        gradient checkpointing does) give exactly the same result.
        """
        cache = DynamicCache(config=model.config)
        dtype = model.dtype
        keys, values = self.keys.to(dtype), self.values.to(dtype)
        for layer_idx in range(self.num_layers):
            k = keys[layer_idx].unsqueeze(0).expand(batch_size, -1, -1, -1)
            v = values[layer_idx].unsqueeze(0).expand(batch_size, -1, -1, -1)
            cache.update(k, v, layer_idx)
        if read_only:
            for layer in cache.layers:
                layer.__class__ = _ReadOnlyLayer
        return cache

    # ------------------------------------------------------------------ save / load

    def save(self, path: str | Path) -> Path:
        """Save as a single `.safetensors` file (tensors + JSON metadata)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {**self.metadata, "format_version": FORMAT_VERSION, "saved_at": time.time()}
        save_file(
            {
                "keys": self.keys.detach().to(torch.bfloat16).contiguous().cpu(),
                "values": self.values.detach().to(torch.bfloat16).contiguous().cpu(),
            },
            str(path),
            metadata={
                "cartridge": json.dumps(metadata),
                "num_frozen_tokens": str(self.num_frozen_tokens),
            },
        )
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> Cartridge:
        path = Path(path)
        tensors = load_file(str(path))
        raw = read_header(path)
        metadata = json.loads(raw.get("cartridge", "{}"))
        cartridge = cls(
            tensors["keys"],
            tensors["values"],
            num_frozen_tokens=int(raw.get("num_frozen_tokens", 1)),
            metadata=metadata,
        )
        return cartridge.to(device)

    def __repr__(self) -> str:
        name = self.metadata.get("name", "unnamed")
        return (
            f"Cartridge(name={name!r}, model={self.metadata.get('model')!r}, tokens={self.num_tokens}, "
            f"layers={self.num_layers}, size={self.size_bytes() / 2**20:.1f} MiB)"
        )


class _ReadOnlyLayer(DynamicLayer):
    """A cache layer that returns [cached + new] keys/values but never stores the new ones."""

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        return torch.cat([self.keys, key_states], dim=-2), torch.cat([self.values, value_states], dim=-2)


def read_header(path: str | Path) -> dict[str, str]:
    """Read a `.safetensors` file's metadata without loading any tensors."""
    return _read_raw_header(path).get("__metadata__", {}) or {}


@dataclass
class CartridgeInfo:
    """What's in a cartridge file, read from its header without loading the tensors."""

    path: Path
    name: str
    model: str | None
    num_tokens: int
    metadata: dict[str, Any]


def peek(path: str | Path) -> CartridgeInfo:
    path = Path(path)
    header = _read_raw_header(path)
    if "keys" not in header:
        raise ValueError(f"{path} is not a kvpack cartridge")
    metadata = json.loads((header.get("__metadata__") or {}).get("cartridge", "{}"))
    return CartridgeInfo(
        path=path,
        name=metadata.get("name") or path.stem,
        model=metadata.get("model"),
        num_tokens=header["keys"]["shape"][2],
        metadata=metadata,
    )


def _read_raw_header(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(header_len))


def check_model_supported(model: PreTrainedModel) -> None:
    """Cartridges need every layer to be ordinary full attention with a growing cache."""
    cache = DynamicCache(config=model.config)
    if not cache.layers or not all(type(layer) is DynamicLayer for layer in cache.layers):
        raise UnsupportedModelError(
            f"{model.config.model_type} uses sliding-window or non-attention layers, which cartridges "
            "don't support yet. Try a Llama- or Qwen3-style model."
        )
