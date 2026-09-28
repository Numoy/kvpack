"""Turning documents into one text, and cutting that text into chunks."""

from __future__ import annotations

import random
from pathlib import Path

from transformers import PreTrainedTokenizerBase

from .sources import TEXT_SUFFIXES, FolderSource, Snapshot, SourceError

__all__ = ["TEXT_SUFFIXES", "Chunker", "load_corpus"]


def load_corpus(paths: list[str | Path]) -> str:
    """Concatenate every readable text file under `paths` into one string.

    Each file is preceded by a `## File: <name>` header so the model (and the
    questions generated about it) can refer to documents by name. PDFs are
    supported when `pypdf` is installed (`pip install kvpack[pdf]`). For Git
    repositories and websites, see `kvpack.sources`.
    """
    snapshots = []
    for p in map(Path, paths):
        if not p.exists():
            raise FileNotFoundError(p)
        try:
            snapshots.append(FolderSource(p).fetch())
        except SourceError as e:
            raise FileNotFoundError(str(e)) from None
    try:
        return Snapshot.merge(snapshots).corpus()
    except ValueError:
        raise ValueError(f"No readable text files found in {[str(p) for p in paths]}") from None


class Chunker:
    """Samples random, token-length-bounded windows of the corpus."""

    def __init__(self, text: str, tokenizer: PreTrainedTokenizerBase, min_tokens: int = 512, max_tokens: int = 1024):
        self.text = text
        self.tokenizer = tokenizer
        self.tokens = tokenizer.encode(text, add_special_tokens=False)
        self.min_tokens, self.max_tokens = min_tokens, max_tokens

    def __len__(self) -> int:
        return len(self.tokens)

    def sample(self, rng: random.Random) -> str:
        size = rng.randint(self.min_tokens, self.max_tokens)
        if size >= len(self.tokens):
            return self.text
        start = rng.randint(0, len(self.tokens) - size)
        return self.tokenizer.decode(self.tokens[start : start + size])
