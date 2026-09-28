"""The whole recipe in one function: documents in, trained cartridge out."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

from transformers import PreTrainedModel

from .cartridge import Cartridge
from .chat_format import ChatFormat
from .synthesize import Generator, SelfStudyDataset, synthesize
from .train import TrainHistory, train


@dataclass
class BuildResult:
    cartridge: Cartridge
    dataset: SelfStudyDataset
    history: TrainHistory


def build(
    model: PreTrainedModel,
    chat_format: ChatFormat,
    corpus: str,
    *,
    name: str = "cartridge",
    num_tokens: int = 2048,
    num_samples: int = 1024,
    eval_fraction: float = 0.05,
    generator: Generator | None = None,
    dataset: SelfStudyDataset | None = None,
    synth_batch_size: int = 16,
    train_batch_size: int = 8,
    lr: float = 2e-2,
    epochs: int = 1,
    gradient_checkpointing: bool = True,
    seed: int = 0,
    sources: list[dict] | None = None,
    fingerprint: str | None = None,
    on_synth_progress: Callable[[int], None] | None = None,
    on_train_step: Callable[[int, int, float], None] | None = None,
) -> BuildResult:
    """Synthesize self-study data for `corpus`, then train a cartridge on it.

    Pass `dataset` to skip synthesis and reuse existing data. `sources` and
    `fingerprint` (from a `kvpack.sources.Snapshot`) are stored in the cartridge so
    `kvpack sync` can rebuild it when the sources change.
    """
    if dataset is None:
        dataset = synthesize(
            model, chat_format, corpus, num_samples,
            generator=generator, batch_size=synth_batch_size, seed=seed, on_progress=on_synth_progress,
        )  # fmt: skip
    train_set, eval_set = dataset.split(eval_fraction, seed=seed)

    cartridge = Cartridge.from_text(model, chat_format, corpus, num_tokens=num_tokens)
    history = train(
        model, cartridge, train_set,
        eval_examples=eval_set, lr=lr, epochs=epochs, batch_size=train_batch_size, seed=seed,
        gradient_checkpointing=gradient_checkpointing, on_step=on_train_step,
    )  # fmt: skip

    corpus_tokens = len(chat_format.tokenizer.encode(corpus, add_special_tokens=False))
    cartridge.metadata.update(
        name=name,
        model=model.config.name_or_path,
        corpus_tokens=corpus_tokens,
        corpus_sha256=hashlib.sha256(corpus.encode()).hexdigest(),
        sources=sources or [],
        sources_fingerprint=fingerprint,
        num_tokens=num_tokens,
        num_samples=len(dataset),
        eval_conversations=len(eval_set),
        lr=lr,
        epochs=epochs,
        eval_no_context=vars(history.eval_no_context) if history.eval_no_context else None,
        eval_before=vars(history.eval_before) if history.eval_before else None,
        eval_after=vars(history.eval_after) if history.eval_after else None,
    )
    return BuildResult(cartridge=cartridge, dataset=dataset, history=history)
