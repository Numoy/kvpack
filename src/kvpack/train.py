"""Training a cartridge with context distillation.

The model's weights stay frozen. Only the cartridge's keys and values receive
gradients. The loss asks: "with only the cartridge in front of it, does the model
predict the same next tokens it predicted when the real chunk was in context?"

    loss = - sum_k  p_teacher(token_k) * log p_student(token_k)

averaged over every answer token, where the teacher's top-k tokens come from the
self-study dataset.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel

from .cartridge import Cartridge
from .models import hidden_states_at, layer_checkpointing
from .synthesize import Example

# Rows of vocabulary-sized logits materialized at once in the loss. With Qwen's 151k
# vocabulary, 1,024 rows of float32 logits take ~600 MB.
LOSS_CHUNK_TOKENS = 1024


@dataclass
class Metrics:
    loss: float  # distillation cross-entropy (lower is better)
    agreement: float  # fraction of answer tokens where student and teacher agree on the top-1 token

    def __str__(self) -> str:
        return f"loss {self.loss:.3f} | token agreement {self.agreement:.1%}"


@dataclass
class TrainHistory:
    losses: list[float] = field(default_factory=list)
    eval_no_context: Metrics | None = None
    eval_before: Metrics | None = None
    eval_after: Metrics | None = None


def distillation_loss(
    model: PreTrainedModel, cartridge: Cartridge | None, examples: Sequence[Example]
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Summed loss, number of top-1 agreements, and number of answer tokens for a batch.

    With `cartridge=None` the model sees the question with no context at all, which
    gives a useful "how much does the model know already?" baseline.
    """
    device = model.device
    seqs = [ex.prompt_ids + ex.answer_ids for ex in examples]
    width = max(map(len, seqs))
    input_ids = torch.tensor([s + [0] * (width - len(s)) for s in seqs], device=device)
    token_mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=device)

    positions = [
        (b, len(ex.prompt_ids) - 1 + j)  # logits at position i predict token i + 1
        for b, ex in enumerate(examples)
        for j in range(len(ex.answer_ids))
    ]
    topk_ids = torch.cat([ex.topk_ids for ex in examples]).to(device).long()
    teacher_probs = torch.cat([ex.topk_logprobs for ex in examples]).to(device).float().exp()

    if cartridge is None:
        cache, attention_mask = None, token_mask
    else:
        cache = cartridge.to_cache(model, batch_size=len(examples), read_only=True)
        cartridge_mask = torch.ones(len(examples), cartridge.num_tokens, dtype=token_mask.dtype, device=device)
        attention_mask = torch.cat([cartridge_mask, token_mask], dim=1)

    hidden = hidden_states_at(model, input_ids, attention_mask, torch.tensor(positions, device=device), cache)

    # The output layer and softmax run in chunks, and are recomputed during the backward
    # pass instead of stored, so memory doesn't grow with (answer tokens x vocabulary).
    loss, agreements = hidden.new_zeros((), dtype=torch.float32), 0
    for start in range(0, len(positions), LOSS_CHUNK_TOKENS):
        chunk = slice(start, start + LOSS_CHUNK_TOKENS)
        args = (model.lm_head, hidden[chunk], topk_ids[chunk], teacher_probs[chunk])
        chunk_loss, chunk_agreements = (
            checkpoint(_chunk_loss, *args, use_reentrant=False) if hidden.requires_grad else _chunk_loss(*args)
        )
        loss = loss + chunk_loss
        agreements += chunk_agreements
    return loss, agreements, len(positions)


def _chunk_loss(
    lm_head: torch.nn.Module, hidden: torch.Tensor, topk_ids: torch.Tensor, teacher_probs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    student_logprobs = torch.log_softmax(lm_head(hidden).float(), dim=-1)
    loss = -(teacher_probs * student_logprobs.gather(-1, topk_ids)).sum()
    agreements = (student_logprobs.argmax(-1) == topk_ids[:, 0]).sum().detach()
    return loss, agreements


@torch.no_grad()
def evaluate(
    model: PreTrainedModel, cartridge: Cartridge | None, examples: Sequence[Example], batch_size: int = 8
) -> Metrics:
    total_loss, total_agree, total_tokens = 0.0, 0, 0
    for i in range(0, len(examples), batch_size):
        loss, agree, n = distillation_loss(model, cartridge, examples[i : i + batch_size])
        total_loss, total_agree, total_tokens = total_loss + loss.item(), total_agree + int(agree), total_tokens + n
    return Metrics(loss=total_loss / max(total_tokens, 1), agreement=total_agree / max(total_tokens, 1))


def train(
    model: PreTrainedModel,
    cartridge: Cartridge,
    examples: Sequence[Example],
    *,
    eval_examples: Sequence[Example] = (),
    lr: float = 2e-2,
    epochs: int = 1,
    batch_size: int = 8,
    seed: int = 0,
    gradient_checkpointing: bool = True,
    on_step: Callable[[int, int, float], None] | None = None,
) -> TrainHistory:
    """Optimize `cartridge` in place. Returns the loss curve and eval metrics.

    `on_step(step, total_steps, loss)` is called after every optimizer step.
    The defaults (Adam, lr 2e-2, constant schedule) follow the Cartridges paper.
    `gradient_checkpointing` trades one extra forward pass per step for far less activation memory.
    """
    cartridge.to(model.device)
    history = TrainHistory()
    if eval_examples:
        history.eval_no_context = evaluate(model, None, eval_examples, batch_size)
        history.eval_before = evaluate(model, cartridge, eval_examples, batch_size)

    optimizer = torch.optim.Adam([p for p in cartridge.parameters() if p.requires_grad], lr=lr)
    rng = random.Random(seed)
    steps_per_epoch = (len(examples) + batch_size - 1) // batch_size
    total_steps, step = epochs * steps_per_epoch, 0

    with layer_checkpointing(model) if gradient_checkpointing else contextlib.nullcontext():
        for _ in range(epochs):
            order = list(examples)
            rng.shuffle(order)
            for i in range(0, len(order), batch_size):
                loss_sum, _, n_tokens = distillation_loss(model, cartridge, order[i : i + batch_size])
                loss = loss_sum / n_tokens
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                step += 1
                history.losses.append(loss.item())
                if on_step:
                    on_step(step, total_steps, loss.item())

    if eval_examples:
        history.eval_after = evaluate(model, cartridge, eval_examples, batch_size)
    return history
