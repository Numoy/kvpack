"""Chatting with a model that has a cartridge (or a document, or nothing) in front of it."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from .cartridge import Cartridge
from .chat_format import ChatFormat, Message


@dataclass
class Completion:
    text: str
    token_ids: list[int]  # generated tokens, including the end-of-turn token if the model stopped
    finish_reason: str  # "stop" if the model ended its turn, "length" if it hit max_new_tokens


def generate(model: PreTrainedModel, chat_format: ChatFormat, messages: list[Message], **kwargs) -> str:
    """Generate the assistant's reply to `messages` and return its text. See `complete`."""
    return complete(model, chat_format, messages, **kwargs).text


@torch.no_grad()
def complete(
    model: PreTrainedModel,
    chat_format: ChatFormat,
    messages: list[Message],
    *,
    cartridge: Cartridge | None = None,
    context: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.8,
    streamer: TextIteratorStreamer | None = None,
    cancel: threading.Event | None = None,
) -> Completion:
    """Generate the assistant's reply to `messages`.

    Exactly one of three modes applies:
      * `cartridge=...`: the cartridge stands in for the system prompt.
      * `context=...`:   the text is placed in the system prompt (the "full context" baseline).
      * neither:         a plain chat with no extra knowledge.

    Setting `cancel` from another thread stops generation after the current token
    (finish_reason "cancelled"), e.g. when a streaming client disconnects.
    """
    if cartridge is not None and context is not None:
        raise ValueError("Pass either a cartridge or a context, not both.")
    tokenizer = chat_format.tokenizer

    cache = None
    if cartridge is not None:
        ids = chat_format.conversation_ids(messages)
        cache = cartridge.to_cache(model)
        # `generate` expects input_ids to cover the cached positions too. The cartridge has
        # no real tokens, so we put placeholders there; they're never read because those
        # positions are already in the cache.
        ids = [tokenizer.pad_token_id] * cartridge.num_tokens + ids
    elif context is not None:
        ids = chat_format.prompt_ids(context, messages)
    else:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False, **chat_format.template_kwargs
        )

    input_ids = torch.tensor([ids], device=model.device)
    sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p} if temperature > 0 else {
        "do_sample": False, "temperature": None, "top_p": None, "top_k": None}  # fmt: skip
    stop_ids = [chat_format.end_of_turn_id, tokenizer.eos_token_id]
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        past_key_values=cache,
        max_new_tokens=max_new_tokens,
        eos_token_id=stop_ids,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([_Cancel(cancel)]) if cancel else None,
        **sampling,
    )
    new_ids = out[0, len(ids) :].tolist()
    stopped = any(t in stop_ids for t in new_ids)
    if stopped:  # drop anything after the first stop token (padding)
        new_ids = new_ids[: next(i for i, t in enumerate(new_ids) if t in stop_ids) + 1]
    return Completion(
        text=tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
        token_ids=new_ids,
        finish_reason="stop" if stopped else "cancelled" if cancel and cancel.is_set() else "length",
    )


class _Cancel(StoppingCriteria):
    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device)
