"""Turning chat messages into token ids, split around the system prompt.

A cartridge *replaces the system prompt*. So we need to tokenize a conversation in
two separate pieces:

    [ system prompt tokens ]  [ everything after the system prompt ]
      ^ the cartridge             ^ what we feed the model at chat time
        stands in for these

Chat templates differ between model families, so instead of hard-coding special
tokens we render the model's own template with a marker string and split on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from transformers import PreTrainedTokenizerBase

# A string that will never appear in a real document or chat template.
_MARKER = "CARTRIDGE_MARKER"

Message = dict[str, str]  # {"role": "user" | "assistant" | "system", "content": "..."}


class UnsupportedModelError(ValueError):
    """Raised when a model's chat template can't be used with cartridges."""


@dataclass
class ChatFormat:
    """Tokenizes system prompts and conversations separately for one tokenizer."""

    tokenizer: PreTrainedTokenizerBase
    system_head_ids: list[int]  # tokens before the system prompt text
    system_tail_ids: list[int]  # tokens after the system prompt text
    end_of_turn_id: int  # token that ends an assistant message
    template_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizerBase) -> ChatFormat:
        if tokenizer.chat_template is None:
            raise UnsupportedModelError(
                "This tokenizer has no chat template. Use an instruction-tuned model "
                "(e.g. Qwen/Qwen3-4B or meta-llama/Llama-3.1-8B-Instruct)."
            )
        # Qwen3 thinks out loud by default; cartridges are trained on direct answers.
        # Templates that don't know this variable simply ignore it.
        template_kwargs = {"enable_thinking": False}

        system_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": _MARKER}], tokenize=False, **template_kwargs
        )
        if system_text.count(_MARKER) != 1:
            raise UnsupportedModelError(
                "This model's chat template has no separate system prompt, "
                "so there is nothing for a cartridge to replace."
            )
        head, tail = system_text.split(_MARKER)

        # Find the token that ends an assistant turn (e.g. <|im_end|> or <|eot_id|>).
        convo_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": _MARKER},
            ],
            tokenize=False,
            **template_kwargs,
        )
        after_answer = convo_text.split(_MARKER)[-1]
        after_ids = tokenizer.encode(after_answer, add_special_tokens=False)
        end_of_turn_id = after_ids[0] if after_ids else tokenizer.eos_token_id

        return cls(
            tokenizer=tokenizer,
            system_head_ids=tokenizer.encode(head, add_special_tokens=False),
            system_tail_ids=tokenizer.encode(tail, add_special_tokens=False),
            end_of_turn_id=end_of_turn_id,
            template_kwargs=template_kwargs,
        )

    def system_ids(self, content: str, max_tokens: int | None = None) -> list[int]:
        """Token ids of a complete system prompt containing `content`.

        If `max_tokens` is given, the content is truncated so that the whole system
        prompt (including the template's special tokens) fits in `max_tokens`.
        """
        body = self.tokenizer.encode(content, add_special_tokens=False)
        if max_tokens is not None:
            budget = max_tokens - len(self.system_head_ids) - len(self.system_tail_ids)
            if budget <= 0:
                raise ValueError(f"max_tokens={max_tokens} is too small for this chat template.")
            body = body[:budget]
        return self.system_head_ids + body + self.system_tail_ids

    def conversation_ids(self, messages: list[Message], add_generation_prompt: bool = True) -> list[int]:
        """Token ids of `messages` as they appear *after* the system prompt.

        With `add_generation_prompt=True` the ids end with the assistant header, ready
        for the model to write its reply.
        """
        system = {"role": "system", "content": _MARKER}
        system_text = self.tokenizer.apply_chat_template([system], tokenize=False, **self.template_kwargs)
        full_text = self.tokenizer.apply_chat_template(
            [system, *messages],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **self.template_kwargs,
        )
        if not full_text.startswith(system_text):
            raise UnsupportedModelError("The chat template renders the system prompt differently mid-conversation.")
        return self.tokenizer.encode(full_text[len(system_text) :], add_special_tokens=False)

    def prompt_ids(self, system: str, messages: list[Message]) -> list[int]:
        """Full prompt: a system prompt with `system` followed by `messages`."""
        return self.system_ids(system) + self.conversation_ids(messages)
