"""Self-study: the model quizzes itself about the corpus to create training data.

For every training example:

  1. Pick a random chunk of the corpus and a seed prompt ("ask a question",
     "ask for a summary", ...).
  2. Asker:    the model reads the chunk and writes a question about it.
  3. Answerer: the model reads the chunk and answers that question.
  4. Teacher:  we record the answerer's top-k next-token probabilities at every
               answer token. This is the target the cartridge is trained to match
               *without* seeing the chunk ("context distillation").

Generation (steps 2-3) can run on the local Hugging Face model or on any
OpenAI-compatible server (vLLM, SGLang, llama.cpp, ...) for much higher
throughput. Step 4 always runs locally, so the targets come from exactly the
weights the cartridge will be used with.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import PreTrainedModel

from .chat_format import ChatFormat, Message
from .corpus import Chunker
from .models import hidden_states_at
from .seeds import SEED_TYPES, sample_seed_prompt

CONTEXT_TEMPLATE = "You are in a conversation about the following user information.\n\n<info>\n{chunk}\n</info>"


# --------------------------------------------------------------------------- data


@dataclass
class Example:
    seed_type: str
    question: str
    answer: str
    prompt_ids: list[int]  # the question, formatted as it follows the cartridge
    answer_ids: list[int]  # the answer tokens the loss is computed on
    topk_ids: torch.Tensor | None = None  # [len(answer_ids), k] int32
    topk_logprobs: torch.Tensor | None = None  # [len(answer_ids), k] float16


class SelfStudyDataset(list[Example]):
    """A list of examples that can be saved to / loaded from a directory.

    Layout: `examples.jsonl` (readable text + token ids) and `teacher.safetensors`
    (the teacher's top-k predictions, concatenated in example order).
    """

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "examples.jsonl", "w") as f:
            for ex in self:
                row = {k: v for k, v in asdict(ex).items() if k not in ("topk_ids", "topk_logprobs")}
                f.write(json.dumps(row) + "\n")
        save_file(
            {
                "topk_ids": torch.cat([ex.topk_ids for ex in self]).contiguous(),
                "topk_logprobs": torch.cat([ex.topk_logprobs for ex in self]).contiguous(),
            },
            str(directory / "teacher.safetensors"),
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> SelfStudyDataset:
        directory = Path(directory)
        teacher = load_file(str(directory / "teacher.safetensors"))
        data, offset = cls(), 0
        with open(directory / "examples.jsonl") as f:
            for line in f:
                ex = Example(**json.loads(line))
                n = len(ex.answer_ids)
                ex.topk_ids = teacher["topk_ids"][offset : offset + n]
                ex.topk_logprobs = teacher["topk_logprobs"][offset : offset + n]
                offset += n
                data.append(ex)
        return data

    def split(self, eval_fraction: float, seed: int = 0) -> tuple[SelfStudyDataset, SelfStudyDataset]:
        items = list(self)
        random.Random(seed).shuffle(items)
        n_eval = int(len(items) * eval_fraction)
        if eval_fraction > 0 and len(items) >= 2:
            n_eval = max(1, n_eval)  # always hold something out to report quality
        return SelfStudyDataset(items[n_eval:]), SelfStudyDataset(items[:n_eval])


# --------------------------------------------------------------------------- generators


class LocalGenerator:
    """Generates with the local Hugging Face model (batched, left-padded)."""

    def __init__(self, model: PreTrainedModel, chat_format: ChatFormat):
        self.model, self.chat_format = model, chat_format
        tok = chat_format.tokenizer
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        self.stop_ids = sorted({i for i in [*eos, tok.eos_token_id, chat_format.end_of_turn_id] if i is not None})
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else self.stop_ids[0]

    @torch.no_grad()
    def generate(
        self, requests: list[tuple[str, list[Message]]], max_new_tokens: int, temperature: float
    ) -> list[list[int]]:
        prompts = [self.chat_format.prompt_ids(system, messages) for system, messages in requests]
        width = max(map(len, prompts))
        input_ids = torch.tensor([[self.pad_id] * (width - len(p)) + p for p in prompts], device=self.model.device)
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=self.model.device)
        sampling = {"do_sample": True, "temperature": temperature, "top_p": 0.95} if temperature > 0 else {
            "do_sample": False, "temperature": None, "top_p": None, "top_k": None}  # fmt: skip
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.stop_ids,
            pad_token_id=self.pad_id,
            **sampling,
        )
        return [self._cut_at_stop(row.tolist()) for row in out[:, width:]]

    def _cut_at_stop(self, ids: list[int]) -> list[int]:
        for i, t in enumerate(ids):
            if t in self.stop_ids:
                return ids[:i] + [self.chat_format.end_of_turn_id]  # keep one end-of-turn token
        return ids


class OpenAIGenerator:
    """Generates with any OpenAI-compatible `/v1/chat/completions` endpoint.

    The server must serve the *same* model you train the cartridge for.
    """

    def __init__(
        self,
        chat_format: ChatFormat,
        base_url: str,
        model: str,
        api_key: str = "none",
        concurrency: int = 64,
        transport=None,  # an httpx transport, e.g. to talk to an in-process app in tests
    ):
        self.chat_format = chat_format
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.concurrency = model, api_key, concurrency
        self.transport = transport

    def generate(
        self, requests: list[tuple[str, list[Message]]], max_new_tokens: int, temperature: float
    ) -> list[list[int]]:
        return asyncio.run(self._generate_all(requests, max_new_tokens, temperature))

    async def _generate_all(self, requests, max_new_tokens, temperature) -> list[list[int]]:
        import httpx

        semaphore = asyncio.Semaphore(self.concurrency)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=600, headers=headers, transport=self.transport) as client:

            async def one(system: str, messages: list[Message]) -> list[int]:
                body = {
                    "model": self.model,
                    "messages": [{"role": "system", "content": system}, *messages],
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                    "chat_template_kwargs": self.chat_format.template_kwargs,  # vLLM/SGLang: disable thinking
                }
                async with semaphore:
                    r = await client.post(self.url, json=body)
                r.raise_for_status()
                choice = r.json()["choices"][0]
                text = choice["message"]["content"] or ""
                text = text.split("</think>")[-1].strip()  # drop reasoning if the server added any
                ids = self.chat_format.tokenizer.encode(text, add_special_tokens=False)
                return ids + [self.chat_format.end_of_turn_id] if choice.get("finish_reason") == "stop" else ids

            return await asyncio.gather(*(one(s, m) for s, m in requests))


Generator = LocalGenerator | OpenAIGenerator


# --------------------------------------------------------------------------- teacher


@torch.no_grad()
def teacher_topk(
    model: PreTrainedModel,
    chat_format: ChatFormat,
    contexts: list[str],
    examples: list[Example],
    top_k: int = 20,
) -> None:
    """Fill in `topk_ids` / `topk_logprobs` for a batch of examples (in place).

    The teacher sees `[system prompt with the chunk] + question + answer` and we
    keep its top-k log-probabilities for each answer token.
    """
    seqs, positions = [], []
    for b, (context, ex) in enumerate(zip(contexts, examples, strict=True)):
        prefix = chat_format.system_ids(context) + ex.prompt_ids
        seqs.append(prefix + ex.answer_ids)
        # logits at position i predict token i + 1
        positions += [(b, len(prefix) - 1 + j) for j in range(len(ex.answer_ids))]

    width = max(map(len, seqs))
    pad = chat_format.tokenizer.pad_token_id or 0
    input_ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], device=model.device)
    mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=model.device)
    hidden = hidden_states_at(model, input_ids, mask, torch.tensor(positions, device=model.device))
    logprobs, ids = [], []
    for start in range(0, len(hidden), 1024):  # bound memory: 1,024 x vocabulary floats at a time
        chunk_logprobs = torch.log_softmax(model.lm_head(hidden[start : start + 1024]).float(), dim=-1)
        top = chunk_logprobs.topk(top_k, dim=-1)
        logprobs.append(top.values)
        ids.append(top.indices)
    logprobs, ids = torch.cat(logprobs), torch.cat(ids)

    offset = 0
    for ex in examples:
        n = len(ex.answer_ids)
        ex.topk_ids = ids[offset : offset + n].to(torch.int32).cpu()
        ex.topk_logprobs = logprobs[offset : offset + n].to(torch.float16).cpu()
        offset += n


# --------------------------------------------------------------------------- main loop


def synthesize(
    model: PreTrainedModel,
    chat_format: ChatFormat,
    corpus: str,
    num_samples: int,
    *,
    generator: Generator | None = None,
    batch_size: int = 16,
    chunk_tokens: tuple[int, int] = (512, 1024),
    max_question_tokens: int = 256,
    max_answer_tokens: int = 512,
    question_temperature: float = 0.6,
    answer_temperature: float = 0.0,
    top_k: int = 20,
    teacher_batch_size: int = 4,
    seed_types: tuple[str, ...] = SEED_TYPES,
    seed: int = 0,
    on_progress: Callable[[int], None] | None = None,
) -> SelfStudyDataset:
    """Generate `num_samples` self-study conversations about `corpus`."""
    generator = generator or LocalGenerator(model, chat_format)
    tokenizer = chat_format.tokenizer
    chunker = Chunker(corpus, tokenizer, *chunk_tokens)
    rng = random.Random(seed)
    dataset = SelfStudyDataset()
    empty_batches = 0

    while len(dataset) < num_samples:
        n = min(batch_size, num_samples - len(dataset))
        contexts = [CONTEXT_TEMPLATE.format(chunk=chunker.sample(rng)) for _ in range(n)]
        seeds = [sample_seed_prompt(rng, seed_types) for _ in range(n)]

        # Asker: write a question about the chunk.
        question_ids = generator.generate(
            [(ctx, [{"role": "user", "content": prompt}]) for ctx, (_, prompt) in zip(contexts, seeds, strict=True)],
            max_new_tokens=max_question_tokens,
            temperature=question_temperature,
        )
        questions = [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in question_ids]
        keep = [i for i, q in enumerate(questions) if q]
        contexts = [contexts[i] for i in keep]
        seeds = [seeds[i] for i in keep]
        questions = [questions[i] for i in keep]

        # Answerer: answer it, with the chunk in context.
        answer_ids = generator.generate(
            [(ctx, [{"role": "user", "content": q}]) for ctx, q in zip(contexts, questions, strict=True)],
            max_new_tokens=max_answer_tokens,
            temperature=answer_temperature,
        )

        batch = [
            Example(
                seed_type=seed_type,
                question=q,
                answer=tokenizer.decode(a, skip_special_tokens=True).strip(),
                prompt_ids=chat_format.conversation_ids([{"role": "user", "content": q}]),
                answer_ids=a,
            )
            for (seed_type, _), q, a in zip(seeds, questions, answer_ids, strict=True)
            if a
        ]
        contexts = [ctx for ctx, a in zip(contexts, answer_ids, strict=True) if a]

        # Teacher: record the answerer's next-token distributions.
        for i in range(0, len(batch), teacher_batch_size):
            teacher_topk(
                model, chat_format, contexts[i : i + teacher_batch_size], batch[i : i + teacher_batch_size], top_k
            )

        empty_batches = 0 if batch else empty_batches + 1
        if empty_batches >= 3:
            raise RuntimeError("The generator returned no usable questions/answers three times in a row.")
        dataset.extend(batch)
        if on_progress:
            on_progress(len(batch))

    return dataset
