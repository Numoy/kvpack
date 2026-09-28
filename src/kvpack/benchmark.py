"""Comparing a cartridge with RAG, the full documents, and no context, on your own questions.

Each question lists the facts a correct answer must contain (`must_include`). An
item may give alternatives separated by "|" ("UV|ultraviolet"). Grading is a
transparent keyword check, not an LLM judge: easy to audit, strict about facts,
lenient about wording.

For every setup we also record how many tokens of context the model holds
(which is what drives GPU memory per conversation) and the time to first token.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from transformers import PreTrainedModel

from .cartridge import Cartridge
from .chat_format import ChatFormat
from .generate import complete
from .rag import BM25Retriever

SETUPS = ("cartridge", "rag", "full", "none")
ALL_SETUPS = (*SETUPS, "cartridge+rag")
LABELS = {
    "cartridge": "Cartridge",
    "cartridge+rag": "Cartridge + RAG",
    "rag": "RAG (BM25)",
    "full": "Full documents",
    "none": "No context",
}


@dataclass
class Question:
    question: str
    must_include: list[str]
    answer: str | None = None


def load_questions(path: str | Path) -> list[Question]:
    questions = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            questions.append(Question(row["question"], row["must_include"], row.get("answer")))
    return questions


def is_correct(answer: str, must_include: list[str]) -> bool:
    """True if every required fact (or one of its alternatives) appears in the answer."""
    text = answer.lower()
    for item in must_include:
        alternatives = [a.strip().lower() for a in item.split("|") if a.strip()]
        # word boundaries, so "9" doesn't match inside "2039"
        if not any(re.search(rf"(?<![\w.]){re.escape(a)}(?![\w])", text) for a in alternatives):
            return False
    return True


@dataclass
class SetupResult:
    setup: str
    correct: int = 0
    total: int = 0
    context_tokens: list[int] = field(default_factory=list)
    ttft_ms: list[float] = field(default_factory=list)
    answers: list[dict] = field(default_factory=list)
    skipped: str | None = None

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def summary(self, kv_bytes_per_token: int) -> dict:
        tokens = statistics.mean(self.context_tokens) if self.context_tokens else 0
        return {
            "setup": self.setup,
            "label": LABELS[self.setup],
            "correct": self.correct,
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "context_tokens": round(tokens),
            "kv_cache_mib": round(tokens * kv_bytes_per_token / 2**20, 2),
            "ttft_ms": round(statistics.median(self.ttft_ms)) if self.ttft_ms else None,
            "skipped": self.skipped,
        }


def with_excerpt(context: str, question: Question) -> str:
    return f"Relevant excerpt from the documents:\n\n{context}\n\nQuestion: {question.question}"


def kv_bytes_per_token(model: PreTrainedModel) -> int:
    config = model.config.get_text_config()
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    return 2 * config.num_hidden_layers * kv_heads * head_dim * model.dtype.itemsize


def run_benchmark(
    model: PreTrainedModel,
    chat_format: ChatFormat,
    corpus: str,
    cartridge: Cartridge,
    questions: list[Question],
    *,
    setups: tuple[str, ...] = SETUPS,
    rag_chunks: int = 4,
    rag_chunk_tokens: int = 256,
    max_new_tokens: int = 200,
    ttft_samples: int = 5,
    on_answer: Callable[[str, Question, str, bool], None] | None = None,
) -> dict:
    """Answer every question with every setup. Returns a JSON-serializable report."""
    needs_retrieval = any("rag" in name for name in setups)
    retriever = BM25Retriever(corpus, chat_format.tokenizer, chunk_tokens=rag_chunk_tokens) if needs_retrieval else None
    full_tokens = len(chat_format.system_ids(corpus))
    window = getattr(model.config, "max_position_embeddings", None)
    results = {name: SetupResult(name) for name in setups}

    if "full" in setups and window and full_tokens + max_new_tokens + 512 > window:
        results["full"].skipped = f"documents ({full_tokens:,} tokens) don't fit the {window:,}-token context window"

    for i, q in enumerate(questions):
        messages = [{"role": "user", "content": q.question}]
        for name in setups:
            r = results[name]
            if r.skipped:
                continue
            if name == "cartridge":
                kwargs, tokens = {"cartridge": cartridge}, cartridge.num_tokens
            elif name == "full":
                kwargs, tokens = {"context": corpus}, full_tokens
            elif name == "rag":
                context = retriever.context(q.question, k=rag_chunks)
                kwargs, tokens = {"context": context}, len(chat_format.system_ids(context))
            elif name == "cartridge+rag":
                # the cartridge carries the whole corpus; the retrieved excerpt goes with the question
                context = retriever.context(q.question, k=rag_chunks)
                kwargs = {"cartridge": cartridge, "messages": [{"role": "user", "content": with_excerpt(context, q)}]}
                tokens = cartridge.num_tokens + len(chat_format.tokenizer.encode(context, add_special_tokens=False))
            else:
                kwargs, tokens = {}, 0
            prompt = kwargs.pop("messages", messages)

            if i < ttft_samples:  # time to first token: prefill plus one decoding step
                started = time.perf_counter()
                complete(model, chat_format, prompt, max_new_tokens=1, temperature=0, **kwargs)
                r.ttft_ms.append((time.perf_counter() - started) * 1000)

            answer = complete(model, chat_format, prompt, max_new_tokens=max_new_tokens, temperature=0, **kwargs).text
            ok = is_correct(answer, q.must_include)
            r.total += 1
            r.correct += ok
            r.context_tokens.append(tokens)
            r.answers.append({"question": q.question, "answer": answer, "correct": ok})
            if on_answer:
                on_answer(name, q, answer, ok)

    per_token = kv_bytes_per_token(model)
    return {
        "model": model.config.name_or_path,
        "cartridge_tokens": cartridge.num_tokens,
        "corpus_tokens": full_tokens,
        "questions": len(questions),
        "rag": {"retriever": "bm25", "chunks": rag_chunks, "chunk_tokens": rag_chunk_tokens},
        "kv_bytes_per_token": per_token,
        "results": [results[name].summary(per_token) for name in setups],
        "answers": {name: results[name].answers for name in setups},
    }
