"""A small retrieval baseline, so cartridges can be compared against RAG on your data.

BM25 over fixed-size token chunks: the classic lexical retriever. It needs no
extra model or service, which makes the comparison easy to reproduce. Embedding
retrievers can do better on paraphrased questions, so treat this as a reasonable
floor for RAG, not its ceiling.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from transformers import PreTrainedTokenizerBase

_WORD = re.compile(r"\w+", re.UNICODE)


def _terms(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


class BM25Retriever:
    def __init__(
        self,
        corpus: str,
        tokenizer: PreTrainedTokenizerBase,
        chunk_tokens: int = 256,
        overlap_tokens: int = 32,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        ids = tokenizer.encode(corpus, add_special_tokens=False)
        step = max(1, chunk_tokens - overlap_tokens)
        self.chunks = [tokenizer.decode(ids[i : i + chunk_tokens]) for i in range(0, max(len(ids), 1), step)]
        self.k1, self.b = k1, b
        self._docs = [Counter(_terms(c)) for c in self.chunks]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg_len = sum(self._lengths) / max(len(self._lengths), 1)
        df = Counter(term for doc in self._docs for term in doc)
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = _terms(query)
        out = []
        for doc, length in zip(self._docs, self._lengths, strict=True):
            s = 0.0
            for t in terms:
                if t in doc:
                    tf = doc[t]
                    s += (
                        self._idf[t]
                        * tf
                        * (self.k1 + 1)
                        / (tf + self.k1 * (1 - self.b + self.b * length / self._avg_len))
                    )
            out.append(s)
        return out

    def retrieve(self, query: str, k: int = 4) -> list[str]:
        """The `k` best chunks, returned in document order (which reads more naturally)."""
        scores = self.scores(query)
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in sorted(best)]

    def context(self, query: str, k: int = 4) -> str:
        return "\n\n---\n\n".join(self.retrieve(query, k))
