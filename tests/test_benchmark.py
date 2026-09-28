from conftest import DOCUMENT

from kvpack import Cartridge
from kvpack.benchmark import Question, is_correct, kv_bytes_per_token, run_benchmark
from kvpack.rag import BM25Retriever


def test_is_correct_needs_every_fact_and_accepts_alternatives():
    facts = ["Old Gerda", "UV|ultraviolet"]
    assert is_correct("It's Old Gerda, after UV exposure.", facts)
    assert is_correct("old gerda; ultraviolet light", facts)
    assert not is_correct("Old Gerda only", facts)


def test_numbers_match_on_word_boundaries():
    assert is_correct("The window is 9 hours.", ["9"])
    assert not is_correct("The season is 2039.", ["9"])
    assert not is_correct("It takes 1.9 days.", ["9"])


def test_bm25_finds_the_relevant_chunk(tokenizer):
    corpus = " ".join(f"Filler sentence number {i} about nothing." for i in range(200))
    corpus += " The bell rung at the lake is called Old Gerda. " + corpus
    retriever = BM25Retriever(corpus, tokenizer, chunk_tokens=64, overlap_tokens=8)
    assert "Old Gerda" in retriever.retrieve("What is the bell at the lake called?", k=1)[0]


def test_kv_bytes_per_token(model):
    # 2 (K and V) x 2 layers x 2 KV heads x 16 dims x 4 bytes (float32)
    assert kv_bytes_per_token(model) == 2 * 2 * 2 * 16 * 4


def test_run_benchmark_reports_every_setup(model, chat_format):
    cartridge = Cartridge.from_text(model, chat_format, DOCUMENT, num_tokens=24)
    questions = [Question("Who is the Station Lead?", ["Oskar"]), Question("How much power?", ["410"])]
    seen = []
    report = run_benchmark(
        model, chat_format, DOCUMENT, cartridge, questions, max_new_tokens=3, rag_chunk_tokens=32,
        on_answer=lambda setup, q, answer, ok: seen.append(setup),
    )  # fmt: skip
    assert [r["setup"] for r in report["results"]] == ["cartridge", "rag", "full", "none"]
    by_setup = {r["setup"]: r for r in report["results"]}
    assert by_setup["cartridge"]["context_tokens"] == 24
    assert by_setup["none"]["context_tokens"] == 0
    assert by_setup["full"]["context_tokens"] > by_setup["cartridge"]["context_tokens"]
    assert all(r["total"] == 2 and r["ttft_ms"] is not None for r in report["results"])
    assert len(seen) == 8
