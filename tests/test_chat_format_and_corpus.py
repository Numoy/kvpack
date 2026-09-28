import pytest

from kvpack import load_corpus
from kvpack.corpus import Chunker


def test_system_prompt_is_wrapped_in_template(chat_format, tokenizer):
    text = tokenizer.decode(chat_format.system_ids("Some docs"))
    assert text == "<|im_start|>system\nSome docs<|im_end|>\n"


def test_system_prompt_truncation(chat_format):
    ids = chat_format.system_ids("word " * 500, max_tokens=40)
    assert len(ids) == 40
    assert ids[-len(chat_format.system_tail_ids) :] == chat_format.system_tail_ids


def test_system_prompt_too_small_budget(chat_format):
    with pytest.raises(ValueError):
        chat_format.system_ids("hello", max_tokens=2)


def test_conversation_follows_system_prompt(chat_format, tokenizer):
    ids = chat_format.conversation_ids([{"role": "user", "content": "Hi"}])
    text = tokenizer.decode(ids)
    assert text.startswith("<|im_start|>user\nHi<|im_end|>")
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")  # thinking disabled


def test_prompt_is_system_plus_conversation(chat_format):
    messages = [{"role": "user", "content": "Hi"}]
    assert chat_format.prompt_ids("Docs", messages) == (
        chat_format.system_ids("Docs") + chat_format.conversation_ids(messages)
    )


def test_end_of_turn_token(chat_format, tokenizer):
    assert tokenizer.decode([chat_format.end_of_turn_id]) == "<|im_end|>"


def test_load_corpus_reads_folders_and_skips_hidden_files(tmp_path):
    (tmp_path / "a.md").write_text("Alpha")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("Beta")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    corpus = load_corpus([tmp_path])
    assert "Alpha" in corpus and "Beta" in corpus
    assert "secret" not in corpus and "PNG" not in corpus
    assert "## File:" in corpus


def test_single_file_has_no_header(tmp_path):
    (tmp_path / "a.md").write_text("Just this.")
    assert load_corpus([tmp_path / "a.md"]) == "Just this."


def test_load_corpus_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus([tmp_path / "missing"])
    (tmp_path / "x.png").write_bytes(b"\x89PNG")
    with pytest.raises(ValueError):
        load_corpus([tmp_path])


def test_chunker_respects_bounds(tokenizer):
    import random

    chunker = Chunker("word " * 1000, tokenizer, min_tokens=10, max_tokens=20)
    rng = random.Random(0)
    for _ in range(20):
        n = len(tokenizer.encode(chunker.sample(rng), add_special_tokens=False))
        assert 8 <= n <= 22  # decode/re-encode can shift token boundaries slightly
