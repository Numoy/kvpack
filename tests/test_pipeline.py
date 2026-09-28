"""Synthesis, training and the end-to-end `build`, using a stub generator for speed."""

import contextlib

import torch
from conftest import DOCUMENT

from kvpack import Cartridge, SelfStudyDataset, build, evaluate, synthesize, train


class StubGenerator:
    """Returns canned questions and answers instead of running generation."""

    def __init__(self, chat_format):
        self.tokenizer = chat_format.tokenizer
        self.end = chat_format.end_of_turn_id
        self.calls = 0

    def generate(self, requests, max_new_tokens, temperature):
        self.calls += 1
        text = "Who is the Station Lead?" if self.calls % 2 else "The Station Lead is Oskar Lindqvist-Mbeki."
        return [self.tokenizer.encode(text, add_special_tokens=False) + [self.end] for _ in requests]


def make_dataset(model, chat_format, n=6) -> SelfStudyDataset:
    return synthesize(
        model, chat_format, DOCUMENT, n, generator=StubGenerator(chat_format), batch_size=3, chunk_tokens=(16, 32)
    )


def test_synthesize_records_teacher_distributions(model, chat_format):
    dataset = make_dataset(model, chat_format)
    assert len(dataset) == 6
    ex = dataset[0]
    assert ex.question == "Who is the Station Lead?"
    assert ex.answer == "The Station Lead is Oskar Lindqvist-Mbeki."
    assert ex.answer_ids[-1] == chat_format.end_of_turn_id
    assert ex.topk_ids.shape == ex.topk_logprobs.shape == (len(ex.answer_ids), 20)
    # top-k log-probabilities are sorted and valid
    assert torch.all(ex.topk_logprobs[:, :-1] >= ex.topk_logprobs[:, 1:])
    assert torch.all(ex.topk_logprobs <= 0)


def test_dataset_save_load_roundtrip(model, chat_format, tmp_path):
    dataset = make_dataset(model, chat_format)
    dataset.save(tmp_path / "data")
    loaded = SelfStudyDataset.load(tmp_path / "data")
    assert len(loaded) == len(dataset)
    for a, b in zip(dataset, loaded, strict=True):
        assert a.question == b.question and a.answer_ids == b.answer_ids
        torch.testing.assert_close(a.topk_logprobs, b.topk_logprobs)


def test_training_lowers_loss_and_keeps_frozen_tokens(model, chat_format):
    dataset = make_dataset(model, chat_format)
    cartridge = Cartridge.from_text(model, chat_format, DOCUMENT[:100], num_tokens=12)
    frozen_before = cartridge.frozen_keys.clone()
    trainable_before = cartridge.trainable_keys.detach().clone()

    before = evaluate(model, cartridge, dataset)
    train(model, cartridge, dataset, epochs=15, batch_size=6, lr=5e-2)
    after = evaluate(model, cartridge, dataset)

    assert after.loss < before.loss
    assert torch.equal(cartridge.frozen_keys, frozen_before)
    assert not torch.equal(cartridge.trainable_keys.detach(), trainable_before)


def test_model_weights_are_not_trained(model, chat_format):
    dataset = make_dataset(model, chat_format, n=3)
    weights_before = {k: v.clone() for k, v in model.state_dict().items()}
    cartridge = Cartridge.from_text(model, chat_format, DOCUMENT, num_tokens=12)
    train(model, cartridge, dataset, batch_size=3)
    for k, v in model.state_dict().items():
        assert torch.equal(v, weights_before[k]), k


def test_build_end_to_end(model, chat_format, tmp_path):
    result = build(
        model, chat_format, DOCUMENT,
        name="kestrel", num_tokens=16, num_samples=8, eval_fraction=0.25,
        generator=StubGenerator(chat_format), synth_batch_size=4, train_batch_size=4,
    )  # fmt: skip
    meta = result.cartridge.metadata
    assert meta["name"] == "kestrel" and meta["num_samples"] == 8
    assert result.history.eval_before is not None and result.history.eval_after is not None
    assert len(result.history.losses) == 2  # 6 training examples / batch size 4

    path = result.cartridge.save(tmp_path / "kestrel.safetensors")
    assert Cartridge.load(path).metadata["eval_after"]["loss"] == result.history.eval_after.loss


def test_chunked_loss_matches_unchunked(model, chat_format, monkeypatch):
    """Chunking + checkpointing the loss is a memory optimization; it must not change the math."""
    import importlib

    train_module = importlib.import_module("kvpack.train")  # `kvpack.train` is also a function

    dataset = make_dataset(model, chat_format, n=3)
    results = []
    for chunk_tokens in (10_000, 7):
        monkeypatch.setattr(train_module, "LOSS_CHUNK_TOKENS", chunk_tokens)
        cartridge = Cartridge.from_text(model, chat_format, DOCUMENT, num_tokens=12)
        loss, agreements, n = train_module.distillation_loss(model, cartridge, dataset)
        loss.backward()
        results.append((loss.detach(), int(agreements), n, cartridge.trainable_keys.grad.clone()))

    (loss_a, agree_a, n_a, grad_a), (loss_b, agree_b, n_b, grad_b) = results
    assert n_a == n_b and agree_a == agree_b
    torch.testing.assert_close(loss_a, loss_b)
    torch.testing.assert_close(grad_a, grad_b)


def test_layer_checkpointing_gives_identical_gradients(model, chat_format):
    """Checkpointing is purely a memory optimization: gradients must not change."""
    from kvpack.models import layer_checkpointing
    from kvpack.train import distillation_loss

    dataset = make_dataset(model, chat_format, n=3)
    grads = []
    for use_checkpointing in (False, True):
        cartridge = Cartridge.from_text(model, chat_format, DOCUMENT, num_tokens=12)
        with layer_checkpointing(model) if use_checkpointing else contextlib.nullcontext():
            loss, _, _ = distillation_loss(model, cartridge, dataset)
            loss.backward()
        grads.append((cartridge.trainable_keys.grad.clone(), cartridge.trainable_values.grad.clone()))

    torch.testing.assert_close(grads[0], grads[1])
    assert "forward" not in model.model.layers[0].__dict__  # the wrapper is removed afterwards
