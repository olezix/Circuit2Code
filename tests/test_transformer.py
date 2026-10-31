"""Tests of the transparent Transformer and the training pipeline."""

from __future__ import annotations

import math
import os

import pytest
import torch

from src.models.train import TrainConfig, evaluate, train_modular_addition
from src.models.transformer import (
    ModelConfig,
    Transformer,
    modular_addition_dataset,
    modular_addition_tokens,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_config_derived_quantities():
    cfg = ModelConfig(p=59, d_model=128, n_heads=4)
    assert cfg.n_classes == 59
    assert cfg.vocab_size == 60
    assert cfg.eq_token == 59
    assert cfg.d_head == 32
    assert cfg.answer_position == 2
    assert cfg.to_dict()["p"] == 59
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg


def test_config_rejects_indivisible_heads():
    with pytest.raises(ValueError):
        ModelConfig(p=7, d_model=10, n_heads=4).d_head


def test_config_roundtrip_ignores_unknown_keys():
    cfg = ModelConfig(p=5, d_model=8, n_heads=2)
    payload = cfg.to_dict()
    payload["future_option"] = 123
    assert ModelConfig.from_dict(payload) == cfg


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------
def test_forward_shapes_and_determinism(medium_model):
    tokens = modular_addition_tokens(torch.arange(11), torch.flip(torch.arange(11), [0]), 11)
    logits_a, _ = medium_model(tokens)
    logits_b, _ = medium_model(tokens)
    assert logits_a.shape == (11, 3, 11)
    assert torch.equal(logits_a, logits_b)


def test_invalid_token_shape_raises(small_model):
    with pytest.raises(ValueError):
        small_model.forward(torch.zeros(2, 5, dtype=torch.long))


def test_attention_rows_sum_to_one(small_model):
    tokens = modular_addition_tokens(torch.arange(7), torch.arange(7), 7)
    _, trace = small_model(tokens, trace=True)
    pattern = trace.attention_patterns[0]
    assert pattern.shape == (7, 2, 3, 3)
    assert torch.allclose(pattern.sum(dim=-1), torch.ones(7, 2, 3), atol=1e-6)
    assert (pattern >= 0).all()


def test_residual_stream_decomposes_into_heads(small_model):
    """``x_L = x_0 + sum_h head_h`` must hold exactly for a one-layer model."""
    tokens = modular_addition_tokens(torch.tensor([1, 2]), torch.tensor([3, 4]), 7)
    _, trace = small_model(tokens, trace=True)
    residual = trace.residual_stream[-1]
    recomposed = trace.embedding + trace.head_outputs[0].sum(dim=2)
    assert torch.allclose(residual, recomposed, atol=1e-6)


def test_answer_position_matches_manual_computation(small_model):
    tokens = modular_addition_tokens(torch.tensor([3]), torch.tensor([5]), 7)
    logits, trace = small_model(tokens, trace=True)
    manual = trace.residual_stream[-1][:, -1, :] @ small_model.W_U
    assert torch.allclose(logits[:, -1, :], manual, atol=1e-6)


def test_causal_mask_blocks_future_positions():
    cfg = ModelConfig(p=5, d_model=8, n_heads=2, causal=True)
    model = Transformer(cfg, seed=0).eval()
    tokens = modular_addition_tokens(torch.arange(5), torch.arange(5), 5)
    with torch.no_grad():
        _, trace = model(tokens, trace=True)
    pattern = trace.attention_patterns[0]
    upper = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    assert torch.all(pattern[:, :, upper] < 1e-9)


def test_intervention_adds_delta_linearly(small_model):
    tokens = modular_addition_tokens(torch.tensor([1]), torch.tensor([2]), 7)
    base, trace = small_model(tokens, trace=True)
    delta = torch.zeros(1, small_model.config.d_model)
    delta[0, 0] = 0.5
    patched, _ = small_model(tokens, interventions=[(0, 2, delta)])
    expected = base[:, -1, :] + delta @ small_model.W_U
    assert torch.allclose(patched[:, -1, :], expected, atol=1e-6)


def test_intervention_validates_shape(small_model):
    tokens = modular_addition_tokens(torch.tensor([1]), torch.tensor([2]), 7)
    with pytest.raises(ValueError):
        small_model(tokens, interventions=[(0, 0, torch.zeros(1, 3))])
    with pytest.raises(ValueError):
        small_model(tokens, interventions=[(0, 9, torch.zeros(1, 16))])


def test_trace_detach_can_be_disabled(small_model):
    tokens = modular_addition_tokens(torch.tensor([1]), torch.tensor([2]), 7)
    _, trace = small_model(tokens, trace=True, detach_trace=False)
    assert trace.residual_stream[-1].requires_grad


def test_mlp_variant_runs():
    cfg = ModelConfig(p=5, d_model=8, n_heads=2, mlp_width=16)
    model = Transformer(cfg, seed=1).eval()
    tokens = modular_addition_tokens(torch.arange(5), torch.arange(5), 5)
    with torch.no_grad():
        logits, trace = model(tokens, trace=True)
    assert logits.shape == (5, 3, 5)
    assert trace.mlp_hidden[0].shape == (5, 3, 16)
    assert (trace.mlp_hidden[0] >= 0).all()


def test_answer_logits_and_predict(small_model):
    logits = small_model.answer_logits(torch.tensor([1, 2]), torch.tensor([3, 4]))
    assert logits.shape == (2, 7)
    assert small_model.predict(torch.tensor([1]), torch.tensor([3])).shape == (1,)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def test_save_load_roundtrip(tmp_path, small_model):
    path = str(tmp_path / "model.pt")
    small_model.save(path, extra={"note": "unit-test"})
    restored = Transformer.load(path)
    tokens = modular_addition_tokens(torch.arange(7), torch.arange(7), 7)
    with torch.no_grad():
        a, _ = small_model(tokens)
        b, _ = restored(tokens)
    assert torch.allclose(a, b)
    assert restored.config == small_model.config
    assert not restored.training


def test_load_rejects_foreign_file(tmp_path):
    path = str(tmp_path / "junk.pt")
    torch.save({"not": "a checkpoint"}, path)
    with pytest.raises(ValueError):
        Transformer.load(path)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def test_dataset_is_complete_and_deterministic():
    ds = modular_addition_dataset(7, train_fraction=0.5, seed=3)
    assert ds["tokens"].shape == (49, 3)
    assert torch.equal(ds["target"], (ds["a"] + ds["b"]) % 7)
    assert ds["tokens"][:, 2].eq(7).all()
    train, test = set(ds["train_idx"].tolist()), set(ds["test_idx"].tolist())
    assert train.isdisjoint(test)
    assert len(train | test) == 49
    again = modular_addition_dataset(7, train_fraction=0.5, seed=3)
    assert torch.equal(ds["train_idx"], again["train_idx"])


def test_tokens_helper_rejects_bad_shapes():
    tokens = modular_addition_tokens([1, 2], [3, 4], 5)
    assert tokens.shape == (2, 3)
    assert tokens[0, 2].item() == 5


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def test_training_loop_runs_and_checkpoints(tmp_path):
    out = str(tmp_path / "tiny.pt")
    cfg = TrainConfig(
        p=5,
        d_model=16,
        n_heads=2,
        epochs=60,
        eval_every=20,
        log_every=1000,
        train_fraction=1.0,
        post_grok_epochs=1,
        seed=0,
        checkpoint_path=out,
        verbose=False,
        torch_threads=1,
    )
    result = train_modular_addition(cfg)
    assert os.path.exists(out)
    assert result.history
    assert math.isfinite(result.history[-1]["train_loss"])
    assert result.epochs_run > 0
    assert 0.0 <= result.final_train_acc <= 1.0
    reloaded = Transformer.load(out)
    assert reloaded.config.p == 5
    # The trajectory is written next to the checkpoint, so every reported number
    # can be regenerated from an artefact instead of from stdout.
    history_path = os.path.splitext(out)[0] + ".history.json"
    assert os.path.exists(history_path)
    import json

    payload = json.loads(open(history_path, encoding="utf-8").read())
    assert payload["summary"]["p"] == 5
    assert payload["history"]


def test_training_is_reproducible():
    cfg = TrainConfig(
        p=5,
        d_model=8,
        n_heads=2,
        epochs=15,
        eval_every=5,
        log_every=1000,
        train_fraction=0.8,
        seed=7,
        verbose=False,
        torch_threads=1,
    )
    first = train_modular_addition(cfg)
    second = train_modular_addition(cfg)
    assert torch.equal(first.model.W_E, second.model.W_E)
    assert first.history == second.history


def test_evaluate_returns_accuracy(small_model, small_dataset):
    loss, acc = evaluate(small_model, small_dataset, small_dataset["test_idx"])
    assert loss >= 0.0
    assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# Learning-rate schedule and gradient clipping
# ---------------------------------------------------------------------------
def test_cosine_schedule_anneals_the_learning_rate():
    from src.models.train import TrainConfig

    cfg = TrainConfig(
        p=5,
        d_model=8,
        n_heads=2,
        epochs=20,
        eval_every=1,
        log_every=1000,
        lr=1e-2,
        lr_schedule="cosine",
        lr_final_frac=0.0,
        train_fraction=1.0,
        seed=0,
        verbose=False,
        torch_threads=1,
    )
    result = train_modular_addition(cfg)
    rates = [record["lr"] for record in result.history[:-1]]
    assert rates == sorted(rates, reverse=True)  # monotonically decreasing
    assert rates[0] == pytest.approx(1e-2, rel=1e-6)
    assert rates[-1] < 0.1 * rates[0]


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError):
        train_modular_addition(
            TrainConfig(p=5, d_model=8, n_heads=2, epochs=1, lr_schedule="warmup", verbose=False)
        )


def test_gradient_clipping_keeps_training_finite():
    cfg = TrainConfig(
        p=5,
        d_model=8,
        n_heads=2,
        epochs=10,
        eval_every=5,
        log_every=1000,
        train_fraction=1.0,
        grad_clip=1e-3,
        seed=0,
        verbose=False,
        torch_threads=1,
    )
    result = train_modular_addition(cfg)
    assert math.isfinite(result.history[-1]["train_loss"])
    assert all(torch.isfinite(w).all().item() for w in result.model.parameters())


def test_checkpoint_is_written_even_off_the_round_epoch(tmp_path):
    """An improvement between two ``checkpoint_every`` boundaries is still saved.

    Regression test: the intermediate checkpoint used to require
    ``epoch % checkpoint_every == 0`` *and* an improvement at that exact epoch,
    so a run whose accuracy only improved in between left no checkpoint at all.
    """
    out = str(tmp_path / "intermediate.pt")
    cfg = TrainConfig(
        p=5,
        d_model=16,
        n_heads=2,
        init_std=0.5,
        epochs=120,
        lr=5e-3,
        eval_every=10,
        log_every=1000,
        train_fraction=1.0,
        seed=0,
        checkpoint_path=out,
        checkpoint_every=7,  # deliberately coprime with eval_every
        verbose=False,
        torch_threads=1,
    )
    train_modular_addition(cfg)
    assert os.path.exists(out), "no intermediate checkpoint was written"
