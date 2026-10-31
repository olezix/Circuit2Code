"""Tests of activation caching, patching, ablation and Fourier circuit discovery."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.interpretability.circuit_finder import (
    direct_logit_attribution,
    dft_spectrum,
    effective_logit_map,
    effective_qk,
    extract_circuit,
    find_active_heads,
    fourier_coefficients,
    head_ablation_table,
    qk_matrix,
    ov_matrix,
    reconstruct_from_coefficients,
    top_frequencies,
)
from src.interpretability.probing import (
    attribution_patch,
    build_activation_cache,
    metric_logit_diff,
    patch_head,
    patch_residual,
    zero_head,
)
from src.models.transformer import modular_addition_tokens


# ---------------------------------------------------------------------------
# Fourier toolbox
# ---------------------------------------------------------------------------
def test_fourier_decomposition_is_exact_for_arbitrary_vectors():
    rng = np.random.default_rng(0)
    for p in (7, 11, 12):
        values = rng.normal(size=p)
        coeffs = fourier_coefficients(values, p)
        assert coeffs.shape == (p // 2 + 1, 2)
        reconstructed = reconstruct_from_coefficients(coeffs, p)
        assert np.allclose(reconstructed, values, atol=1e-12)


def test_fourier_recovers_planted_frequency():
    p = 59
    x = np.arange(p)
    planted = 0.7 * np.cos(2 * math.pi * 5 * x / p) - 0.3 * np.sin(2 * math.pi * 9 * x / p)
    coeffs = fourier_coefficients(planted, p)
    assert abs(coeffs[5, 0] - 0.7) < 1e-12
    assert abs(coeffs[9, 1] + 0.3) < 1e-12
    assert top_frequencies(planted, p, k=2)[0][0] in (5, 9)
    assert abs(sum(e for _, e in top_frequencies(planted, p, k=2)) - 1.0) < 1e-9


def test_top_frequencies_handles_constant_signal():
    assert top_frequencies(np.full(11, 3.0), 11) == []


def test_dft_spectrum_peaks_at_planted_frequency():
    p = 19
    j = np.arange(p)[:, None]
    c = np.arange(p)[None, :]
    matrix = np.cos(2 * math.pi * 4 * (j - c) / p)
    spectrum = dft_spectrum(matrix, p)
    peak = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    assert peak[0] % p == 4 and peak[1] % p == p - 4
    with pytest.raises(ValueError):
        dft_spectrum(np.zeros((p, p + 1)), p)


# ---------------------------------------------------------------------------
# Effective circuits
# ---------------------------------------------------------------------------
def test_effective_matrices_match_manual_computation(small_model):
    w_e = small_model.W_E.detach().numpy()
    w_q = small_model.W_Q[0][:, 0, :].detach().numpy()
    w_k = small_model.W_K[0][:, 0, :].detach().numpy()
    w_v = small_model.W_V[0][:, 0, :].detach().numpy()
    w_o = small_model.W_O[0][:, 0, :].detach().numpy()
    w_u = small_model.W_U.detach().numpy()
    scale = 1 / math.sqrt(small_model.config.d_head)
    expected_qk = scale * (w_e @ w_q @ w_k.T @ w_e.T)
    expected_ov = w_e @ (w_v @ w_o.T) @ w_u
    assert np.allclose(qk_matrix(small_model, 0, 0), scale * (w_q @ w_k.T))
    assert np.allclose(ov_matrix(small_model, 0, 0), w_v @ w_o.T)
    assert np.allclose(effective_qk(small_model, 0, 0), expected_qk)
    assert np.allclose(effective_logit_map(small_model, 0, 0), expected_ov)


def test_direct_logit_attribution_is_additive(small_model, small_dataset):
    tokens = small_dataset["tokens"][:5]
    attribution = direct_logit_attribution(small_model, tokens)
    with torch.no_grad():
        logits, _ = small_model(tokens)
    head_total = attribution["head_total"].sum(dim=(0, 2))  # (B, p)
    bias = small_model.W_E[small_model.config.eq_token] @ small_model.W_U
    recomposed = bias.unsqueeze(0) + head_total
    assert torch.allclose(recomposed, logits[:, -1, :], atol=1e-5)


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------
def test_activation_cache_has_all_entries(small_model):
    tokens = modular_addition_tokens(torch.arange(7), torch.arange(7), 7)
    cache = build_activation_cache(small_model, tokens)
    cfg = small_model.config
    assert cache.residual(1, 2).shape == (7, cfg.d_model)
    assert cache.attention(0, 1, 2).shape == (7, cfg.seq_len)
    assert cache["logits.2"].shape == (7, cfg.p)
    assert any(key.startswith("head.0.1.") for key in cache.keys())


def test_patch_residual_with_identical_inputs_is_identity(small_model):
    tokens = modular_addition_tokens(torch.tensor([1, 2]), torch.tensor([3, 4]), 7)
    patched = patch_residual(small_model, tokens, tokens, layer=1, position=2)
    with torch.no_grad():
        expected = small_model(tokens)[0][:, -1, :]
    assert torch.allclose(patched, expected, atol=1e-6)


def test_patch_residual_recovers_clean_behaviour(small_model):
    clean = modular_addition_tokens(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), 7)
    corrupted = modular_addition_tokens(torch.tensor([6, 5, 4]), torch.tensor([3, 2, 1]), 7)
    with torch.no_grad():
        clean_logits = small_model(clean)[0][:, -1, :]
    patched = patch_residual(small_model, clean, corrupted, layer=1, position=2)
    assert torch.allclose(patched, clean_logits, atol=1e-5)


def test_patch_head_and_attribution_shapes(small_model):
    clean = modular_addition_tokens(torch.tensor([1, 2]), torch.tensor([4, 5]), 7)
    corrupted = modular_addition_tokens(torch.tensor([6, 5]), torch.tensor([3, 2]), 7)
    patched = patch_head(small_model, clean, corrupted, layer=0, head=0)
    assert patched.shape == (2, 7)
    attribution = attribution_patch(small_model, clean, corrupted, layer=0)
    assert attribution.shape == (2,)
    assert small_model.W_Q[0].grad is None  # no gradient leakage into parameters


def test_zero_head_restores_weights(small_model):
    before = small_model.W_O[0].detach().clone()
    with zero_head(small_model, 0, 0):
        assert torch.count_nonzero(small_model.W_O[0][:, 0, :]) == 0
        assert torch.equal(small_model.W_O[0][:, 1, :], before[:, 1, :])
    assert torch.equal(small_model.W_O[0], before)
    with pytest.raises(IndexError):
        with zero_head(small_model, 0, 99):
            pass


def test_metric_logit_diff_is_linear(small_model):
    logits = torch.zeros(3, 7)
    logits[:, 2] = 1.0
    target = torch.tensor([2, 2, 2])
    assert metric_logit_diff(logits, target).item() == pytest.approx(1.0)
    alt = torch.tensor([0, 0, 0])
    logits[:, 0] = 1.0
    assert metric_logit_diff(logits, target, alt).item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Circuit discovery
# ---------------------------------------------------------------------------
def test_head_ablation_table_is_complete(small_model, small_dataset):
    table = head_ablation_table(small_model, small_dataset, small_dataset["test_idx"])
    assert len(table) == small_model.config.n_heads * small_model.config.n_layers
    for row in table:
        assert 0.0 <= row["accuracy"] <= 1.0
        assert "accuracy_drop" in row and "logit_diff_drop" in row


def test_find_active_heads_falls_back_to_worst_head():
    table = [
        {"layer": 0.0, "head": 0.0, "accuracy_drop": 0.0},
        {"layer": 0.0, "head": 1.0, "accuracy_drop": 0.01},
    ]
    assert find_active_heads(table) == [(0, 1)]
    assert find_active_heads(table, threshold=0.0) == [(0, 0), (0, 1)]


def test_extract_circuit_reports_structure(medium_model, medium_dataset):
    spec = extract_circuit(medium_model, medium_dataset)
    assert spec.p == 11
    assert spec.heads, "at least one head must be reported"
    payload = spec.to_dict()
    assert payload["p"] == 11
    assert isinstance(payload["frequencies"], list)
    assert "Circuit(" in spec.summary()
    for report in spec.heads:
        assert report.spectral_energy_top3 >= 0.0
        assert report.row_frequency_profile.shape[0] >= 2
