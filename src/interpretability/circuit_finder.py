"""Automated circuit discovery for modular-arithmetic Transformers.

The module answers three questions, in this order:

1. **Which heads matter?**  Zero-ablation of each head's output projection and
   direct logit attribution (DLA) rank the heads by their causal contribution to
   the correct-answer logit.
2. **What do the heads compute?**  For every head we form the two effective
   circuits of Elhage et al. (2021) -- the bilinear form read directly off the
   weights,

   .. math:: W_{QK} = W_Q W_K^\\top, \\qquad W_{OV} = W_V W_O^\\top

   and their token-space counterparts
   :math:`W_E W_{QK} W_E^\\top` (attention scores between *tokens*) and
   :math:`W_E W_{OV} W_U` (the map from an attended token to logits).
3. **In which basis are they sparse?**  A discrete Fourier transform of those
   token-space matrices.  A grokked network concentrates almost all of its
   spectral energy on a handful of frequencies :math:`k`, which is the
   fingerprint of the trigonometric algorithm, and those frequencies are what
   the synthesizer consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from src.interpretability.probing import metric_logit_diff, zero_head
from src.models.transformer import Transformer

__all__ = [
    "fourier_coefficients",
    "reconstruct_from_coefficients",
    "top_frequencies",
    "dft_spectrum",
    "qk_matrix",
    "ov_matrix",
    "effective_qk",
    "effective_logit_map",
    "direct_logit_attribution",
    "head_ablation_table",
    "find_active_heads",
    "HeadReport",
    "CircuitSpec",
    "extract_circuit",
]


# ---------------------------------------------------------------------------
# Discrete Fourier toolbox (exact interpolation on Z_p)
# ---------------------------------------------------------------------------
def fourier_coefficients(values: np.ndarray, p: int) -> np.ndarray:
    """Exact real Fourier coefficients of a vector sampled on :math:`\\mathbb{Z}_p`.

    Returns an array ``coeffs`` of shape ``(p // 2 + 1, 2)`` such that

    .. math::
        f(x) = a_0 + \\sum_{k=1}^{\\lfloor p/2 \\rfloor}
               \\bigl(a_k \\cos(2\\pi k x/p) + b_k \\sin(2\\pi k x/p)\\bigr)

    with ``coeffs[k] = (a_k, b_k)``.  The representation is exact (up to floating
    point round-off) for every :math:`x \\in \\{0, \\dots, p-1\\}` because the
    :math:`p` real basis functions span :math:`\\mathbb{R}^p`.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] != p:
        raise ValueError(f"expected {p} samples, got {values.shape[0]}")
    spec = np.fft.rfft(values) / p
    n_freq = p // 2 + 1
    coeffs = np.zeros((n_freq, 2), dtype=np.float64)
    coeffs[0, 0] = spec[0].real
    for k in range(1, n_freq):
        a, b = 2.0 * spec[k].real, -2.0 * spec[k].imag
        if p % 2 == 0 and k == p // 2:
            a, b = a / 2.0, b / 2.0
        coeffs[k] = (a, b)
    return coeffs


def reconstruct_from_coefficients(coeffs: np.ndarray, p: int) -> np.ndarray:
    """Inverse of :func:`fourier_coefficients` on the grid ``0..p-1``."""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    x = np.arange(p)
    out = np.full(p, coeffs[0, 0], dtype=np.float64)
    for k in range(1, coeffs.shape[0]):
        angle = 2.0 * math.pi * k * x / p
        out += coeffs[k, 0] * np.cos(angle) + coeffs[k, 1] * np.sin(angle)
    return out


def top_frequencies(
    values: np.ndarray,
    p: int,
    k: int = 3,
    *,
    include_dc: bool = False,
) -> List[Tuple[int, float]]:
    """Return the ``k`` dominant frequencies of ``values`` with their energy.

    Energy of frequency ``j`` is :math:`a_j^2 + b_j^2`; the list is sorted by
    decreasing energy and normalised so that energies sum to the energy of the
    non-DC part of the signal.
    """
    coeffs = fourier_coefficients(values, p)
    energy = (coeffs[:, 0] ** 2 + coeffs[:, 1] ** 2).astype(np.float64)
    start = 0 if include_dc else 1
    energy = energy[start:]
    total = float(energy.sum())
    # Reject signals whose non-constant part is at round-off level relative to
    # the mean (a pure constant has no meaningful spectrum).
    scale = max(1.0, float(coeffs[0, 0]) ** 2)
    if total <= 1e-24 * scale:
        return []
    idx = np.argsort(energy)[::-1][:k]
    return [(int(i + start), float(energy[i] / total)) for i in idx]


def dft_spectrum(matrix: np.ndarray, p: int) -> np.ndarray:
    """Normalised 2-D power spectrum of a ``(p, p)`` matrix over :math:`\\mathbb{Z}_p^2`.

    ``spectrum[k1, k2]`` is :math:`|\\hat M(k_1, k_2)|^2 / p^4`, where
    :math:`\\hat M` is the unnormalised 2-D DFT.  Rows and columns are assumed
    to be indexed by :math:`\\mathbb{Z}_p`.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (p, p):
        raise ValueError(f"expected a ({p}, {p}) matrix, got {matrix.shape}")
    return np.abs(np.fft.fft2(matrix)) ** 2 / float(p) ** 4


# ---------------------------------------------------------------------------
# Effective circuits
# ---------------------------------------------------------------------------
def _head_slices(model: Transformer, layer: int, head: int) -> Dict[str, np.ndarray]:
    return {
        "Q": model.W_Q[layer][:, head, :].detach().cpu().numpy(),
        "K": model.W_K[layer][:, head, :].detach().cpu().numpy(),
        "V": model.W_V[layer][:, head, :].detach().cpu().numpy(),
        "O": model.W_O[layer][:, head, :].detach().cpu().numpy(),
    }


def qk_matrix(model: Transformer, layer: int, head: int) -> np.ndarray:
    """``W_Q W_K^T`` for one head, including the attention scale factor."""
    w = _head_slices(model, layer, head)
    scale = 1.0 / math.sqrt(model.config.d_head) if model.config.attn_scale == "sqrt" else 1.0
    return scale * (w["Q"] @ w["K"].T)


def ov_matrix(model: Transformer, layer: int, head: int) -> np.ndarray:
    """``W_V W_O^T`` for one head (residual space to residual space)."""
    w = _head_slices(model, layer, head)
    return w["V"] @ w["O"].T


def effective_qk(model: Transformer, layer: int, head: int) -> np.ndarray:
    """Token-space attention bilinear form ``W_E W_QK W_E^T`` (vocab x vocab)."""
    we = model.W_E.detach().cpu().numpy()
    return we @ qk_matrix(model, layer, head) @ we.T


def effective_logit_map(model: Transformer, layer: int, head: int) -> np.ndarray:
    """Token-to-logit map ``W_E W_OV W_U`` of shape ``(vocab, n_classes)``.

    ``M[v, c]`` is the contribution of a unit attention weight on source token
    ``v`` to the logit of class ``c``.
    """
    we = model.W_E.detach().cpu().numpy()
    wu = model.W_U.detach().cpu().numpy()
    return we @ ov_matrix(model, layer, head) @ wu


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
@torch.no_grad()
def direct_logit_attribution(
    model: Transformer,
    tokens: Tensor,
    *,
    layer: Optional[int] = None,
) -> Dict[str, Tensor]:
    """Per-head, per-source-token direct logit attribution at the answer position.

    Because the architecture is bias-free and the residual stream is linear, the
    answer logits decompose *exactly* as

    .. math:: \\ell_c = (E_{eq} W_U)_c + \\sum_h \\sum_j P_h(\\text{ans}, j)\\,
              (E_j W_{Vh} W_{Oh} W_U)_c

    Returns a dict with

    ``head_total``  (L, B, H, p)    contribution of each head summed over sources
    ``token``       (L, B, T, H, p) contribution of each (head, source token)
    """
    cfg = model.config
    logits, trace = model(tokens, trace=True)
    assert trace is not None
    layers = range(cfg.n_layers) if layer is None else [layer]
    answer = cfg.answer_position
    head_total = []
    token_level = []
    for lay in layers:
        w_v = model.W_V[lay]  # (d, H, dh)
        w_o = model.W_O[lay]  # (d, H, dh)
        w_u = model.W_U        # (d, p)
        # ``per_head_maps[v, h, c]`` is the contribution of a unit attention
        # weight on source token v to the logit of class c, through head h:
        #   W_E (V,d) -> W_V (d,H,dh) -> W_O (d,H,dh) -> W_U (d,p).
        embed = model.W_E.detach()  # (V, d)
        z = torch.einsum("vd,dhk->vhk", embed, w_v)        # (V,H,dh)
        y = torch.einsum("vhk,dhk->vhd", z, w_o)           # (V,H,d)
        per_head_maps = torch.einsum("vhd,dc->vhc", y, w_u)  # (V,H,p)
        pattern = trace.attention_patterns[lay][:, :, answer, :]  # (B,H,T)
        contrib = torch.einsum("bht,vhc->bthc", pattern, per_head_maps)
        hs = trace.head_outputs[lay][:, answer, :, :] @ model.W_U  # (B,H,p)
        head_total.append(hs.detach())
        token_level.append(contrib[:, :, :, :].detach())
    return {
        "head_total": torch.stack(head_total) if head_total else torch.empty(0),
        "token": torch.stack(token_level) if token_level else torch.empty(0),
        "logits": logits.detach(),
    }


@torch.no_grad()
def head_ablation_table(
    model: Transformer,
    dataset: Dict[str, Tensor],
    index: torch.Tensor,
) -> List[Dict[str, float]]:
    """Zero-ablate every head and report accuracy / logit-difference damage."""
    cfg = model.config
    tokens = dataset["tokens"][index]
    target = dataset["target"][index]
    model.eval()

    clean_logits = model(tokens)[0][:, cfg.answer_position, :]
    clean_acc = float((clean_logits.argmax(-1) == target).float().mean())
    clean_metric = float(metric_logit_diff(clean_logits, target))

    rows: List[Dict[str, float]] = []
    for layer in range(cfg.n_layers):
        for head in range(cfg.n_heads):
            with zero_head(model, layer, head):
                logits = model(tokens)[0][:, cfg.answer_position, :]
            acc = float((logits.argmax(-1) == target).float().mean())
            rows.append(
                {
                    "layer": float(layer),
                    "head": float(head),
                    "accuracy": acc,
                    "accuracy_drop": clean_acc - acc,
                    "logit_diff": float(metric_logit_diff(logits, target)),
                    "logit_diff_drop": clean_metric - float(metric_logit_diff(logits, target)),
                }
            )
    return rows


def find_active_heads(
    table: Sequence[Dict[str, float]],
    *,
    threshold: float = 0.05,
) -> List[Tuple[int, int]]:
    """Heads whose zero-ablation costs at least ``threshold`` accuracy."""
    heads = [
        (int(row["layer"]), int(row["head"]))
        for row in table
        if row["accuracy_drop"] >= threshold
    ]
    # If nothing crosses the threshold (e.g. a tiny random model), fall back to
    # the most damaging head so that downstream stages always have a circuit.
    if not heads and table:
        worst = max(table, key=lambda r: r["accuracy_drop"])
        heads = [(int(worst["layer"]), int(worst["head"]))]
    return sorted(heads)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
@dataclass
class HeadReport:
    """Everything the synthesizer needs to know about a single head."""

    layer: int
    head: int
    accuracy_drop: float
    logit_diff_drop: float
    top_frequencies: List[Tuple[int, float]]
    spectral_energy_top3: float
    row_frequency_profile: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(1))

    def to_dict(self) -> Dict[str, object]:
        return {
            "layer": self.layer,
            "head": self.head,
            "accuracy_drop": round(self.accuracy_drop, 6),
            "logit_diff_drop": round(self.logit_diff_drop, 6),
            "top_frequencies": [(int(k), round(e, 6)) for k, e in self.top_frequencies],
            "spectral_energy_top3": round(self.spectral_energy_top3, 6),
        }


@dataclass
class CircuitSpec:
    """The extracted circuit: active heads, their spectra and effective maps."""

    p: int
    heads: List[HeadReport]
    frequencies: List[int]
    logit_maps: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict, repr=False)
    qk_maps: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict, repr=False)
    is_periodic: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "p": self.p,
            "active_heads": [h.to_dict() for h in self.heads],
            "frequencies": list(self.frequencies),
            "is_periodic": self.is_periodic,
            "n_heads": len(self.heads),
        }

    def summary(self) -> str:
        head_list = ", ".join(f"L{h.layer}H{h.head}" for h in self.heads) or "<none>"
        freqs = ", ".join(str(k) for k in self.frequencies) or "<none>"
        return (
            f"Circuit(p={self.p}, heads=[{head_list}], "
            f"frequencies=[{freqs}], periodic={self.is_periodic})"
        )


def _matrix_frequency_profile(matrix: np.ndarray, p: int) -> np.ndarray:
    """Energy of each frequency in the *row* index, summed over columns."""
    matrix = np.asarray(matrix, dtype=np.float64)
    spec = np.fft.rfft(matrix, axis=0) / p
    energy = (np.abs(spec) ** 2).sum(axis=1)
    total = float(energy[1:].sum()) if energy.shape[0] > 1 else 0.0
    if total <= 0.0:
        return np.zeros_like(energy)
    energy = energy.copy()
    energy[0] = 0.0
    return energy / total


def extract_circuit(
    model: Transformer,
    dataset: Dict[str, Tensor],
    index: Optional[torch.Tensor] = None,
    *,
    top_k: int = 3,
    ablation_threshold: float = 0.05,
    periodicity_threshold: float = 0.90,
) -> CircuitSpec:
    """End-to-end circuit extraction (Phase 2 of the pipeline).

    Parameters
    ----------
    model:
        A trained :class:`~src.models.transformer.Transformer`.
    dataset:
        Output of :func:`~src.models.transformer.modular_addition_dataset`.
    index:
        Examples used for the ablation study.  Defaults to the *full* dataset,
        which measures the circuit on the whole domain rather than only on the
        training split.
    """
    p = model.config.p
    if index is None:
        index = torch.arange(dataset["tokens"].shape[0])
    table = head_ablation_table(model, dataset, index)
    active = find_active_heads(table, threshold=ablation_threshold)
    table_by_head = {(int(r["layer"]), int(r["head"])): r for r in table}

    heads: List[HeadReport] = []
    logit_maps: Dict[Tuple[int, int], np.ndarray] = {}
    qk_maps: Dict[Tuple[int, int], np.ndarray] = {}
    for (layer, head) in active:
        lm = effective_logit_map(model, layer, head)
        qk = effective_qk(model, layer, head)
        profile = _matrix_frequency_profile(lm[:p, :], p)
        # frequency energy of the logit map, combined over classes
        coeffs = np.fft.rfft(lm[:p, :], axis=0) / p
        energy = (np.abs(coeffs) ** 2).sum(axis=1)
        energy[0] = 0.0
        total = float(energy.sum())
        top = []
        if total > 0:
            order = np.argsort(energy)[::-1][:top_k]
            top = [(int(k), float(energy[k] / total)) for k in order]
        heads.append(
            HeadReport(
                layer=layer,
                head=head,
                accuracy_drop=float(table_by_head[(layer, head)]["accuracy_drop"]),
                logit_diff_drop=float(table_by_head[(layer, head)]["logit_diff_drop"]),
                top_frequencies=top,
                spectral_energy_top3=float(sum(e for _, e in top)),
                row_frequency_profile=profile,
            )
        )
        logit_maps[(layer, head)] = lm
        qk_maps[(layer, head)] = qk

    # Aggregate frequency support of the circuit.
    agg = np.zeros(p // 2 + 1)
    for report in heads:
        for k, e in report.top_frequencies:
            if k < agg.shape[0]:
                agg[k] += e
    frequencies = sorted({k for k, e in sorted(enumerate(agg), key=lambda t: -t[1])[:top_k] if e > 0})
    is_periodic = bool(heads) and all(h.spectral_energy_top3 >= periodicity_threshold for h in heads)
    return CircuitSpec(
        p=p,
        heads=heads,
        frequencies=frequencies,
        logit_maps=logit_maps,
        qk_maps=qk_maps,
        is_periodic=is_periodic,
    )
