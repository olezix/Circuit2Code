"""Mechanistic interpretability: activation caching, patching, circuit discovery."""

from __future__ import annotations

from src.interpretability.circuit_finder import (
    CircuitSpec,
    HeadReport,
    dft_spectrum,
    extract_circuit,
    find_active_heads,
    head_ablation_table,
    top_frequencies,
)
from src.interpretability.probing import (
    ActivationCache,
    add_head,
    metric_logit_diff,
    patch_head,
    patch_residual,
    zero_head,
)

__all__ = [
    "ActivationCache",
    "add_head",
    "metric_logit_diff",
    "patch_head",
    "patch_residual",
    "zero_head",
    "CircuitSpec",
    "HeadReport",
    "dft_spectrum",
    "extract_circuit",
    "find_active_heads",
    "head_ablation_table",
    "top_frequencies",
]
