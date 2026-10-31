"""Formal verification of decompiled circuits with Z3."""

from __future__ import annotations

from src.verification.verifier import (
    ModuloAdditionSpec,
    ProofResult,
    VerificationReport,
    VerifyConfig,
    Verifier,
    synthesize_and_verify,
)
from src.verification.z3_translator import (
    IntervalArray,
    Z3Translator,
    compute_logit_bounds,
    translate_program,
)

__all__ = [
    "IntervalArray",
    "Z3Translator",
    "compute_logit_bounds",
    "translate_program",
    "ModuloAdditionSpec",
    "ProofResult",
    "VerificationReport",
    "VerifyConfig",
    "Verifier",
    "synthesize_and_verify",
]
