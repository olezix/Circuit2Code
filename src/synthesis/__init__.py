"""Neuro-symbolic synthesis: IR, basis extraction and the synthesizer."""

from __future__ import annotations

from src.synthesis.ast_nodes import (
    ArgMax,
    BinOp,
    Const,
    Cos,
    Expr,
    IfElse,
    Linear,
    Mod,
    Neg,
    Not,
    Program,
    Sin,
    Softmax,
    Sum,
    Threshold,
    Var,
    simplify,
    walk,
)
from src.synthesis.program_synthesizer import (
    SynthesisResult,
    build_fourier_circuit,
    model_predictions,
    synthesize,
)
from src.synthesis.symbolic_reg import (
    FourierFit,
    fit_fourier,
    fit_fourier_greedy,
    fit_periodic_shift,
)

__all__ = [
    "ArgMax",
    "BinOp",
    "Const",
    "Cos",
    "Expr",
    "IfElse",
    "Linear",
    "Mod",
    "Neg",
    "Not",
    "Program",
    "Sin",
    "Softmax",
    "Sum",
    "Threshold",
    "Var",
    "simplify",
    "walk",
    "SynthesisResult",
    "build_fourier_circuit",
    "model_predictions",
    "synthesize",
    "FourierFit",
    "fit_fourier",
    "fit_fourier_greedy",
    "fit_periodic_shift",
]
