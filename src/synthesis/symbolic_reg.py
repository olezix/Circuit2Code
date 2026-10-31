"""Symbolic basis extraction: Fourier, modular-shift and threshold bases.

The neural circuit is a composition of linear maps and (soft)max, but on the
*finite* ring :math:`\\mathbb{Z}_p` every function it computes has an alternative
sparse description in a well-chosen basis.  Three bases suffice for the tasks in
this repository:

Fourier basis
    ``1, cos(2πkx/p), sin(2πkx/p)`` for ``k = 1..p//2``.  This is the basis in
    which grokked modular-arithmetic networks are sparse (Nanda et al., 2023).

Modular-shift basis
    A matrix ``M[j, c]`` with ``M[j, c] = g((j - c) mod p)`` for a single vector
    ``g``.  This is the fingerprint of an algorithm that *transmits* a token
    value into a class value, e.g. an OV circuit that adds the attended token to
    the accumulated sum.

Threshold (ReLU) basis
    ``1, x, relu(x - t)`` at a fixed set of knots ``t``.  This is the basis of
    piecewise-linear MLP neurons and is what the SMT translation needs for MLPs
    to remain exact.

Everything is deterministic: no randomness, fixed least-squares solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.synthesis.ast_nodes import (
    BinOp,
    Const,
    Cos,
    Expr,
    Sin,
    Sum,
    Var,
)

__all__ = [
    "FourierFit",
    "ShiftFit",
    "PiecewiseLinearFit",
    "fourier_design_matrix",
    "fit_fourier",
    "fit_fourier_greedy",
    "fit_periodic_shift",
    "fit_piecewise_linear",
    "fourier_term_expr",
    "periodicity_score",
]


# ---------------------------------------------------------------------------
# Fourier basis
# ---------------------------------------------------------------------------
def fourier_design_matrix(p: int, freqs: Sequence[int]) -> np.ndarray:
    """Design matrix of the real Fourier basis on ``x = 0..p-1``.

    Columns are ``[1, cos(2πk₁x/p), sin(2πk₁x/p), cos(2πk₂x/p), ...]`` so the
    matrix has shape ``(p, 1 + 2 * len(freqs))``.
    """
    x = np.arange(p, dtype=np.float64)
    columns = [np.ones(p)]
    for k in freqs:
        if not 0 < int(k) <= p // 2:
            raise ValueError(f"frequency {k} outside 1..{p // 2}")
        angle = 2.0 * math.pi * int(k) * x / p
        columns.append(np.cos(angle))
        columns.append(np.sin(angle))
    return np.stack(columns, axis=1)


@dataclass
class FourierFit:
    """A real Fourier series fitted on :math:`\\mathbb{Z}_p`."""

    p: int
    freqs: Tuple[int, ...]
    a0: float
    cos_coeffs: np.ndarray
    sin_coeffs: np.ndarray
    residual_max: float = 0.0
    residual_rms: float = 0.0

    @property
    def n_terms(self) -> int:
        return len(self.freqs)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.full_like(x, self.a0, dtype=np.float64)
        for k, a, b in zip(self.freqs, self.cos_coeffs, self.sin_coeffs):
            angle = 2.0 * math.pi * k * x / self.p
            out = out + a * np.cos(angle) + b * np.sin(angle)
        return out

    def to_expr(self, var: str = "x") -> Expr:
        """Convert the fitted series into an IR expression."""
        terms: List[Expr] = [Const(self.a0)]
        for k, a, b in zip(self.freqs, self.cos_coeffs, self.sin_coeffs):
            omega = Const(2.0 * math.pi * k / self.p)
            angle = omega * Var(var)
            if abs(a) > 0:
                terms.append(Const(float(a)) * Cos(angle))
            if abs(b) > 0:
                terms.append(Const(float(b)) * Sin(angle))
        return Sum(terms) if len(terms) > 1 else terms[0]

    def to_dict(self) -> dict:
        return {
            "p": self.p,
            "freqs": list(self.freqs),
            "a0": self.a0,
            "cos": self.cos_coeffs.tolist(),
            "sin": self.sin_coeffs.tolist(),
            "residual_max": self.residual_max,
            "residual_rms": self.residual_rms,
        }


def fit_fourier(values: np.ndarray, p: int, freqs: Sequence[int]) -> FourierFit:
    """Least-squares fit of ``values`` (length ``p``) in the given Fourier basis.

    For a *complete* frequency set the fit is exact to machine precision; for a
    truncated set ``residual_max`` reports the worst-case reconstruction error.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] != p:
        raise ValueError(f"expected {p} samples, got {values.shape[0]}")
    freqs = tuple(int(k) for k in freqs)
    design = fourier_design_matrix(p, freqs)
    coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
    a0 = float(coeffs[0])
    cos_c = np.array(coeffs[1::2], dtype=np.float64)
    sin_c = np.array(coeffs[2::2], dtype=np.float64)
    approx = design @ coeffs
    err = np.abs(approx - values)
    return FourierFit(
        p=p,
        freqs=freqs,
        a0=a0,
        cos_coeffs=cos_c,
        sin_coeffs=sin_c,
        residual_max=float(err.max()) if err.size else 0.0,
        residual_rms=float(np.sqrt(np.mean(err**2))) if err.size else 0.0,
    )


def fit_fourier_greedy(
    values: np.ndarray,
    p: int,
    *,
    max_terms: int = 8,
    tol: float = 1e-9,
    candidate_freqs: Optional[Iterable[int]] = None,
) -> FourierFit:
    """Greedy forward selection of Fourier frequencies (``O(max_terms * p^2)``).

    At each step the *spectral energy* of the current residual picks the next
    candidate.  The choice is driven by L2 energy rather than by the L-inf
    residual: a least-squares refit can leave the maximum residual unchanged
    (or momentarily worse), which would stall a max-norm greedy loop even on
    functions that the full basis represents exactly.  The procedure stops when
    the residual falls below ``tol`` or ``max_terms`` frequencies are selected,
    which is the sparsity pattern the synthesizer reports as a closed form.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    candidates = list(candidate_freqs) if candidate_freqs is not None else list(
        range(1, p // 2 + 1)
    )
    chosen: List[int] = []
    best = fit_fourier(values, p, chosen)
    grid = np.arange(p, dtype=np.float64)
    for _ in range(int(max_terms)):
        if best.residual_max <= tol:
            break
        residual = values - best.evaluate(grid)
        energy = np.abs(np.fft.rfft(residual)) ** 2
        available = [k for k in candidates if k not in chosen]
        # Largest residual energy first; ties broken by the lower frequency so
        # that the procedure is deterministic.
        available.sort(key=lambda k: (-float(energy[k]) if k < energy.size else 0.0, k))
        # Accept the highest-energy candidate.  Selection is driven by the L2
        # residual spectrum -- a least-squares fit is only L2-optimal, so
        # demanding that the L-inf residual improve at *every* step would stall
        # the loop on perfectly representable functions.
        if not available:
            break
        chosen.append(available[0])
        best = fit_fourier(values, p, chosen)
    return best


# ---------------------------------------------------------------------------
# Modular-shift basis
# ---------------------------------------------------------------------------
@dataclass
class ShiftFit:
    """Result of testing ``M[j, c] == g((j - c) mod p)`` for one vector ``g``."""

    p: int
    values: np.ndarray
    residual_max: float

    @property
    def is_shift_invariant(self) -> bool:
        return self.residual_max <= 1e-9 * max(1.0, float(np.abs(self.values).max()))

    def to_dict(self) -> dict:
        return {
            "p": self.p,
            "residual_max": float(self.residual_max),
            "shift_invariant": bool(self.is_shift_invariant),
            "values": [float(v) for v in self.values],
        }


def fit_periodic_shift(matrix: np.ndarray, p: int) -> ShiftFit:
    """Test whether a ``(p, p)`` matrix only depends on ``(j - c) mod p``.

    The best such ``g`` is the circular mean over the diagonals, and the residual
    is the worst-case deviation from it.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (p, p):
        raise ValueError(f"expected a ({p}, {p}) matrix, got {matrix.shape}")
    diagonals = np.zeros((p, p))
    counts = np.zeros(p)
    for j in range(p):
        for c in range(p):
            d = (j - c) % p
            diagonals[d, j] = matrix[j, c]
            counts[d] += 1
    g = np.array(
        [diagonals[d, : int(counts[d])].mean() for d in range(p)], dtype=np.float64
    )
    residual = 0.0
    for j in range(p):
        for c in range(p):
            residual = max(residual, abs(matrix[j, c] - g[(j - c) % p]))
    return ShiftFit(p=p, values=g, residual_max=float(residual))


def periodicity_score(matrix: np.ndarray, p: int) -> float:
    """Fraction of the Frobenius energy explained by the best shift model."""
    matrix = np.asarray(matrix, dtype=np.float64)
    fit = fit_periodic_shift(matrix, p)
    model = np.array([[fit.values[(j - c) % p] for c in range(p)] for j in range(p)])
    denom = float(np.sum(matrix**2))
    if denom <= 0:
        return 0.0
    return 1.0 - float(np.sum((matrix - model) ** 2)) / denom


# ---------------------------------------------------------------------------
# Piecewise-linear (ReLU) basis
# ---------------------------------------------------------------------------
@dataclass
class PiecewiseLinearFit:
    """``f(x) = b + w x + Σ_i v_i relu(x - t_i)``."""

    knots: Tuple[float, ...]
    bias: float
    slope: float
    weights: np.ndarray
    residual_max: float = 0.0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = self.bias + self.slope * x
        for t, v in zip(self.knots, self.weights):
            out = out + v * np.maximum(x - t, 0.0)
        return out

    def to_expr(self, var: str = "x") -> Expr:
        from src.synthesis.ast_nodes import Threshold  # local import: avoid cycle

        terms: List[Expr] = [Const(self.bias), Const(self.slope) * Var(var)]
        for t, v in zip(self.knots, self.weights):
            # relu(z) = z * step(z)
            z = Var(var) - Const(t)
            terms.append(Const(float(v)) * (z * Threshold(z, 0.0)))
        return Sum(terms)


def fit_piecewise_linear(
    x: np.ndarray,
    values: np.ndarray,
    knots: Sequence[float],
) -> PiecewiseLinearFit:
    """Least-squares fit of a ReLU network with one hidden layer (fixed knots)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.shape != values.shape:
        raise ValueError("x and values must have the same shape")
    knots = tuple(float(t) for t in knots)
    columns = [np.ones_like(x), x]
    for t in knots:
        columns.append(np.maximum(x - t, 0.0))
    design = np.stack(columns, axis=1)
    coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
    approx = design @ coeffs
    return PiecewiseLinearFit(
        knots=knots,
        bias=float(coeffs[0]),
        slope=float(coeffs[1]),
        weights=np.array(coeffs[2:], dtype=np.float64),
        residual_max=float(np.abs(approx - values).max()),
    )


# ---------------------------------------------------------------------------
# IR helpers
# ---------------------------------------------------------------------------
def fourier_term_expr(k: int, p: int, var: str = "x") -> Tuple[Expr, Expr]:
    """The pair ``(cos(2πkx/p), sin(2πkx/p))`` as IR expressions."""
    if not 0 < int(k) <= p // 2:
        raise ValueError(f"frequency {k} outside 1..{p // 2}")
    omega = Const(2.0 * math.pi * int(k) / p)
    angle = omega * Var(var)
    return Cos(angle), Sin(angle)
