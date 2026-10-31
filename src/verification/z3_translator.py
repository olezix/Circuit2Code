"""Translation of PyTorch layers and IR programs into Z3 constraints.

Two very different kinds of object are translated here, and the distinction
matters for what a proof means:

**Discrete IR programs** (:func:`translate_program`) are translated *exactly*
into quantifier-free integer arithmetic (``Int`` + Z3's Euclidean ``Mod``).
Equivalence and specification proofs about these programs are exact, with no
floating point anywhere.

**The neural network** is translated into exact rational linear arithmetic
(``Real``) with floating point entering only through *sound enclosures*.  The
attention probabilities are the only non-algebraic ingredients of the forward
pass; each of them becomes a ``Real`` variable constrained to an interval that
provably contains the value produced by the IEEE-754 forward pass.  The interval
comes from :func:`compute_logit_bounds`, which propagates rigorous rounding
error bounds (Higham's :math:`\\gamma_n` model) through every matmul, softmax and
residual addition.  Consequently ``unsat`` from a verification query means:

    *for every input in the domain and every attention assignment inside the
    certified enclosure, the network's argmax is the program's output.*

No floating point comparison ever decides the verdict: the enclosure endpoints
are converted to exact rationals (:class:`fractions.Fraction`) before they reach
Z3, and Z3 reasons about them symbolically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import z3

from src.models.transformer import Transformer
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
    Sin,
    Softmax,
    Sum,
    Threshold,
    Var,
)

__all__ = [
    "IntervalArray",
    "Z3Translator",
    "compute_logit_bounds",
    "compute_attention_values",
    "translate_program",
    "rational_ref",
    "gamma",
]

ULP = 2.0**-53


def gamma(n: int) -> float:
    """Higham's :math:`\\gamma_n = nu/(1-nu)`: relative error of an ``n``-term dot product."""
    if n <= 0:
        return 0.0
    nu = n * ULP
    return nu / (1.0 - nu)


def _outward(x: np.ndarray, ulps: int = 2) -> np.ndarray:
    """Widen an array outward by ``ulps`` units in the last place."""
    out = np.asarray(x, dtype=np.float64)
    for _ in range(ulps):
        out = np.nextafter(out, -np.inf)
    return out


def _outward_up(x: np.ndarray, ulps: int = 2) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64)
    for _ in range(ulps):
        out = np.nextafter(out, np.inf)
    return out


def rational_ref(value: float) -> z3.ArithRef:
    """Exact Z3 rational for the IEEE-754 double ``value``."""
    frac = Fraction.from_float(float(value))
    return z3.RealVal(z3.Q(frac.numerator, frac.denominator))


# ---------------------------------------------------------------------------
# Rigorous interval arrays
# ---------------------------------------------------------------------------
@dataclass
class IntervalArray:
    """Tensor of closed intervals with element-wise ``lo``/``hi`` bounds."""

    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self) -> None:
        if self.lo.shape != self.hi.shape:
            raise ValueError("lo and hi must have the same shape")
        if np.any(self.lo > self.hi):
            raise ValueError("some interval has lo > hi")

    @staticmethod
    def exact(values: np.ndarray) -> "IntervalArray":
        values = np.asarray(values, dtype=np.float64)
        return IntervalArray(values, values.copy())

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.lo.shape

    def width(self) -> np.ndarray:
        return self.hi - self.lo

    def __add__(self, other: "IntervalArray") -> "IntervalArray":
        return IntervalArray(_outward(self.lo + other.lo), _outward_up(self.hi + other.hi))


def _matmul_interval(x: IntervalArray, weight: np.ndarray) -> IntervalArray:
    """``x @ W`` for an interval tensor ``x`` (..., n) and an exact matrix ``W`` (n, m)."""
    w = np.asarray(weight, dtype=np.float64)
    n = w.shape[0]
    wp = np.maximum(w, 0.0)
    wn = np.minimum(w, 0.0)
    lo = x.lo @ wp + x.hi @ wn
    hi = x.hi @ wp + x.lo @ wn
    mag = np.abs(x.lo) @ np.abs(w) + np.abs(x.hi) @ np.abs(w)
    err = gamma(n) * mag
    lo = _outward(lo - err)
    hi = _outward_up(hi + err)
    return IntervalArray(lo, hi)


def _softmax_interval(scores: IntervalArray, axis: int = -1) -> IntervalArray:
    """Sound softmax enclosure, using the shift-invariance of softmax."""
    lo, hi = scores.lo, scores.hi
    shift = np.max(np.where(np.isfinite(hi), hi, -700.0), axis=axis, keepdims=True)
    # exp is monotone; pad by 2 ulps in each direction.
    e_lo = _outward(np.exp(np.clip(lo - shift, -700, 700)))
    e_hi = _outward_up(np.exp(np.clip(hi - shift, -700, 700)))
    sum_lo = np.sum(e_lo, axis=axis, keepdims=True) * (1.0 - 4 * ULP)
    sum_hi = np.sum(e_hi, axis=axis, keepdims=True) * (1.0 + 4 * ULP)
    # p_j = e_j / sum(e): the smallest value divides the smallest numerator by
    # the largest denominator, and vice versa.  No correlation tightening is
    # applied, so that the enclosure is sound by construction alone.
    p_lo = _outward(e_lo / sum_hi)
    p_hi = _outward_up(e_hi / sum_lo)
    p_lo = np.clip(p_lo, 0.0, 1.0)
    p_hi = np.clip(p_hi, 0.0, 1.0)
    p_lo = np.minimum(p_lo, p_hi)
    return IntervalArray(p_lo, p_hi)


# ---------------------------------------------------------------------------
# Network enclosure
# ---------------------------------------------------------------------------
def _model_weights(model: Transformer) -> Dict[str, np.ndarray]:
    """All parameters as float64 NumPy arrays (the exact quantities Z3 sees)."""
    out = {
        "W_E": model.W_E.detach().double().cpu().numpy(),
        "W_U": model.W_U.detach().double().cpu().numpy(),
    }
    for layer in range(model.config.n_layers):
        out[f"W_Q{layer}"] = model.W_Q[layer].detach().double().cpu().numpy()
        out[f"W_K{layer}"] = model.W_K[layer].detach().double().cpu().numpy()
        out[f"W_V{layer}"] = model.W_V[layer].detach().double().cpu().numpy()
        out[f"W_O{layer}"] = model.W_O[layer].detach().double().cpu().numpy()
        if model.config.mlp_width > 0:
            out[f"W_in{layer}"] = model.W_in[layer].detach().double().cpu().numpy()
            out[f"W_out{layer}"] = model.W_out[layer].detach().double().cpu().numpy()
    return out


def _flatten_heads(w: np.ndarray) -> np.ndarray:
    """``(d, H, dh) -> (d, H*dh)``."""
    d, h, dh = w.shape
    return w.reshape(d, h * dh)


@dataclass
class NetworkBounds:
    """Certified enclosures of every stage of the forward pass."""

    logits: IntervalArray        # (B, p)
    attention: IntervalArray     # (B, H, T, T) optional/last layer
    residuals: List[IntervalArray]
    tokens: np.ndarray

    def attention_box(self, batch_index: int, layer: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        lo = self.attention.lo[batch_index]
        hi = self.attention.hi[batch_index]
        return lo, hi


def compute_logit_bounds(
    model: Transformer,
    tokens: np.ndarray,
    *,
    keep_attention: bool = True,
) -> NetworkBounds:
    """Certified enclosures of the answer-position logits for a batch of inputs.

    The enclosure is computed with rigorous rounding-error budgets at every
    matmul (``gamma_n * sum |x| |w|``), an interval softmax, and outward-rounded
    additions -- so it contains the value that the IEEE-754 float64 forward pass
    computes, and *a fortiori* the float32 model's value up to the documented
    float32/float64 gap.
    """
    cfg = model.config
    w = _model_weights(model)
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (batch, seq_len)")
    B, T = tokens.shape
    H, dh, d = cfg.n_heads, cfg.d_head, cfg.d_model
    scale = 1.0 / math.sqrt(dh) if cfg.attn_scale == "sqrt" else 1.0

    x = IntervalArray.exact(w["W_E"][tokens])
    residuals = [x]
    last_attention: Optional[IntervalArray] = None

    for layer in range(cfg.n_layers):
        # ---- QKV projections -------------------------------------------
        wq = _flatten_heads(w[f"W_Q{layer}"])
        wk = _flatten_heads(w[f"W_K{layer}"])
        wv = _flatten_heads(w[f"W_V{layer}"])
        # The output projection contracts the concatenated head axis:
        # out[d] = sum_{h,k} z[h,k] W_O[d,h,k]  ==  z_flat @ W_O_flat.T
        wo = _flatten_heads(w[f"W_O{layer}"]).T

        q_flat = _matmul_interval(x, wq)   # (B,T,H*dh)
        k_flat = _matmul_interval(x, wk)
        v_flat = _matmul_interval(x, wv)
        q = IntervalArray(
            q_flat.lo.reshape(B, T, H, dh), q_flat.hi.reshape(B, T, H, dh)
        )
        k = IntervalArray(k_flat.lo.reshape(B, T, H, dh), k_flat.hi.reshape(B, T, H, dh))
        v = IntervalArray(v_flat.lo.reshape(B, T, H, dh), v_flat.hi.reshape(B, T, H, dh))

        # ---- scores (B, H, T, T) ---------------------------------------
        # sum over the head dimension k of q[t,k] * k[s,k];
        # broadcast q over s and k over t, then contract the last axis.
        q_lo = q.lo[:, :, None, :, :]   # (B,T,1,H,dh)
        q_hi = q.hi[:, :, None, :, :]
        k_lo = k.lo[:, None, :, :, :]   # (B,1,S,H,dh)
        k_hi = k.hi[:, None, :, :, :]
        p_lo = np.minimum(
            np.minimum(q_lo * k_lo, q_lo * k_hi), np.minimum(q_hi * k_lo, q_hi * k_hi)
        )
        p_hi = np.maximum(
            np.maximum(q_lo * k_lo, q_lo * k_hi), np.maximum(q_hi * k_lo, q_hi * k_hi)
        )
        s_lo = p_lo.sum(axis=-1) * scale
        s_hi = p_hi.sum(axis=-1) * scale
        mag = (np.abs(q_lo) * np.abs(k_hi) + np.abs(q_hi) * np.abs(k_lo)).sum(axis=-1)
        err = gamma(dh + 1) * mag * abs(scale)
        scores = IntervalArray(
            _outward(s_lo - err).transpose(0, 3, 1, 2),
            _outward_up(s_hi + err).transpose(0, 3, 1, 2),
        )  # (B, H, T, T)

        if cfg.causal:
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores.lo[:, :, mask] = -np.inf
            scores.hi[:, :, mask] = -np.inf

        pattern = _softmax_interval(scores, axis=-1)
        last_attention = pattern

        # ---- attention output ------------------------------------------
        # Corner method over the source-token axis: z[t] = sum_s P[t, s] v[s].
        c1 = np.einsum("bhts,bshk->bthk", pattern.lo, v.lo)
        c2 = np.einsum("bhts,bshk->bthk", pattern.lo, v.hi)
        c3 = np.einsum("bhts,bshk->bthk", pattern.hi, v.lo)
        c4 = np.einsum("bhts,bshk->bthk", pattern.hi, v.hi)
        z_lo = np.minimum(np.minimum(c1, c2), np.minimum(c3, c4))
        z_hi = np.maximum(np.maximum(c1, c2), np.maximum(c3, c4))
        mag = np.einsum("bhts,bshk->bthk", np.abs(pattern.hi), np.abs(v.lo) + np.abs(v.hi))
        err = gamma(T) * mag
        z_lo = _outward(z_lo - err)
        z_hi = _outward_up(z_hi + err)

        z_flat = IntervalArray(
            z_lo.reshape(B, T, H * dh), z_hi.reshape(B, T, H * dh)
        )
        head_out = _matmul_interval(z_flat, wo)
        x = x + head_out

        if cfg.mlp_width > 0:
            win = w[f"W_in{layer}"]
            wout = w[f"W_out{layer}"]
            pre = _matmul_interval(x, win)
            hidden = IntervalArray(np.maximum(pre.lo, 0.0), np.maximum(pre.hi, 0.0))
            x = x + _matmul_interval(hidden, wout)

        residuals.append(x)

    logits = _matmul_interval(x, w["W_U"])
    return NetworkBounds(logits=logits, attention=last_attention, residuals=residuals, tokens=tokens)


def compute_attention_values(model: Transformer, tokens: np.ndarray) -> np.ndarray:
    """Float64 attention patterns of the last layer, shape ``(B, H, T, T)``."""
    import copy

    token_t = torch.from_numpy(np.asarray(tokens, dtype=np.int64))
    # Reuse the differentiable forward pass in float64 for the point values.
    model64 = copy.deepcopy(model).double()
    with torch.no_grad():
        _, trace = model64(token_t, trace=True)
    assert trace is not None
    return trace.attention_patterns[-1].cpu().numpy()


# ---------------------------------------------------------------------------
# Symbolic encoding of the network for Z3
# ---------------------------------------------------------------------------
@dataclass
class SymbolicEncoding:
    """A per-input symbolic encoding of the network's answer logits."""

    variables: List[z3.ArithRef]
    constraints: List[z3.BoolRef]
    logits: List[z3.ArithRef]
    layout: List[Tuple[int, int]]  # (head, position) for each variable


class Z3Translator:
    """Translates IR programs and (one-layer) networks into Z3 expressions."""

    def __init__(self, p: int) -> None:
        self.p = int(p)

    # -- IR -> Z3 ----------------------------------------------------------
    def translate(
        self,
        expr: Expr,
        env: Dict[str, z3.ExprRef],
        *,
        sort: str = "int",
    ) -> z3.ExprRef:
        """Translate an IR expression into Z3.

        ``sort="int"`` produces an ``Int`` expression (exact), ``sort="real"`` a
        ``Real`` expression.  Trigonometry, softmax and argmax have no SMT
        counterpart and raise :class:`NotImplementedError`; those programs are
        handled by the enclosure-based verifier instead.
        """
        if sort not in ("int", "real"):
            raise ValueError("sort must be 'int' or 'real'")

        def rec(node: Expr) -> z3.ExprRef:
            if isinstance(node, Const):
                return self._const(node.value, sort)
            if isinstance(node, Var):
                if node.name not in env:
                    raise KeyError(f"unbound variable {node.name!r}")
                return env[node.name]
            if isinstance(node, Neg):
                return -rec(node.operand)
            if isinstance(node, BinOp):
                lhs, rhs = rec(node.left), rec(node.right)
                if node.op == "+":
                    return lhs + rhs
                if node.op == "-":
                    return lhs - rhs
                if node.op == "*":
                    return lhs * rhs
                if node.op == "/":
                    if sort == "int":
                        raise NotImplementedError("integer division is not part of the IR")
                    return lhs / rhs
                if node.op == "%":
                    if sort != "int":
                        raise NotImplementedError("modulo requires the integer sort")
                    return self._mod(lhs, rhs)
                raise NotImplementedError(f"operator {node.op!r} has no SMT translation")
            if isinstance(node, Mod):
                if sort != "int":
                    raise NotImplementedError("Modulo requires the integer sort")
                return self._mod(rec(node.operand), self._const(node.modulus, "int"))
            if isinstance(node, Sum):
                acc: z3.ExprRef = self._const(0.0, sort)
                for term in node.terms:
                    acc = acc + rec(term)
                return acc
            if isinstance(node, IfElse):
                # Comparisons are encoded through the boolean fragment.
                from src.synthesis.ast_nodes import Compare
                from src.synthesis.ast_nodes import BoolOp as _BoolOp
                from src.synthesis.ast_nodes import Not as _Not

                cond = self._translate_bool(node.condition, env, sort)
                return z3.If(cond, rec(node.then), rec(node.otherwise))
            raise NotImplementedError(
                f"node {type(node).__name__} has no exact SMT translation"
            )

        return rec(expr)

    def _const(self, value: float, sort: str) -> z3.ExprRef:
        if sort == "int":
            if float(value) != int(value):
                raise ValueError(f"{value} is not an integer")
            return z3.IntVal(int(value))
        return rational_ref(value)

    def _mod(self, lhs: z3.ExprRef, rhs: z3.ExprRef) -> z3.ExprRef:
        return lhs % rhs

    def _translate_bool(self, node: Expr, env: Dict[str, z3.ExprRef], sort: str) -> z3.BoolRef:
        from src.synthesis.ast_nodes import BoolOp as _BoolOp
        from src.synthesis.ast_nodes import Compare
        from src.synthesis.ast_nodes import Not as _Not

        if isinstance(node, Compare):
            lhs, rhs = self.translate(node.left, env, sort=sort), self.translate(
                node.right, env, sort=sort
            )
            return {
                "<": lhs < rhs,
                "<=": lhs <= rhs,
                ">": lhs > rhs,
                ">=": lhs >= rhs,
                "==": lhs == rhs,
                "!=": lhs != rhs,
            }[node.op]
        if isinstance(node, _BoolOp):
            args = [self._translate_bool(a, env, sort) for a in node.args]
            return z3.And(*args) if node.op == "and" else z3.Or(*args)
        if isinstance(node, _Not):
            return z3.Not(self._translate_bool(node.operand, env, sort))
        raise NotImplementedError(f"{type(node).__name__} is not a boolean expression")

    # -- network -> Z3 ------------------------------------------------------
    def encode_attention(
        self,
        bounds: NetworkBounds,
        batch_index: int,
        *,
        slack: float = 0.0,
    ) -> Tuple[List[z3.ArithRef], List[z3.BoolRef], List[Tuple[int, int]]]:
        """Declare one Real variable per attention probability of one input.

        ``slack`` widens the certified enclosure by that many probability units
        (clipped to ``[0, 1]``).  Widening an enclosure keeps the encoding sound
        and gives a strictly stronger theorem.
        """
        if bounds.attention is None:
            raise ValueError("attention bounds were not retained")
        lo = bounds.attention.lo[batch_index] - slack   # (H, T, T)
        hi = bounds.attention.hi[batch_index] + slack
        lo = np.clip(lo, 0.0, 1.0)
        hi = np.clip(hi, 0.0, 1.0)
        n_heads, seq_len, _ = lo.shape
        variables: List[z3.ArithRef] = []
        constraints: List[z3.BoolRef] = []
        layout: List[Tuple[int, int]] = []
        for h in range(n_heads):
            head_vars = []
            for j in range(seq_len):
                name = f"p_{batch_index}_{h}_{j}"
                var = z3.Real(name)
                variables.append(var)
                layout.append((h, j))
                head_vars.append(var)
                constraints.append(var >= rational_ref(max(0.0, float(lo[h, -1, j]))))
                constraints.append(var <= rational_ref(min(1.0, float(hi[h, -1, j]))))
            constraints.append(z3.Sum(head_vars) == rational_ref(1.0))
        return variables, constraints, layout

    def encode_network_logits(
        self,
        model: Transformer,
        bounds: NetworkBounds,
        batch_index: int,
        attention_vars: List[z3.ArithRef],
        layout: Sequence[Tuple[int, int]],
    ) -> List[z3.ArithRef]:
        """Exact-rational linear expression of the answer logits in the attention vars."""
        cfg = model.config
        if cfg.n_layers != 1 or cfg.mlp_width > 0:
            raise NotImplementedError(
                "symbolic encoding is implemented for one-layer attention-only models"
            )
        w = _model_weights(model)
        p = cfg.p
        answer = cfg.answer_position
        tokens = bounds.tokens[batch_index]
        w_v = w["W_V0"]  # (d, H, dh)
        w_o = w["W_O0"]
        w_u = w["W_U"]
        embed = w["W_E"]

        maps = np.einsum("vhd,dc->vhc", np.einsum("vhk,dhk->vhd",
                                                  np.einsum("vd,dhk->vhk", embed, w_v), w_o), w_u)
        bias = embed[cfg.eq_token] @ w_u
        var_lookup = {key: var for key, var in zip(layout, attention_vars)}

        logits: List[z3.ArithRef] = []
        for c in range(p):
            acc = rational_ref(float(bias[c]))
            for h in range(cfg.n_heads):
                for j in range(cfg.seq_len):
                    var = var_lookup.get((h, j))
                    if var is None:
                        continue
                    token = int(tokens[j])
                    coeff = float(maps[token, h, c])
                    if coeff != 0.0:
                        acc = acc + rational_ref(coeff) * var
            logits.append(acc)
        return logits


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------
def translate_program(
    program,
    *,
    sort: str = "int",
    solver_vars: Optional[Dict[str, z3.ExprRef]] = None,
) -> Tuple[z3.ExprRef, Dict[str, z3.ExprRef]]:
    """Translate every parameter of ``program`` and its body into Z3.

    Returns the body expression and the variable environment.  Integer programs
    are translated exactly; real-valued programs use exact rational constants.
    """
    translator = Z3Translator(program.p)
    env = dict(solver_vars or {})
    for name in program.params:
        if name not in env:
            env[name] = z3.Int(name) if sort == "int" else z3.Real(name)
    return translator.translate(program.body, env, sort=sort), env
