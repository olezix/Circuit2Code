"""Circuit2Code Intermediate Representation (IR).

The IR is a small, typed expression language that sits between the extracted
circuit and the emitted code.  It has three independent evaluators:

``eval(env)``
    Vectorised NumPy semantics -- the *reference* semantics of the IR, used by
    the synthesizer to search the program space and by the tests as an oracle.
``interval(env)``
    Outward-rounded interval semantics (:class:`Interval`) used to build sound
    rational enclosures for the Z3 prover.
``to_python()`` / ``to_c()``
    Source emission (implemented in :mod:`src.codegen`, which dispatches on the
    node type).

Design constraints
------------------
* No node has hidden state; two structurally equal trees behave identically.
* Every node is immutable (frozen dataclass) so expressions can be shared.
* The language is deliberately tiny: a decompiled program should be readable by
  a human, therefore there is no generic ``Call`` node -- composite operations
  are explicit primitives (``Softmax``, ``Linear``, ``ArgMax``, ...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import sympy

__all__ = [
    "Interval",
    "Expr",
    "Const",
    "Var",
    "Neg",
    "BinOp",
    "Mod",
    "Sin",
    "Cos",
    "Sum",
    "Linear",
    "Softmax",
    "SoftmaxComponent",
    "Threshold",
    "VectorIndex",
    "ArgMax",
    "IfElse",
    "Compare",
    "BoolOp",
    "Not",
    "Program",
    "walk",
    "simplify",
]


# ---------------------------------------------------------------------------
# Interval arithmetic
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Interval:
    """Closed interval ``[lo, hi]`` of IEEE doubles with outward rounding.

    Every operation widens its result by :data:`ULP_PAD` units in the last place,
    which makes the implementation *sound* (never excludes a true value) despite
    relying on hardware rounding.  The Z3 layer converts the endpoints to exact
    rationals, so the proof never sees a floating point number.
    """

    lo: float
    hi: float

    ULP_PAD = 2

    def __post_init__(self) -> None:
        if not (self.lo <= self.hi):
            raise ValueError(f"invalid interval [{self.lo}, {self.hi}]")

    # -- constructors ------------------------------------------------------
    @staticmethod
    def of(value: float) -> "Interval":
        return Interval(float(value), float(value))

    @staticmethod
    def box(lo: float, hi: float) -> "Interval":
        return Interval(float(lo), float(hi))

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _widen(value: float) -> Tuple[float, float]:
        if math.isnan(value):
            raise ValueError("interval arithmetic produced NaN")
        if math.isinf(value):
            return (value, value)
        n = Interval.ULP_PAD
        lo, hi = value, value
        for _ in range(n):
            lo = math.nextafter(lo, -math.inf)
            hi = math.nextafter(hi, math.inf)
        return lo, hi

    def __add__(self, other: Union["Interval", float]) -> "Interval":
        other = as_interval(other)
        lo, hi = self._widen(self.lo + other.lo)
        lo2, hi2 = self._widen(self.hi + other.hi)
        return Interval(min(lo, lo2), max(hi, hi2))

    def __sub__(self, other: Union["Interval", float]) -> "Interval":
        other = as_interval(other)
        return self.__add__(Interval(-other.hi, -other.lo))

    def __neg__(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def __mul__(self, other: Union["Interval", float]) -> "Interval":
        other = as_interval(other)
        products = [
            self._widen(self.lo * other.lo),
            self._widen(self.lo * other.hi),
            self._widen(self.hi * other.lo),
            self._widen(self.hi * other.hi),
        ]
        return Interval(min(p[0] for p in products), max(p[1] for p in products))

    def __truediv__(self, other: Union["Interval", float]) -> "Interval":
        other = as_interval(other)
        if other.lo <= 0.0 <= other.hi:
            raise ZeroDivisionError(f"interval division by {other} containing zero")
        lo = min(self._widen(1.0 / other.lo)[0], self._widen(1.0 / other.hi)[0])
        hi = max(self._widen(1.0 / other.lo)[1], self._widen(1.0 / other.hi)[1])
        return self * Interval(lo, hi)

    def __pow__(self, exponent: int) -> "Interval":
        if exponent < 0:
            raise ValueError("negative interval powers are not supported")
        result = Interval.of(1.0)
        for _ in range(exponent):
            result = result * self
        return result

    @property
    def mid(self) -> float:
        return 0.5 * (self.lo + self.hi)

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi

    def to_fractions(self) -> Tuple[Fraction, Fraction]:
        return Fraction.from_float(self.lo), Fraction.from_float(self.hi)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return f"[{self.lo:.6g}, {self.hi:.6g}]"


def _contains_angle(lo: float, hi: float, angle: float, period: float = 2.0 * math.pi) -> bool:
    """Whether ``[lo, hi]`` contains ``angle + k * period`` for some integer ``k``."""
    k = math.ceil((lo - angle) / period)
    return angle + k * period <= hi


def _sin_interval(inner: Interval) -> Interval:
    """Sound enclosure of ``sin`` over an interval (endpoints padded outward)."""
    lo, hi = inner.lo, inner.hi
    if hi - lo >= 2.0 * math.pi:
        return Interval.box(-1.0, 1.0)
    candidates = [math.sin(lo), math.sin(hi)]
    if _contains_angle(lo, hi, math.pi / 2.0):
        candidates.append(1.0)
    if _contains_angle(lo, hi, -math.pi / 2.0):
        candidates.append(-1.0)
    low = min(candidates)
    high = max(candidates)
    lo_w = min(Interval._widen(low)[0], Interval._widen(high)[0])
    hi_w = max(Interval._widen(low)[1], Interval._widen(high)[1])
    return Interval(lo_w, hi_w)


def as_interval(value: Union[Interval, float, int]) -> Interval:
    if isinstance(value, Interval):
        return value
    return Interval.of(float(value))


# ---------------------------------------------------------------------------
# Expression base class
# ---------------------------------------------------------------------------
class Expr:
    """Base class of all IR nodes."""

    __slots__ = ()

    # -- semantics ---------------------------------------------------------
    def eval(self, env: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def interval(self, env: Mapping[str, Interval]) -> Any:
        raise NotImplementedError

    # -- structure ---------------------------------------------------------
    def children(self) -> Tuple["Expr", ...]:
        return ()

    def vars(self) -> List[str]:
        out: List[str] = []
        for child in self.children():
            out.extend(child.vars())
        if isinstance(self, Var) and self.name not in out:
            out.append(self.name)
        return out

    def replace(self, old: "Expr", new: "Expr") -> "Expr":
        """Substitute every occurrence of ``old`` by ``new`` (structurally).

        The identity fast-path avoids rebuilding trees when the caller already
        holds the node it means to replace; the structural check makes the
        operation usable from outside the module (``expr.replace(Var("a"),
        Var("b"))``), where the caller usually builds a fresh literal.
        """
        if self is old or self == old:
            return new
        return self

    def _key(self) -> Tuple:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Expr) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    # -- rendering ---------------------------------------------------------
    def render(self, precedence: int = 0) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.render()})"

    def __str__(self) -> str:
        return self.render()

    # -- convenience -------------------------------------------------------
    def __add__(self, other: Union["Expr", float]) -> "Expr":
        return BinOp("+", self, as_expr(other))

    def __radd__(self, other: Union["Expr", float]) -> "Expr":
        return BinOp("+", as_expr(other), self)

    def __sub__(self, other: Union["Expr", float]) -> "Expr":
        return BinOp("-", self, as_expr(other))

    def __rsub__(self, other: Union["Expr", float]) -> "Expr":
        return BinOp("-", as_expr(other), self)

    def __mul__(self, other: Union["Expr", float]) -> "Expr":
        return BinOp("*", self, as_expr(other))

    def __rmul__(self, other: Union["Expr", float]) -> "Expr":
        return BinOp("*", as_expr(other), self)

    def __truediv__(self, other: Union["Expr", float]) -> "Expr":
        return BinOp("/", self, as_expr(other))

    def __neg__(self) -> "Expr":
        return Neg(self)

    def to_sympy(self) -> sympy.Expr:
        raise NotImplementedError(f"{type(self).__name__} has no SymPy counterpart")


# ---------------------------------------------------------------------------
# Leaves
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Const(Expr):
    value: float

    def eval(self, env: Mapping[str, Any]) -> float:
        return float(self.value)

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        return Interval.of(self.value)

    def _key(self) -> Tuple:
        return ("const", float(self.value))

    def render(self, precedence: int = 0) -> str:
        return repr(float(self.value))

    def to_sympy(self) -> sympy.Expr:
        return sympy.Float(self.value) if not float(self.value).is_integer() else sympy.Integer(
            int(self.value)
        )


@dataclass(frozen=True, eq=False)
class Var(Expr):
    name: str

    def eval(self, env: Mapping[str, Any]) -> Any:
        if self.name not in env:
            raise KeyError(f"unbound variable {self.name!r}")
        return env[self.name]

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        if self.name not in env:
            raise KeyError(f"unbound variable {self.name!r}")
        return as_interval(env[self.name])

    def vars(self) -> List[str]:
        return [self.name]

    def _key(self) -> Tuple:
        return ("var", self.name)

    def render(self, precedence: int = 0) -> str:
        return self.name

    def to_sympy(self) -> sympy.Expr:
        return sympy.Symbol(self.name)


def as_expr(value: Union[Expr, int, float, np.floating, np.integer]) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return Const(float(value))
    raise TypeError(f"cannot convert {type(value).__name__} to an IR expression")


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------
_PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "%": 2, "**": 3}


@dataclass(frozen=True, eq=False)
class Neg(Expr):
    operand: Expr

    def eval(self, env: Mapping[str, Any]) -> Any:
        return -self.operand.eval(env)

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        return -self.operand.interval(env)

    def children(self) -> Tuple[Expr, ...]:
        return (self.operand,)

    def replace(self, old: Expr, new: Expr) -> Expr:
        if self is old:
            return new
        return Neg(self.operand.replace(old, new))

    def _key(self) -> Tuple:
        return ("neg", self.operand)

    def render(self, precedence: int = 0) -> str:
        text = f"-{self.operand.render(_PREC['*'])}"
        return f"({text})" if precedence > _PREC["*"] else text

    def to_sympy(self) -> sympy.Expr:
        return -self.operand.to_sympy()


@dataclass(frozen=True, eq=False)
class BinOp(Expr):
    """Binary arithmetic node; ``op`` is one of ``+ - * / % **``."""

    op: str
    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        if self.op not in _PREC:
            raise ValueError(f"unsupported operator {self.op!r}")

    def eval(self, env: Mapping[str, Any]) -> Any:
        lhs, rhs = self.left.eval(env), self.right.eval(env)
        if self.op == "+":
            return np.add(lhs, rhs)
        if self.op == "-":
            return np.subtract(lhs, rhs)
        if self.op == "*":
            return np.multiply(lhs, rhs)
        if self.op == "/":
            return np.true_divide(lhs, rhs)
        if self.op == "%":
            return np.mod(lhs, rhs)
        return np.power(lhs, rhs)

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        lhs = self.left.interval(env)
        rhs = self.right.interval(env)
        if self.op == "+":
            return lhs + rhs
        if self.op == "-":
            return lhs - rhs
        if self.op == "*":
            return lhs * rhs
        if self.op == "/":
            return lhs / rhs
        if self.op == "%":
            raise NotImplementedError("interval modulo is not supported")
        if self.op == "**":
            exponent = int(rhs.lo)
            if rhs.lo != rhs.hi or exponent < 0:
                raise NotImplementedError("only non-negative constant exponents are supported")
            return lhs**exponent
        raise ValueError(self.op)  # pragma: no cover

    def children(self) -> Tuple[Expr, ...]:
        return (self.left, self.right)

    def replace(self, old: Expr, new: Expr) -> Expr:
        if self is old:
            return new
        return BinOp(self.op, self.left.replace(old, new), self.right.replace(old, new))

    def _key(self) -> Tuple:
        return ("binop", self.op, self.left, self.right)

    def render(self, precedence: int = 0) -> str:
        p = _PREC[self.op]
        text = (
            f"{self.left.render(p)} {self.op} {self.right.render(p + (1 if self.op in ('-', '/') else 0))}"
        )
        return f"({text})" if p < precedence else text

    def to_sympy(self) -> sympy.Expr:
        lhs, rhs = self.left.to_sympy(), self.right.to_sympy()
        table = {
            "+": lambda: lhs + rhs,
            "-": lambda: lhs - rhs,
            "*": lambda: lhs * rhs,
            "/": lambda: lhs / rhs,
            "%": lambda: sympy.Mod(lhs, rhs),
            "**": lambda: lhs**rhs,
        }
        return table[self.op]()


@dataclass(frozen=True, eq=False)
class Mod(Expr):
    """Euclidean remainder ``operand mod modulus`` (floored, like Python's ``%``)."""

    operand: Expr
    modulus: int

    def __post_init__(self) -> None:
        if int(self.modulus) <= 0:
            raise ValueError("modulus must be a positive integer")

    def eval(self, env: Mapping[str, Any]) -> Any:
        return np.mod(self.operand.eval(env), int(self.modulus))

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        return Interval.box(0.0, float(self.modulus - 1))

    def children(self) -> Tuple[Expr, ...]:
        return (self.operand,)

    def replace(self, old: Expr, new: Expr) -> Expr:
        if self is old:
            return new
        return Mod(self.operand.replace(old, new), self.modulus)

    def _key(self) -> Tuple:
        return ("mod", self.operand, int(self.modulus))

    def render(self, precedence: int = 0) -> str:
        text = f"({self.operand.render(_PREC['%'])} % {int(self.modulus)})"
        return text if precedence <= _PREC["%"] else f"({text})"

    def to_sympy(self) -> sympy.Expr:
        return sympy.Mod(self.operand.to_sympy(), int(self.modulus))


@dataclass(frozen=True, eq=False)
class Sin(Expr):
    operand: Expr

    def eval(self, env: Mapping[str, Any]) -> Any:
        return np.sin(self.operand.eval(env))

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        return _sin_interval(self.operand.interval(env))

    def children(self) -> Tuple[Expr, ...]:
        return (self.operand,)

    def replace(self, old: Expr, new: Expr) -> Expr:
        if self is old:
            return new
        return Sin(self.operand.replace(old, new))

    def _key(self) -> Tuple:
        return ("sin", self.operand)

    def render(self, precedence: int = 0) -> str:
        return f"sin({self.operand.render()})"

    def to_sympy(self) -> sympy.Expr:
        return sympy.sin(self.operand.to_sympy())


@dataclass(frozen=True, eq=False)
class Cos(Expr):
    operand: Expr

    def eval(self, env: Mapping[str, Any]) -> Any:
        return np.cos(self.operand.eval(env))

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        return _sin_interval(self.operand.interval(env) + math.pi / 2.0)

    def children(self) -> Tuple[Expr, ...]:
        return (self.operand,)

    def replace(self, old: Expr, new: Expr) -> Expr:
        if self is old:
            return new
        return Cos(self.operand.replace(old, new))

    def _key(self) -> Tuple:
        return ("cos", self.operand)

    def render(self, precedence: int = 0) -> str:
        return f"cos({self.operand.render()})"

    def to_sympy(self) -> sympy.Expr:
        return sympy.cos(self.operand.to_sympy())


@dataclass(frozen=True, eq=False)
class Sum(Expr):
    terms: Tuple[Expr, ...]

    def __init__(self, terms: Iterable[Expr]) -> None:
        object.__setattr__(self, "terms", tuple(as_expr(t) for t in terms))

    def eval(self, env: Mapping[str, Any]) -> Any:
        total = 0.0
        for term in self.terms:
            total = total + term.eval(env)
        return total

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        total = Interval.of(0.0)
        for term in self.terms:
            total = total + term.interval(env)
        return total

    def children(self) -> Tuple[Expr, ...]:
        return self.terms

    def replace(self, old: Expr, new: Expr) -> Expr:
        if self is old:
            return new
        return Sum(t.replace(old, new) for t in self.terms)

    def _key(self) -> Tuple:
        return ("sum", self.terms)

    def render(self, precedence: int = 0) -> str:
        if not self.terms:
            return "0.0"
        text = " + ".join(t.render(_PREC["+"]) for t in self.terms)
        return f"({text})" if precedence > _PREC["+"] else text

    def to_sympy(self) -> sympy.Expr:
        return sympy.Add(*[t.to_sympy() for t in self.terms])


# ---------------------------------------------------------------------------
# Linear algebra / non-linear primitives
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Linear(Expr):
    """Matrix-vector product ``weights @ [args...]`` producing a vector.

    ``weights`` has shape ``(out_features, len(args))``; the node evaluates to a
    vector of ``out_features`` entries.  This is the only place where a table of
    numbers can enter the IR, and the code generator emits it as a static array
    (C) or a literal tuple (Python).
    """

    weights: np.ndarray
    args: Tuple[Expr, ...]
    name: str = "linear"

    def __init__(self, weights, args: Iterable[Expr], name: str = "linear") -> None:
        w = np.asarray(weights, dtype=np.float64)
        args = tuple(as_expr(a) for a in args)
        if w.ndim != 2 or w.shape[1] != len(args):
            raise ValueError(
                f"Linear weights must have shape (out, {len(args)}), got {w.shape}"
            )
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "name", name)

    def eval(self, env: Mapping[str, Any]) -> np.ndarray:
        raw = [np.asarray(a.eval(env), dtype=np.float64) for a in self.args]
        vec = np.stack(np.broadcast_arrays(*raw), axis=-1)
        return vec @ self.weights.T

    def interval(self, env: Mapping[str, Interval]) -> List[Interval]:
        columns = [a.interval(env) for a in self.args]
        out: List[Interval] = []
        for row in self.weights:
            acc = Interval.of(0.0)
            for w, col in zip(row, columns):
                acc = acc + col * Interval.of(float(w))
            out.append(acc)
        return out

    def children(self) -> Tuple[Expr, ...]:
        return self.args

    def replace(self, old: Expr, new: Expr) -> Expr:
        if self is old:
            return new
        return Linear(self.weights, (a.replace(old, new) for a in self.args), self.name)

    def _key(self) -> Tuple:
        return ("linear", self.weights.tobytes(), tuple(self.args), self.name)

    def render(self, precedence: int = 0) -> str:
        return f"{self.name}({', '.join(a.render() for a in self.args)})"


@dataclass(frozen=True, eq=False)
class Softmax(Expr):
    """Softmax over a fixed-length vector of scalar expressions."""

    args: Tuple[Expr, ...]

    def __init__(self, args: Iterable[Expr]) -> None:
        args = tuple(as_expr(a) for a in args)
        if not args:
            raise ValueError("softmax needs at least one argument")
        object.__setattr__(self, "args", args)

    def eval(self, env: Mapping[str, Any]) -> np.ndarray:
        raw = [np.asarray(a.eval(env), dtype=np.float64) for a in self.args]
        values = np.stack(np.broadcast_arrays(*raw), axis=-1)
        shifted = values - values.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    def interval(self, env: Mapping[str, Interval]) -> List[Interval]:
        # Sound but coarse: probabilities live in [0, 1] and sum to one.
        return [Interval.box(0.0, 1.0) for _ in self.args]

    def children(self) -> Tuple[Expr, ...]:
        return self.args

    def replace(self, old: Expr, new: Expr) -> Expr:
        if self is old:
            return new
        return Softmax(a.replace(old, new) for a in self.args)

    def _key(self) -> Tuple:
        return ("softmax", self.args)

    def render(self, precedence: int = 0) -> str:
        return f"softmax([{', '.join(a.render() for a in self.args)}])"


@dataclass(frozen=True, eq=False)
class SoftmaxComponent(Expr):
    """One component of a ``Softmax`` vector.

    This is a *projection*, not a second softmax: it evaluates to
    ``softmax(parent)[index]``.  It exists so that a program can share a single
    softmax across many summands (one per class) without duplicating it.
    """

    parent: "Softmax"
    index: int

    def __post_init__(self) -> None:
        if not isinstance(self.parent, Softmax):
            raise TypeError("SoftmaxComponent.parent must be a Softmax node")
        if not 0 <= int(self.index) < len(self.parent.args):
            raise IndexError("softmax component index out of range")

    def eval(self, env: Mapping[str, Any]) -> Any:
        probs = np.asarray(self.parent.eval(env), dtype=np.float64)
        return probs[..., int(self.index)]

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        return Interval.box(0.0, 1.0)

    def children(self) -> Tuple[Expr, ...]:
        return (self.parent,)

    def replace(self, old: Expr, new: Expr) -> Expr:
        if self is old:
            return new
        return SoftmaxComponent(self.parent.replace(old, new), self.index)  # type: ignore[arg-type]

    def _key(self) -> Tuple:
        return ("softmax_component", self.parent, int(self.index))

    def render(self, precedence: int = 0) -> str:
        return f"softmax(...)[{int(self.index)}]"


@dataclass(frozen=True, eq=False)
class VectorIndex(Expr):
    """Component ``index`` of a vector-valued expression (see :class:`Linear`)."""

    vector: Expr
    index: int

    def __post_init__(self) -> None:
        if int(self.index) < 0:
            raise ValueError("vector index must be non-negative")

    def eval(self, env: Mapping[str, Any]) -> Any:
        return np.asarray(self.vector.eval(env))[..., int(self.index)]

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        value = self.vector.interval(env)
        if isinstance(value, Interval):  # pragma: no cover - defensive
            return value
        return value[int(self.index)]

    def children(self) -> Tuple[Expr, ...]:
        return (self.vector,)

    def replace(self, old: Expr, new: Expr) -> Expr:
        if self is old:
            return new
        return VectorIndex(self.vector.replace(old, new), self.index)

    def _key(self) -> Tuple:
        return ("vindex", self.vector, int(self.index))

    def render(self, precedence: int = 0) -> str:
        inner = self.vector.render(_PREC["**"] + 1)
        return f"{inner}[{int(self.index)}]"


@dataclass(frozen=True, eq=False)
class Threshold(Expr):
    """Step function ``1.0 if operand >= threshold else 0.0``."""

    operand: Expr
    threshold: float = 0.0

    def eval(self, env: Mapping[str, Any]) -> Any:
        return (np.asarray(self.operand.eval(env)) >= self.threshold).astype(np.float64)

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        inner = self.operand.interval(env)
        if inner.hi < self.threshold:
            return Interval.of(0.0)
        if inner.lo >= self.threshold:
            return Interval.of(1.0)
        return Interval.box(0.0, 1.0)

    def children(self) -> Tuple[Expr, ...]:
        return (self.operand,)

    def replace(self, old: Expr, new: Expr) -> Expr:
        if self is old:
            return new
        return Threshold(self.operand.replace(old, new), self.threshold)

    def _key(self) -> Tuple:
        return ("threshold", self.operand, float(self.threshold))

    def render(self, precedence: int = 0) -> str:
        return f"step({self.operand.render()} - {self.threshold!r})"


@dataclass(frozen=True, eq=False)
class ArgMax(Expr):
    """Argmax over a fixed-length vector; returns an integer-valued expression."""

    args: Tuple[Expr, ...]

    def __init__(self, args: Iterable[Expr]) -> None:
        args = tuple(as_expr(a) for a in args)
        if not args:
            raise ValueError("argmax needs at least one argument")
        object.__setattr__(self, "args", args)

    def eval(self, env: Mapping[str, Any]) -> Any:
        raw = [np.asarray(a.eval(env), dtype=np.float64) for a in self.args]
        values = np.stack(np.broadcast_arrays(*raw), axis=-1)
        return np.argmax(values, axis=-1)

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        return Interval.box(0.0, float(len(self.args) - 1))

    def children(self) -> Tuple[Expr, ...]:
        return self.args

    def replace(self, old: Expr, new: Expr) -> Expr:
        if self is old:
            return new
        return ArgMax(a.replace(old, new) for a in self.args)

    def _key(self) -> Tuple:
        return ("argmax", self.args)

    def render(self, precedence: int = 0) -> str:
        return f"argmax([{', '.join(a.render() for a in self.args)}])"


@dataclass(frozen=True, eq=False)
class Compare(Expr):
    """Comparison node returning a boolean array (``< <= > >= == !=``)."""

    op: str
    left: Expr
    right: Expr

    OPS = ("<", "<=", ">", ">=", "==", "!=")

    def __post_init__(self) -> None:
        if self.op not in self.OPS:
            raise ValueError(f"unsupported comparison {self.op!r}")

    def eval(self, env: Mapping[str, Any]) -> Any:
        lhs, rhs = self.left.eval(env), self.right.eval(env)
        return {
            "<": lambda: lhs < rhs,
            "<=": lambda: lhs <= rhs,
            ">": lambda: lhs > rhs,
            ">=": lambda: lhs >= rhs,
            "==": lambda: lhs == rhs,
            "!=": lambda: lhs != rhs,
        }[self.op]()

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        raise NotImplementedError("comparisons do not have an interval value")

    def children(self) -> Tuple[Expr, ...]:
        return (self.left, self.right)

    def replace(self, old: Expr, new: Expr) -> Expr:
        if self is old:
            return new
        return Compare(self.op, self.left.replace(old, new), self.right.replace(old, new))

    def _key(self) -> Tuple:
        return ("cmp", self.op, self.left, self.right)

    def render(self, precedence: int = 0) -> str:
        return f"({self.left.render()} {self.op} {self.right.render()})"


@dataclass(frozen=True, eq=False)
class BoolOp(Expr):
    op: str
    args: Tuple[Expr, ...]

    def __init__(self, op: str, args: Iterable[Expr]) -> None:
        if op not in ("and", "or"):
            raise ValueError(f"unsupported boolean operator {op!r}")
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "args", tuple(as_expr(a) for a in args))

    def eval(self, env: Mapping[str, Any]) -> Any:
        values = [np.asarray(a.eval(env), dtype=bool) for a in self.args]
        out = values[0]
        for value in values[1:]:
            out = np.logical_and(out, value) if self.op == "and" else np.logical_or(out, value)
        return out

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        raise NotImplementedError("boolean nodes do not have an interval value")

    def children(self) -> Tuple[Expr, ...]:
        return self.args

    def replace(self, old: Expr, new: Expr) -> Expr:
        if self is old:
            return new
        return BoolOp(self.op, (a.replace(old, new) for a in self.args))

    def _key(self) -> Tuple:
        return (self.op, self.args)

    def render(self, precedence: int = 0) -> str:
        return f"({' %s ' % self.op.join(a.render() for a in self.args)})"


@dataclass(frozen=True, eq=False)
class Not(Expr):
    operand: Expr

    def eval(self, env: Mapping[str, Any]) -> Any:
        return np.logical_not(np.asarray(self.operand.eval(env), dtype=bool))

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        raise NotImplementedError("boolean nodes do not have an interval value")

    def children(self) -> Tuple[Expr, ...]:
        return (self.operand,)

    def replace(self, old: Expr, new: Expr) -> Expr:
        if self is old:
            return new
        return Not(self.operand.replace(old, new))

    def _key(self) -> Tuple:
        return ("not", self.operand)

    def render(self, precedence: int = 0) -> str:
        return f"(not {self.operand.render()})"


@dataclass(frozen=True, eq=False)
class IfElse(Expr):
    """Ternary conditional with eager (vectorised) semantics."""

    condition: Expr
    then: Expr
    otherwise: Expr

    def eval(self, env: Mapping[str, Any]) -> Any:
        cond = np.asarray(self.condition.eval(env), dtype=bool)
        return np.where(cond, self.then.eval(env), self.otherwise.eval(env))

    def interval(self, env: Mapping[str, Interval]) -> Interval:
        a = self.then.interval(env)
        b = self.otherwise.interval(env)
        return Interval.box(min(a.lo, b.lo), max(a.hi, b.hi))

    def children(self) -> Tuple[Expr, ...]:
        return (self.condition, self.then, self.otherwise)

    def replace(self, old: Expr, new: Expr) -> Expr:
        if self is old:
            return new
        return IfElse(
            self.condition.replace(old, new),
            self.then.replace(old, new),
            self.otherwise.replace(old, new),
        )

    def _key(self) -> Tuple:
        return ("if", self.condition, self.then, self.otherwise)

    def render(self, precedence: int = 0) -> str:
        return (
            f"({self.then.render()} if {self.condition.render()} "
            f"else {self.otherwise.render()})"
        )


# ---------------------------------------------------------------------------
# Traversal, simplification, programs
# ---------------------------------------------------------------------------
def walk(expr: Expr) -> Iterable[Expr]:
    """Pre-order traversal over the expression DAG (each node visited once)."""
    seen: set = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        stack.extend(reversed(node.children()))


def simplify(expr: Expr, *, max_nodes: int = 400) -> Expr:
    """Simplify an expression with SymPy when it is purely algebraic.

    Nodes with no SymPy counterpart (``ArgMax``, ``Softmax``, ``Threshold``,
    boolean nodes, and ``Mod`` under a non-algebraic context) are left untouched;
    their children are simplified recursively.  The function is conservative: it
    only returns a *different* tree when SymPy can prove the rewrite, and the
    result is re-checked numerically against the original.
    """
    try:
        sym = expr.to_sympy()
    except (NotImplementedError, TypeError, sympy.SympyError):
        return expr
    if len(list(walk(expr))) > max_nodes:
        return expr
    try:
        simplified = sympy.simplify(sympy.expand(sym))
    except Exception:  # pragma: no cover - SymPy is allowed to give up
        return expr
    rebuilt = _from_sympy(simplified)
    if rebuilt is None or rebuilt == expr:
        return rebuilt or expr
    # Numeric equivalence check on a deterministic grid.
    rng = np.random.default_rng(0)
    names = sorted(set(expr.vars()))
    env = {name: rng.uniform(0.5, 5.5, size=8) for name in names}
    try:
        before = np.asarray(expr.eval(env), dtype=np.float64)
        after = np.asarray(rebuilt.eval(env), dtype=np.float64)
    except Exception:  # pragma: no cover - defensive
        return expr
    if before.shape != after.shape or not np.allclose(before, after, atol=1e-9, rtol=1e-9):
        return expr
    return rebuilt


def _from_sympy(expr: sympy.Expr) -> Optional[Expr]:
    """Best-effort conversion of a SymPy expression back into the IR."""
    if expr.is_Number:
        return Const(float(expr))
    if expr.is_Symbol:
        return Var(str(expr))
    if isinstance(expr, sympy.Add):
        args = [_from_sympy(a) for a in expr.args]
        if any(a is None for a in args):
            return None
        if len(args) == 2:
            return BinOp("+", args[0], args[1])
        return Sum(args)  # type: ignore[arg-type]
    if isinstance(expr, sympy.Mul):
        coeff, rest = expr.as_coeff_Mul()
        factors = sympy.Mul.make_args(rest)
        nodes = [_from_sympy(f) for f in factors]
        if any(node is None for node in nodes):
            return None
        built = nodes[0]
        for node in nodes[1:]:
            built = BinOp("*", built, node)  # type: ignore[arg-type]
        if coeff != 1:
            return BinOp("*", Const(float(coeff)), built)  # type: ignore[arg-type]
        return built
    if isinstance(expr, sympy.Pow):
        base = _from_sympy(expr.args[0])
        exponent = expr.args[1]
        if base is None or not exponent.is_Integer:
            return None
        exp_int = int(exponent)
        if exp_int < 0:
            return BinOp("/", Const(1.0), base ** Const(float(-exp_int)))
        return base ** Const(float(exp_int))
    if isinstance(expr, sympy.Mod):
        inner = _from_sympy(expr.args[0])
        modulus = expr.args[1]
        if inner is None or not modulus.is_Integer:
            return None
        return Mod(inner, int(modulus))
    if isinstance(expr, sympy.sin):
        inner = _from_sympy(expr.args[0])
        return None if inner is None else Sin(inner)
    if isinstance(expr, sympy.cos):
        inner = _from_sympy(expr.args[0])
        return None if inner is None else Cos(inner)
    return None


@dataclass
class Program:
    """A complete decompiled program: a single function over integer inputs.

    Parameters
    ----------
    name:
        Function name used by the emitters.
    params:
        Ordered input parameter names (``["a", "b"]``).
    body:
        Expression computing the answer.
    p:
        Modulus of the task (part of the contract, not of the expression).
    docstring / provenance:
        Human-readable metadata that ends up in the emitted source.
    """

    name: str
    params: List[str]
    body: Expr
    p: int
    docstring: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __call__(self, **kwargs: Any) -> Any:
        missing = set(self.params) - set(kwargs)
        if missing:
            raise KeyError(f"missing arguments {sorted(missing)}")
        return self.body.eval(kwargs)

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        return self.body.eval(env)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "params": list(self.params),
            "p": self.p,
            "docstring": self.docstring,
            "body": self.body.render(),
            "provenance": dict(self.provenance),
        }

    def __repr__(self) -> str:
        return f"Program({self.name}{tuple(self.params)} -> {self.body.render()})"
