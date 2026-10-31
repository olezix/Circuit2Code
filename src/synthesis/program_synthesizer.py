"""From weights and attention patterns to a symbolic program.

The synthesizer runs in two stages.

**Stage 1 -- circuit-grounded algorithm identification.**
The :class:`~src.interpretability.circuit_finder.CircuitSpec` provides (i) the
heads that causally implement the task, (ii) the Fourier support of each head's
OV circuit, and (iii) the attention allocation at the answer position.  From
these features the synthesizer *instantiates* a small DSL of algorithmic motifs
(``a + b``, ``a - b``, ``b - a``, ``a * b``, ``a``, ``b``, constant) and checks
each candidate against the model over the **entire** finite domain.  A motif is
accepted only if it reproduces the model's argmax on every one of the
:math:`p^2` inputs, which is what makes the subsequent proof a *certification*
rather than a statistical claim.

**Stage 2 -- faithful circuit decompilation (always available).**
If no discrete motif is exact, the synthesizer emits the trig circuit itself:

.. math::
    s_{h,j} = \\sum_k a_{h,j,k}\\cos(2\\pi k j/p) + b_{h,j,k}\\sin(2\\pi k j/p)
    \\quad\\Longrightarrow\\quad
    P = \\mathrm{softmax}(s)
    \\quad\\Longrightarrow\\quad
    \\ell_c = \\beta_c + \\sum_{h,j} P_{h,j}\\,g_{h,c}(j)

where :math:`g_{h,c}` is the Fourier series of the head's effective logit map
``W_E W_OV W_U``.  This program is a *structural* decompilation of the network:
it exposes the learned frequencies and routing instead of the raw weight
matrices.

Both outputs are IR :class:`~src.synthesis.ast_nodes.Program` objects that the
code generators and the Z3 verifier consume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.interpretability.circuit_finder import CircuitSpec, extract_circuit
from src.interpretability.circuit_finder import direct_logit_attribution
from src.models.transformer import Transformer
from src.synthesis.ast_nodes import (
    ArgMax,
    BinOp,
    Const,
    Cos,
    Expr,
    Mod,
    Program,
    Sin,
    Linear,
    Softmax,
    SoftmaxComponent,
    Sum,
    Var,
    VectorIndex,
    simplify,
    walk,
)
from src.synthesis.symbolic_reg import (
    FourierFit,
    fit_fourier,
    fit_fourier_greedy,
    fit_periodic_shift,
    fourier_design_matrix,
)

__all__ = [
    "Motif",
    "SynthesisResult",
    "model_predictions",
    "evaluate_program_on_domain",
    "synthesize",
    "build_fourier_circuit",
    "MOTIFS",
]


# ---------------------------------------------------------------------------
# Motif DSL
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Motif:
    """A named candidate algorithm in the discrete DSL."""

    name: str
    description: str
    build: Callable[[int], Expr]
    #: Which circuit features the motif is compatible with.  Used only to order
    #: the candidate list, never to filter it (soundness is decided by the
    #: full-domain check).
    tags: Tuple[str, ...] = ()

    def make(self, p: int, params: Sequence[str] = ("a", "b")) -> Expr:
        return self.build(p)

    def node_count(self, p: int) -> int:
        return sum(1 for _ in walk(self.build(p)))


def _mod(expr: Expr, p: int) -> Expr:
    return Mod(expr, p)


MOTIFS: Tuple[Motif, ...] = (
    Motif(
        "add_mod",
        "(a + b) mod p",
        lambda p: _mod(Var("a") + Var("b"), p),
        ("sum", "shift"),
    ),
    Motif(
        "sub_mod",
        "(a - b) mod p",
        lambda p: _mod(Var("a") - Var("b"), p),
        ("difference",),
    ),
    Motif(
        "sub_mod_rev",
        "(b - a) mod p",
        lambda p: _mod(Var("b") - Var("a"), p),
        ("difference",),
    ),
    Motif(
        "mul_mod",
        "(a * b) mod p",
        lambda p: _mod(Var("a") * Var("b"), p),
        ("product",),
    ),
    Motif("first", "a", lambda p: Var("a"), ("identity",)),
    Motif("second", "b", lambda p: Var("b"), ("identity",)),
)


# ---------------------------------------------------------------------------
# Full-domain evaluation
# ---------------------------------------------------------------------------
def _domain_inputs(p: int) -> Tuple[np.ndarray, np.ndarray]:
    grid = np.arange(p)
    a = np.repeat(grid, p).astype(np.int64)
    b = np.tile(grid, p).astype(np.int64)
    return a, b


@torch.no_grad()
def model_predictions(model: Transformer, p: Optional[int] = None) -> np.ndarray:
    """Model argmax over the complete domain, returned as a ``(p, p)`` array."""
    p = model.config.p if p is None else p
    a, b = _domain_inputs(p)
    logits = model.answer_logits(torch.from_numpy(a), torch.from_numpy(b))
    return logits.argmax(dim=-1).cpu().numpy().reshape(p, p)


def evaluate_program_on_domain(program: Program, p: Optional[int] = None) -> np.ndarray:
    """Evaluate a program over the complete domain, returned as ``(p, p)``."""
    p = program.p if p is None else p
    a, b = _domain_inputs(p)
    values = np.asarray(program.evaluate({"a": a, "b": b}))
    return values.reshape(p, p)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class SynthesisResult:
    """Everything produced by :func:`synthesize`."""

    program: Program
    motif: Optional[str]
    full_domain_accuracy: float
    train_accuracy: float
    test_accuracy: float
    candidate_scores: Dict[str, float] = field(default_factory=dict)
    circuit: Optional[CircuitSpec] = None
    diagnostics: Dict[str, object] = field(default_factory=dict)
    is_faithful_fallback: bool = False

    @property
    def n_nodes(self) -> int:
        return sum(1 for _ in walk(self.program.body))

    @property
    def truncation_error(self) -> float:
        """Worst-case logit error of a faithful circuit (``0`` for a motif)."""
        error = self.program.provenance.get("truncation_error")
        if isinstance(error, dict):
            return float(error.get("logit", 0.0))
        return 0.0

    def narrative(self) -> str:
        if self.motif:
            return (
                f"The circuit computes the discrete algorithm `{self.motif}` "
                f"({self.program.body.render()}); it agrees with the model on "
                f"{self.full_domain_accuracy * 100:.2f}% of the {self.program.p ** 2} "
                f"inputs."
            )
        return (
            "No discrete motif is exact; the faithful trig circuit was emitted "
            f"(agreement {self.full_domain_accuracy * 100:.2f}%)."
        )

    def to_dict(self) -> Dict[str, object]:
        payload = self.program.to_dict()
        payload.update(
            {
                "motif": self.motif,
                "full_domain_accuracy": round(self.full_domain_accuracy, 8),
                "train_accuracy": round(self.train_accuracy, 8),
                "test_accuracy": round(self.test_accuracy, 8),
                "candidate_scores": {k: round(v, 8) for k, v in self.candidate_scores.items()},
                "n_nodes": self.n_nodes,
                "is_faithful_fallback": self.is_faithful_fallback,
                "diagnostics": self.diagnostics,
                "narrative": self.narrative(),
            }
        )
        return payload


# ---------------------------------------------------------------------------
# Circuit diagnostics
# ---------------------------------------------------------------------------
@torch.no_grad()
def _attention_profile(model: Transformer, p: int) -> Dict[str, object]:
    """Mean attention at the answer position, per head and per source position."""
    a, b = _domain_inputs(p)
    tokens = torch.stack(
        [
            torch.from_numpy(a),
            torch.from_numpy(b),
            torch.full((a.shape[0],), p, dtype=torch.long),
        ],
        dim=1,
    )
    _, trace = model(tokens, trace=True)
    assert trace is not None
    profile: Dict[str, object] = {}
    answer = model.config.answer_position
    for layer, pattern in enumerate(trace.attention_patterns):
        # (B, H, T) attention of the answer query over source tokens
        rows = pattern[:, :, answer, :].mean(dim=0).cpu().numpy()  # (H, T)
        profile[f"layer{layer}"] = [
            {"head": h, "attention": [float(v) for v in rows[h]]} for h in range(rows.shape[0])
        ]
    return profile


def _head_features(model: Transformer, circuit: CircuitSpec) -> Dict[str, object]:
    """Summarise each active head: frequency support and shift-invariance."""
    p = circuit.p
    rows: List[Dict[str, object]] = []
    for report in circuit.heads:
        lm = circuit.logit_maps[(report.layer, report.head)]
        shift = fit_periodic_shift(lm[:p, :], p)
        rows.append(
            {
                "layer": report.layer,
                "head": report.head,
                "accuracy_drop": report.accuracy_drop,
                "top_frequencies": [[int(k), float(e)] for k, e in report.top_frequencies],
                "shift_invariant": bool(shift.is_shift_invariant),
                "shift_residual": float(shift.residual_max),
            }
        )
    return {"heads": rows, "is_periodic": circuit.is_periodic, "frequencies": circuit.frequencies}


# ---------------------------------------------------------------------------
# Faithful trig circuit
# ---------------------------------------------------------------------------
def _select_frequencies(
    matrix: np.ndarray,
    p: int,
    *,
    max_freqs: Optional[int],
    tol: float,
) -> Tuple[List[int], float]:
    """Greedy Fourier support of a ``(p, C)`` map, by residual energy.

    Starting from the DC term only, the frequency with the largest residual
    energy (across all columns) is added until the worst-case reconstruction
    error drops below ``tol`` or the basis is exhausted (``K = p//2``).
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    columns = matrix.shape[1]
    limit = p // 2 if max_freqs is None else min(int(max_freqs), p // 2)
    chosen: List[int] = []
    grid = np.arange(p, dtype=np.float64)
    residual_max = float(np.abs(matrix - matrix.mean(axis=0, keepdims=True)).max())
    while len(chosen) < limit:
        fits = [fit_fourier(matrix[:, c], p, chosen) for c in range(columns)]
        residual_max = max(f.residual_max for f in fits)
        if residual_max <= tol:
            break
        recon = np.stack([f.evaluate(grid) for f in fits], axis=1)
        spectrum = np.abs(np.fft.rfft(matrix - recon, axis=0)) ** 2
        energy = spectrum[1:].sum(axis=1)
        if energy.size == 0 or energy.max() <= 0.0:
            break
        # Take the largest residual frequency that is not already in the basis
        # (residual leakage can keep re-proposing an already selected one).
        order = np.argsort(energy)[::-1]
        candidate = next((int(idx) + 1 for idx in order if int(idx) + 1 not in chosen), None)
        if candidate is None:  # pragma: no cover - basis exhausted
            break
        chosen.append(candidate)
    # Report the residual of the *final* basis (the loop's value predates the
    # last addition).
    final = [fit_fourier(matrix[:, c], p, chosen) for c in range(columns)]
    residual_max = max(f.residual_max for f in final) if final else 0.0
    return sorted(chosen), float(residual_max)


def _fourier_readout(matrix: np.ndarray, p: int, freqs: Sequence[int]) -> np.ndarray:
    """Least-squares Fourier readout matrix ``W`` of shape ``(out, 1 + 2|K|)``.

    ``W @ phi(x)`` reproduces the columns of ``matrix`` at every ``x`` in
    :math:`\mathbb{Z}_p`, where ``phi(x) = [1, cos ω_k x, sin ω_k x]_k``.
    """
    design = fourier_design_matrix(p, freqs)              # (p, 1 + 2K)
    coeffs, *_ = np.linalg.lstsq(design, np.asarray(matrix, dtype=np.float64), rcond=None)
    return coeffs.T                                        # (C, 1 + 2K)


def _fourier_features(freqs: Sequence[int], p: int, var: str) -> List[Expr]:
    """IR list ``[1, cos(ω_k var), sin(ω_k var), ...]`` for the readout matrix."""
    features: List[Expr] = [Const(1.0)]
    for k in freqs:
        omega = Const(2.0 * math.pi * int(k) / p)
        angle = omega * Var(var)
        features.append(Cos(angle))
        features.append(Sin(angle))
    return features


def build_fourier_circuit(
    model: Transformer,
    circuit: CircuitSpec,
    *,
    max_freqs: Optional[int] = None,
    tol: float = 1e-10,
) -> Program:
    """Decompile the network into the explicit trig/softmax circuit (Stage 2).

    The emitted program exposes the *learned* structure rather than the raw
    weight matrices:

    .. math::
        \phi(x) = [1, \cos\omega_k x, \sin\omega_k x]_{k \in K}
        \qquad
        G_h(x) = W_h \phi(x)
        \qquad
        \ell_c = \beta_c + \sum_h \sum_{j} P_{h,j}\, G_h(\mathrm{tok}_j)_c

    :math:`W_h` is the head's token-to-class map expressed in the Fourier basis,
    so a grokked network yields a small :math:`|K|` and a memorising network
    yields :math:`|K| = \lfloor p/2 \rfloor` (a full basis, which is exact for
    *any* map on :math:`\mathbb{Z}_p`).  The truncation error is reported in the
    program's provenance.
    """
    cfg = model.config
    p = cfg.p
    if cfg.n_layers != 1:
        raise ValueError("faithful circuit extraction currently supports n_layers == 1")
    if cfg.causal:
        raise ValueError("faithful circuit extraction requires non-causal attention")
    if cfg.mlp_width > 0:
        raise ValueError("faithful circuit extraction supports attention-only models")

    layer = 0
    # Faithfulness requires *every* head: restricting to the causally dominant
    # ones would change the argmax.  The circuit specification is used for the
    # reported frequencies and for interpretation, not to drop computation.
    heads = list(range(cfg.n_heads))
    w_e = model.W_E.detach().double().cpu().numpy()
    w_u = model.W_U.detach().double().cpu().numpy()
    w_q = model.W_Q[layer].detach().double().cpu().numpy()
    w_k = model.W_K[layer].detach().double().cpu().numpy()
    w_v = model.W_V[layer].detach().double().cpu().numpy()
    w_o = model.W_O[layer].detach().double().cpu().numpy()
    scale = 1.0 / math.sqrt(cfg.d_head) if cfg.attn_scale == "sqrt" else 1.0

    # ---- key scores as functions of the source token ---------------------
    eq_vec = w_e[cfg.eq_token]
    score_tables: Dict[int, np.ndarray] = {}
    for head in heads:
        q = eq_vec @ w_q[:, head, :]
        k_all = w_e @ w_k[:, head, :]
        score_tables[head] = scale * (k_all @ q)

    # ---- token-to-class maps and their Fourier readouts ------------------
    logit_maps: Dict[int, np.ndarray] = {}
    for head in heads:
        z = w_e @ w_v[:, head, :]
        y = z @ w_o[:, head, :].T
        logit_maps[head] = y @ w_u

    freq_sets: Dict[int, List[int]] = {}
    readouts: Dict[int, np.ndarray] = {}
    score_fits: Dict[int, FourierFit] = {}
    residual = {"score": 0.0, "logit": 0.0}
    for head in heads:
        score_fits[head] = fit_fourier_greedy(
            score_tables[head][:p], p, max_terms=cfg.p // 2, tol=tol
        )
        residual["score"] = max(residual["score"], score_fits[head].residual_max)
        freqs, logit_residual = _select_frequencies(
            logit_maps[head][:p, :], p, max_freqs=max_freqs, tol=tol
        )
        freq_sets[head] = freqs
        readouts[head] = _fourier_readout(logit_maps[head][:p, :], p, freqs)
        residual["logit"] = max(residual["logit"], logit_residual)

    # ---- assemble the IR -------------------------------------------------
    bias = w_e[cfg.eq_token] @ w_u
    score_exprs: List[List[Expr]] = []
    for head in heads:
        score_exprs.append(
            [
                score_fits[head].to_expr("a"),
                score_fits[head].to_expr("b"),
                Const(float(score_tables[head][cfg.eq_token])),
            ]
        )

    readout_vars: Dict[int, Expr] = {}
    for idx, head in enumerate(heads):
        for var in ("a", "b"):
            readout_vars[(head, var)] = Linear(
                readouts[head], _fourier_features(freq_sets[head], p, var), name=f"G{idx}_{var}"
            )

    logits: List[Expr] = []
    for c in range(p):
        terms: List[Expr] = [Const(float(bias[c]))]
        for head_idx, head in enumerate(heads):
            attention = Softmax(score_exprs[head_idx])
            attn_a = SoftmaxComponent(attention, 0)
            attn_b = SoftmaxComponent(attention, 1)
            attn_eq = SoftmaxComponent(attention, 2)
            terms.append(attn_a * VectorIndex(readout_vars[(head, "a")], c))
            terms.append(attn_b * VectorIndex(readout_vars[(head, "b")], c))
            terms.append(attn_eq * Const(float(logit_maps[head][cfg.eq_token, c])))
        logits.append(Sum(terms))

    body = ArgMax(logits)
    full_basis = any(len(freq_sets[h]) >= p // 2 for h in heads)
    diagnostics = {
        "frequencies": {str(h): list(freq_sets[h]) for h in heads},
        "score_frequencies": {str(h): list(score_fits[h].freqs) for h in heads},
        "max_score_residual": residual["score"],
        "max_logit_residual": residual["logit"],
        "full_basis": full_basis,
        "active_heads": [f"L{circuit.heads[0].layer}H{h}" for h in heads] if heads else [],
    }
    provenance = {
        "kind": "fourier_circuit",
        "frequencies": {str(h): list(freq_sets[h]) for h in heads},
        "truncation_error": residual,
        "full_basis": full_basis,
        "n_readout_parameters": int(sum(r.size for r in readouts.values())),
    }
    return Program(
        name="decompiled_circuit",
        params=["a", "b"],
        body=body,
        p=p,
        docstring=(
            "Trigonometric circuit decompiled from the Transformer's weights:\n"
            "attention is a softmax over Fourier-encoded key scores, and every\n"
            "class logit is a Fourier readout of the attended tokens."
        ),
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def synthesize(
    model: Transformer,
    dataset: Optional[Dict[str, torch.Tensor]] = None,
    circuit: Optional[CircuitSpec] = None,
    *,
    max_freqs: Optional[int] = None,
    tol: float = 1e-10,
    allow_fallback: bool = True,
    ablation_threshold: float = 0.05,
) -> SynthesisResult:
    """Synthesize an IR program from a trained model.

    Parameters
    ----------
    model:
        Trained Transformer (attention-only recommended).
    dataset:
        Dataset dict from :func:`~src.models.transformer.modular_addition_dataset`.
        Used to report train/test agreement; when omitted, agreement is reported
        on the whole domain only.
    circuit:
        A pre-computed :class:`CircuitSpec`; extracted on the fly when omitted.
    allow_fallback:
        When no discrete motif reproduces the model exactly, emit the faithful
        trig circuit instead of raising.
    """
    p = model.config.p
    model.eval()
    if circuit is None:
        if dataset is None:
            from src.models.transformer import modular_addition_dataset

            dataset = modular_addition_dataset(p)
        circuit = extract_circuit(model, dataset, ablation_threshold=ablation_threshold)

    predictions = model_predictions(model, p)
    a, b = _domain_inputs(p)
    env = {"a": a, "b": b}

    candidate_scores: Dict[str, float] = {}
    best: Optional[Tuple[Motif, Expr, Program]] = None
    for motif in MOTIFS:
        expr = motif.build(p)
        values = np.asarray(expr.eval(env)).reshape(p, p)
        accuracy = float((values == predictions).mean())
        candidate_scores[motif.name] = accuracy
        if accuracy == 1.0 and best is None:
            best = (motif, expr, Program(
                name="add_mod_p" if motif.name == "add_mod" else motif.name,
                params=["a", "b"],
                body=expr,
                p=p,
                docstring=f"Decompiled algorithm: {motif.description}.",
                provenance={"kind": "motif", "motif": motif.name},
            ))

    diagnostics: Dict[str, object] = {
        "attention": _attention_profile(model, p),
        "circuit": _head_features(model, circuit),
        "candidate_scores": candidate_scores,
        "protocol": "full-domain exhaustive identification over p*p inputs",
    }

    train_acc = test_acc = 1.0
    if best is not None:
        motif, expr, program = best
        body = simplify(expr)
        program = Program(
            name=program.name,
            params=program.params,
            body=body,
            p=p,
            docstring=program.docstring,
            provenance=program.provenance,
        )
        values = np.asarray(body.eval(env)).reshape(p, p)
        full_acc = float((values == predictions).mean())
        if dataset is not None:
            train_tokens = dataset["tokens"][dataset["train_idx"]]
            test_tokens = dataset["tokens"][dataset["test_idx"]]
            train_a = train_tokens[:, 0].numpy()
            train_b = train_tokens[:, 1].numpy()
            train_target = dataset["target"][dataset["train_idx"]].numpy()
            if test_tokens.numel() == 0:
                # No held-out split (train_fraction == 1.0): reuse the training
                # examples so the reported metrics stay well defined.
                test_a, test_b, test_target = train_a, train_b, train_target
            else:
                test_a = test_tokens[:, 0].numpy()
                test_b = test_tokens[:, 1].numpy()
                test_target = dataset["target"][dataset["test_idx"]].numpy()
            train_acc = float(
                (np.asarray(body.eval({"a": train_a, "b": train_b})) == train_target).mean()
            )
            test_acc = float(
                (np.asarray(body.eval({"a": test_a, "b": test_b})) == test_target).mean()
            )
        return SynthesisResult(
            program=program,
            motif=motif.name,
            full_domain_accuracy=full_acc,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            candidate_scores=candidate_scores,
            circuit=circuit,
            diagnostics=diagnostics,
        )

    if not allow_fallback:
        raise RuntimeError(
            "no discrete motif reproduces the model on the full domain; "
            f"best was {max(candidate_scores, key=candidate_scores.get)} "
            f"with {max(candidate_scores.values()):.4f} accuracy"
        )

    program = build_fourier_circuit(model, circuit, max_freqs=max_freqs, tol=tol)
    values = evaluate_program_on_domain(program, p)
    full_acc = float((values == predictions).mean())
    diagnostics["fourier"] = dict(program.provenance)
    return SynthesisResult(
        program=program,
        motif=None,
        full_domain_accuracy=full_acc,
        train_accuracy=full_acc,
        test_accuracy=full_acc,
        candidate_scores=candidate_scores,
        circuit=circuit,
        diagnostics=diagnostics,
        is_faithful_fallback=True,
    )
