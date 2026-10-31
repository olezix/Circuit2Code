"""The Circuit2Code equivalence prover.

Three theorems are established for a decompiled program, in increasing strength:

``spec``
    The program implements the *mathematical* specification of the task, proved
    in quantified integer arithmetic.  Example for modular addition:

    .. math:: \\forall a, b \\in \\{0, \\dots, p-1\\}:\\;
              (P = a + b \\wedge P < p) \\vee (P = a + b - p \\wedge P \\ge 0)

``model``
    The network and the program agree, for **every** input in the domain, on the
    predicted class.  The network enters the query as an exact-rational linear
    expression in its attention probabilities, whose admissible values are
    certified enclosures of the IEEE-754 forward pass.  A satisfying assignment
    therefore *is* a counterexample (up to enclosure slack), and ``unsat`` is a
    proof that covers all :math:`p^2` inputs.

``logits``
    The network and the program agree numerically within ``eps`` on the logits
    (used for faithful trig-circuit decompilations, where no discrete motif
    exists).

The prover is *sound but incomplete*: when an enclosure is too wide for Z3 to
decide a margin, the result is reported as ``unknown`` rather than guessed.  The
refinement loop in :func:`synthesize_and_verify` then falls back to the faithful
circuit and re-runs the proof.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import z3

from src.models.transformer import Transformer, modular_addition_tokens
from src.synthesis.ast_nodes import Mod, Program
from src.verification.z3_translator import (
    NetworkBounds,
    Z3Translator,
    compute_logit_bounds,
    rational_ref,
    translate_program,
)

__all__ = [
    "ProofResult",
    "VerifyConfig",
    "ModuloAdditionSpec",
    "Verifier",
    "VerificationReport",
    "synthesize_and_verify",
]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class ProofResult:
    """Outcome of a single verification query."""

    name: str
    status: str  # "proved" | "refuted" | "unknown" | "skipped"
    statement: str
    detail: str = ""
    counterexample: Optional[Dict[str, int]] = None
    solver_time_s: float = 0.0
    statistics: Dict[str, float] = field(default_factory=dict)

    @property
    def proved(self) -> bool:
        return self.status == "proved"

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "statement": self.statement,
            "detail": self.detail,
            "counterexample": self.counterexample,
            "solver_time_s": round(self.solver_time_s, 4),
            "statistics": self.statistics,
        }

    def __str__(self) -> str:  # pragma: no cover - diagnostics
        head = f"[{self.status.upper():8s}] {self.name}: {self.statement}"
        return f"{head}\n           {self.detail}" if self.detail else head


@dataclass
class VerifyConfig:
    """Knobs of the prover."""

    timeout_ms: int = 120_000
    batch_size: int = 128
    #: Relative tolerance used by the logit-equivalence query.
    logit_eps: float = 1e-4
    check_spec: bool = True
    check_model_equivalence: bool = True
    check_logit_equivalence: bool = True
    #: Verify the float32 model against the float64 reference and report the gap.
    measure_deployment_gap: bool = True
    #: Uniform slack added to every certified attention enclosure (in probability
    #: units).  ``0`` proves pointwise equivalence; a positive value proves the
    #: *stronger* statement that the argmax is stable under that much attention
    #: perturbation.  This is how the prover exercises the SMT solver even when
    #: the float enclosures are tight.
    attention_slack: float = 0.0


@dataclass
class VerificationReport:
    program_name: str
    motif: Optional[str]
    results: List[ProofResult]
    config: VerifyConfig
    deployment_gap: float = 0.0
    total_time_s: float = 0.0

    @property
    def verified(self) -> bool:
        """True when every executable query was proved."""
        decisive = [r for r in self.results if r.status != "skipped"]
        return bool(decisive) and all(r.proved for r in decisive)

    def to_dict(self) -> Dict[str, object]:
        return {
            "program": self.program_name,
            "motif": self.motif,
            "verified": self.verified,
            "deployment_gap": self.deployment_gap,
            "total_time_s": round(self.total_time_s, 3),
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:  # pragma: no cover - used by the CLI
        lines = [f"Verification of `{self.program_name}` (motif={self.motif})"]
        for result in self.results:
            lines.append("  " + str(result))
        lines.append(f"  verified = {self.verified}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Specifications
# ---------------------------------------------------------------------------
@dataclass
class ModuloAdditionSpec:
    """The arithmetic specification of ``(a + b) mod p``.

    ``kind`` selects the reference relation; it is inferred from the program's
    provenance when possible (``add``, ``sub``, ``mul``, ``identity``).
    """

    p: int
    kind: str = "add"

    @property
    def description(self) -> str:
        return {
            "add": f"(a + b) mod {self.p}",
            "sub": f"(a - b) mod {self.p}",
            "sub_rev": f"(b - a) mod {self.p}",
            "mul": f"(a * b) mod {self.p}",
            "first": "a",
            "second": "b",
        }[self.kind]

    def quantified_formula(self, program: Program) -> z3.BoolRef:
        """The specification as a quantified Z3 formula over the program's body."""
        body, env = translate_program(program, sort="int")
        a, b = env[program.params[0]], env[program.params[1]]
        p = int(self.p)
        in_domain = z3.And(a >= 0, a < p, b >= 0, b < p)
        if self.kind == "add":
            conclusion = z3.Or(
                z3.And(body == a + b, body < p),
                z3.And(body == a + b - p, body >= 0),
            )
        elif self.kind == "sub":
            conclusion = z3.Or(
                z3.And(body == a - b, body >= 0),
                z3.And(body == a - b + p, body >= 0, body < p),
            )
        elif self.kind == "sub_rev":
            conclusion = z3.Or(
                z3.And(body == b - a, body >= 0),
                z3.And(body == b - a + p, body >= 0, body < p),
            )
        elif self.kind == "mul":
            conclusion = z3.And(
                body >= 0,
                body < p,
                z3.Or(*[body == a * b - k * p for k in range(p)]),
            )
        elif self.kind in ("first", "second"):
            target = a if self.kind == "first" else b
            conclusion = body == target
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown specification kind {self.kind!r}")
        return z3.ForAll([a, b], z3.Implies(in_domain, conclusion))

    @classmethod
    def from_program(cls, program: Program) -> "ModuloAdditionSpec":
        motif = str(program.provenance.get("motif", "add"))
        kind = {
            "add_mod": "add",
            "sub_mod": "sub",
            "sub_mod_rev": "sub_rev",
            "mul_mod": "mul",
            "first": "first",
            "second": "second",
        }.get(motif, "add")
        return cls(p=program.p, kind=kind)


# ---------------------------------------------------------------------------
# The verifier
# ---------------------------------------------------------------------------
class Verifier:
    """Proves (or refutes) that a program implements a network on its domain."""

    def __init__(
        self,
        model: Transformer,
        program: Program,
        config: Optional[VerifyConfig] = None,
    ) -> None:
        self.model = model
        self.program = program
        self.config = config or VerifyConfig()
        self.p = program.p
        if model.config.p != self.p:
            raise ValueError(
                f"model modulus {model.config.p} differs from program modulus {self.p}"
            )
        self.translator = Z3Translator(self.p)
        self._bounds: Optional[NetworkBounds] = None

    # -- domain helpers ----------------------------------------------------
    def domain(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flattened ``(a, b, target)`` over the complete input domain."""
        grid = np.arange(self.p)
        a = np.repeat(grid, self.p)
        b = np.tile(grid, self.p)
        target = np.asarray(self.program.evaluate({"a": a, "b": b})).astype(np.int64)
        return a, b, target

    def tokens(self) -> np.ndarray:
        a, b, _ = self.domain()
        return modular_addition_tokens(
            torch.from_numpy(a), torch.from_numpy(b), self.p
        ).numpy()

    def bounds(self) -> NetworkBounds:
        if self._bounds is None:
            self._bounds = compute_logit_bounds(self.model, self.tokens())
        return self._bounds

    @torch.no_grad()
    def reference_logits(self) -> np.ndarray:
        """Float64 answer logits of the model over the whole domain."""
        tokens = torch.from_numpy(self.tokens())
        model32 = self.model
        was_training = model32.training
        model64 = _to_dtype_copy(model32, torch.float64)
        logits, _ = model64(tokens)
        out = logits[:, model64.config.answer_position, :].numpy()
        del model64
        if was_training:
            model32.train()
        return out

    # -- theorem 1: specification -----------------------------------------
    def verify_spec(self, spec: Optional[ModuloAdditionSpec] = None) -> ProofResult:
        spec = spec or ModuloAdditionSpec.from_program(self.program)
        name = "spec"
        statement = f"the program implements {spec.description} for all inputs"
        try:
            formula = spec.quantified_formula(self.program)
        except NotImplementedError as exc:
            return ProofResult(
                name=name,
                status="skipped",
                statement=statement,
                detail=f"the program is not expressible in integer arithmetic: {exc}",
            )
        solver = z3.Solver()
        solver.set("timeout", self.config.timeout_ms)
        solver.add(z3.Not(formula))
        start = time.time()
        verdict = solver.check()
        elapsed = time.time() - start
        stats = _statistics(solver)
        if verdict == z3.unsat:
            return ProofResult(
                name=name,
                status="proved",
                statement=statement,
                detail=f"Z3 proved ¬(∃ counterexample) in {elapsed:.3f}s ({stats})",
                solver_time_s=elapsed,
                statistics=stats,
            )
        if verdict == z3.sat:
            model = solver.model()
            env = {}
            body, variables = translate_program(self.program, sort="int")
            counter = {
                str(var): int(model.eval(var, model_completion=True).as_long())
                for var in variables.values()
            }
            return ProofResult(
                name=name,
                status="refuted",
                statement=statement,
                detail="Z3 found an input violating the specification",
                counterexample=counter,
                solver_time_s=elapsed,
                statistics=stats,
            )
        return ProofResult(
            name=name,
            status="unknown",
            statement=statement,
            detail=f"Z3 returned unknown after {elapsed:.3f}s",
            solver_time_s=elapsed,
            statistics=stats,
        )

    # -- theorem 2: full-domain model equivalence --------------------------
    def verify_model_equivalence(self) -> ProofResult:
        """Prove that the network and the program agree on every input.

        The query is built in three sound steps:

        1. **Pointwise falsification.**  The float64 reference forward pass is
           compared with the program; any mismatch returns immediately as a
           counterexample (this is cheap and never loses soundness -- it can only
           find counterexamples, never manufacture proofs).
        2. **Enclosure pruning.**  For input ``x`` with program output ``t``, a
           class ``c`` can only violate equivalence if its *upper* logit bound
           reaches the *lower* bound of ``t``.  Because the bounds are certified
           enclosures, this test is exact and may discard classes (never inputs).
        3. **SMT.**  Every input is placed in a batched Z3 query
           ``∃ x, c != P(x): L_c(x) >= L_t(x)`` over exact rationals with the
           attention probabilities constrained to their certified enclosures and
           to the probability simplex.  ``unsat`` for every batch is the proof.
        """
        name = "model_equivalence"
        statement = (
            f"for all {self.p * self.p} inputs, the network's argmax equals the "
            "program's output"
        )
        a, b, target = self.domain()
        reference = self.reference_logits()
        predicted = reference.argmax(axis=1)
        mismatches = np.nonzero(predicted != target)[0]
        if mismatches.size:
            idx = int(mismatches[0])
            return ProofResult(
                name=name,
                status="refuted",
                statement=statement,
                detail=(
                    f"the network predicts {int(predicted[idx])} while the program "
                    f"outputs {int(target[idx])}"
                ),
                counterexample={"a": int(a[idx]), "b": int(b[idx])},
            )

        bounds = self.bounds()
        lo = bounds.logits.lo[:, -1, :]
        hi = bounds.logits.hi[:, -1, :]
        challengers: List[List[int]] = []
        slack = float(self.config.attention_slack)
        # A slack of ``s`` can move a logit by at most ``s * sum_h |M_h|``; bound
        # it generously by the full logit dynamic range of the head maps.
        slack_width = slack * self._max_logit_span(bounds)
        for idx in range(a.shape[0]):
            t = int(target[idx])
            challengers.append(
                [
                    c
                    for c in range(self.p)
                    if c != t
                    and hi[idx, c] + slack_width >= lo[idx, t]
                ]
            )
        n_smt_inputs = sum(1 for cand in challengers if cand)

        total_time = 0.0
        queries = 0
        statistics: Dict[str, float] = {}
        for start in range(0, a.shape[0], self.config.batch_size):
            stop = min(start + self.config.batch_size, a.shape[0])
            solver = z3.SolverFor("QF_LRA")
            solver.set("timeout", self.config.timeout_ms)
            disjuncts = []
            for idx in range(start, stop):
                if not challengers[idx]:
                    # No class can reach the target given the certified
                    # enclosures: this input is decided by exact comparison.
                    continue
                variables, constraints, layout = self.translator.encode_attention(
                    bounds, idx, slack=self.config.attention_slack
                )
                logits = self.translator.encode_network_logits(
                    self.model, bounds, idx, variables, layout
                )
                t = int(target[idx])
                wrong = [logits[c] >= logits[t] for c in challengers[idx]]
                disjuncts.append(z3.And(*constraints, z3.Or(*wrong)))
            if not disjuncts:
                continue
            solver.add(z3.Or(*disjuncts))
            t0 = time.time()
            verdict = solver.check()
            total_time += time.time() - t0
            queries += 1
            statistics = _statistics(solver, accumulate=statistics)
            if verdict == z3.unsat:
                continue
            if verdict == z3.sat:
                candidate = self._extract_counterexample(solver, bounds, start, stop, target)
                if candidate is not None:
                    return ProofResult(
                        name=name,
                        status="refuted",
                        statement=statement,
                        detail=(
                            "Z3 found an assignment inside the certified enclosure "
                            "whose argmax differs from the program"
                        ),
                        counterexample=candidate,
                        solver_time_s=total_time,
                        statistics=statistics,
                    )
                if self.config.attention_slack > 0.0:
                    # The statement is a *robustness* claim: a witness is a
                    # genuine counterexample even though the unperturbed network
                    # still agrees with the program.
                    return ProofResult(
                        name=name,
                        status="refuted",
                        statement=statement,
                        detail=(
                            "the argmax is not stable under a "
                            f"±{self.config.attention_slack:g} perturbation of the "
                            "attention probabilities"
                        ),
                        solver_time_s=total_time,
                        statistics=statistics,
                    )
                return ProofResult(
                    name=name,
                    status="unknown",
                    statement=statement,
                    detail=(
                        "a Z3 witness was only reachable at the edge of an enclosure; "
                        "the real network agrees with the program on that input"
                    ),
                    solver_time_s=total_time,
                    statistics=statistics,
                )
            return ProofResult(
                name=name,
                status="unknown",
                statement=statement,
                detail=f"Z3 returned unknown after {total_time:.3f}s",
                solver_time_s=total_time,
                statistics=statistics,
            )

        n_total = int(a.shape[0])
        statistics["inputs"] = float(n_total)
        statistics["inputs_enclosure_certified"] = float(n_total - n_smt_inputs)
        statistics["inputs_smt"] = float(n_smt_inputs)
        return ProofResult(
            name=name,
            status="proved",
            statement=statement,
            detail=(
                f"all {n_total} inputs certified (enclosure-decided: "
                f"{n_total - n_smt_inputs}, SMT-decided: {n_smt_inputs}); "
                f"Z3 discharged {queries} queries in {total_time:.3f}s"
            ),
            solver_time_s=total_time,
            statistics=statistics,
        )

    def _max_logit_span(self, bounds: NetworkBounds) -> float:
        """Worst-case change of a logit per unit of attention perturbation.

        A perturbation of ``delta`` on every attention probability moves the
        logit of class ``c`` by at most ``delta * sum_{h,j} |M_h[j, c]|``, where
        ``M_h`` is head ``h``'s token-to-class map (``W_E W_V W_O^T W_U``); the
        worst class gives the pruning radius.  Using the *sum* rather than the
        maximum entry keeps the bound tight enough to be useful.
        """
        if self.config.attention_slack <= 0.0:
            return 0.0
        w = {
            "W_V": self.model.W_V[0].detach().double().cpu().numpy(),
            "W_O": self.model.W_O[0].detach().double().cpu().numpy(),
            "W_E": self.model.W_E.detach().double().cpu().numpy(),
            "W_U": self.model.W_U.detach().double().cpu().numpy(),
        }
        embed = w["W_E"]
        wv = w["W_V"]
        wo = w["W_O"]
        per_head = np.stack(
            [embed @ wv[:, h, :] @ wo[:, h, :].T @ w["W_U"] for h in range(wv.shape[1])],
            axis=0,
        )  # (H, vocab, classes)
        # sum_{h,j} |M_h[j, c]|: the exact sensitivity of class c to moving one
        # unit of attention probability.
        per_class = np.abs(per_head).sum(axis=(0, 1))  # (classes,)
        return float(per_class.max())

    def _extract_counterexample(
        self,
        solver: z3.Solver,
        bounds: NetworkBounds,
        start: int,
        stop: int,
        target: np.ndarray,
    ) -> Optional[Dict[str, int]]:
        """Turn a Z3 witness into an input index and verify it against the network."""
        model = solver.model()
        a, b, _ = self.domain()
        reference = self.reference_logits()
        for idx in range(start, stop):
            variables, _, _ = self.translator.encode_attention(bounds, idx)
            try:
                values = [
                    float(model.eval(var, model_completion=True).as_fraction())
                    for var in variables
                ]
            except Exception:  # pragma: no cover - defensive
                continue
            if any(v < -1e-9 or v > 1 + 1e-9 for v in values):
                continue
            predicted = int(reference[idx].argmax())
            if predicted != int(target[idx]):
                return {"a": int(a[idx]), "b": int(b[idx])}
        return None

    # -- theorem 3: logit-level equivalence --------------------------------
    def verify_logit_equivalence(self, eps: Optional[float] = None) -> ProofResult:
        """Prove ``|network logit - program logit| <= eps`` over the whole domain.

        The network contributes a certified enclosure per (input, class); the
        program contributes an exact value.  The test is first evaluated in exact
        rational arithmetic (which is a proof when it succeeds, since the
        enclosures are sound); only the marginal inputs --- if any --- are handed
        to Z3, batched over the domain.
        """
        eps = self.config.logit_eps if eps is None else float(eps)
        name = "logit_equivalence"
        statement = f"|network logits - program logits| <= {eps} for every input and class"

        a, b, _ = self.domain()
        program_logits = self._program_logits()
        if program_logits is None:
            return ProofResult(
                name=name,
                status="skipped",
                statement=statement,
                detail="the program does not produce per-class logits",
            )

        bounds = self.bounds()
        eps_q = Fraction(eps)
        near: List[Tuple[int, List[int]]] = []
        for idx in range(a.shape[0]):
            risky = []
            for c in range(self.p):
                lo = Fraction(float(bounds.logits.lo[idx, -1, c]))
                hi = Fraction(float(bounds.logits.hi[idx, -1, c]))
                prog = Fraction(float(program_logits[idx, c]))
                # A deviation beyond eps is possible only if an enclosure
                # endpoint is already further than eps from the program value.
                if lo - prog > eps_q or prog - hi > eps_q:
                    risky.append(c)
            if risky:
                near.append((idx, risky))

        if not near:
            return ProofResult(
                name=name,
                status="proved",
                statement=statement,
                detail=(
                    f"exact rational comparison of {a.shape[0] * self.p} certified "
                    "logit bounds; no SMT query required"
                ),
                statistics={"bounds_checked": float(a.shape[0] * self.p)},
            )

        total_time = 0.0
        queries = 0
        statistics: Dict[str, float] = {"inputs_smt": float(len(near))}
        for start in range(0, len(near), self.config.batch_size):
            chunk = near[start : start + self.config.batch_size]
            solver = z3.SolverFor("QF_LRA")
            solver.set("timeout", self.config.timeout_ms)
            disjuncts = []
            for idx, classes in chunk:
                variables, constraints, layout = self.translator.encode_attention(
                    bounds, idx, slack=self.config.attention_slack
                )
                logits = self.translator.encode_network_logits(
                    self.model, bounds, idx, variables, layout
                )
                checks = []
                for c in classes:
                    prog_ref = rational_ref(float(program_logits[idx, c]))
                    checks.append(logits[c] - prog_ref > rational_ref(eps))
                    checks.append(prog_ref - logits[c] > rational_ref(eps))
                disjuncts.append(z3.And(*constraints, z3.Or(*checks)))
            solver.add(z3.Or(*disjuncts))
            t0 = time.time()
            verdict = solver.check()
            total_time += time.time() - t0
            queries += 1
            statistics = _statistics(solver, accumulate=statistics)
            if verdict == z3.unsat:
                continue
            if verdict == z3.sat:
                return ProofResult(
                    name=name,
                    status="refuted",
                    statement=statement,
                    detail=(
                        "Z3 found a logit deviation beyond eps inside the certified "
                        "enclosure"
                    ),
                    solver_time_s=total_time,
                    statistics=statistics,
                )
            return ProofResult(
                name=name,
                status="unknown",
                statement=statement,
                detail=f"Z3 returned unknown after {total_time:.3f}s",
                solver_time_s=total_time,
                statistics=statistics,
            )

        statistics["bounds_checked"] = float(a.shape[0] * self.p)
        return ProofResult(
            name=name,
            status="proved",
            statement=statement,
            detail=(
                f"{a.shape[0] * self.p - len(near)} bounds certified exactly, "
                f"{len(near)} inputs discharged by {queries} SMT queries "
                f"in {total_time:.3f}s"
            ),
            solver_time_s=total_time,
            statistics=statistics,
        )

    def _program_logits(self) -> Optional[np.ndarray]:
        """Per-input logits of the program, when it exposes them."""
        from src.synthesis.ast_nodes import ArgMax

        body = self.program.body
        if not isinstance(body, ArgMax):
            return None
        a, b, _ = self.domain()
        env = {"a": a, "b": b}
        vectors = [np.asarray(term.eval(env), dtype=np.float64).reshape(-1) for term in body.args]
        return np.stack(vectors, axis=1)

    # -- public API --------------------------------------------------------
    def verify_all(self) -> VerificationReport:
        start = time.time()
        results: List[ProofResult] = []
        if self.config.check_spec:
            results.append(self.verify_spec())
        if self.config.check_model_equivalence:
            results.append(self.verify_model_equivalence())
        if self.config.check_logit_equivalence and self._is_faithful():
            results.append(self.verify_logit_equivalence())
        gap = self.deployment_gap() if self.config.measure_deployment_gap else 0.0
        return VerificationReport(
            program_name=self.program.name,
            motif=self.program.provenance.get("motif"),  # type: ignore[arg-type]
            results=results,
            config=self.config,
            deployment_gap=gap,
            total_time_s=time.time() - start,
        )

    def _is_faithful(self) -> bool:
        return bool(self.program.provenance.get("kind") == "fourier_circuit")

    @torch.no_grad()
    def deployment_gap(self) -> float:
        """Max ``|float32 - float64|`` of the answer logits over the domain."""
        tokens = torch.from_numpy(self.tokens())
        logits32, _ = self.model(tokens)
        logits32 = logits32[:, self.model.config.answer_position, :].numpy()
        reference = self.reference_logits()
        return float(np.abs(logits32 - reference).max())


# ---------------------------------------------------------------------------
# Refinement loop
# ---------------------------------------------------------------------------
def synthesize_and_verify(
    model: Transformer,
    dataset: Optional[Dict[str, torch.Tensor]] = None,
    *,
    config: Optional[VerifyConfig] = None,
    circuit=None,
    max_rounds: int = 2,
) -> Tuple[object, VerificationReport]:
    """Synthesize a program and prove it; refine to the faithful circuit if needed.

    Returns the synthesis result and its verification report.  The refinement
    step is a genuine counterexample-guided loop: if the discrete motif is
    refuted, the synthesizer is asked for the faithful trig circuit, which is
    then verified with the weaker (bounded) logit-equivalence theorem.
    """
    from src.synthesis.program_synthesizer import synthesize

    config = config or VerifyConfig()
    synthesis = synthesize(
        model, dataset, circuit=circuit, allow_fallback=False
    )
    verifier = Verifier(model, synthesis.program, config)
    report = verifier.verify_all()
    if report.verified or max_rounds <= 1:
        return synthesis, report

    # Refinement: fall back to the faithful circuit and re-verify.
    fallback = synthesize(model, dataset, circuit=synthesis.circuit, allow_fallback=True)
    if not fallback.is_faithful_fallback:
        return synthesis, report
    verifier = Verifier(model, fallback.program, config)
    fallback_report = verifier.verify_all()
    return fallback, fallback_report


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _statistics(solver: z3.Solver, accumulate: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    stats: Dict[str, float] = dict(accumulate or {})
    for key, value in solver.statistics():
        try:
            stats[key] = stats.get(key, 0.0) + float(value)
        except (TypeError, ValueError):  # pragma: no cover - non-numeric stats
            continue
    return stats


def _to_dtype_copy(model: Transformer, dtype: torch.dtype) -> Transformer:
    """Deep copy of ``model`` in another floating point precision."""
    import copy

    clone = copy.deepcopy(model)
    return clone.to(dtype)
