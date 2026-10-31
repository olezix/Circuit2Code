"""Tests of the Z3 translation and the equivalence prover."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import z3

from src.models.transformer import ModelConfig, Transformer, modular_addition_dataset
from src.synthesis.ast_nodes import BinOp, Const, Mod, Program, Var
from src.synthesis.program_synthesizer import synthesize
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
    compute_attention_values,
    compute_logit_bounds,
    rational_ref,
    translate_program,
)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
def test_translate_program_is_exact():
    program = Program("add", ["a", "b"], Mod(Var("a") + Var("b"), 7), 7)
    body, env = translate_program(program, sort="int")
    solver = z3.Solver()
    solver.add(env["a"] == 3, env["b"] == 6)
    solver.check()
    assert solver.model().eval(body).as_long() == 2


def test_translate_rejects_floats_in_integer_mode():
    program = Program("half", ["a"], BinOp("/", Var("a"), Const(2.0)), 7)
    with pytest.raises(NotImplementedError):
        translate_program(program, sort="int")


def test_rational_reference_is_exact():
    value = 0.1
    ref = rational_ref(value)
    solver = z3.Solver()
    solver.add(ref * z3.RealVal(10) == 1)
    assert solver.check() == z3.unsat  # 0.1 is not exactly 1/10 in binary


def test_interval_array_sanity():
    box = IntervalArray(np.zeros((2, 2)), np.ones((2, 2)))
    assert box.shape == (2, 2)
    assert box.width().max() == 1.0
    with pytest.raises(ValueError):
        IntervalArray(np.ones((2,)), np.zeros((2,)))


# ---------------------------------------------------------------------------
# Enclosures
# ---------------------------------------------------------------------------
def test_logit_enclosure_contains_the_float64_forward_pass(medium_model):
    p = medium_model.config.p
    a = np.repeat(np.arange(p), p)
    b = np.tile(np.arange(p), p)
    tokens = torch.stack([torch.from_numpy(a), torch.from_numpy(b), torch.full_like(torch.from_numpy(a), p)], dim=1)
    bounds = compute_logit_bounds(medium_model, tokens.numpy())
    model64 = Transformer(medium_model.config, seed=0).double()
    model64.load_state_dict({k: v.double() for k, v in medium_model.state_dict().items()})
    with torch.no_grad():
        logits = model64(tokens)[0][:, -1, :].numpy()
    lo = bounds.logits.lo[:, -1, :]
    hi = bounds.logits.hi[:, -1, :]
    assert np.all(lo <= logits) and np.all(hi >= logits)
    assert (hi - lo).max() < 1e-10


def test_attention_enclosure_contains_the_softmax(medium_model):
    p = medium_model.config.p
    a = np.repeat(np.arange(p), p)
    b = np.tile(np.arange(p), p)
    tokens = np.stack([a, b, np.full_like(a, p)], axis=1)
    bounds = compute_logit_bounds(medium_model, tokens)
    attention = compute_attention_values(medium_model, tokens)
    assert np.all(bounds.attention.lo <= attention)
    assert np.all(bounds.attention.hi >= attention)


def test_enclosure_is_deterministic(medium_model):
    p = medium_model.config.p
    tokens = np.stack([np.zeros(p, int), np.arange(p), np.full(p, p)], axis=1)
    first = compute_logit_bounds(medium_model, tokens).logits
    second = compute_logit_bounds(medium_model, tokens).logits
    assert np.array_equal(first.lo, second.lo)
    assert np.array_equal(first.hi, second.hi)


# ---------------------------------------------------------------------------
# Specification proofs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "motif,body",
    [
        ("add_mod", Mod(Var("a") + Var("b"), 7)),
        ("sub_mod", Mod(Var("a") - Var("b"), 7)),
        ("mul_mod", Mod(Var("a") * Var("b"), 7)),
        ("first", Var("a")),
        ("second", Var("b")),
    ],
)
def test_spec_proofs_succeed(small_model, motif, body):
    program = Program("f", ["a", "b"], body, 7, provenance={"motif": motif})
    verifier = Verifier(small_model, program, VerifyConfig(check_model_equivalence=False))
    result = verifier.verify_spec(ModuloAdditionSpec.from_program(program))
    assert result.status == "proved"
    assert "proved" in result.detail


def test_spec_detects_wrong_program(small_model):
    # An "addition" spec applied to a subtraction program must be refuted.
    program = Program(
        "sub", ["a", "b"], Mod(Var("a") - Var("b"), 7), 7, provenance={"motif": "sub_mod"}
    )
    verifier = Verifier(small_model, program)
    result = verifier.verify_spec(ModuloAdditionSpec(p=7, kind="add"))
    assert result.status == "refuted"
    assert result.counterexample is not None
    assert set(result.counterexample) == {"a", "b"}


def test_spec_is_skipped_for_real_valued_programs(medium_model, medium_dataset):
    fallback = synthesize(medium_model, medium_dataset)
    verifier = Verifier(medium_model, fallback.program)
    result = verifier.verify_spec()
    assert result.status in ("skipped", "unknown")


# ---------------------------------------------------------------------------
# Model equivalence
# ---------------------------------------------------------------------------
def test_model_equivalence_for_a_wrong_program(medium_model):
    program = Program("first", ["a", "b"], Var("a"), 11, provenance={"motif": "first"})
    verifier = Verifier(medium_model, program, VerifyConfig(check_spec=False))
    result = verifier.verify_model_equivalence()
    assert result.status == "refuted"
    assert result.counterexample is not None
    a, b = result.counterexample["a"], result.counterexample["b"]
    with torch.no_grad():
        logits = medium_model.answer_logits(torch.tensor([a]), torch.tensor([b]))
    assert int(logits.argmax()) != a


def test_model_equivalence_for_the_faithful_circuit(medium_model, medium_dataset):
    synthesis = synthesize(medium_model, medium_dataset)
    verifier = Verifier(medium_model, synthesis.program, VerifyConfig(batch_size=64))
    result = verifier.verify_model_equivalence()
    assert result.status == "proved", result.detail
    assert result.statistics["inputs"] == 121.0
    assert result.solver_time_s >= 0.0


def test_model_equivalence_uses_smt_under_attention_slack():
    """Widened enclosures force the prover through the SMT path.

    ``attention_slack`` proves the stronger statement that the argmax is stable
    under that much perturbation of the attention distribution, so every input
    must be decided by Z3 rather than by the (tight) float enclosure.
    """
    cfg = ModelConfig(p=5, d_model=16, n_heads=2, init_std=0.2)
    model = Transformer(cfg, seed=3).eval()
    dataset = modular_addition_dataset(5, train_fraction=0.6, seed=0)
    synthesis = synthesize(model, dataset)
    verifier = Verifier(
        model, synthesis.program, VerifyConfig(batch_size=25, attention_slack=0.2)
    )
    result = verifier.verify_model_equivalence()
    assert result.status == "proved", result.detail
    assert result.statistics["inputs_smt"] > 0
    assert result.statistics["inputs_smt"] + result.statistics["inputs_enclosure_certified"] == 25.0


def test_logit_equivalence_proof(medium_model, medium_dataset):
    synthesis = synthesize(medium_model, medium_dataset)
    verifier = Verifier(medium_model, synthesis.program, VerifyConfig())
    result = verifier.verify_logit_equivalence(eps=1e-3)
    assert result.status == "proved", result.detail


def test_logit_equivalence_detects_too_tight_bound(medium_model, medium_dataset):
    from src.synthesis.program_synthesizer import build_fourier_circuit

    circuit = synthesize(medium_model, medium_dataset).circuit
    truncated = build_fourier_circuit(medium_model, circuit, max_freqs=1)
    verifier = Verifier(medium_model, truncated, VerifyConfig())
    assert verifier.verify_logit_equivalence(eps=1e-9).status == "refuted"
    assert verifier.verify_logit_equivalence(eps=1.0).status == "proved"


@pytest.mark.parametrize(
    "kind,body",
    [
        ("add", Mod(Var("a") + Var("b"), 7)),
    ],
)
def test_full_report_add_motif(addition_model, kind, body):
    program = Program(
        "add_mod_p", ["a", "b"], body, 7, provenance={"motif": "add_mod", "kind": "motif"}
    )
    verifier = Verifier(addition_model, program)
    report = verifier.verify_all()
    assert isinstance(report, VerificationReport)
    assert report.verified, report.summary()
    assert report.deployment_gap >= 0.0
    payload = report.to_dict()
    assert payload["verified"] is True
    assert payload["motif"] == "add_mod"


def test_synthesize_and_verify_end_to_end(addition_model):
    from src.models.transformer import modular_addition_dataset

    dataset = modular_addition_dataset(7, train_fraction=1.0, seed=0)
    synthesis, report = synthesize_and_verify(
        addition_model, dataset, config=VerifyConfig(batch_size=64)
    )
    assert synthesis.motif == "add_mod"
    assert report.verified
    assert report.results[0].status == "proved"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def addition_model():
    from src.models.train import TrainConfig, train_modular_addition

    result = train_modular_addition(
        TrainConfig(
            p=7,
            d_model=32,
            n_heads=2,
            init_std=0.5,
            epochs=1500,
            lr=5e-3,
            weight_decay=1e-3,
            eval_every=500,
            log_every=10_000,
            train_fraction=1.0,
            post_grok_epochs=1,
            seed=0,
            verbose=False,
            torch_threads=1,
        )
    )
    return result.model


def test_attention_slack_reports_a_robustness_counterexample():
    """A witness under slack refutes the *robustness* claim, it is not 'unknown'."""
    cfg = ModelConfig(p=11, d_model=32, n_heads=2, init_std=0.5)
    model = Transformer(cfg, seed=42).eval()
    dataset = modular_addition_dataset(11, train_fraction=0.5, seed=42)
    synthesis = synthesize(model, dataset)
    verifier = Verifier(model, synthesis.program, VerifyConfig(attention_slack=0.2))
    result = verifier.verify_model_equivalence()
    assert result.status == "refuted", result.detail
    assert "not stable" in result.detail


def test_attention_slack_pruning_radius_is_tight():
    """The pruning radius must scale with the true logit sensitivity, not with p."""
    model = Transformer(ModelConfig(p=11, d_model=32, n_heads=2, init_std=0.5), seed=42).eval()
    dataset = modular_addition_dataset(11, train_fraction=0.5, seed=42)
    synthesis = synthesize(model, dataset)
    tight = Verifier(model, synthesis.program, VerifyConfig(attention_slack=0.0))
    loose = Verifier(model, synthesis.program, VerifyConfig(attention_slack=0.01))
    assert tight._max_logit_span(tight.bounds()) == 0.0
    span = loose._max_logit_span(loose.bounds())
    assert 0.0 < span < 1e4  # finite, and far below slack * p * H * max|M|
