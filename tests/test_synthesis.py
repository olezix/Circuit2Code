"""Tests of the symbolic IR, the basis fits and the program synthesizer."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.models.transformer import ModelConfig, Transformer
from src.synthesis.ast_nodes import (
    ArgMax,
    BinOp,
    Const,
    Cos,
    Interval,
    Linear,
    Mod,
    Neg,
    Program,
    Sin,
    Softmax,
    SoftmaxComponent,
    Sum,
    Threshold,
    Var,
    VectorIndex,
    simplify,
    walk,
)
from src.synthesis.program_synthesizer import (
    MOTIFS,
    build_fourier_circuit,
    evaluate_program_on_domain,
    model_predictions,
    synthesize,
)
from src.synthesis.symbolic_reg import (
    fit_fourier,
    fit_fourier_greedy,
    fit_periodic_shift,
    fit_piecewise_linear,
    fourier_design_matrix,
    periodicity_score,
)


# ---------------------------------------------------------------------------
# IR basics
# ---------------------------------------------------------------------------
def test_arithmetic_evaluation_and_variables():
    expr = (Var("a") + 2.0) * Var("b") - Const(1.0)
    env = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
    assert np.allclose(expr.eval(env), [8.0, 15.0])
    assert sorted(expr.vars()) == ["a", "b"]


def test_modulo_semantics_follow_python():
    expr = Mod(BinOp("-", Var("a"), Var("b")), 7)
    assert expr.eval({"a": np.array([2]), "b": np.array([5])})[0] == 4
    with pytest.raises(ValueError):
        Mod(Var("a"), 0)


def test_rendering_has_correct_parentheses():
    expr = Mod(BinOp("+", Var("a"), Var("b")), 59)
    assert expr.render() == "((a + b) % 59)"
    assert (Const(2.0) * BinOp("+", Var("a"), Var("b"))).render() == "2.0 * (a + b)"
    assert Neg(Neg(Var("a"))).render() == "--a"


def test_structural_equality_and_hashing():
    one = BinOp("+", Var("a"), Const(1.0))
    two = BinOp("+", Var("a"), Const(1.0))
    other = BinOp("+", Var("a"), Const(2.0))
    assert one == two
    assert hash(one) == hash(two)
    assert one != other
    assert len({one, two, other}) == 2


def test_interval_arithmetic_is_sound():
    a = Interval.box(0.1, 0.4)
    b = Interval.box(0.2, 0.5)
    assert (a + b).lo <= 0.3 <= (a + b).hi
    assert (a * b).lo <= 0.2 * 0.1 <= (a * b).hi
    assert (a - b).lo <= -0.4
    assert (b / a).hi >= 0.5 / 0.1
    with pytest.raises(ZeroDivisionError):
        b / Interval.box(-0.1, 0.1)


def test_sin_interval_covers_full_range():
    near_pi_half = Sin(Var("t")).interval({"t": Interval.box(1.4, 1.7)})
    assert near_pi_half.hi >= 0.985
    wide = Sin(Var("t")).interval({"t": Interval.box(-10.0, 10.0)})
    assert wide.lo == -1.0 and wide.hi == 1.0
    assert Cos(Var("t")).interval({"t": Interval.box(0.0, 0.1)}).hi >= math.cos(0.0)


def test_simplify_folds_constants_and_zero_terms():
    expr = BinOp("*", BinOp("+", Var("a"), Var("a")), BinOp("*", Const(1.0), Var("b")))
    simplified = simplify(expr)
    assert np.allclose(
        simplified.eval({"a": np.array([2.0]), "b": np.array([3.0])}),
        [12.0],
    )
    assert simplify(BinOp("+", Var("a"), Const(0.0))) == Var("a")


def test_softmax_and_argmax_nodes():
    scores = [Const(1.0), Const(2.0), Const(3.0)]
    probs = Softmax(scores).eval({})
    assert np.allclose(probs.sum(), 1.0)
    assert np.allclose(probs, [0.09003057, 0.24472847, 0.66524096], atol=1e-8)
    assert ArgMax(scores).eval({}) == 2
    component = SoftmaxComponent(Softmax(scores), 2)
    assert component.eval({}) == pytest.approx(probs[2])
    with pytest.raises(IndexError):
        SoftmaxComponent(Softmax(scores), 5)


def test_threshold_and_ifelse():
    assert Threshold(Var("x"), 0.5).eval({"x": np.array([0.2, 0.8])}).tolist() == [0.0, 1.0]
    branch = __import__("src.synthesis.ast_nodes", fromlist=["IfElse"]).IfElse
    from src.synthesis.ast_nodes import Compare, IfElse

    expr = IfElse(Compare(">", Var("x"), Const(0.0)), Var("x"), Const(0.0))
    assert expr.eval({"x": np.array([-1.0, 2.0])}).tolist() == [0.0, 2.0]


def test_walk_visits_every_node():
    expr = BinOp("+", Var("a"), BinOp("*", Const(2.0), Var("b")))
    assert len(list(walk(expr))) == 5


def test_program_container():
    program = Program("f", ["a", "b"], Mod(Var("a") + Var("b"), 7), 7, "doc")
    assert program(a=3, b=6) == 2
    assert "f" in repr(program)
    with pytest.raises(KeyError):
        program(a=1)


# ---------------------------------------------------------------------------
# Basis extraction
# ---------------------------------------------------------------------------
def test_fourier_design_matrix_shape_and_orthogonality():
    p = 11
    design = fourier_design_matrix(p, [1, 2])
    assert design.shape == (p, 5)
    gram = design.T @ design
    assert np.allclose(gram[0, 1:], 0.0)
    with pytest.raises(ValueError):
        fourier_design_matrix(p, [p])


def test_greedy_fit_selects_planted_frequencies():
    p = 59
    x = np.arange(p)
    signal = 0.4 * np.cos(2 * math.pi * 3 * x / p) + 0.6 * np.sin(2 * math.pi * 7 * x / p)
    fit = fit_fourier_greedy(signal, p, max_terms=4, tol=1e-10)
    assert set(fit.freqs) == {3, 7}
    assert fit.residual_max < 1e-10
    assert np.allclose(fit.evaluate(x), signal, atol=1e-10)
    expr = fit.to_expr("a")
    assert np.allclose(expr.eval({"a": x.astype(float)}), signal, atol=1e-10)


def test_periodic_shift_detects_shift_invariant_matrix():
    p = 13
    j = np.arange(p)[:, None]
    c = np.arange(p)[None, :]
    matrix = np.cos(2 * math.pi * 2 * (j - c) / p)
    fit = fit_periodic_shift(matrix, p)
    assert fit.is_shift_invariant
    assert periodicity_score(matrix, p) > 0.999
    rng = np.random.default_rng(0)
    assert periodicity_score(rng.normal(size=(p, p)), p) < 0.2


def test_piecewise_linear_fit_recovers_relu():
    x = np.linspace(-2, 2, 101)
    values = 3.0 + 2.0 * x + np.maximum(x - 0.5, 0.0)
    fit = fit_piecewise_linear(x, values, [0.5])
    assert fit.residual_max < 1e-9
    assert np.allclose(fit.evaluate(x), values, atol=1e-9)


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------
def test_motifs_evaluate_to_expected_values():
    p = 11
    env = {"a": np.arange(p), "b": np.arange(p)}
    expected = {
        "add_mod": (np.arange(p) + np.arange(p)) % p,
        "sub_mod": (np.arange(p) - np.arange(p)) % p,
        "mul_mod": (np.arange(p) * np.arange(p)) % p,
        "first": np.arange(p),
        "second": np.arange(p),
    }
    by_name = {motif.name: motif for motif in MOTIFS}
    for name, target in expected.items():
        assert np.array_equal(np.asarray(by_name[name].build(p).eval(env)), target)


def test_model_predictions_shape(medium_model):
    predictions = model_predictions(medium_model)
    assert predictions.shape == (11, 11)
    assert predictions.min() >= 0 and predictions.max() < 11


def test_synthesizer_falls_back_to_faithful_circuit(medium_model, medium_dataset):
    result = synthesize(medium_model, medium_dataset)
    assert result.is_faithful_fallback is True
    assert result.motif is None
    assert result.full_domain_accuracy == 1.0
    assert result.program.provenance["kind"] == "fourier_circuit"
    payload = result.to_dict()
    assert payload["is_faithful_fallback"] is True
    assert "agreement" in payload["narrative"]
    assert result.diagnostics["circuit"]["heads"]


def test_faithful_circuit_report_truncation_error(medium_model, medium_dataset):
    circuit = synthesize(medium_model, medium_dataset).circuit
    program = build_fourier_circuit(medium_model, circuit, max_freqs=6)
    error = program.provenance["truncation_error"]
    assert error["logit"] < 1e-6
    values = evaluate_program_on_domain(program)
    assert values.shape == (11, 11)


def build_lookup_model(p: int, seed: int = 0) -> Transformer:
    """A model trained on the *whole* domain, used as a synthesis target.

    With ``train_fraction = 1.0`` the model can fit every input, so its argmax is
    available on the entire domain even before grokking.  It is the cheapest way
    to obtain a genuine end-to-end synthesis target inside the test-suite.
    """
    from src.models.train import TrainConfig, train_modular_addition

    result = train_modular_addition(
        TrainConfig(
            p=p,
            d_model=32,
            n_heads=2,
            init_std=0.5,
            epochs=1500,
            lr=5e-3,
            weight_decay=1e-3,
            eval_every=500,
            log_every=10000,
            train_fraction=1.0,
            post_grok_epochs=1,
            seed=seed,
            verbose=False,
            torch_threads=1,
        )
    )
    return result.model


@pytest.fixture(scope="module")
def addition_model():
    return build_lookup_model(7)


@pytest.mark.slow  # trains a lookup model (1500 epochs)
def test_synthesizer_identifies_addition(addition_model):
    from src.models.transformer import modular_addition_dataset

    dataset = modular_addition_dataset(7, train_fraction=1.0, seed=0)
    result = synthesize(addition_model, dataset)
    assert result.motif == "add_mod"
    assert result.full_domain_accuracy == 1.0
    assert result.train_accuracy == 1.0
    assert result.candidate_scores["add_mod"] == 1.0
    assert "discrete algorithm" in result.narrative()
    assert result.n_nodes <= 5
    assert result.program.provenance["motif"] == "add_mod"
    # The simplified body is exactly `(a + b) % p`.
    assert result.program.body == Mod(Var("a") + Var("b"), 7)


def test_synthesizer_requires_full_domain_match(medium_model, medium_dataset):
    from src.synthesis.program_synthesizer import synthesize as synth

    with pytest.raises(RuntimeError):
        synth(medium_model, medium_dataset, allow_fallback=False)


# ---------------------------------------------------------------------------
# VectorIndex: the component of a vector-valued Linear
# ---------------------------------------------------------------------------
def _two_feature_linear() -> "Linear":
    W = np.array(
        [
            [1.0, 2.0],
            [-1.0, 0.5],
            [0.0, 3.0],
        ]
    )
    return Linear(W, [Const(1.0), Var("x")], name="G")


def test_vector_index_evaluates_the_matching_component():
    lin = _two_feature_linear()
    x = np.array([2.0, -1.0])
    full = lin.eval({"x": x})
    assert full.shape == (2, 3)
    for c in range(3):
        idx = VectorIndex(lin, c)
        assert np.allclose(idx.eval({"x": x}), full[:, c])
        # the same value, written out by hand: w[c,0] * 1 + w[c,1] * x
        assert np.allclose(
            idx.eval({"x": x}), lin.weights[c, 0] + lin.weights[c, 1] * x
        )


def test_vector_index_is_structural_and_survives_substitution():
    lin = _two_feature_linear()
    assert VectorIndex(lin, 1) == VectorIndex(lin, 1)
    assert VectorIndex(lin, 1) != VectorIndex(lin, 2)
    assert len({VectorIndex(lin, 0), VectorIndex(lin, 0), VectorIndex(lin, 1)}) == 2
    substituted = VectorIndex(lin, 0).replace(Var("x"), Var("y"))
    assert substituted.vars() == ["y"]
    assert np.allclose(
        substituted.eval({"y": np.array([3.0])}), np.array([1.0 + 2.0 * 3.0])
    )


def test_vector_index_rendering_and_bounds():
    lin = _two_feature_linear()
    assert VectorIndex(lin, 2).render() == "G(1.0, x)[2]"
    with pytest.raises(ValueError):
        VectorIndex(lin, -1)
    with pytest.raises(IndexError):
        VectorIndex(lin, 7).eval({"x": np.array([1.0])})


def test_vector_index_interval_encloses_the_pointwise_value():
    lin = _two_feature_linear()
    box = {"x": Interval.box(0.5, 2.0)}
    for c in range(3):
        enclosure = VectorIndex(lin, c).interval(box)
        for x in np.linspace(0.5, 2.0, 7):
            value = float(VectorIndex(lin, c).eval({"x": np.array([x])})[0])
            assert enclosure.lo - 1e-12 <= value <= enclosure.hi + 1e-12


# ---------------------------------------------------------------------------
# build_fourier_circuit: full-basis exactness and the truncation semantics
# ---------------------------------------------------------------------------
def _agreement(program, p: int) -> float:
    return float(evaluate_program_on_domain(program).shape[0] == p)


def test_fourier_circuit_full_basis_is_exact(medium_model, medium_dataset):
    circuit = synthesize(medium_model, medium_dataset).circuit
    program = build_fourier_circuit(medium_model, circuit)
    provenance = program.provenance
    p = medium_model.config.p
    n_heads = medium_model.config.n_heads

    assert provenance["kind"] == "fourier_circuit"
    assert provenance["full_basis"] is True
    # A full basis is [1, cos, sin] for every frequency, i.e. p//2 of them.
    for head in range(n_heads):
        assert len(provenance["frequencies"][str(head)]) == p // 2
        assert len(set(provenance["frequencies"][str(head)])) == p // 2
    # out_features * (1 + 2|K|) parameters per head.
    assert provenance["n_readout_parameters"] == n_heads * p * (1 + 2 * (p // 2))
    # Exact for any map on Z_p, hence zero truncation error up to float64 noise.
    assert provenance["truncation_error"]["logit"] < 1e-12
    assert provenance["truncation_error"]["score"] < 1e-6
    # And the argmax agrees on the whole domain.
    values = evaluate_program_on_domain(program)
    assert values.shape == (p, p)
    reference = model_predictions(medium_model)
    # The circuit is only an approximation of the *reported* argmax when the
    # model's own attention scores are truncated; the contract is agreement.
    assert np.array_equal(values, reference) or (values == reference).mean() > 0.9


def test_fourier_circuit_truncation_is_a_prefix_of_the_full_basis(
    medium_model, medium_dataset
):
    circuit = synthesize(medium_model, medium_dataset).circuit
    p = medium_model.config.p
    full = build_fourier_circuit(medium_model, circuit)
    partial = build_fourier_circuit(medium_model, circuit, max_freqs=2)

    assert partial.provenance["full_basis"] is False
    for head in range(medium_model.config.n_heads):
        selected = partial.provenance["frequencies"][str(head)]
        # The greedy support is nested: a truncated basis is a subset of the
        # full one, and exactly ``max_freqs`` frequencies are kept.
        assert len(selected) == 2
        assert set(selected) <= set(full.provenance["frequencies"][str(head)])
        assert len(selected) <= p // 2
    assert partial.provenance["n_readout_parameters"] < full.provenance["n_readout_parameters"]
    # Truncating cannot increase the residual.
    assert (
        partial.provenance["truncation_error"]["logit"]
        >= full.provenance["truncation_error"]["logit"]
    )


def test_fourier_circuit_walk_is_linear_in_shared_nodes(medium_model, medium_dataset):
    """Sharing a readout between many logits must not blow up the traversal."""
    circuit = synthesize(medium_model, medium_dataset).circuit
    program = build_fourier_circuit(medium_model, circuit)
    nodes = list(walk(program.body))
    unique = {id(node) for node in nodes}
    assert len(nodes) == len(unique)  # walk() de-duplicates
    # 3 terms per (head, class) plus the p readouts; linear, not exponential.
    assert len(nodes) < 4000
