"""Tests of the Python and C code generators.

Every generated artifact is executed and compared against the IR's reference
semantics over the *complete* input domain.  The C tests compile with
``-std=c99 -O3 -Wall -Wextra -pedantic`` and fail on any diagnostic.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pytest
import torch

from src.codegen import emit_c, emit_python
from src.codegen.common import fmt_number, is_integer_program
from src.synthesis import synthesize
from src.synthesis.ast_nodes import BinOp, Const, Mod, Program, Var


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _program(name: str, body, p: int, motif: str = "test") -> Program:
    return Program(name, ["a", "b"], body, p, provenance={"motif": motif})


def _domain(p: int):
    a = np.repeat(np.arange(p), p)
    b = np.tile(np.arange(p), p)
    return a, b


def _run_python(source: str, func: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    namespace: dict = {}
    exec(compile(source, "<generated>", "exec"), namespace)  # noqa: S102 - intended
    fn = namespace[func]
    return np.array([fn(int(x), int(y)) for x, y in zip(a, b)])


def _run_c(source: str, func: str, p: int, workdir: str) -> np.ndarray:
    harness = (
        "#include <stdio.h>\n"
        + source
        + "\nint main(void)\n{\n"
        + f"    for (int a = 0; a < {p}; ++a)\n"
        + f"        for (int b = 0; b < {p}; ++b)\n"
        + '            printf("%.17g\\n", (double)' + func + "(a, b));\n"
        + "    return 0;\n}\n"
    )
    c_path = os.path.join(workdir, "program.c")
    exe_path = os.path.join(workdir, "program")
    with open(c_path, "w", encoding="utf-8") as handle:
        handle.write(harness)
    compile_result = subprocess.run(
        [
            "gcc",
            "-std=c99",
            "-O3",
            "-Wall",
            "-Wextra",
            "-pedantic",
            "-Werror",
            "-o",
            exe_path,
            c_path,
            "-lm",
        ],
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run = subprocess.run([exe_path], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    return np.array([float(value) for value in run.stdout.split()])


needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc is not available")


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def test_fmt_number_prefers_readable_integers():
    assert fmt_number(59.0) == "59"
    assert fmt_number(59.0, float_context=True) == "59.0"
    assert fmt_number(0.25) == "0.25"
    assert fmt_number(1e-7, float_context=True).startswith("1e-07")


def test_is_integer_program():
    assert is_integer_program(Mod(Var("a") + Var("b"), 59))
    assert not is_integer_program(BinOp("/", Var("a"), Const(2.0)))
    assert not is_integer_program(Const(0.5))


# ---------------------------------------------------------------------------
# Python emission
# ---------------------------------------------------------------------------
def test_python_motif_program_is_concise_and_correct():
    program = _program("add_mod_p", Mod(Var("a") + Var("b"), 59), 59, motif="add_mod")
    source = emit_python(program)
    assert "return (a + b) % 59" in source
    assert "import math" not in source
    a, b = _domain(59)
    expected = (a + b) % 59
    np.testing.assert_array_equal(_run_python(source, "add_mod_p", a, b), expected)


def test_python_emission_is_deterministic():
    program = _program("add_mod_p", Mod(Var("a") + Var("b"), 59), 59)
    assert emit_python(program) == emit_python(program)


def test_python_handles_subtraction_and_negative_values():
    program = _program("sub_mod_p", Mod(Var("a") - Var("b"), 7), 7, motif="sub_mod")
    a, b = _domain(7)
    np.testing.assert_array_equal(
        _run_python(emit_python(program), "sub_mod_p", a, b), (a - b) % 7
    )


def test_python_fallback_circuit_matches_reference(medium_model, medium_dataset):
    synthesis = synthesize(medium_model, medium_dataset)
    source = emit_python(synthesis.program)
    assert "def _softmax" in source and "def _argmax" in source
    p = medium_model.config.p
    a, b = _domain(p)
    expected = np.asarray(synthesis.program.evaluate({"a": a, "b": b}))
    np.testing.assert_array_equal(
        _run_python(source, synthesis.program.name, a, b).astype(int), expected
    )


# ---------------------------------------------------------------------------
# C emission
# ---------------------------------------------------------------------------
@needs_gcc
def test_c_motif_program_matches_reference():
    program = _program("add_mod_p", Mod(Var("a") + Var("b"), 59), 59, motif="add_mod")
    source = emit_c(program, include_main=False)
    assert "#include <math.h>" not in source
    assert "malloc" not in source
    a, b = _domain(59)
    with tempfile.TemporaryDirectory() as tmp:
        values = _run_c(source, "add_mod_p", 59, tmp)
    np.testing.assert_array_equal(values.astype(int), (a + b) % 59)


@needs_gcc
def test_c_fallback_circuit_matches_reference(medium_model, medium_dataset):
    synthesis = synthesize(medium_model, medium_dataset)
    source = emit_c(synthesis.program, include_main=False)
    assert "#include <math.h>" in source
    p = medium_model.config.p
    a, b = _domain(p)
    expected = np.asarray(synthesis.program.evaluate({"a": a, "b": b})).astype(int)
    with tempfile.TemporaryDirectory() as tmp:
        values = _run_c(source, synthesis.program.name, p, tmp)
    np.testing.assert_array_equal(values.astype(int), expected)


@needs_gcc
def test_c_program_with_main_compiles_and_runs():
    program = _program("add_mod_p", Mod(Var("a") + Var("b"), 11), 11, motif="add_mod")
    source = emit_c(program)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "standalone.c")
        exe = os.path.join(tmp, "standalone")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)
        result = subprocess.run(
            ["gcc", "-std=c99", "-O3", "-Wall", "-Wextra", "-pedantic", "-o", exe, path],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        run = subprocess.run([exe], capture_output=True, text=True)
    assert run.returncode == 0
    assert "add_mod_p(1, 2) = 3" in run.stdout
    assert "add_mod_p(10, 1) = 0" in run.stdout


def test_c_source_has_no_dynamic_allocation(medium_model, medium_dataset):
    synthesis = synthesize(medium_model, medium_dataset)
    source = emit_c(synthesis.program)
    for forbidden in ("malloc", "calloc", "free(", "realloc"):
        assert forbidden not in source
    assert not re.search(r"\bwhile\s*\(", source)


def test_generated_c_declares_expected_headers():
    program = _program("add_mod_p", Mod(Var("a") + Var("b"), 5), 5)
    source = emit_c(program, include_main=False)
    assert "#include <stdio.h>" not in source  # no main, no I/O
    assert "int add_mod_p(int a, int b)" in source
