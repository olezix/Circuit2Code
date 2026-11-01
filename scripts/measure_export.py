#!/usr/bin/env python
"""Measure the code that Circuit2Code emits.

The paper quotes the size of the generated files, the time ``gcc`` needs for
them, and the fact that the C and the Python agree on every input.  Those are
measurements, so they are produced here and stored as JSON instead of being
typed into the text by hand:

    python scripts/measure_export.py \\
        --checkpoint artifacts/checkpoints/g70.pt artifacts/checkpoints/t30.pt \\
        --train-fraction 0.7 0.3 --out artifacts/export_measurements.json

The script exports the checkpoint, compiles the C with the flags the README
advertises, runs both artefacts over the whole domain and compares them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.codegen import emit_c, emit_python  # noqa: E402
from src.interpretability import extract_circuit  # noqa: E402
from src.models.transformer import (  # noqa: E402
    Transformer,
    modular_addition_dataset,
)
from src.synthesis import synthesize  # noqa: E402

CFLAGS = ["-std=c99", "-O3", "-Wall", "-Wextra", "-pedantic", "-Werror"]


def _run_python(source: str, func: str, p: int) -> np.ndarray:
    namespace: dict = {}
    exec(compile(source, "<generated>", "exec"), namespace)  # noqa: S102 - intended
    fn = namespace[func]
    return np.array([float(fn(int(a), int(b))) for a in range(p) for b in range(p)])


def _run_c(source: str, func: str, p: int, workdir: str) -> tuple[np.ndarray, float, str]:
    """Compile ``source`` with a main() harness, run it, return values+timing+stderr."""
    body = source[: source.rindex("int main(void)")]
    harness = (
        "#include <stdio.h>\n"
        + body
        + "\nint main(void)\n{\n"
        + f"    for (int a = 0; a < {p}; ++a)\n"
        + f"        for (int b = 0; b < {p}; ++b)\n"
        + '            printf("%.17g\\n", (double)' + func + "(a, b));\n"
        + "    return 0;\n}\n"
    )
    c_path = os.path.join(workdir, "harness.c")
    exe_path = os.path.join(workdir, "harness")
    with open(c_path, "w", encoding="utf-8") as handle:
        handle.write(harness)
    started = time.time()
    compile_result = subprocess.run(
        ["gcc", *CFLAGS, "-o", exe_path, c_path, "-lm"], capture_output=True, text=True
    )
    compile_s = time.time() - started
    if compile_result.returncode != 0:  # pragma: no cover - a hard failure
        raise RuntimeError(f"gcc failed:\n{compile_result.stderr}")
    run = subprocess.run([exe_path], capture_output=True, text=True, check=False)
    values = np.array([float(value) for value in run.stdout.split()])
    return values, compile_s, compile_result.stderr


def measure(checkpoint: str, train_fraction: float, seed: int = 3) -> dict:
    model = Transformer.load(checkpoint)
    p = model.config.p
    dataset = modular_addition_dataset(p, train_fraction=train_fraction, seed=seed)
    circuit = extract_circuit(model, dataset)
    synthesis = synthesize(model, dataset, circuit=circuit)
    program = synthesis.program

    py_source = emit_python(program)
    c_source = emit_c(program)
    func = program.name

    workdir = tempfile.mkdtemp(prefix="c2c-export-")
    try:
        c_values, compile_s, gcc_stderr = _run_c(c_source, func, p, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    py_values = _run_python(py_source, func, p)

    return {
        "checkpoint": os.path.relpath(checkpoint, os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "p": p,
        "motif": synthesis.motif,
        "n_nodes": synthesis.n_nodes,
        "n_inputs": int(p * p),
        "python_bytes": len(py_source.encode()),
        "c_bytes": len(c_source.encode()),
        "c_lines": len(c_source.splitlines()),
        "gcc_flags": " ".join(CFLAGS),
        "gcc_warnings": len([line for line in gcc_stderr.splitlines() if line.strip()]),
        "gcc_seconds": round(compile_s, 2),
        "max_abs_difference": float(np.max(np.abs(c_values - py_values))),
        "bit_identical": bool(np.array_equal(c_values, py_values)),
        "allocates": "malloc" in c_source or "calloc" in c_source,
        "includes": sorted(set(re.findall(r"#include <([^>]+)>", c_source))),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        default=["artifacts/checkpoints/g70.pt", "artifacts/checkpoints/t30.pt"],
    )
    parser.add_argument("--train-fraction", nargs="+", type=float, default=[0.7, 0.3])
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--out", default="artifacts/export_measurements.json")
    args = parser.parse_args(argv)

    if len(args.checkpoint) != len(args.train_fraction):
        parser.error("--checkpoint and --train-fraction must have the same length")

    records = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as handle:
            previous = json.load(handle)
        for record in previous.get("measurements", [previous]):
            records[record["checkpoint"]] = record

    for checkpoint, fraction in zip(args.checkpoint, args.train_fraction):
        record = measure(checkpoint, fraction, args.seed)
        records[record["checkpoint"]] = record
        print(json.dumps(record, indent=2))

    payload = {
        "generated_by": "scripts/measure_export.py",
        "seed": args.seed,
        "measurements": [records[key] for key in sorted(records)],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
