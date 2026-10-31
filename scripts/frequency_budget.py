#!/usr/bin/env python
"""Measure how many Fourier frequencies a decompilation needs, per checkpoint.

For every checkpoint given on the command line this script reports

* held-out accuracy of the network,
* whether the token -> logit maps are shift invariant (``circuit.is_periodic``),
* the number of frequencies the greedy basis selection keeps per head,
* the full-domain agreement between the faithful trigonometric circuit and the
  network, and the certified truncation error of the circuit.

The point of the experiment is that a *generalising* network compresses, while a
memorising network needs the full basis ``|K| = floor(p/2)`` -- i.e. the size of
the recovered circuit is itself a measure of generalisation.

Usage::

    python scripts/frequency_budget.py artifacts/checkpoints/add59_cos.pt:0.3 \
        artifacts/checkpoints/cos13_60k.pt:0.5 --out artifacts/frequency_budget.json

A trailing ``:FRACTION`` overrides ``--train-fraction`` for that checkpoint, so
checkpoints trained on different splits can be compared in one table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.interpretability import extract_circuit  # noqa: E402
from src.models.transformer import Transformer, modular_addition_dataset  # noqa: E402
from src.synthesis import synthesize  # noqa: E402


def measure(path: str, train_fraction: float, seed: int) -> dict:
    model = Transformer.load(path)
    p = model.config.p
    dataset = modular_addition_dataset(p, train_fraction=train_fraction, seed=seed)
    test_idx = dataset["test_idx"]
    with torch.no_grad():
        logits = model(dataset["tokens"][test_idx])[0][:, model.config.answer_position, :]
        test_acc = float((logits.argmax(-1) == dataset["target"][test_idx]).float().mean())

    circuit = extract_circuit(model, dataset)
    result = synthesize(model, dataset, circuit=circuit)
    provenance = result.program.provenance
    freqs = provenance.get("frequencies", {})
    return {
        "checkpoint": path,
        "p": p,
        "n_heads": model.config.n_heads,
        "d_model": model.config.d_model,
        "mlp_width": model.config.mlp_width,
        "params": model.n_params,
        "test_accuracy": round(test_acc, 6),
        "motif": result.motif,
        "train_fraction": train_fraction,
        "is_periodic": circuit.is_periodic,
        "frequencies_per_head": {k: len(v) for k, v in freqs.items()},
        "max_frequencies_per_head": max((len(v) for v in freqs.values()), default=0),
        "full_basis": provenance.get("full_basis"),
        "truncation_error": {
            k: float(v) for k, v in provenance.get("truncation_error", {}).items()
        },
        "full_domain_agreement": round(result.full_domain_accuracy, 6),
        "n_nodes": result.n_nodes,
        "n_readout_parameters": provenance.get("n_readout_parameters"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints",
        nargs="+",
        help="checkpoint files, optionally suffixed with ':train_fraction'",
    )
    parser.add_argument("--train-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="write the table as JSON here")
    args = parser.parse_args(argv)

    rows = []
    for spec in args.checkpoints:
        path, fraction = spec, args.train_fraction
        if ":" in spec and not spec.endswith(".pt"):
            path, raw = spec.rsplit(":", 1)
            fraction = float(raw)
        rows.append(measure(path, fraction, args.seed))
    header = (
        f"{'checkpoint':38s} {'p':>3s} {'train':>5s} {'test':>6s} {'periodic':>8s} "
        f"{'|K|':>4s} {'nodes':>6s} {'agree':>6s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{os.path.basename(row['checkpoint']):38s} {row['p']:3d} "
            f"{row['train_fraction']:5.2f} {row['test_accuracy']:6.3f} "
            f"{str(row['is_periodic']):>8s} "
            f"{row['max_frequencies_per_head']:4d} {row['n_nodes']:6d} "
            f"{row['full_domain_agreement']:6.3f}"
        )
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"train_fraction": args.train_fraction, "rows": rows}, handle, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
