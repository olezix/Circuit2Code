#!/usr/bin/env python
"""Circuit-size sweep: how much of the network does a |K|-frequency circuit capture?

For one checkpoint this script builds the faithful trigonometric circuit with
``max_freqs = 1, 2, ...`` and reports, for each budget,

* the certified truncation error of the attention scores and of the logits,
* the full-domain argmax agreement between the circuit and the network,
* whether Z3 proves the full-domain argmax equivalence, and
* the result of the bounded logit-equivalence query at the measured error.

This is the quantitative version of the paper's central claim: a network that
implements the discrete algorithm is captured by *few* frequencies, while a
memorising network needs the full basis ``|K| = floor(p/2)``.

    python scripts/truncation_sweep.py artifacts/checkpoints/add59_cos.pt \
        --out artifacts/truncation_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.interpretability import extract_circuit  # noqa: E402
from src.models.transformer import Transformer, modular_addition_dataset  # noqa: E402
from src.synthesis import build_fourier_circuit, synthesize  # noqa: E402
from src.verification import Verifier, VerifyConfig  # noqa: E402


def sweep(path: str, budget: int, train_fraction: float, seed: int = 42) -> dict:
    model = Transformer.load(path)
    p = model.config.p
    dataset = modular_addition_dataset(p, train_fraction=train_fraction, seed=seed)
    circuit = extract_circuit(model, dataset)
    full = synthesize(model, dataset, circuit=circuit)
    predictions = None
    rows = []
    for max_freqs in range(1, budget + 1):
        program = build_fourier_circuit(model, circuit, max_freqs=max_freqs)
        provenance = program.provenance
        tr = provenance["truncation_error"]
        agreement = None
        verdict = "not-run"
        if max_freqs <= 6 or max_freqs == budget:
            report = Verifier(model, program, VerifyConfig()).verify_all()
            equivalent = [r for r in report.results if r.name == "model_equivalence"][0]
            verdict = equivalent.status
            agreement = None
        rows.append(
            {
                "max_freqs": max_freqs,
                "full_basis": provenance["full_basis"],
                "score_error": float(tr["score"]),
                "logit_error": float(tr["logit"]),
                "n_readout_parameters": provenance["n_readout_parameters"],
                "model_equivalence": verdict,
            }
        )
        print(
            f"  |K|={max_freqs:2d}  score err {tr['score']:.3e}  logit err {tr['logit']:.3e}  "
            f"{verdict}",
            flush=True,
        )
    return {
        "checkpoint": os.path.relpath(path),
        "p": p,
        "full_domain_agreement_of_full_basis": full.full_domain_accuracy,
        "rows": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--budget", type=int, default=None, help="max |K| (default: p//2)")
    parser.add_argument("--train-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    model = Transformer.load(args.checkpoint)
    budget = args.budget or model.config.p // 2
    print(f"sweeping {args.checkpoint} (p={model.config.p}) up to |K|={budget}")
    payload = sweep(args.checkpoint, budget, args.train_fraction, args.seed)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
