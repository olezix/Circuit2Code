#!/usr/bin/env python
"""Regenerate every number quoted in the paper and in README section 9.

The script measures the same quantities the paper reports -- training accuracy,
circuit statistics, decompilation agreement, truncation error, and the outcome
of each Z3 proof obligation -- and writes them to a JSON file.  Run it after
training so that no number in the write-up is hand-copied:

    python scripts/paper_tables.py --out artifacts/paper_tables.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.interpretability import extract_circuit  # noqa: E402
from src.models.train import TrainConfig, train_modular_addition  # noqa: E402
from src.models.transformer import (  # noqa: E402
    ModelConfig,
    Transformer,
    modular_addition_dataset,
)
from src.synthesis import synthesize  # noqa: E402
from src.synthesis.symbolic_reg import fit_periodic_shift  # noqa: E402
from src.verification import Verifier, VerifyConfig  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "artifacts", "checkpoints")

#: The seed every reported checkpoint was trained with (``make train SEED=3``).
#: Reusing it for the data split makes the reported "held-out" accuracy the
#: accuracy on the inputs the optimiser never saw.
SPLIT_SEED = 3


def _accuracy(model: Transformer, dataset) -> dict:
    with torch.no_grad():
        tokens = dataset["tokens"]
        logits = model(tokens)[0][:, model.config.answer_position, :]
        pred = logits.argmax(-1)
    target = dataset["target"]
    train_idx, test_idx = dataset["train_idx"], dataset["test_idx"]
    out = {
        "train_accuracy": float((pred[train_idx] == target[train_idx]).float().mean()),
        "full_domain_accuracy": float((pred == target).float().mean()),
    }
    if test_idx.numel():
        out["test_accuracy"] = float((pred[test_idx] == target[test_idx]).float().mean())
    else:
        out["test_accuracy"] = out["train_accuracy"]
    return out


def analyse_checkpoint(path: str, train_fraction: float, seed: int = 42) -> dict:
    model = Transformer.load(path)
    p = model.config.p
    dataset = modular_addition_dataset(p, train_fraction=train_fraction, seed=seed)
    circuit = extract_circuit(model, dataset)

    t0 = time.time()
    synthesis = synthesize(model, dataset, circuit=circuit)
    synthesis_s = time.time() - t0

    verification = Verifier(model, synthesis.program, VerifyConfig()).verify_all()
    row = {
        "checkpoint": os.path.relpath(path, ROOT),
        "p": p,
        "d_model": model.config.d_model,
        "n_heads": model.config.n_heads,
        "mlp_width": model.config.mlp_width,
        "n_params": model.n_params,
        "train_fraction": train_fraction,
        "accuracy": _accuracy(model, dataset),
        "circuit": {
            "heads": [f"L{h.layer}H{h.head}" for h in circuit.heads],
            "shift_residual": {
                f"L{layer}H{head}": float(
                    fit_periodic_shift(circuit.logit_maps[(layer, head)][:p, :], p).residual_max
                )
                for (layer, head) in circuit.logit_maps
            },
            "spectral_energy_top3": {
                f"L{h.layer}H{h.head}": h.spectral_energy_top3 for h in circuit.heads
            },
            "frequencies": list(circuit.frequencies),
            "is_periodic": circuit.is_periodic,
            "head_reports": [h.to_dict() for h in circuit.heads],
        },
        "synthesis": {
            "motif": synthesis.motif,
            "is_faithful_fallback": synthesis.is_faithful_fallback,
            "full_domain_accuracy": synthesis.full_domain_accuracy,
            "n_nodes": synthesis.n_nodes,
            "truncation_error": synthesis.program.provenance.get("truncation_error"),
            "frequencies_per_head": {
                k: len(v) for k, v in synthesis.program.provenance.get("frequencies", {}).items()
            },
            "full_basis": synthesis.program.provenance.get("full_basis"),
            "n_readout_parameters": synthesis.program.provenance.get("n_readout_parameters"),
            "seconds": round(synthesis_s, 2),
            "body": synthesis.program.body.render()[:400],
        },
        "verification": {
            "verified": verification.verified,
            "deployment_gap": verification.deployment_gap,
            "total_time_s": round(verification.total_time_s, 2),
            "results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "detail": r.detail,
                    "solver_time_s": round(r.solver_time_s, 4),
                    "statistics": {
                        k: v for k, v in r.statistics.items() if k.startswith("inputs")
                    },
                }
                for r in verification.results
            ],
        },
    }
    return row


def robustness_row(path: str, train_fraction: float, slack: float = 0.2) -> dict:
    model = Transformer.load(path)
    p = model.config.p
    dataset = modular_addition_dataset(p, train_fraction=train_fraction, seed=42)
    circuit = extract_circuit(model, dataset)
    synthesis = synthesize(model, dataset, circuit=circuit)
    t0 = time.time()
    report = Verifier(
        model, synthesis.program, VerifyConfig(attention_slack=slack)
    ).verify_all()
    return {
        "checkpoint": os.path.relpath(path, ROOT),
        "slack": slack,
        "verified": report.verified,
        "seconds": round(time.time() - t0, 2),
        "results": [
            {"name": r.name, "status": r.status, "detail": r.detail} for r in report.results
        ],
    }


def random_network_row(p: int = 11, seed: int = 42) -> dict:
    cfg = ModelConfig(p=p, d_model=32, n_heads=2, init_std=0.5)
    model = Transformer(cfg, seed=seed).eval()
    dataset = modular_addition_dataset(p, train_fraction=0.5, seed=seed)
    circuit = extract_circuit(model, dataset)
    synthesis = synthesize(model, dataset, circuit=circuit)
    report = Verifier(model, synthesis.program, VerifyConfig()).verify_all()
    return {
        "p": p,
        "n_params": model.n_params,
        "frequencies_per_head": {
            k: len(v) for k, v in synthesis.program.provenance.get("frequencies", {}).items()
        },
        "truncation_error": synthesis.program.provenance.get("truncation_error"),
        "full_domain_accuracy": synthesis.full_domain_accuracy,
        "n_nodes": synthesis.n_nodes,
        "verified": report.verified,
        "results": [{"name": r.name, "status": r.status, "detail": r.detail} for r in report.results],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(ROOT, "artifacts", "paper_tables.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="restrict to these row keys (e.g. --only p113_grok p59_grok)",
    )
    args = parser.parse_args(argv)

    payload: dict = {"generated_by": "scripts/paper_tables.py", "rows": {}, "extra": {}}
    #: key -> (checkpoint, train fraction, data-split seed).  Every reported
    #: checkpoint was trained with ``--seed 3``, so the same seed is used here:
    #: "held-out" then means the fraction of the domain the optimiser never saw,
    #: not a different random subset.  ``args.seed`` only drives the random
    #: (untrained) network row, where there is no training split to match.
    specs = {
        "p59_grok": ("g70.pt", 0.7, SPLIT_SEED),
        "p59_partial": ("t30.pt", 0.3, SPLIT_SEED),
        "p59_memoriser": ("m30.pt", 0.3, SPLIT_SEED),
        "p13_cosine": ("cos13_60k.pt", 0.5, SPLIT_SEED),
        "p23_cosine": ("cos2.pt", 0.5, SPLIT_SEED),
        "p113_grok": ("g113.pt", 0.7, SPLIT_SEED),
    }
    for key, (name, fraction, split_seed) in specs.items():
        if args.only and key not in args.only:
            continue
        path = os.path.join(CKPT, name)
        if not os.path.exists(path):
            print(f"skip {name} (missing)")
            continue
        print(f"analysing {name} ...", flush=True)
        payload["rows"][key] = analyse_checkpoint(path, fraction, split_seed)

    # Robustness (attention slack) is exercised on a small domain: with slack the
    # enclosure-based pruning no longer applies and every input enters the solver.
    want_robustness = not args.only or "p13_cosine" in args.only
    for key, (name, fraction) in (("p13", ("cos13_60k.pt", 0.5)),) if want_robustness else ():
        path = os.path.join(CKPT, name)
        if os.path.exists(path):
            for slack in (0.001, 0.01):
                print(f"robustness on {name} (slack={slack}) ...", flush=True)
                payload["extra"][f"robustness_{key}_{slack}"] = robustness_row(
                    path, fraction, slack
                )

    if not args.only:
        print("random p=11 ...", flush=True)
        payload["extra"]["random_p11"] = random_network_row(11, args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
