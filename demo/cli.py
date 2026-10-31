"""Circuit2Code command-line interface.

Five commands mirror the five stages of the pipeline::

    circuit2code train            --p 59 --epochs 40000 --out ckpt.pt
    circuit2code extract-circuit  --checkpoint ckpt.pt
    circuit2code decompile        --checkpoint ckpt.pt --out artifacts/decompiled
    circuit2code verify           --checkpoint ckpt.pt --json artifacts/verification.json
    circuit2code export           --checkpoint ckpt.pt --out artifacts/decompiled

The CLI is a thin, *complete* wrapper: it contains no logic of its own beyond
argument parsing, formatting (``rich``) and file I/O, so every number it prints
can be reproduced from the Python API in a few lines.

Exit codes: ``0`` success, ``1`` a verification obligation failed, ``2`` bad
usage.  ``--json`` makes each command emit a machine-readable artefact instead
of human-readable text (the two are always consistent).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.codegen import emit_c, emit_python
from src.interpretability import extract_circuit
from src.models.train import TrainConfig, train_modular_addition
from src.models.transformer import Transformer, modular_addition_dataset
from src.synthesis import synthesize
from src.verification import VerificationReport, Verifier, VerifyConfig

console = Console()
# Errors are printed without line wrapping so that a message such as
# "checkpoint ... not found" stays greppable in a terminal or a CI log.
err_console = Console(stderr=True, soft_wrap=True)

DEFAULT_CHECKPOINT = os.path.join("artifacts", "checkpoints", "add59.pt")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _load_model(path: str) -> Transformer:
    if not os.path.exists(path):
        raise SystemExit(
            f"checkpoint {path!r} not found -- run `circuit2code train` first "
            "(or pass --checkpoint)"
        )
    model = Transformer.load(path)
    model.eval()
    return model


def _dataset(model: Transformer, train_fraction: float, seed: int) -> Dict[str, torch.Tensor]:
    return modular_addition_dataset(
        model.config.p, train_fraction=train_fraction, seed=seed
    )


def _save_json(payload: Dict[str, Any], path: Optional[str]) -> None:
    if not path:
        return
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _train_table(history: Sequence[Dict[str, float]], every: int = 1) -> Table:
    table = Table(title="Grokking trajectory", header_style="bold")
    for column in ("epoch", "train loss", "train acc", "test loss", "test acc"):
        table.add_column(column, justify="right")
    for record in history[::every]:
        table.add_row(
            f"{int(record['epoch'])}",
            f"{record['train_loss']:.4f}",
            f"{record['train_acc']:.3f}",
            f"{record['test_loss']:.4f}",
            f"{record['test_acc']:.3f}",
        )
    return table


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> int:
    config = TrainConfig(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_width=args.mlp_width,
        init_std=args.init_std,
        train_fraction=args.train_fraction,
        epochs=args.epochs,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        post_grok_epochs=args.post_grok_epochs,
        checkpoint_path=args.out,
        checkpoint_every=args.checkpoint_every,
        torch_threads=args.torch_threads,
        verbose=not args.quiet,
    )
    result = train_modular_addition(config)

    summary = result.summary()
    if args.json:
        payload = {"summary": summary, "history": result.history}
        _save_json(payload, args.json)
        return 0

    console.print(_train_table(result.history, every=1))
    grok = summary["grokking_epoch"]
    console.print(
        Panel(
            f"[bold]p[/bold] = {args.p}   [bold]params[/bold] = {result.model.n_params:,}\n"
            f"[bold]train acc[/bold] = {summary['final_train_acc']:.3f}   "
            f"[bold]test acc[/bold] = {summary['final_test_acc']:.3f}\n"
            f"[bold]grokked at[/bold] = "
            f"{'never' if grok is None else f'epoch {int(grok)}'}   "
            f"[bold]time[/bold] = {summary['elapsed_s']:.1f}s",
            title="Training finished",
            border_style="green" if (summary["final_test_acc"] or 0) > 0.99 else "yellow",
        )
    )
    if summary["final_test_acc"] < 0.99:
        console.print(
            "[yellow]note:[/yellow] held-out accuracy is below 100%.  The network has "
            "not finished grokking; `decompile` will still produce a faithful "
            "trigonometric circuit, but no discrete algorithm can be claimed."
        )
    return 0


# ---------------------------------------------------------------------------
# extract-circuit
# ---------------------------------------------------------------------------
def cmd_extract(args: argparse.Namespace) -> int:
    model = _load_model(args.checkpoint)
    dataset = _dataset(model, args.train_fraction, args.seed)
    circuit = extract_circuit(model, dataset)

    if args.json:
        payload = {
            "p": model.config.p,
            "summary": circuit.summary(),
            "heads": [
                {
                    "layer": head.layer,
                    "head": head.head,
                    "accuracy_drop": head.accuracy_drop,
                    "logit_diff_drop": head.logit_diff_drop,
                    "top_frequencies": [(int(k), float(e)) for k, e in head.top_frequencies],
                    "spectral_energy_top3": head.spectral_energy_top3,
                }
                for head in circuit.heads
            ],
            "frequencies": list(circuit.frequencies),
            "is_periodic": circuit.is_periodic,
        }
        _save_json(payload, args.json)
        return 0

    table = Table(title=f"Circuit of p={model.config.p}", header_style="bold")
    table.add_column("head")
    table.add_column("Δacc (zero-ablation)", justify="right")
    table.add_column("Δlogit", justify="right")
    table.add_column("top frequencies")
    table.add_column("concentration", justify="right")
    for head in circuit.heads:
        table.add_row(
            f"L{head.layer}H{head.head}",
            f"{head.accuracy_drop:+.4f}",
            f"{head.logit_diff_drop:+.4f}",
            ", ".join(str(int(k)) for k, _ in head.top_frequencies) or "-",
            f"{head.spectral_energy_top3:.3f}",
        )
    console.print(table)
    console.print(
        f"[bold]circuit[/bold]: {circuit.summary()}\n"
        f"frequencies with the most energy: {', '.join(str(k) for k in circuit.frequencies)}"
    )
    if not circuit.is_periodic:
        console.print(
            "[yellow]note:[/yellow] the token→logit maps are not shift invariant, which is "
            "the fingerprint of a *memorising* rather than a Fourier circuit."
        )
    return 0


# ---------------------------------------------------------------------------
# decompile
# ---------------------------------------------------------------------------
def cmd_decompile(args: argparse.Namespace) -> int:
    model = _load_model(args.checkpoint)
    dataset = _dataset(model, args.train_fraction, args.seed)
    circuit = extract_circuit(model, dataset)
    result = synthesize(model, dataset, circuit=circuit, max_freqs=args.max_freqs)

    outdir = args.out
    os.makedirs(outdir, exist_ok=True)
    python_source = emit_python(result.program)
    c_source = emit_c(result.program, include_main=True)
    paths = {
        "python": os.path.join(outdir, f"{result.program.name}.py"),
        "c": os.path.join(outdir, f"{result.program.name}.c"),
    }
    with open(paths["python"], "w", encoding="utf-8") as handle:
        handle.write(python_source)
    with open(paths["c"], "w", encoding="utf-8") as handle:
        handle.write(c_source)

    payload = result.to_dict()
    payload["files"] = paths
    _save_json(payload, args.json or os.path.join(outdir, "synthesis.json"))

    if args.json:
        return 0

    description = (
        f"discrete algorithm [bold]{result.motif}[/bold]"
        if result.motif
        else "[bold]faithful trigonometric circuit[/bold] (no discrete motif)"
    )
    console.print(
        Panel(
            f"{description}\n"
            f"full-domain agreement with the network: "
            f"[bold]{result.full_domain_accuracy * 100:.2f}%[/bold] "
            f"({result.n_nodes} IR nodes)\n"
            f"truncation error: {result.truncation_error:.3e}\n"
            f"wrote {paths['python']} and {paths['c']}",
            title="Decompilation",
            border_style="green" if result.motif else "cyan",
        )
    )
    console.print(Panel(result.narrative(), title="What the program computes", border_style="blue"))
    console.print(
        Panel(python_source, title=f"{paths['python']}", border_style="magenta", expand=False)
    )
    return 0


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
def _report_table(report: VerificationReport) -> Table:
    table = Table(title="Z3 proof obligations", header_style="bold")
    table.add_column("theorem")
    table.add_column("status")
    table.add_column("statement", overflow="fold")
    table.add_column("detail", overflow="fold")
    for result in report.results:
        colour = {"proved": "green", "refuted": "red", "unknown": "yellow"}.get(
            result.status, "dim"
        )
        table.add_row(
            result.name,
            f"[{colour}]{result.status}[/{colour}]",
            result.statement,
            result.detail,
        )
    return table


def cmd_verify(args: argparse.Namespace) -> int:
    model = _load_model(args.checkpoint)
    dataset = _dataset(model, args.train_fraction, args.seed)
    circuit = extract_circuit(model, dataset)
    result = synthesize(model, dataset, circuit=circuit, max_freqs=args.max_freqs)
    config = VerifyConfig(
        timeout_ms=args.timeout_ms,
        batch_size=args.batch_size,
        logit_eps=args.eps,
        attention_slack=args.slack,
    )
    report = Verifier(model, result.program, config).verify_all()

    payload = report.to_dict()
    payload["program"] = {
        "motif": result.motif,
        "full_domain_accuracy": result.full_domain_accuracy,
        "n_nodes": result.n_nodes,
        "is_faithful_fallback": result.is_faithful_fallback,
    }
    _save_json(payload, args.json)
    if args.json:
        return 0 if report.verified else 1

    console.print(_report_table(report))
    console.print(
        Panel(
            f"[bold]verified[/bold] = {report.verified}\n"
            f"float32↔float64 deployment gap = {report.deployment_gap:.3e}\n"
            f"total solver time = {report.total_time_s:.2f}s",
            border_style="green" if report.verified else "red",
        )
    )
    return 0 if report.verified else 1


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def cmd_export(args: argparse.Namespace) -> int:
    model = _load_model(args.checkpoint)
    dataset = _dataset(model, args.train_fraction, args.seed)
    circuit = extract_circuit(model, dataset)
    result = synthesize(model, dataset, circuit=circuit, max_freqs=args.max_freqs)

    os.makedirs(args.out, exist_ok=True)
    python_path = os.path.join(args.out, f"{result.program.name}.py")
    c_path = os.path.join(args.out, f"{result.program.name}.c")
    with open(python_path, "w", encoding="utf-8") as handle:
        handle.write(emit_python(result.program))
    with open(c_path, "w", encoding="utf-8") as handle:
        handle.write(emit_c(result.program, include_main=not args.no_main))

    if args.json:
        _save_json(
            {
                "program": result.program.name,
                "motif": result.motif,
                "python": python_path,
                "c": c_path,
                "n_nodes": result.n_nodes,
            },
            args.json,
        )
        return 0

    console.print(
        Panel(
            f"emitted [bold]{python_path}[/bold] and [bold]{c_path}[/bold]\n"
            f"build with: gcc -std=c99 -O3 -Wall -Wextra -pedantic {c_path} -o prog -lm",
            title="Export",
            border_style="green",
        )
    )
    return 0


# ---------------------------------------------------------------------------
# demo (end-to-end pipeline)
# ---------------------------------------------------------------------------
def cmd_demo(args: argparse.Namespace) -> int:
    console.rule("[bold]1/5 train")
    train_args = argparse.Namespace(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=1,
        mlp_width=0,
        init_std=args.init_std,
        train_fraction=args.train_fraction,
        epochs=args.epochs,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        grad_clip=None,
        weight_decay=args.weight_decay,
        seed=args.seed,
        log_every=max(1, args.epochs // 10),
        eval_every=max(1, args.epochs // 40),
        post_grok_epochs=max(1, args.epochs // 20),
        out=args.checkpoint,
        checkpoint_every=0,
        torch_threads=args.torch_threads,
        quiet=False,
        json=None,
    )
    if cmd_train(train_args) != 0:
        return 1

    console.rule("[bold]2/5 extract circuit")
    common = argparse.Namespace(
        checkpoint=args.checkpoint,
        train_fraction=args.train_fraction,
        seed=args.seed,
        max_freqs=None,
        json=None,
    )
    extract_args = argparse.Namespace(**vars(common))
    extract_args.json = None
    cmd_extract(extract_args)

    console.rule("[bold]3/5 decompile")
    decompile_args = argparse.Namespace(**vars(common))
    decompile_args.out = os.path.join(args.out, "decompiled")
    decompile_args.json = None
    cmd_decompile(decompile_args)

    console.rule("[bold]4/5 verify")
    verify_args = argparse.Namespace(**vars(common))
    verify_args.json = None
    verify_args.timeout_ms = 120_000
    verify_args.batch_size = 128
    verify_args.eps = 1e-4
    verify_args.slack = 0.0
    status = cmd_verify(verify_args)

    console.rule("[bold]5/5 export")
    export_args = argparse.Namespace(**vars(common))
    export_args.out = os.path.join(args.out, "exported")
    export_args.no_main = False
    cmd_export(export_args)
    return status


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="circuit2code",
        description="Neuro-symbolic decompilation of algorithmic Transformers.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="model checkpoint")
        p.add_argument("--train-fraction", type=float, default=0.3, help="fraction of the domain used for training")
        p.add_argument("--seed", type=int, default=42, help="dataset split seed")
        p.add_argument("--json", default=None, help="write machine-readable output to this path")
        p.add_argument("--max-freqs", type=int, default=None, help="maximum Fourier frequencies per head (default: all)")

    train = sub.add_parser("train", help="train the reference model")
    train.add_argument("--p", type=int, default=59)
    train.add_argument("--d-model", type=int, default=128)
    train.add_argument("--n-heads", type=int, default=4)
    train.add_argument("--n-layers", type=int, default=1)
    train.add_argument("--mlp-width", type=int, default=0)
    train.add_argument("--init-std", type=float, default=0.02)
    train.add_argument("--train-fraction", type=float, default=0.3)
    train.add_argument("--epochs", type=int, default=40000)
    train.add_argument("--lr", type=float, default=1e-2)
    train.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine")
    train.add_argument("--grad-clip", type=float, default=None)
    train.add_argument("--weight-decay", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--log-every", type=int, default=500)
    train.add_argument("--eval-every", type=int, default=250)
    train.add_argument("--post-grok-epochs", type=int, default=2000)
    train.add_argument("--torch-threads", type=int, default=2)
    train.add_argument("--checkpoint-every", type=int, default=1000)
    train.add_argument("--out", default=DEFAULT_CHECKPOINT)
    train.add_argument("--quiet", action="store_true")
    train.add_argument("--json", default=None)
    train.set_defaults(func=cmd_train)

    extract = sub.add_parser("extract-circuit", help="identify the algorithmic circuit")
    add_common(extract)
    extract.set_defaults(func=cmd_extract)

    decompile = sub.add_parser("decompile", help="synthesize a symbolic program")
    add_common(decompile)
    decompile.add_argument("--out", default=os.path.join("artifacts", "decompiled"))
    decompile.set_defaults(func=cmd_decompile)

    verify = sub.add_parser("verify", help="prove equivalence with Z3")
    add_common(verify)
    verify.add_argument("--eps", type=float, default=1e-4, help="logit tolerance")
    verify.add_argument("--slack", type=float, default=0.0, help="attention robustness slack")
    verify.add_argument("--timeout-ms", type=int, default=120_000)
    verify.add_argument("--batch-size", type=int, default=128)
    verify.set_defaults(func=cmd_verify)

    export = sub.add_parser("export", help="emit Python and C sources")
    add_common(export)
    export.add_argument("--out", default=os.path.join("artifacts", "exported"))
    export.add_argument("--no-main", action="store_true", help="omit the C main() driver")
    export.set_defaults(func=cmd_export)

    demo = sub.add_parser("demo", help="run the whole pipeline end to end")
    demo.add_argument("--p", type=int, default=5)
    demo.add_argument("--d-model", type=int, default=16)
    demo.add_argument("--n-heads", type=int, default=2)
    demo.add_argument("--init-std", type=float, default=0.5)
    demo.add_argument("--train-fraction", type=float, default=1.0)
    demo.add_argument("--epochs", type=int, default=600)
    demo.add_argument("--lr", type=float, default=5e-3)
    demo.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    demo.add_argument("--weight-decay", type=float, default=1e-3)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--torch-threads", type=int, default=1)
    demo.add_argument("--checkpoint", default=os.path.join("artifacts", "checkpoints", "demo.pt"))
    demo.add_argument("--out", default=os.path.join("artifacts", "demo"))
    demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except SystemExit as exc:  # checkpoint missing / bad usage
        err_console.print(f"[red]{exc}[/red]")
        return int(exc.code) if isinstance(exc.code, int) else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
