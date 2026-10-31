"""End-to-end tests of the command-line interface.

The CLI is exercised exactly as a user would: ``main(argv)`` is called with real
arguments, real checkpoints are written to ``tmp_path``, and the JSON artefacts
are parsed back.  No network, no environment variables, no global state.
"""

from __future__ import annotations

import json
import os

import pytest

from demo.cli import build_parser, main

#: Every test in this module trains a small model, so `make test-fast` skips it.
pytestmark = pytest.mark.slow

TINY = [
    "--p", "5",
    "--d-model", "16",
    "--n-heads", "2",
    "--init-std", "0.5",
    "--epochs", "600",
    "--lr", "5e-3",
    "--lr-schedule", "constant",
    "--weight-decay", "1e-3",
    "--train-fraction", "1.0",
    "--torch-threads", "1",
    "--log-every", "600",
    "--eval-every", "150",
    "--post-grok-epochs", "100",
    "--checkpoint-every", "0",
]


def test_parser_exposes_every_stage():
    parser = build_parser()
    actions = [
        action
        for action in parser._actions
        if hasattr(action, "choices") and action.choices
    ]
    commands = set()
    for action in actions:
        commands.update(action.choices)
    assert {"train", "extract-circuit", "decompile", "verify", "export", "demo"} <= commands


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    path = str(tmp_path_factory.mktemp("ckpt") / "add5.pt")
    assert main(["train", *TINY, "--out", path, "--json", path + ".json"]) == 0
    return path


def test_train_writes_history_json(tmp_path):
    out = str(tmp_path / "add5.pt")
    assert main(["train", *TINY, "--out", out, "--json", str(tmp_path / "history.json")]) == 0
    payload = json.loads((tmp_path / "history.json").read_text())
    assert payload["summary"]["final_train_acc"] >= 0.99
    assert payload["summary"]["final_test_acc"] >= 0.99
    assert payload["summary"]["grokking_epoch"] is not None
    assert len(payload["history"]) >= 2
    assert os.path.exists(out)


def test_extract_decompile_verify_export(checkpoint, tmp_path):
    circuit_json = tmp_path / "circuit.json"
    assert main([
        "extract-circuit", "--checkpoint", checkpoint, "--train-fraction", "1.0",
        "--json", str(circuit_json),
    ]) == 0
    circuit = json.loads(circuit_json.read_text())
    assert circuit["p"] == 5
    assert circuit["heads"] and circuit["frequencies"]

    outdir = tmp_path / "decompiled"
    assert main([
        "decompile", "--checkpoint", checkpoint, "--train-fraction", "1.0",
        "--out", str(outdir), "--json", str(tmp_path / "synthesis.json"),
    ]) == 0
    synthesis = json.loads((tmp_path / "synthesis.json").read_text())
    assert synthesis["motif"] == "add_mod"
    assert synthesis["full_domain_accuracy"] == 1.0
    assert (outdir / "add_mod_p.py").exists() and (outdir / "add_mod_p.c").exists()

    assert main([
        "verify", "--checkpoint", checkpoint, "--train-fraction", "1.0",
        "--json", str(tmp_path / "verification.json"),
    ]) == 0
    report = json.loads((tmp_path / "verification.json").read_text())
    assert report["verified"] is True
    statuses = {r["name"]: r["status"] for r in report["results"]}
    assert statuses["spec"] == "proved"
    assert statuses["model_equivalence"] == "proved"

    assert main([
        "export", "--checkpoint", checkpoint, "--train-fraction", "1.0",
        "--out", str(tmp_path / "exported"), "--json", str(tmp_path / "export.json"),
    ]) == 0
    exported = json.loads((tmp_path / "export.json").read_text())
    assert exported["motif"] == "add_mod"
    assert os.path.exists(exported["python"]) and os.path.exists(exported["c"])


def test_verify_exits_non_zero_when_a_proof_fails(checkpoint, tmp_path):
    """An unrealistic robustness slack makes the claim false, not the tool lax."""
    status = main([
        "verify", "--checkpoint", checkpoint, "--train-fraction", "1.0",
        "--slack", "10.0", "--json", str(tmp_path / "robust.json"),
    ])
    report = json.loads((tmp_path / "robust.json").read_text())
    assert report["verified"] is False
    assert status == 1


def test_missing_checkpoint_exits_with_message(tmp_path, capsys):
    status = main(["extract-circuit", "--checkpoint", str(tmp_path / "nope.pt")])
    assert status != 0
    assert "not found" in capsys.readouterr().err


def test_decompile_fallback_for_untrained_model(tmp_path):
    """An untrained network has no discrete motif: the faithful circuit is used."""
    from src.models.transformer import ModelConfig, Transformer

    path = str(tmp_path / "random.pt")
    Transformer(ModelConfig(p=5, d_model=16, n_heads=2, init_std=0.5), seed=7).save(path)
    outdir = tmp_path / "fallback"
    assert main([
        "decompile", "--checkpoint", path, "--train-fraction", "0.5",
        "--out", str(outdir), "--json", str(tmp_path / "fallback.json"),
    ]) == 0
    payload = json.loads((tmp_path / "fallback.json").read_text())
    assert payload["motif"] is None
    assert payload["is_faithful_fallback"] is True
    assert payload["provenance"]["kind"] == "fourier_circuit"
