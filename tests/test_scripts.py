"""Tests of the measurement scripts.

These are the scripts that produce every number in README section 9 and in the
paper, so they are tested like library code: the measurement helpers are called
directly on a tiny checkpoint and their invariants are asserted.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer import ModelConfig, Transformer  # noqa: E402
from scripts import frequency_budget, paper_tables, truncation_sweep  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    path = str(tmp_path_factory.mktemp("ckpt") / "random5.pt")
    Transformer(ModelConfig(p=5, d_model=16, n_heads=2, init_std=0.3), seed=11).save(path)
    return path


def test_frequency_budget_row_is_complete(checkpoint):
    row = frequency_budget.measure(checkpoint, train_fraction=0.5, seed=42)
    assert row["p"] == 5
    assert row["params"] > 0
    assert 0.0 <= row["test_accuracy"] <= 1.0
    assert row["max_frequencies_per_head"] <= 2  # p // 2
    assert row["full_domain_agreement"] == 1.0  # the faithful circuit is exact
    assert set(row["truncation_error"]) == {"score", "logit"}
    assert row["truncation_error"]["logit"] < 1e-6


def test_frequency_budget_cli_writes_json(checkpoint, tmp_path):
    out = tmp_path / "budget.json"
    assert frequency_budget.main([f"{checkpoint}:0.5", "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["train_fraction"] == 0.3  # global default, overridden per checkpoint
    assert payload["rows"][0]["train_fraction"] == 0.5
    assert payload["rows"][0]["checkpoint"] == checkpoint


def test_truncation_sweep_reports_monotone_errors(checkpoint, tmp_path):
    out = tmp_path / "sweep.json"
    assert truncation_sweep.main([checkpoint, "--budget", "1", "--train-fraction", "0.5",
                                  "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    rows = payload["rows"]
    # The full basis (p // 2 = 2 for p=5) is always appended for reference.
    assert [r["max_freqs"] for r in rows] == [1, 2]
    assert all(r["logit_error"] >= 0.0 for r in rows)
    assert rows[-1]["agreement"] == 1.0  # the full basis is exact
    assert rows[0]["agreement"] <= 1.0
    assert rows[0]["model_equivalence"] in {
        "proved",
        "refuted",
        "unknown",
        "skipped",
        "refuted-by-counting",
        "not-run",
    }


def test_paper_tables_analyse_random_network(checkpoint):
    row = paper_tables.analyse_checkpoint(checkpoint, train_fraction=0.5)
    assert row["p"] == 5
    assert row["synthesis"]["is_faithful_fallback"] is True
    assert row["synthesis"]["full_domain_accuracy"] == 1.0
    assert row["verification"]["verified"] is True
    names = {r["name"] for r in row["verification"]["results"]}
    assert "model_equivalence" in names
    # Only the heads that survive the ablation threshold have a logit map.
    residual_heads = set(row["circuit"]["shift_residual"])
    assert residual_heads and residual_heads <= {"L0H0", "L0H1"}
    assert residual_heads == set(row["circuit"]["spectral_energy_top3"])
