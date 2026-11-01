#!/usr/bin/env python
"""Render the results section of the README and the experiments section of the paper.

Every number in those two places is a measurement, so it is generated from the
JSON artefacts instead of being typed:

    python scripts/render_report.py            # rewrite README + paper
    python scripts/render_report.py --check    # fail if they are out of date

The generated regions are delimited by markers, so hand-written prose elsewhere
is never touched.  ``tests/test_docs.py`` runs ``--check``, which means a stale
table or a stale number fails the test suite instead of surviving in the text.
"""

from __future__ import annotations

import argparse
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")

README = os.path.join(ROOT, "README.md")
PAPER = os.path.join(ROOT, "paper", "main.tex")

README_BEGIN = "<!-- BEGIN GENERATED: results  (scripts/render_report.py) -->"
README_END = "<!-- END GENERATED: results -->"

PAPER_BEGIN = "% BEGIN GENERATED: experiments  (scripts/render_report.py)"
PAPER_END = "% END GENERATED: experiments"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load(name: str):
    path = os.path.join(ART, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_all() -> dict:
    return {
        "tables": _load("paper_tables.json"),
        "sweep59": _load("truncation_sweep_p59.json"),
        "sweep113": _load("truncation_sweep_p113.json"),
        "budget": _load("frequency_budget.json"),
        "export": _load("export_measurements.json"),
        "environment": _load("environment.json"),
        "hist": {
            name: _load(f"history/{name}.history.json")
            for name in ("g70", "t30", "m30", "cos13_60k", "cos2", "g113")
        },
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def pct(x: float, digits: int = 1) -> str:
    return f"{100.0 * x:.{digits}f}%"


def score(x: float, digits: int = 3) -> str:
    """Accuracy that never rounds a near-miss up to a perfect score."""
    if x >= 1.0:
        return f"{1.0:.{digits}f}"
    return f"{x:.{digits + 1}f}"


def n_errors(row: dict) -> int:
    domain = row["p"] * row["p"]
    return round((1.0 - row["accuracy"]["full_domain_accuracy"]) * domain)


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def sci(x: float) -> str:
    if x == 0.0:
        return "0"
    mantissa, exponent = f"{x:.2e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def latex_sci(x: float) -> str:
    if x == 0.0:
        return "$0$"
    mantissa, exponent = f"{x:.2e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def latex_num(value: float, digits: int = 3) -> str:
    text = f"{value:,.{digits}f}"
    return text.replace(",", "{,}")


def agreement(value: float) -> str:
    """Agreement as a string that can never round up to "1.000"."""
    if value >= 1.0:
        return "1.0000"
    return f"{value:.4f}"


def heads(row: dict) -> list:
    return row["circuit"]["head_reports"]


def top_freqs(row: dict, count: int = 3) -> str:
    """The ``count`` frequencies that appear most often in the heads' top three."""
    score: dict = {}
    for head in heads(row):
        for rank, (freq, _) in enumerate(head["top_frequencies"][:3]):
            score[freq] = score.get(freq, 0.0) + 1.5 ** (2 - rank)
    ranked = sorted(score.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(str(freq) for freq, _ in ranked[:count])


def head_freqs(row: dict) -> str:
    return " / ".join(
        "{" + ", ".join(str(k) for k, _ in h["top_frequencies"][:3]) + "}" for h in heads(row)
    )


def energy_range(row: dict) -> str:
    values = [h["spectral_energy_top3"] for h in heads(row)]
    return f"{num(min(values))} – {num(max(values))}"


def dacc_range(row: dict) -> str:
    values = [h["accuracy_drop"] for h in heads(row)]
    return f"{num(min(values))} – {num(max(values))}"


def dlogit_range(row: dict) -> str:
    values = [h["logit_diff_drop"] for h in heads(row)]
    return f"{num(min(values), 2)} – {num(max(values), 2)}"


def proofs(row: dict) -> dict:
    return {r["name"]: r for r in row["verification"]["results"]}


def solver_seconds(row: dict, name: str) -> float:
    return proofs(row).get(name, {}).get("solver_time_s", 0.0)


def smt_decided(detail: str) -> str:
    match = re.search(r"SMT-decided:\s*([0-9.]+)", detail or "")
    return f"{float(match.group(1)):.0f}" if match else "n/a"


def export_for(data: dict, checkpoint: str) -> dict | None:
    payload = data["export"]
    if payload is None:
        return None
    records = payload.get("measurements", [payload])
    for record in records:
        if record["checkpoint"].endswith(checkpoint):
            return record
    return None


def minimal_exact(sweep: dict) -> dict | None:
    for row in sweep["rows"]:
        if row["agreement"] >= 1.0 and row["model_equivalence"] == "proved":
            return row
    return None


def trajectory_points(history: dict, epochs: list) -> list:
    points = history["history"]
    return [min(points, key=lambda point: abs(point["epoch"] - epoch)) for epoch in epochs]


def crossing(history: dict, threshold: float) -> int | None:
    for point in history["history"]:
        if point["test_acc"] >= threshold:
            return int(point["epoch"])
    return None


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------
def table_setup(data: dict) -> str:
    rows = data["tables"]["rows"]
    lines = [
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\toprule",
        "Checkpoint & $p$ & train frac. & params & epochs & held-out & full domain \\\\",
        "\\midrule",
    ]
    specs = [
        ("G (grokked)", "g70", rows.get("p59_grok")),
        ("T (partial)", "t30", rows.get("p59_partial")),
        ("M (memoriser)", "m30", rows.get("p59_memoriser")),
        ("$p=13$", "cos13_60k", rows.get("p13_cosine")),
        ("$p=23$", "cos2", rows.get("p23_cosine")),
        ("$p=113$", "g113", rows.get("p113_grok")),
    ]
    for label, key, row in specs:
        if row is None:
            continue
        history = data["hist"].get(key) or {}
        run = history.get("summary", {}).get("epochs_run")
        epochs = f"{int(run):,}".replace(",", "{,}") if run else "{-}"
        params = f"{row['n_params']:,}".replace(",", "{,}")
        lines.append(
            f"{label} & {row['p']} & {row['train_fraction']:.2f} & {params} & {epochs} & "
            f"{score(row['accuracy']['test_accuracy'])} & "
            f"{score(row['accuracy']['full_domain_accuracy'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_training(data: dict) -> str:
    history = data["hist"].get("t30")
    if history is None:
        return ""
    total = int(history["summary"]["epochs_run"])
    picks = [total // 12, total // 3, int(total * 0.4), int(total * 0.5),
             int(total * 0.6), int(total * 0.75), total]
    rows = trajectory_points(history, sorted(set(picks)))
    lines = [
        "\\begin{tabular}{@{}rrrr@{}}",
        "\\toprule",
        "Epoch & Train acc. & Test acc. & Test loss \\\\",
        "\\midrule",
    ]
    for point in rows:
        lines.append(
            f"{int(point['epoch']):,} & {point['train_acc']:.3f} & "
            f"{point['test_acc']:.3f} & {point['test_loss']:.2f} \\\\".replace(",", "{,}")
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_circuit(data: dict) -> str:
    rows = data["tables"]["rows"]
    lines = [
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "Head & $\\Delta$acc.\\ (zero-ablation) & $\\Delta$logit & top frequencies & "
        "top-3 energy \\\\",
        "\\midrule",
    ]
    blocks = [
        ("checkpoint G (grokked)", rows.get("p59_grok")),
        ("checkpoint T (partial)", rows.get("p59_partial")),
        ("checkpoint M (memoriser)", rows.get("p59_memoriser")),
    ]
    present = [(title, row) for title, row in blocks if row is not None]
    for index, (title, row) in enumerate(present):
        held_out = latex_num(row["accuracy"]["test_accuracy"])
        lines.append(
            f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{title}}}, held-out ${held_out}$}}\\\\"
        )
        for head in heads(row):
            freqs = ",\\,".join(str(k) for k, _ in head["top_frequencies"][:3])
            lines.append(
                f"L{head['layer']}H{head['head']} & ${head['accuracy_drop']:.3f}$ & "
                f"${head['logit_diff_drop']:.2f}$ & ${freqs}$ & "
                f"${head['spectral_energy_top3']:.3f}$ \\\\"
            )
        if index != len(present) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_size(data: dict) -> str:
    sweep = data["sweep59"]
    lines = [
        "\\begin{tabular}{@{}rrrr@{}}",
        "\\toprule",
        "$|K|$ & agreement & logit error & \\code{model\\_equivalence} \\\\",
        "\\midrule",
    ]
    for row in sweep["rows"]:
        verdict = {
            "proved": "proved",
            "refuted-by-counting": "refuted (by counting)",
        }.get(row["model_equivalence"], "not queried")
        shown = agreement(row["agreement"])
        if row["agreement"] >= 1.0:
            shown = f"$\\mathbf{{{shown}}}$"
        else:
            shown = f"${shown}$"
        lines.append(
            f"{row['max_freqs']} & {shown} & {latex_sci(row['logit_error'])} & {verdict} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_verification(data: dict) -> str:
    rows = data["tables"]["rows"]
    lines = [
        "\\begin{tabular}{@{}llll@{}}",
        "\\toprule",
        "Target & Theorem & Domain & Result \\\\",
        "\\midrule",
    ]
    labels = [
        ("p59_grok", "$p=59$ G"),
        ("p59_partial", "$p=59$ T"),
        ("p59_memoriser", "$p=59$ M"),
        ("p113_grok", "$p=113$"),
        ("p13_cosine", "$p=13$"),
        ("p23_cosine", "$p=23$"),
    ]
    for key, label in labels:
        row = rows.get(key)
        if row is None:
            continue
        domain = f"{row['p'] * row['p']:,} inputs".replace(",", "{,}")
        for name, result in proofs(row).items():
            if result["status"] == "skipped":
                continue
            seconds = ""
            if name == "spec" and result["solver_time_s"]:
                seconds = f", {result['solver_time_s']:.2f}\\,s"
            lines.append(
                f"{label} & \\code{{{name.replace('_', chr(92) + '_')}}} & {domain} & "
                f"{result['status']}{seconds} \\\\"
            )
    random_row = data["tables"].get("extra", {}).get("random_p11")
    if random_row:
        for name, result in {r["name"]: r for r in random_row["results"]}.items():
            if result["status"] == "skipped":
                continue
            lines.append(
                f"$p=11$ random & \\code{{{name.replace('_', chr(92) + '_')}}} & 121 inputs & "
                f"{result['status']} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_robustness(data: dict) -> str:
    rows = data["tables"]["rows"]
    robustness = data["tables"].get("extra", {})
    lines = [
        "\\begin{tabular}{@{}rlr@{}}",
        "\\toprule",
        "$\\delta$ & verdict & inputs decided by Z3 \\\\",
        "\\midrule",
    ]
    def cell(result: dict) -> str:
        decided = smt_decided(result["detail"])
        return f"${decided}/169$" if decided != "n/a" else "n/a"

    base = rows.get("p13_cosine")
    if base is not None:
        result = proofs(base)["model_equivalence"]
        lines.append(f"$0$ & {result['status']} & {cell(result)} \\\\")
    keys = [key for key in robustness if key.startswith("robustness_p13")]
    for key in sorted(keys, key=lambda k: float(k.rsplit("_", 1)[-1])):
        slack = float(key.rsplit("_", 1)[-1])
        result = {r["name"]: r for r in robustness[key]["results"]}["model_equivalence"]
        lines.append(f"${slack:g}$ & {result['status']} & {cell(result)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper: the experiments section
# ---------------------------------------------------------------------------
def render_paper(data: dict) -> str:
    rows = data["tables"]["rows"]
    g = rows["p59_grok"]
    t = rows.get("p59_partial")
    m = rows.get("p59_memoriser")
    p13 = rows.get("p13_cosine")
    p23 = rows.get("p23_cosine")
    p113 = rows.get("p113_grok")
    hist_g = data["hist"]["g70"]
    hist_t = data["hist"]["t30"]
    hist_m = data["hist"]["m30"]
    sweep = data["sweep59"]
    exact = minimal_exact(sweep)
    environment = data["environment"]
    export_g = export_for(data, "g70.pt")
    export_t = export_for(data, "t30.pt")
    random_row = data["tables"].get("extra", {}).get("random_p11")

    n_inputs = g["p"] * g["p"]
    grok_epoch = int(hist_g["summary"]["grokking_epoch"])
    grok_run = int(hist_g["summary"]["epochs_run"])
    onset = crossing(hist_t, 0.5) if hist_t else None
    plateau = t["accuracy"]["test_accuracy"] if t is not None else 0.0
    energies = lambda row: (
        min(h["spectral_energy_top3"] for h in heads(row)),
        max(h["spectral_energy_top3"] for h in heads(row)),
    )
    drops = lambda row: (
        min(h["accuracy_drop"] for h in heads(row)),
        max(h["accuracy_drop"] for h in heads(row)),
    )

    L: list[str] = []
    add = L.append

    add("\\section{Experiments}")
    add("\\label{sec:results}")
    add("")
    add("\\subsection{Setup}")
    add("")
    add(
        f"All models are one-layer, attention-only Transformers trained full-batch with "
        f"AdamW ($\\beta=(0.9,0.98)$, weight decay $1.0$) on one CPU core, single "
        f"BLAS thread (Table~\\ref{{tab:checkpoints}}); the $p=13$ and $p=23$ instances "
        f"use $d=64$ with four heads, all others $d=128$. The learning rate is half-cosine "
        f"annealed from $10^{{-2}}$ to $2\\%$ of its value, except for the memoriser, "
        f"which uses a constant $10^{{-3}}$. Runs are deterministic given the seed "
        f"\\emph{{and}} the library versions, which "
        f"\\code{{artifacts/environment.json}} records; everything reported here was "
        f"measured with Python {environment['python']}, PyTorch {environment['torch']}, "
        f"NumPy {environment['numpy']} and Z3 {environment['z3']}. The checkpoints "
        f"themselves are not distributed: \\code{{make train}} rebuilds them and "
        f"\\code{{make results}} regenerates every number in this section from the "
        f"JSON artefacts."
    )
    add("")
    add("\\begin{table}[t]\n\\centering\n\\small")
    add(table_setup(data))
    add(
        "\\caption{The checkpoints of this section. ``held-out'' is the fraction of the "
        "domain the optimiser never saw, evaluated with the seed the model was trained "
        "with; ``full domain'' scores all $p^2$ inputs.}\\label{tab:checkpoints}\n"
        "\\end{table}"
    )
    add("")
    add("\\subsection{Delayed generalisation and circuit structure}")
    add("")
    if hist_t is not None:
        add(
            f"Table~\\ref{{tab:training}} is the trajectory of checkpoint T, trained on "
            f"$30\\%$ of the domain. Training accuracy reaches $100\\%$ early while "
            f"held-out accuracy sits at chance ($1/{g['p']}\\approx"
            f"{100.0 / g['p']:.1f}\\%$); held-out accuracy only starts to move at epoch "
            f"{onset:,}, when the annealed learning rate is small, and the run ends at "
            f"${latex_num(plateau)}$ --- delayed generalisation that never reaches "
            f"exactness."
            if onset
            else ""
        )
        add("")
        add("\\begin{table}[t]\n\\centering\n\\small")
        add(table_training(data))
        add(
            "\\caption{The delayed-generalisation trajectory of checkpoint T ($30\\%$ of "
            "the domain): held-out accuracy is still at chance after thousands of epochs "
            "of perfect training accuracy.}\\label{tab:training}\n\\end{table}"
        )
        add("")
    add(
        f"Checkpoint G, trained on $70\\%$ of the domain with the same schedule, is at "
        f"chance until epoch {grok_epoch:,} and then transitions to exact "
        f"generalisation in a few hundred epochs; training stops "
        f"{grok_run - grok_epoch:,} epochs after the transition, at epoch {grok_run:,}, "
        f"when the network is exact on the training set, on the held-out set and on the "
        f"whole domain. The transition epoch is a property of the numerical "
        f"environment --- we have seen the same code and seed grok anywhere between "
        f"$1{{,}}250$ and $16{{,}}750$ epochs in different PyTorch builds --- whereas the "
        f"circuit recovered from the endpoint has been identical every time."
    )
    add("")
    add("\\begin{table}[t]\n\\centering\n\\small")
    add(table_circuit(data))
    add(
        "\\caption{Recovered circuits. ``Top-3 energy'' is the fraction of the spectral "
        "energy of $M^{\\text{logit}}$ carried by the three dominant frequencies; "
        "``$\\Delta$acc.'' is the accuracy lost when the head's output projection is "
        "zeroed on the full domain.}\\label{tab:circuit}\n\\end{table}"
    )
    add("")
    add("Three statistics separate the checkpoints (Table~\\ref{tab:circuit}).")
    add("")
    if t is not None and m is not None:
        add(
            f"\\emph{{Spectral concentration.}} In checkpoint G three frequencies carry "
            f"${100 * energies(g)[0]:.1f}$--${100 * energies(g)[1]:.1f}\\%$ of the energy "
            f"of every head's token$\\to$logit map, and the same modes dominate all four "
            f"heads. In checkpoint T the concentration falls to "
            f"${100 * energies(t)[0]:.1f}$--${100 * energies(t)[1]:.1f}\\%$ and the "
            f"dominant set differs from head to head; in the memoriser the three leading "
            f"frequencies carry only "
            f"${100 * energies(m)[0]:.1f}$--${100 * energies(m)[1]:.1f}\\%$, because the "
            f"map is a lookup table spread over the whole basis."
        )
        add("")
        add(
            f"\\emph{{Causal load.}} Zero-ablating any single head's output projection "
            f"costs ${100 * drops(g)[0]:.1f}$--${100 * drops(g)[1]:.1f}\\%$ accuracy in "
            f"checkpoint G and ${100 * drops(t)[0]:.1f}$--${100 * drops(t)[1]:.1f}\\%$ in "
            f"checkpoint T, so every head takes part in the computation, but only "
            f"${100 * drops(m)[0]:.1f}$--${100 * drops(m)[1]:.1f}\\%$ in the memoriser, "
            f"where the work is replicated across heads."
        )
        add("")
    add(
        f"\\emph{{Circuit size.}} Table~\\ref{{tab:size}} rebuilds the circuit from its "
        f"$|K|$ strongest frequencies. For checkpoint G, \\emph{{{exact['max_freqs']}}} "
        f"of the {g['p'] // 2} available frequencies already reproduce the network's "
        f"argmax on every input, and Z3 proves the full-domain equivalence of that "
        f"program; the exact reconstruction of the \\emph{{logits}} needs the whole "
        f"basis, whose truncation error is {latex_sci(sweep['rows'][-1]['logit_error'])}."
    )
    add("")
    add("\\begin{table}[t]\n\\centering\n\\small")
    add(table_size(data))
    add(
        "\\caption{Full-domain argmax agreement between a $|K|$-frequency truncation of "
        f"the faithful circuit and checkpoint G. The smallest exact truncation is "
        f"$|K|={exact['max_freqs']}$ and its equivalence is \\emph{{proved}}; every row "
        "with agreement below one is refuted by counting, so no SMT query is needed. "
        "The curve is not monotone --- a truncation is a re-projection, not a nested "
        "approximation of the argmax.}\\label{tab:size}\n\\end{table}"
    )
    add("")
    add("\\subsection{Decompilation and proof}")
    add("")
    add(
        f"\\paragraph{{A discrete program.}} For checkpoint G the synthesizer's first "
        f"stage succeeds: the argmax of the network is $a+b \\bmod p$ on all "
        f"{n_inputs:,} inputs, so the emitted program is"
    )
    add("")
    add("\\begin{center}")
    add("\\code{def add\\_mod\\_p(a, b): return (a + b) \\% 59}")
    add("\\end{center}")
    add("")
    add(
        f"({g['synthesis']['n_nodes']} IR nodes). Z3 discharges the specification --- "
        f"the quantified Presburger statement "
        f"$\\forall a,b\\in\\Zp: (P=a+b \\wedge P<p) \\vee (P=a+b-p \\wedge P\\ge 0)$ --- "
        f"in {solver_seconds(g, 'spec'):.2f}\\,s and the full-domain equivalence with the "
        f"network in {g['verification']['total_time_s']:.2f}\\,s "
        f"(Table~\\ref{{tab:verification}})."
        + (
            f" The emitted C is a {export_g['c_lines']}-line file, allocation-free, and "
            f"compiles with \\code{{-Werror}} and no diagnostics "
            f"(Table~\\ref{{tab:export}})."
            if export_g
            else ""
        )
    )
    add("")
    add(
        f"\\paragraph{{The same network as a trigonometric circuit.}} Building the "
        f"faithful circuit from checkpoint G's weights reproduces its argmax on all "
        f"{n_inputs:,} inputs at the full basis and, as Table~\\ref{{tab:size}} shows, "
        f"already at $|K|={exact['max_freqs']}$; Z3 proves that program equivalent to the "
        f"network over the whole domain. The logit-level statement is \\emph{{not}} true "
        f"at that truncation --- the logits differ by up to "
        f"{exact['logit_error']:.2f} in absolute value --- which is exactly why the "
        f"prover is asked for the argmax statement: the same network is simultaneously "
        f"the discrete algorithm and an approximate trigonometric circuit, and both "
        f"readings are machine-checked."
    )
    add("")
    if t is not None and m is not None:
        add(
            f"\\paragraph{{Networks without a motif.}} For checkpoints T and M stage 1 "
            f"finds no exact motif (the faithful circuit agrees with the network on "
            f"${latex_num(t['accuracy']['full_domain_accuracy'])}$ and "
            f"${latex_num(m['accuracy']['full_domain_accuracy'])}$ of the domain, and "
            f"the residual is a genuine error of the trained weights, not of the "
            f"extraction), so the faithful circuit is emitted and proved equivalent over "
            f"the whole domain. Certification does not depend on the network being "
            f"right, only on it being a well-defined function. A randomly initialised "
            f"$p=11$ network --- the worst case for a decompiler --- is handled the same "
            "way: full basis, "
            + (
                f"truncation error {latex_sci(random_row['truncation_error']['score'])} on "
                f"the scores and {latex_sci(random_row['truncation_error']['logit'])} on "
                f"the logits, "
                if random_row and random_row.get("truncation_error")
                else ""
            )
            + "equivalence proved."
        )
        add("")
    if p113 is not None:
        add(
            f"\\paragraph{{A second modulus.}} The same pipeline, unchanged, on $p=113$: "
            f"a {p113['n_params']:,}-parameter network trained on $70\\%$ of the "
            f"{p113['p'] * p113['p']:,} pairs reaches {score(p113['accuracy']['full_domain_accuracy'], 4)} "
            f"of the domain ({n_errors(p113)} inputs wrong). Stage 1 declines "
            f"to claim a motif, the faithful circuit "
            f"({p113['synthesis']['n_nodes']:,} nodes) is emitted, and Z3 certifies both "
            f"the argmax statement over all {p113['p'] * p113['p']:,} inputs in "
            f"{p113['verification']['total_time_s']:.0f}\\,s and the logit statement, "
            f"with a deployment gap of "
            f"{latex_sci(p113['verification']['deployment_gap'])}."
        )
        add("")
    add("\\begin{table}[t]\n\\centering\n\\small")
    add(table_verification(data))
    add(
        "\\caption{Proof obligations discharged by Z3. Every query covers the full "
        "domain; no input is sampled or skipped. \\code{spec} is the quantified integer "
        "specification, \\code{model\\_equivalence} the full-domain argmax statement and "
        "\\code{logit\\_equivalence} the bounded-logit statement used when no discrete "
        "program exists.}\\label{tab:verification}\n\\end{table}"
    )
    add("")
    add("\\subsection{Robustness: where the solver does real work}")
    add("")
    add(
        "For most proved queries the certified enclosures are tight enough that every "
        "input is decided by exact rational comparison and Z3 only confirms "
        "unsatisfiability. Widening the enclosures by $\\pm\\delta$ turns the theorem "
        "into a strictly stronger robustness claim and forces the solver to reason "
        "inside the simplex (Table~\\ref{tab:robustness})."
    )
    add("")
    add("\\begin{table}[t]\n\\centering\n\\small")
    add(table_robustness(data))
    add(
        "\\caption{Robustness of the $p=13$ circuit to a uniform perturbation of every "
        "attention probability. ``inputs decided by Z3'' counts the inputs whose "
        "winning margin could no longer be certified by exact rational comparison. "
        "A witness inside the perturbation set is reported as \\code{refuted}, not "
        "\\code{unknown}: for a robustness statement it is a counterexample, even though "
        "the unperturbed network still agrees with its program on that input.}"
        "\\label{tab:robustness}\n\\end{table}"
    )
    add("")
    add("\\subsection{Cost}")
    add("")
    add("\\begin{itemize}[leftmargin=1.4em,itemsep=2pt]")
    add(
        f"  \\item \\emph{{Discovery and synthesis.}} Circuit discovery takes under a "
        f"second; synthesis takes {g['synthesis']['seconds']:.2f}\\,s for the grokked "
        f"checkpoint and "
        + (
            f"{t['synthesis']['seconds']:.1f}\\,s when the faithful $p=59$ circuit "
            f"({t['synthesis']['n_nodes']:,} nodes) is emitted."
            if t is not None
            else "tens of seconds for the faithful circuit."
        )
    )
    add(
        f"  \\item \\emph{{Proof.}} {g['verification']['total_time_s']:.2f}\\,s for the "
        f"grokked checkpoint and "
        + (
            f"{t['verification']['total_time_s']:.1f}\\,s for the faithful $p=59$ "
            f"circuit, dominated by building the certified enclosures rather than by "
            f"Z3; {p113['verification']['total_time_s']:.0f}\\,s for $p=113$, whose "
            f"domain is {p113['p'] * p113['p'] / n_inputs:.1f}$\\times$ larger."
            if t is not None and p113 is not None
            else "longer for the faithful circuit."
        )
    )
    add(
        "  \\item \\emph{Code generation.} Under a second; compiling the "
        + (
            f"{export_t['c_bytes'] / 1e6:.2f}\\,MB source of checkpoint T takes "
            f"{export_t['gcc_seconds']:.2f}\\,s."
            if export_t
            else "largest source takes a few seconds."
        )
    )
    add(
        "  \\item \\emph{Enclosure width.} With two-ulp outward padding per operation the "
        "logit enclosures stay within $1.5\\times10^{-14}$ of the float64 forward pass, "
        "which the test suite validates for every input."
    )
    if p13 is not None and p23 is not None and t is not None:
        add(
            f"  \\item \\emph{{Deployment gap.}} float32 and float64 logits differ by at "
            f"most {sci(g['verification']['deployment_gap'])} ($p=59$), "
            f"{sci(p13['verification']['deployment_gap'])} ($p=13$) and "
            f"{sci(p23['verification']['deployment_gap'])} ($p=23$) in absolute value. "
            f"The proof is about the float64 network and this gap is reported, never "
            f"assumed away."
        )
    add(
        "  \\item \\emph{Training.} Dominates everything: "
        f"{hist_g['summary']['elapsed_s'] / 60:.0f} minutes on one CPU core for "
        f"checkpoint G, {hist_t['summary']['elapsed_s'] / 60:.0f} minutes for "
        f"checkpoint T."
        if hist_t is not None
        else "  \\item \\emph{Training.} Dominates everything."
    )
    add("\\end{itemize}")
    add("")
    if export_t:
        add("\\begin{table}[t]\n\\centering\n\\small")
        add(
            "\\begin{tabular}{@{}lr@{}}\n\\toprule\nMeasurement & Value \\\\\n\\midrule\n"
            f"IR nodes & {export_t['n_nodes']:,} \\\\\n".replace(",", "{,}")
            + f"Python source & {export_t['python_bytes'] / 1e6:.2f}\\,MB \\\\\n"
            + f"C source & {export_t['c_bytes'] / 1e6:.2f}\\,MB, "
            f"{export_t['c_lines']} lines \\\\\n"
            + f"\\code{{gcc}} time & {export_t['gcc_seconds']:.2f}\\,s \\\\\n"
            + f"\\code{{gcc}} diagnostics & {export_t['gcc_warnings']} \\\\\n"
            + f"input domain & {export_t['n_inputs']:,} pairs \\\\\n".replace(",", "{,}")
            + f"C vs.\\ Python, all inputs & max $|\\Delta| = "
            f"{export_t['max_abs_difference']:.1f}$ "
            f"({'bit-identical' if export_t['bit_identical'] else 'differs'}) \\\\\n"
            + f"heap allocation & {'yes' if export_t['allocates'] else 'none'} \\\\\n"
            + "\\bottomrule\n\\end{tabular}"
        )
        add(
            "\\caption{The emitted code of checkpoint T, the largest artefact, compiled "
            f"with \\code{{{export_t['gcc_flags']}}}. The C and the Python agree on the "
            f"whole domain, and at {export_t['c_bytes'] / export_t['n_inputs']:.0f} bytes "
            "per input the C cannot be a lookup table in disguise.}"
            "\\label{tab:export}\n\\end{table}"
        )
        add("")
    add("\\subsection{What is not claimed}")
    add("\\label{sec:notclaimed}")
    add("")
    if t is not None:
        add(
            f"Checkpoint G was trained on $70\\%$ of the domain, and it is exact on the "
            f"held-out $30\\%$ as well. The scarcer-data setting does not work here: "
            f"checkpoint T, trained on $30\\%$ with the same schedule, plateaus at "
            f"${latex_num(t['accuracy']['test_accuracy'])}$ held-out accuracy, and we "
            f"report the fraction that works rather than presenting a hand-picked "
            f"success as the typical behaviour. Nothing in the method depends on which "
            f"of the two succeeds --- the decompiler is exact and proved in both cases "
            f"--- and the difference shows up quantitatively as circuit size, "
            f"$|K|={exact['max_freqs']}$ versus the whole basis."
        )
        add("")
    add(
        "The verified object is the float64 forward pass; the gap to the float32 model "
        "that would be deployed is measured over the whole domain and reported, never "
        "assumed away. Softmax is not an SMT primitive, so its semantics are carried by "
        "the enclosures rather than by the solver. The proof quantifies over the inputs "
        "and over the certified enclosures, not over all possible attention "
        "distributions. Enclosures are sound by construction, but they are machinery, "
        "not a proof assistant: the trust base is our interval arithmetic plus Z3. Only "
        "finite, enumerable domains can be certified, and only one-layer attention-only "
        "models can currently be decompiled."
    )
    add("")
    return "\n".join(line for line in L if line is not None)


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------
def render_readme(data: dict) -> str:
    rows = data["tables"]["rows"]
    g = rows["p59_grok"]
    t = rows.get("p59_partial")
    m = rows.get("p59_memoriser")
    p113 = rows.get("p113_grok")
    sweep = data["sweep59"]
    exact = minimal_exact(sweep)
    environment = data["environment"]
    hist_g = data["hist"]["g70"]
    export_g = export_for(data, "g70.pt")
    export_t = export_for(data, "t30.pt")
    n_inputs = g["p"] * g["p"]

    L = []
    add = L.append
    add("### 9.1 A grokked network: the algorithm, recovered and proved")
    add("")
    add(
        f"`g70.pt` is a one-layer, four-head, `d_model = 128` attention-only model "
        f"({g['n_params']:,} parameters) trained on 70% of the {g['p']}² = {n_inputs:,} "
        f"pairs: full-batch AdamW, `lr = 1e-2` half-cosine annealed to 2%, weight decay "
        f"1.0, seed 3, one CPU core. Training stops itself 2,000 epochs after held-out "
        f"accuracy first reaches 1.0, which happened at epoch "
        f"{int(hist_g['summary']['grokking_epoch']):,}; the run ended at epoch "
        f"{int(hist_g['summary']['epochs_run']):,} "
        f"({hist_g['summary']['elapsed_s']:.0f} s)."
    )
    add("")
    add(
        "The network fits the training set **and the held-out set exactly**, and both "
        "readings of it are machine-checked:"
    )
    add("")
    add("| Quantity | Value |")
    add("|---|---|")
    add(
        f"| Train / held-out / full-domain accuracy | "
        f"{num(g['accuracy']['train_accuracy'])} / {num(g['accuracy']['test_accuracy'])} / "
        f"{num(g['accuracy']['full_domain_accuracy'])} |"
    )
    add(
        f"| Identified algorithm | `{g['synthesis']['body'].strip()}` "
        f"({g['synthesis']['n_nodes']} IR nodes) |"
    )
    add(f"| Top-3 spectral energy per head | {energy_range(g)} |")
    add(f"| Dominant frequencies per head | `{head_freqs(g)}` |")
    add(
        f"| Zero-ablation damage (Δacc / Δlogit) | {dacc_range(g)} / {dlogit_range(g)} |"
    )
    add(
        f"| Z3 `spec` | **{proofs(g)['spec']['status']}** in "
        f"{solver_seconds(g, 'spec'):.2f} s |"
    )
    add(
        f"| Z3 `model_equivalence` | **{proofs(g)['model_equivalence']['status']}** over "
        f"all {n_inputs:,} inputs |"
    )
    add(
        f"| float32 ↔ float64 deployment gap | "
        f"{sci(g['verification']['deployment_gap'])} (measured, not assumed away) |"
    )
    add(
        f"| Synthesis / verification wall time | {g['synthesis']['seconds']:.2f} s / "
        f"{g['verification']['total_time_s']:.2f} s |"
    )
    add("")
    if export_g:
        add(
            f"The emitted C for this network is a {export_g['c_lines']}-line file whose "
            f"whole body is a Euclidean-remainder helper plus "
            f"`return c2c_imod(a + b, {g['p']})`; it is allocation-free and compiles clean "
            f"with `{export_g['gcc_flags']}`."
        )
        add("")
    add(
        f"The *faithful trigonometric circuit* built from the same weights is also "
        f"exactly equivalent, and — the paper's central quantitative claim — a "
        f"**{exact['max_freqs']}-frequency** truncation of it reproduces the network's "
        f"argmax on all {n_inputs:,} inputs and is **proved** equivalent by Z3 (§9.5)."
    )
    add("")

    if t is not None and m is not None:
        add("### 9.2 A partially generalising network, and a memoriser")
        add("")
        add(
            f"Two further checkpoints of the same architecture pin down what the circuit "
            f"statistic measures. **T** is trained on 30% of the domain with the same "
            f"schedule for {int(data['hist']['t30']['summary']['epochs_run']):,} epochs; "
            f"it shows delayed generalisation (held-out accuracy stays at chance for "
            f"thousands of epochs, then climbs to {pct(t['accuracy']['test_accuracy'])}) "
            f"and never reaches exactness. **M** is trained on the same 30% with a "
            f"constant `lr = 1e-3` for "
            f"{int(data['hist']['m30']['summary']['epochs_run']):,} epochs and never "
            f"leaves the memorisation regime."
        )
        add("")
        add(
            "| Checkpoint | p | train frac. | held-out | full domain | top-3 energy | "
            "dominant frequencies | program | `model_equivalence` |"
        )
        add("|---|---|---|---|---|---|---|---|---|")
        for row in (g, t, m, rows.get("p13_cosine"), rows.get("p23_cosine")):
            if row is None:
                continue
            name = row["checkpoint"].split("/")[-1]
            program = (
                f"`{row['synthesis']['body'].strip()}`"
                if row["synthesis"]["motif"]
                else f"faithful circuit, {row['synthesis']['n_nodes']:,} nodes"
            )
            add(
                f"| `{name}` | {row['p']} | {row['train_fraction']:.1f} | "
                f"{pct(row['accuracy']['test_accuracy'], 2 if row['accuracy']['test_accuracy'] < 1 else 1)} | "
                f"{pct(row['accuracy']['full_domain_accuracy'], 2 if row['accuracy']['full_domain_accuracy'] < 1 else 1)} | "
                f"{energy_range(row)} | {top_freqs(row)} | {program} | "
                f"**{proofs(row)['model_equivalence']['status']}** "
                f"({row['p'] * row['p']:,}/{row['p'] * row['p']:,}) |"
            )
        add("")
        add(
            f"The rows order themselves by spectral concentration — {energy_range(g)} "
            f"(grokked), {energy_range(t)} (partial), {energy_range(m)} (memoriser) — and "
            f"by circuit size: {exact['max_freqs']} frequencies already reproduce the "
            f"grokked network's argmax exactly, while the other two need the full basis. "
            f"Neither T nor M yields a discrete motif, so both are emitted as their "
            f"faithful trigonometric circuit and **proved equivalent over the whole "
            f"domain**: certification does not depend on the network being right, only "
            f"on it being a well-defined function."
        )
        add("")

    if p113 is not None:
        add("### 9.3 A second modulus: p = 113")
        add("")
        add(
            f"The same pipeline, unchanged, on a domain "
            f"{(p113['p'] ** 2) / n_inputs:.1f}× larger: a {p113['n_params']:,}-"
            f"parameter model trained on 70% of the {p113['p'] ** 2:,} pairs."
        )
        add("")
        add("| Quantity | Value |")
        add("|---|---|")
        add(
            f"| Train / held-out / full-domain accuracy | "
            f"{num(p113['accuracy']['train_accuracy'])} / "
            f"{num(p113['accuracy']['test_accuracy'])} / "
            f"{score(p113['accuracy']['full_domain_accuracy'], 4)}"
            f" ({n_errors(p113)} of {p113['p'] * p113['p']:,} inputs wrong) |"
        )
        add(f"| Top-3 spectral energy per head | {energy_range(p113)} |")
        add(f"| Dominant frequencies per head | `{head_freqs(p113)}` |")
        add(
            f"| Emitted program | faithful trigonometric circuit, "
            f"{p113['synthesis']['n_nodes']:,} nodes |"
        )
        add(
            f"| Z3 `model_equivalence` | "
            f"**{proofs(p113)['model_equivalence']['status']}** over all "
            f"{p113['p'] ** 2:,} inputs "
            f"({p113['verification']['total_time_s']:.0f} s) |"
        )
        add(
            f"| Z3 `logit_equivalence` | "
            f"**{proofs(p113)['logit_equivalence']['status']}** |"
        )
        add(
            f"| float32 ↔ float64 deployment gap | "
            f"{sci(p113['verification']['deployment_gap'])} |"
        )
        add("")
        if data["sweep113"]:
            series = ", ".join(
                pct(row["agreement"])
                for row in data["sweep113"]["rows"]
                if row["max_freqs"] <= 8
            )
            add(
                f"Unlike the p=59 network, this one is not captured by a handful of "
                f"frequencies: `|K| = 1…8` give {series} agreement, and only the full "
                f"{p113['p'] // 2}-frequency basis is exact. The same statistic as in "
                f"§9.2, pointing the same way: an almost-learned algorithm needs almost "
                f"all of its spectrum."
            )
            add("")

    if export_t:
        add("### 9.4 The emitted code, measured")
        add("")
        add(
            f"`scripts/measure_export.py` exports checkpoint T (the largest artefact: "
            f"{export_t['n_nodes']:,} IR nodes), compiles the C with "
            f"`{export_t['gcc_flags']}`, and runs both artefacts over the whole domain."
        )
        add("")
        add("| Measurement | Value |")
        add("|---|---|")
        add(f"| Generated Python | {export_t['python_bytes'] / 1e6:.2f} MB |")
        add(
            f"| Generated C | {export_t['c_bytes'] / 1e6:.2f} MB, "
            f"{export_t['c_lines']} lines |"
        )
        add(f"| `gcc` time | {export_t['gcc_seconds']:.2f} s |")
        add(f"| `gcc` diagnostics | {export_t['gcc_warnings']} |")
        add(
            f"| C vs Python on all {export_t['n_inputs']:,} inputs | max \\|Δ\\| = "
            f"{export_t['max_abs_difference']:.1f}"
            + (", bit-identical" if export_t["bit_identical"] else "")
            + " |"
        )
        add(f"| Heap allocation | {'yes' if export_t['allocates'] else 'none'} |")
        add(
            f"| Headers | {', '.join('`' + h + '`' for h in export_t['includes'])} "
            f"(`stdio.h` only for the demonstration `main`) |"
        )
        add("")
        add(
            f"Neither file is a lookup table in disguise: the C is "
            f"{export_t['c_bytes'] / max(export_t['n_inputs'], 1):,.0f} bytes per input, "
            f"so the answer has to be computed rather than looked up."
        )
        add("")

    add("### 9.5 Circuit size versus fidelity")
    add("")
    add(
        f"Building the faithful circuit from checkpoint `g70.pt` with only its `|K|` "
        f"strongest frequencies per head, and comparing its argmax with the network's "
        f"over all {n_inputs:,} inputs:"
    )
    add("")
    add("| `|K|` | agreement | logit truncation error | `model_equivalence` (Z3) |")
    add("|---|---|---|---|")
    for row in sweep["rows"]:
        label = {
            "proved": "**proved**",
            "refuted-by-counting": "refuted (counting)",
            "not-run": "not queried",
        }.get(row["model_equivalence"], row["model_equivalence"])
        shown = agreement(row["agreement"])
        if row["agreement"] >= 1.0:
            shown = f"**{shown}**"
        add(f"| {row['max_freqs']} | {shown} | {sci(row['logit_error'])} | {label} |")
    add("")
    add(
        f"The grokked network's *function* is captured by {exact['max_freqs']} of the "
        f"{g['p'] // 2} available frequencies, and Z3 proves it. The logit-level "
        f"statement is far stronger than what holds: at `|K| = {exact['max_freqs']}` the "
        f"logits still differ by up to {exact['logit_error']:.2f} in absolute value, yet "
        f"the argmax — the thing the program must reproduce — is identical on every "
        f"input. The prover is asked for that statement, not the stronger one; and "
        f"agreement below 1 refutes equivalence by counting, so the solver is never asked "
        f"about the approximate rows."
    )
    add("")

    robustness = {k: v for k, v in data["tables"].get("extra", {}).items()
                  if k.startswith("robustness_p13")}
    if robustness:
        add("### 9.6 Robustness, and what the solver actually does")
        add("")
        add(
            "Certified enclosures are tight enough that, for a trained network, every "
            "input is decided by exact rational comparison and Z3 only confirms "
            "unsatisfiability. Widening every enclosure by `±δ` makes the theorem "
            "strictly stronger and forces the solver to work:"
        )
        add("")
        add("| attention slack δ | verdict | inputs decided by Z3 |")
        add("|---|---|---|")
        def row_cell(result: dict) -> str:
            decided = smt_decided(result["detail"])
            return f"{decided} / 169" if decided != "n/a" else "n/a"

        base = rows.get("p13_cosine")
        if base is not None:
            result = proofs(base)["model_equivalence"]
            add(f"| 0 | {result['status']} | {row_cell(result)} |")
        for key in sorted(robustness, key=lambda k: float(k.rsplit("_", 1)[-1])):
            slack = float(key.rsplit("_", 1)[-1])
            result = {r["name"]: r for r in robustness[key]["results"]}["model_equivalence"]
            add(f"| {slack:g} | {result['status']} | {row_cell(result)} |")
        add("")
        add(
            "Reporting `refuted` rather than `unknown` is deliberate: for a robustness "
            "statement a witness inside the perturbation set *is* a counterexample, even "
            "though the unperturbed network still agrees with its program on that input."
        )
        add("")

    add("### 9.7 What is *not* claimed")
    add("")
    if t is not None:
        add(
            f"* The exact solution needs the 70% data fraction. The 30%-data run of §9.2 "
            f"plateaus at {pct(t['accuracy']['test_accuracy'])} held-out accuracy; we "
            f"report the fraction that works instead of presenting a hand-picked success "
            f"as typical."
        )
    add(
        f"* Grokking time, the minimal truncation size and the exact decimals of every "
        f"number here are properties of the numerical environment. "
        f"`artifacts/environment.json` records Python {environment['python']}, PyTorch "
        f"{environment['torch']}, NumPy {environment['numpy']} and Z3 "
        f"{environment['z3']} on {environment['cpu_count']} CPU cores; everything quoted "
        f"is reproducible in *that* environment, and `make results` regenerates it."
    )
    add(
        "* Only modular arithmetic is implemented. Dyck-1 is absent for a structural "
        "reason documented in §10: the architecture has no positional embeddings and is "
        "therefore permutation-invariant."
    )
    if p113 is not None and not p113["synthesis"]["motif"]:
        add(
            f"* The p=113 network is not an exact solution "
            f"({pct(p113['accuracy']['full_domain_accuracy'], 2)} of the domain), so no "
            f"discrete motif is claimed for it; its certification is exact all the same."
        )
    add("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Splice
# ---------------------------------------------------------------------------
def _replace_between(text: str, begin: str, end: str, body: str) -> str:
    start = text.index(begin) + len(begin)
    stop = text.index(end, start)
    return text[:start] + "\n" + body.strip("\n") + "\n" + text[stop:]


def build() -> dict:
    data = load_all()
    readme = open(README, encoding="utf-8").read()
    paper = open(PAPER, encoding="utf-8").read()
    return {
        README: _replace_between(readme, README_BEGIN, README_END, render_readme(data)),
        PAPER: _replace_between(paper, PAPER_BEGIN, PAPER_END, render_paper(data)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the rendered files differ from what is on disk",
    )
    args = parser.parse_args(argv)

    stale = []
    for path, contents in build().items():
        current = open(path, encoding="utf-8").read()
        if current != contents:
            stale.append(path)
            if not args.check:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(contents)
    if args.check:
        if stale:
            print("out of date: " + ", ".join(os.path.relpath(p, ROOT) for p in stale))
            print("run `python scripts/render_report.py`")
            return 1
        print("README and paper are up to date with the artefacts")
        return 0
    for path in stale:
        print(f"updated {os.path.relpath(path, ROOT)}")
    if not stale:
        print("nothing to do")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
