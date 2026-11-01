# Circuit2Code

**A neuro-symbolic decompiler for algorithmic Transformers.**
Circuit2Code takes a small Transformer, recovers the *circuit* that implements
its function, synthesizes an equivalent symbolic program, and **proves** the
equivalence with the Z3 theorem prover — then emits dependency-free Python and
ANSI C. When the network implements a discrete algorithm the program *is* that
algorithm (e.g. `(a + b) % p`); when it only approximates one, the program is the
network's own trigonometric circuit, exact and certified over the whole input
domain.

```
        Transformer                 circuit                    program              proof
  ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐   ┌──────────────┐
  │ attention-only net │──▶│ heads, W_QK / W_OV │──▶│ IR  (a + b) % p    │──▶│ Z3: ∀ a,b.   │
  │ trained on a+b%p   │   │ Fourier spectrum   │   │ Python + C99       │   │ net = prog   │
  └────────────────────┘   └────────────────────┘   └────────────────────┘   └──────────────┘
```

---

## 1. What is in this repository

| Stage | Module | What it does |
|---|---|---|
| Model | `src/models/transformer.py` | Minimal, fully transparent Transformer (no LayerNorm, explicit weights, complete activation trace). |
| Training | `src/models/train.py` | Deterministic full-batch AdamW loop; grokking on `a + b (mod p)`. |
| Probing | `src/interpretability/probing.py` | Activation cache, denoising patches, attribution patching, weight ablation. |
| Circuit discovery | `src/interpretability/circuit_finder.py` | Zero-ablation, direct logit attribution, `W_QK`/`W_OV`, DFT projection, frequency support. |
| IR | `src/synthesis/ast_nodes.py` | Expression language with NumPy semantics, interval semantics and rendering. |
| Basis extraction | `src/synthesis/symbolic_reg.py` | Fourier, modular-shift and ReLU bases; greedy sparse fitting. |
| Synthesis | `src/synthesis/program_synthesizer.py` | Circuit-grounded motif search + faithful trig-circuit decompilation. |
| Translation | `src/verification/z3_translator.py` | Rigorous enclosures of the float forward pass, exact rational Z3 encoding of layers and IR. |
| Proving | `src/verification/verifier.py` | Specification, full-domain model-equivalence and bounded logit-equivalence proofs; counterexample-guided refinement. |
| Codegen | `src/codegen/emit_python.py`, `emit_c.py` | Idiomatic Python and ISO C99 output. |
| Demo | `demo/cli.py` | `train`, `extract-circuit`, `decompile`, `verify`, `export`. |
| Paper | `paper/main.tex` | NeurIPS-style write-up of the method and results. |

---

## 2. Scientific background

### 2.1 The task and the grokking regime

The primary domain is modular addition: for a prime `p`, learn
`f(a, b) = (a + b) mod p` from a fraction of the `p²` possible inputs. In this
regime a one-layer, attention-only Transformer trained with full-batch AdamW
exhibits **grokking** (Power et al., 2022): training accuracy reaches 100% early
while held-out accuracy stays at chance, and only after many further optimisation
steps does the network generalise. Which data fraction grokks is not obvious and
is worth reporting honestly: in our sweep it is 70% (grokked at epoch 1,250 in
the environment recorded in `artifacts/environment.json`, and at 16,750 in an
older PyTorch build), while every 30%-data run we trained stalled at 77% held-out
accuracy — the textbook "grokks *because* data is scarce" story did not reproduce
here, and §9.7 says so instead of hiding it.

### 2.2 The trigonometric circuit

Nanda et al. (2023) showed that the generalising solution is a discrete Fourier
algorithm. Writing `ω_k = 2πk/p`, the network

1. embeds tokens on a Fourier basis (`cos ω_k a`, `sin ω_k a`),
2. uses the bilinear attention form to *multiply* the two operand embeddings,
   which by the product-to-sum identity produces the frequency `k(a + b)`,
3. reads the result out through a second Fourier basis, so that the logit of
   class `c` is a sum of `cos ω_k(a + b − c)` terms.

Circuit2Code measures this structure directly from the weights:

```
W_QK = W_Q W_Kᵀ                     (what attends to what)
W_OV = W_V W_Oᵀ                     (what is written to the residual stream)
W_E W_QK W_Eᵀ                       (token → token attention scores)
W_E W_OV W_U                        (token → class logits)
```

and projects the token-space matrices onto the DFT basis of `Z_p`, reporting the
frequencies that carry the algorithm.

### 2.3 From circuit to program

Given the frequency support and the shift-invariance of the `W_E W_OV W_U` maps,
the synthesizer instantiates a small DSL of candidate algorithms
(`(a+b) % p`, `(a−b) % p`, `(a·b) % p`, identity projections) and accepts a
candidate only if it reproduces the network's argmax on **every** input in the
domain. If no discrete motif is exact, the synthesizer emits the faithful
trigonometric circuit instead:

```
scores    s_{h,j} = Σ_k a_{h,j,k} cos(ω_k j) + b_{h,j,k} sin(ω_k j)
attention P       = softmax_j(s_{h,j})
class     ℓ_c     = β_c + Σ_{h,j} P_{h,j} · g_{h,c}(j)
answer            = argmax_c ℓ_c
```

where `g_{h,c}` is the truncated Fourier series of head `h`'s token-to-logit map.
Both forms are IR (`Program`) objects consumed by the code generators and the
prover.

---

## 3. Formal verification

Three theorems are discharged by Z3 (`src/verification/verifier.py`).

### 3.1 Specification (`spec`)

For the synthesized discrete program `P(a,b)`, Z3 proves the quantified integer
statement

```
∀ a, b ∈ {0,…,p−1}:  (P = a + b ∧ P < p) ∨ (P = a + b − p ∧ P ≥ 0)
```

which is the *mathematical* definition of modular addition — not a restatement of
the program. This query involves quantified Presburger arithmetic with `mod`, and
is decided in milliseconds.

### 3.2 Full-domain model equivalence (`model_equivalence`)

For every one of the `p²` inputs, the network's argmax must equal `P(a,b)`. The
network is translated into **exact rational linear arithmetic**, in which the
only non-algebraic objects are the attention probabilities. Each of those is a
`Real` variable constrained to a *certified enclosure* of the IEEE-754 forward
pass and to the probability simplex. The enclosures are produced by
`compute_logit_bounds`, which propagates Higham-`γₙ` rounding-error budgets
through every matmul, an interval softmax, and outward-rounded additions:

```
|fl(⟨x, w⟩) − ⟨x, w⟩| ≤ γₙ Σ|xᵢwᵢ|,      γₙ = n·u/(1 − n·u),  u = 2⁻⁵³.
```

The proof query is `∃ x, c ≠ P(x): ℓ_c(x) ≥ ℓ_{P(x)}(x)` with the attention
variables bounded as above, batched over the domain. `unsat` for every batch is
the proof; `sat` yields a candidate counterexample that is checked against the
real network before anything is reported. No floating-point comparison decides
the verdict: the enclosure endpoints reach Z3 as exact rationals
(`fractions.Fraction`).

Optional: `VerifyConfig(attention_slack=δ)` strengthens the theorem to
*robustness* — the argmax stays correct under a `±δ` perturbation of every
attention probability. This is the regime in which the SMT solver does the heavy
lifting rather than the (tight) enclosures.

### 3.3 Bounded logit equivalence (`logit_equivalence`)

For faithful trig-circuit decompilations, where no discrete program exists, the
prover establishes

```
∀ x, ∀ c:  |ℓ_c^{network}(x) − ℓ_c^{program}(x)| ≤ ε
```

again with the network represented by certified rational enclosures and the
program's value exact.

### 3.4 Trust boundary (stated explicitly)

* The verified object is the **float64** forward pass. The gap to the float32
  model actually deployed is measured over the whole domain and reported
  (`deployment_gap`, typically `1e-5` absolute on logits).
* Interval enclosures are sound by construction (outward rounding, 2 ulp padding
  per operation, `γₙ` error budgets). They are validated in `tests/test_verifier.py`
  against the float64 forward pass on every input.
* A `sat` answer from Z3 is never accepted as a counterexample unless the real
  network reproduces it; otherwise the result is reported as `unknown`.

---

## 4. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # or: pip install -e ".[dev]"
```

Python ≥ 3.10, PyTorch ≥ 2.2, `z3-solver` ≥ 4.12, `sympy`, `numpy`, `scipy`,
`rich`, `pytest`. A CUDA device is not required: every result in this repository
was produced CPU-only with one BLAS thread, which is also what makes a run
deterministic (`make train` passes `--torch-threads 1`).

## 5. Quickstart

```bash
# train the reference checkpoint (this is the configuration that groks: 70% of
# the domain, cosine-annealed lr 1e-2, seed 3; it stops itself 2,000 epochs after
# held-out accuracy first reaches 1.0: 1,250 epochs / 4.6 min in the environment
# of artifacts/environment.json, and 16,750 epochs / 48 min in an older PyTorch)
python demo/cli.py train --p 59 --epochs 30000 --train-fraction 0.7 --seed 3 \
    --checkpoint-every 1000 --out artifacts/checkpoints/g70.pt

# 1. locate the circuit (head ablation + DFT projection)
python demo/cli.py extract-circuit --checkpoint artifacts/checkpoints/g70.pt \
    --train-fraction 0.7

# 2. decompile to a symbolic program, emit Python + C
python demo/cli.py decompile --checkpoint artifacts/checkpoints/g70.pt \
    --train-fraction 0.7 --out artifacts/decompiled

# 3. prove equivalence with Z3
python demo/cli.py verify --checkpoint artifacts/checkpoints/g70.pt \
    --train-fraction 0.7 --json artifacts/verification.json
```

The generated C compiles with

```bash
gcc -std=c99 -O3 -Wall -Wextra -pedantic artifacts/decompiled/add_mod_p.c -o add_mod_p
```

and has no heap allocation, no global state and no dependency beyond `math.h`.

One command runs the whole pipeline (train → extract → decompile → verify →
export) on a tiny model in about five seconds:

```bash
python demo/cli.py demo          # writes artifacts/demo/
```

A one-minute end-to-end smoke test (tiny model, full-domain training):

```bash
python demo/cli.py train --p 5 --d-model 16 --n-heads 2 --init-std 0.5 \
    --epochs 800 --lr 5e-3 --weight-decay 1e-3 --train-fraction 1.0 \
    --out /tmp/add5.pt
python demo/cli.py decompile --checkpoint /tmp/add5.pt --train-fraction 1.0 --out /tmp/out
python demo/cli.py verify    --checkpoint /tmp/add5.pt --train-fraction 1.0
```

## 6. Library API

```python
from src.models.train import TrainConfig, train_modular_addition
from src.models.transformer import modular_addition_dataset
from src.interpretability import extract_circuit
from src.synthesis import synthesize
from src.verification import Verifier, VerifyConfig
from src.codegen import emit_python, emit_c

# the recipe that grokks (section 5): 70% of the domain, seed 3
result = train_modular_addition(
    TrainConfig(p=59, epochs=30000, train_fraction=0.7, seed=3)
)
dataset = modular_addition_dataset(59, train_fraction=0.7, seed=3)
circuit = extract_circuit(result.model, dataset)
synth = synthesize(result.model, dataset, circuit=circuit)
report = Verifier(result.model, synth.program, VerifyConfig()).verify_all()

print(circuit.summary())     # Circuit(p=59, heads=[L0H2, L0H0, ...], frequencies=[...])
print(synth.narrative())     # "The circuit computes the discrete algorithm `add_mod` ..."
print(report.verified)       # True  <-- machine-checked over all p^2 inputs
open("add.c", "w").write(emit_c(synth.program))
```

## 7. Reproducibility

* `torch.manual_seed(seed)` before every model construction; the data split is
  generated by its own seeded generator, so initialisation and split are
  independent.
* Full-batch training only: no shuffling, no dropout, no data augmentation.
* `TrainConfig(torch_threads=k)` pins the BLAS thread count, which fixes the
  reduction order: with the same library versions a run is bitwise reproducible
  (thread count and BLAS/library versions are part of the recipe).
* `tests/test_transformer.py::test_training_is_reproducible` trains twice and
  asserts identical weights.

## 8. Testing

```bash
make test          # 118 tests: model, training, probing, circuit discovery, IR,
                   # synthesis, Z3, codegen, CLI
make test-fast     # 110 tests: skips the eight that train a model (marker "slow")
```

`make` uses `.venv/bin/python` when the virtualenv exists and `python3`
otherwise; override with `make test PYTHON=/path/to/python`.

The code-generation tests do not only compare strings: the emitted Python is
`exec`'d and the emitted C is compiled with
`gcc -std=c99 -O3 -Wall -Wextra -pedantic -Werror`, run over the entire input
domain, and compared against the IR reference semantics.

## 9. Results

Everything in this section is generated from JSON artefacts by
`scripts/render_report.py`; the same generator writes the experiments section of
`paper/main.tex`, and `tests/test_docs.py` fails if either file is out of date.
Training checkpoints themselves are **not** committed (`artifacts/checkpoints/`
is ignored) — `make train` rebuilds them, and `make results` rebuilds every
number here.

| Artifact | Contents |
|---|---|
| `artifacts/paper_tables.json` | per-checkpoint accuracy, per-head circuit statistics, synthesis provenance and the Z3 verdict of every proof obligation |
| `artifacts/truncation_sweep_p59.json`, `..._p113.json` | full-domain argmax agreement of `|K|`-frequency truncations (`|K| = 1…12` and `29` for p=59), with the verdict of each exactly-agreeing truncation |
| `artifacts/frequency_budget.json` | the spectral summary of every checkpoint |
| `artifacts/export_measurements.json` | size, compile time and C↔Python agreement of the emitted code |
| `artifacts/environment.json` | Python/PyTorch/Z3/NumPy versions and CPU of the machine that produced all of the above |
| `artifacts/history/*.json` | the training trajectories quoted in §9.1–§9.2 |

```bash
make train      # the three p=59 checkpoints used below (70%, 30%, memoriser)
make results    # every JSON artefact, then regenerate this section and the paper
```

<!-- BEGIN GENERATED: results  (scripts/render_report.py) -->
### 9.1 A grokked network: the algorithm, recovered and proved

`g70.pt` is a one-layer, four-head, `d_model = 128` attention-only model (80,768 parameters) trained on 70% of the 59² = 3,481 pairs: full-batch AdamW, `lr = 1e-2` half-cosine annealed to 2%, weight decay 1.0, seed 3, one CPU core. Training stops itself 2,000 epochs after held-out accuracy first reaches 1.0, which happened at epoch 1,250; the run ended at epoch 3,250 (277 s).

The network fits the training set **and the held-out set exactly**, and both readings of it are machine-checked:

| Quantity | Value |
|---|---|
| Train / held-out / full-domain accuracy | 1.000 / 1.000 / 1.000 |
| Identified algorithm | `((a + b) % 59)` (4 IR nodes) |
| Top-3 spectral energy per head | 0.979 – 0.995 |
| Dominant frequencies per head | `{22, 15, 28} / {22, 15, 28} / {22, 28, 15} / {28, 22, 3}` |
| Zero-ablation damage (Δacc / Δlogit) | 0.757 – 0.888 / 17.00 – 39.80 |
| Z3 `spec` | **proved** in 0.02 s |
| Z3 `model_equivalence` | **proved** over all 3,481 inputs |
| float32 ↔ float64 deployment gap | 6.25e-5 (measured, not assumed away) |
| Synthesis / verification wall time | 0.21 s / 1.42 s |

The emitted C for this network is a 35-line file whose whole body is a Euclidean-remainder helper plus `return c2c_imod(a + b, 59)`; it is allocation-free and compiles clean with `-std=c99 -O3 -Wall -Wextra -pedantic -Werror`.

The *faithful trigonometric circuit* built from the same weights is also exactly equivalent, and — the paper's central quantitative claim — a **10-frequency** truncation of it reproduces the network's argmax on all 3,481 inputs and is **proved** equivalent by Z3 (§9.5).

### 9.2 A partially generalising network, and a memoriser

Two further checkpoints of the same architecture pin down what the circuit statistic measures. **T** is trained on 30% of the domain with the same schedule for 30,000 epochs; it shows delayed generalisation (held-out accuracy stays at chance for thousands of epochs, then climbs to 77.4%) and never reaches exactness. **M** is trained on the same 30% with a constant `lr = 1e-3` for 40,000 epochs and never leaves the memorisation regime.

| Checkpoint | p | train frac. | held-out | full domain | top-3 energy | dominant frequencies | program | `model_equivalence` |
|---|---|---|---|---|---|---|---|---|
| `g70.pt` | 59 | 0.7 | 100.0% | 100.0% | 0.979 – 0.995 | 22, 28, 15 | `((a + b) % 59)` | **proved** (3,481/3,481) |
| `t30.pt` | 59 | 0.3 | 77.39% | 84.17% | 0.928 – 0.958 | 11, 29, 6 | faithful circuit, 5,763 nodes | **proved** (3,481/3,481) |
| `m30.pt` | 59 | 0.3 | 30.28% | 51.19% | 0.241 – 0.373 | 16, 20, 2 | faithful circuit, 5,763 nodes | **proved** (3,481/3,481) |
| `cos13_60k.pt` | 13 | 0.5 | 72.94% | 86.39% | 0.916 – 0.990 | 4, 6, 5 | faithful circuit, 1,255 nodes | **proved** (169/169) |
| `cos2.pt` | 23 | 0.5 | 74.34% | 87.15% | 0.953 – 0.974 | 3, 2, 6 | faithful circuit, 2,235 nodes | **proved** (529/529) |

The rows order themselves by spectral concentration — 0.979 – 0.995 (grokked), 0.928 – 0.958 (partial), 0.241 – 0.373 (memoriser) — and by circuit size: 10 frequencies already reproduce the grokked network's argmax exactly, while the other two need the full basis. Neither T nor M yields a discrete motif, so both are emitted as their faithful trigonometric circuit and **proved equivalent over the whole domain**: certification does not depend on the network being right, only on it being a well-defined function.

### 9.3 A second modulus: p = 113

The same pipeline, unchanged, on a domain 3.7× larger: a 94,592-parameter model trained on 70% of the 12,769 pairs.

| Quantity | Value |
|---|---|
| Train / held-out / full-domain accuracy | 1.000 / 0.999 / 0.99977 (3 of 12,769 inputs wrong) |
| Top-3 spectral energy per head | 0.892 – 0.995 |
| Dominant frequencies per head | `{28, 25, 21} / {28, 25, 21} / {21, 50, 8} / {25, 38, 12}` |
| Emitted program | faithful trigonometric circuit, 11,055 nodes |
| Z3 `model_equivalence` | **proved** over all 12,769 inputs (501 s) |
| Z3 `logit_equivalence` | **proved** |
| float32 ↔ float64 deployment gap | 1.83e-4 |

Unlike the p=59 network, this one is not captured by a handful of frequencies: `|K| = 1…8` give 13.8%, 23.5%, 35.8%, 47.5%, 51.1%, 61.5%, 79.5%, 87.8% agreement, and only the full 56-frequency basis is exact. The same statistic as in §9.2, pointing the same way: an almost-learned algorithm needs almost all of its spectrum.

### 9.4 The emitted code, measured

`scripts/measure_export.py` exports checkpoint T (the largest artefact: 5,763 IR nodes), compiles the C with `-std=c99 -O3 -Wall -Wextra -pedantic -Werror`, and runs both artefacts over the whole domain.

| Measurement | Value |
|---|---|
| Generated Python | 2.21 MB |
| Generated C | 2.08 MB, 869 lines |
| `gcc` time | 2.50 s |
| `gcc` diagnostics | 0 |
| C vs Python on all 3,481 inputs | max \|Δ\| = 0.0, bit-identical |
| Heap allocation | none |
| Headers | `math.h`, `stdio.h` (`stdio.h` only for the demonstration `main`) |

Neither file is a lookup table in disguise: the C is 599 bytes per input, so the answer has to be computed rather than looked up.

### 9.5 Circuit size versus fidelity

Building the faithful circuit from checkpoint `g70.pt` with only its `|K|` strongest frequencies per head, and comparing its argmax with the network's over all 3,481 inputs:

| `|K|` | agreement | logit truncation error | `model_equivalence` (Z3) |
|---|---|---|---|
| 1 | 0.2390 | 6.05e1 | refuted (counting) |
| 2 | 0.5248 | 3.14e1 | refuted (counting) |
| 3 | 0.8995 | 1.40e1 | refuted (counting) |
| 4 | 0.9879 | 1.05e1 | refuted (counting) |
| 5 | 0.9948 | 1.03e1 | refuted (counting) |
| 6 | 0.9954 | 8.08e0 | refuted (counting) |
| 7 | 0.9989 | 8.10e0 | refuted (counting) |
| 8 | 0.9997 | 8.31e0 | refuted (counting) |
| 9 | 0.9994 | 7.66e0 | refuted (counting) |
| 10 | **1.0000** | 6.66e0 | **proved** |
| 11 | 0.9994 | 5.68e0 | refuted (counting) |
| 12 | 0.9997 | 4.88e0 | refuted (counting) |
| 29 | **1.0000** | 9.17e-13 | **proved** |

The grokked network's *function* is captured by 10 of the 29 available frequencies, and Z3 proves it. The logit-level statement is far stronger than what holds: at `|K| = 10` the logits still differ by up to 6.66 in absolute value, yet the argmax — the thing the program must reproduce — is identical on every input. The prover is asked for that statement, not the stronger one; and agreement below 1 refutes equivalence by counting, so the solver is never asked about the approximate rows.

### 9.6 Robustness, and what the solver actually does

Certified enclosures are tight enough that, for a trained network, every input is decided by exact rational comparison and Z3 only confirms unsatisfiability. Widening every enclosure by `±δ` makes the theorem strictly stronger and forces the solver to work:

| attention slack δ | verdict | inputs decided by Z3 |
|---|---|---|
| 0 | proved | 0 / 169 |
| 0.001 | proved | 6 / 169 |
| 0.01 | refuted | n/a |

Reporting `refuted` rather than `unknown` is deliberate: for a robustness statement a witness inside the perturbation set *is* a counterexample, even though the unperturbed network still agrees with its program on that input.

### 9.7 What is *not* claimed

* The exact solution needs the 70% data fraction. The 30%-data run of §9.2 plateaus at 77.4% held-out accuracy; we report the fraction that works instead of presenting a hand-picked success as typical.
* Grokking time, the minimal truncation size and the exact decimals of every number here are properties of the numerical environment. `artifacts/environment.json` records Python 3.11.2, PyTorch 2.15.0+cu132, NumPy 2.4.6 and Z3 5.3.3 on 2 CPU cores; everything quoted is reproducible in *that* environment, and `make results` regenerates it.
* Only modular arithmetic is implemented. Dyck-1 is absent for a structural reason documented in §10: the architecture has no positional embeddings and is therefore permutation-invariant.
* The p=113 network is not an exact solution (99.98% of the domain), so no discrete motif is claimed for it; its certification is exact all the same.
<!-- END GENERATED: results -->

## 10. Limitations

* **Linear access assumed.** The prover evaluates the network through its
  weights, not through black-box queries; white-box verification is the setting.
* **One-layer attention-only models** are supported by the circuit analysis and
  the faithful circuit (the builder raises rather than guessing otherwise). The
  enclosure engine (`compute_logit_bounds`) and the SMT encoder already handle
  multiple layers and a ReLU MLP, so a deeper model can be *proved* against a
  given program, but no decompiler for it is provided.
* **Only modular arithmetic is implemented.** Dyck-1 bracket balancing was on
  the roadmap and is *not* present. It is not a small addition: this
  architecture has no positional embeddings, so it is permutation-invariant and
  literally cannot compute a prefix-dependent language — Dyck support requires
  learned positional embeddings, which change both the forward pass and the SMT
  encoder. The symbolic side (an IR program `min_prefix ≥ 0 ∧ total = 0`) is
  expressible with the existing nodes; it is the model side that is missing.
* **Softmax is not an SMT primitive.** It is handled by certified enclosures;
  the equivalence statement therefore concerns the exact network evaluated at
  the input, with the enclosure carrying the rounding uncertainty. It is not a
  proof about an arbitrary softmax-parameterised network family.
* **Discrete motif search is exhaustive over a small DSL**, not open-ended
  program synthesis. Programs outside the DSL fall back to the faithful circuit.
* Only tasks whose input domain is finite and enumerable can be certified
  exhaustively; modular arithmetic at p=59 (3,481 inputs) and p=113 (12,769
  inputs) are inside that envelope, at 1.4 s and 501 s of proof time
  respectively. Domains grow as `p²`, so p=113 is roughly the practical ceiling
  on one CPU core.

## 11. Repository layout

```
circuit2code/
├── README.md  LICENSE  Makefile  pyproject.toml  requirements.txt  .gitignore
├── CHANGELOG.md  CITATION.cff
├── paper/main.tex         NeurIPS-style write-up (self-contained, 8 pages)
├── src/models             transformer.py, train.py
├── src/interpretability   probing.py, circuit_finder.py
├── src/synthesis          ast_nodes.py, symbolic_reg.py, program_synthesizer.py
├── src/verification       z3_translator.py, verifier.py
├── src/codegen            emit_python.py, emit_c.py, common.py
├── tests/                 test_transformer, test_circuit_finder, test_synthesis,
│                          test_verifier, test_codegen, test_cli, test_scripts,
│                          test_docs
├── scripts/               paper_tables.py, frequency_budget.py, truncation_sweep.py,
│                          measure_export.py, environment.py, render_report.py
├── artifacts/             paper_tables.json, frequency_budget.json,
│                          truncation_sweep_p59.json, truncation_sweep_p113.json,
│                          export_measurements.json, environment.json, history/
├── .github/workflows      test.yml (pytest on 3.10 and 3.12, plus the CLI demo)
└── demo/cli.py            train | extract-circuit | decompile | verify | export | demo
```

## 12. License

MIT. See `pyproject.toml`.
