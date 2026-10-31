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
is worth reporting honestly: in our sweep it is 70% (grokked at epoch 16,750),
while every 30%-data run we trained stalled at 87% held-out accuracy — the
textbook "grokks *because* data is scarce" story did not reproduce here, and
§9.8 says so instead of hiding it.

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
# held-out accuracy first reaches 1.0, which took 16,750 epochs / 48 min on the
# machine used for the paper and 1,250 epochs / 5 min on a newer PyTorch)
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
make test          # 114 tests: model, training, probing, circuit discovery, IR,
                   # synthesis, Z3, codegen, CLI
make test-fast     # 106 tests: skips the eight that train a model (marker "slow")
```

`make` uses `.venv/bin/python` when the virtualenv exists and `python3`
otherwise; override with `make test PYTHON=/path/to/python`.

The code-generation tests do not only compare strings: the emitted Python is
`exec`'d and the emitted C is compiled with
`gcc -std=c99 -O3 -Wall -Wextra -pedantic -Werror`, run over the entire input
domain, and compared against the IR reference semantics.

## 9. Results

Every number below is produced by a script in `scripts/` and stored as JSON; the
tables are transcriptions, not hand measurements.

```bash
python scripts/paper_tables.py        # artifacts/paper_tables.json
python scripts/frequency_budget.py artifacts/checkpoints/g70.pt:0.7 \
        artifacts/checkpoints/add59_cos.pt:0.3 artifacts/checkpoints/g113.pt:0.7
python scripts/truncation_sweep.py artifacts/checkpoints/g70.pt --train-fraction 0.7 \
        --budget 8            # add --no-verify to skip the SMT queries
python scripts/truncation_sweep.py artifacts/checkpoints/g113.pt --train-fraction 0.7 \
        --budget 8 --no-verify
```

The committed evidence is:

| Artifact | Contents |
|---|---|
| `artifacts/paper_tables.json` | every per-checkpoint row: accuracy, per-head circuit statistics, synthesis provenance, and the Z3 verdict for each proof obligation |
| `artifacts/frequency_budget.json` | the spectral summary of six checkpoints, including the p=113 network |
| `artifacts/truncation_sweep_p59.json` | full-domain agreement of the reported p=59 checkpoint for `|K| = 1 … 8` and the full basis, with the Z3 verdict of each exactly-agreeing truncation |
| `artifacts/truncation_sweep_p59_replication.json` | the same sweep for the independent rerun of §9.1 (`|K| = 1 … 12`) |
| `artifacts/truncation_sweep_p113.json` | the same sweep for the p=113 network |
| `artifacts/history/g70_replication.history.json` | training trajectory of the replication run |

None of these files is edited by hand; they are rewritten by the commands above.

### 9.1 A grokked network, decompiled and proved

`g70.pt` is a one-layer, four-head, `d_model = 128` attention-only model
(80,768 parameters) trained on 70% of the `59² = 3481` pairs with full-batch
AdamW at `lr = 1e-2`, half-cosine annealed to 2%, weight decay 1.0, for 18,750
epochs. It fits the training set **and the held-out set exactly** (100% / 100%,
grokking detected at epoch 16,750). The checkpoint is not committed (it is
regenerated by `make train`); every number in the table below is the `p59_grok`
row of `artifacts/paper_tables.json`, and `train` writes the trajectory next to
the checkpoint as `<checkpoint>.history.json`.

| Quantity | Value |
|---|---|
| Parameters | 80,768 |
| Train / held-out / full-domain accuracy | 1.000 / 1.000 / 1.000 |
| Identified algorithm | `(a + b) mod 59` (4 IR nodes) |
| Top-3 spectral energy per head | 0.972 – 0.998 |
| Dominant frequencies | `{22, 15, 7}` (2 heads), `{22, 28, 15}` (2 heads) |
| Zero-ablation damage per head | Δacc 0.826 – 0.840, Δlogit 21.6 – 58.7 |
| Z3 `spec` | **proved** — quantified `(a + b) mod 59` |
| Z3 `model_equivalence` | **proved** over all 3481 inputs |
| float32 ↔ float64 deployment gap | 1.18e-4 (reported, not assumed away) |
| Synthesis / verification wall time | 0.37 s / 1.49 s |

The emitted C is ten lines (a Euclidean-remainder helper plus a one-line
function), allocation-free, and compiles under
`-std=c99 -O3 -Wall -Wextra -pedantic -Werror`.

The *faithful trigonometric
circuit* built from the same weights is also exactly equivalent, and — this is
the paper's central quantitative claim — a **7-frequency** truncation of it
still reproduces the network's argmax on all 3481 inputs and is **proved
equivalent by Z3** (Table in §9.6).

**Independent replication.** Rerunning the same command in this repository's
current environment (PyTorch 2.15 instead of 2.8) grokks *earlier* — 100%
held-out at epoch 1,250, stopped at 3,250 after the 2,000-epoch post-grok
window (checkpoint `artifacts/checkpoints/g70_replication.pt`, trajectory
`artifacts/history/g70_replication.history.json`; neither weights file is
committed). The
result is the same object, not a fluke of one seed:

| Quantity | Original run | Replication |
|---|---|---|
| Grokking / stop epoch | 16,750 / 18,750 | 1,250 / 3,250 |
| Train / held-out / full-domain | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| Dominant frequencies | `{22, 15, 7}`, `{22, 28, 15}` | `{22, 15, 28}` in three heads |
| Top-3 spectral energy | 0.972 – 0.998 | 0.979 – 0.995 |
| Identified algorithm | `(a + b) % 59` (4 nodes) | `(a + b) % 59` (4 nodes) |
| Z3 `spec` / `model_equivalence` | proved / proved (3481) | proved / proved (3481) |
| Frequencies for an exact argmax | 7 of 29 (proved) | 10 of 29 (proved) |
| Deployment gap | 1.18e-4 | 6.2e-5 |

The optimisation path is chaotic (a different PyTorch release moves the
grokking epoch by an order of magnitude) but the *circuit* it lands in is not:
same frequency family, same program, same proof. That is why the paper reports
one trajectory and certifies properties, not training curves.

### 9.2 Other checkpoints: how the statistic tracks the algorithm

| Checkpoint | p | train frac. | held-out acc. | full-domain acc. | top-3 spectral energy | top frequencies | program | `model_equivalence` |
|---|---|---|---|---|---|---|---|---|
| `g70.pt` | 59 | 0.7 | **100%** | 100% | 0.972 – 0.998 | 22, 15, 7 / 22, 28, 15 | `(a+b) % 59` | proved (3481/3481) |
| `add59_cos.pt` | 59 | 0.3 | 87.0% | 90.9% | 0.948 – 0.989 | 6, 11, 15 | faithful circuit, 5,763 nodes | proved (3481/3481) |
| `add59.pt` (memoriser) | 59 | 0.3 | 29.5% | 50.2% | 0.267 – 0.432 | 11, 15, 26 | faithful circuit, 5,763 nodes | proved (3481/3481) |
| `cos13_60k.pt` | 13 | 0.5 | 76.5% | 88.2% | 0.987 – 0.992 | 1, 3, 4 | faithful circuit, 1,255 nodes | proved (169/169) |
| `cos2.pt` | 23 | 0.5 | 72.5% | 86.2% | 0.896 – 0.962 | 6, 8, 11 | faithful circuit, 2,235 nodes | proved (529/529) |
| random `p = 11` | 11 | – | – | 100% (exact) | 1.0 (full basis) | 1 … 5 | faithful circuit, 541 nodes | proved (121/121) |
| `g113.pt` | 113 | 0.7 | 99.6% | 99.9% | 0.916 – 0.996 | 28, 25, 21 / 21, 50, 8 | faithful circuit, 11,055 nodes | proved (12,769/12,769) |

Every row is machine-checked over the *complete* input domain; none of them
relies on an empirical agreement figure alone. Truncation error of the faithful
circuit (score / logit residuals) is `4e-14`–`1e-13` in every case, and the
float32 ↔ float64 deployment gap is `1.2e-4` (p=59), `1.4e-4` (p=13),
`2.1e-4` (p=23).

### 9.3 A second modulus: `p = 113`

The same pipeline on a harder instance: a one-layer, four-head, `d_model = 128`
model (94,592 parameters) trained on 70% of the 12,769 `(a, b)` pairs for 18,000
epochs of a 30,000-epoch half-cosine schedule (the run was stopped at epoch
18,000, before the schedule had annealed; its measurements are the `p113_grok`
row of `artifacts/paper_tables.json`). This network does *not*
generalise exactly — 15 of the 12,769 pairs are still wrong — which makes it
a useful intermediate case between the grokked `g70.pt` and the 87% run
`add59_cos.pt`. Accuracies are measured on the split the model was trained on
(seed 3, so the held-out set is the same 30% the optimiser never saw).

| quantity | value |
|---|---|
| train / held-out / full-domain accuracy | 100.0% / 99.61% / 99.88% (15 errors) |
| top-3 spectral energy per head | 0.916, 0.914, 0.987, 0.996 |
| top frequencies per head | 28, 25, 21 / 28, 25, 21 / 21, 50, 8 / 25, 38, 12 |
| head ablation (Δacc / Δlogit) | 0.913/11.7, 0.911/11.6, 0.887/6.6, 0.928/11.5 |
| stage-1 motif | none (argmax is not exactly `(a+b) mod 113`) |
| emitted program | faithful trigonometric circuit, 11,055 nodes |
| Z3 `model_equivalence` | **proved**, 12,769/12,769 inputs (663 s) |
| Z3 `logit_equivalence` | **proved**, 1,442,897 certified logit comparisons |
| float32 ↔ float64 deployment gap | 7.5e-5 |
| Python / C | 8.18 MB / 7.70 MB, `gcc -O3` clean in 11.1 s, C ≡ Python on all 12,769 inputs |

Reproduce it with

```bash
python demo/cli.py train --p 113 --epochs 30000 --train-fraction 0.7 --seed 3 \
    --torch-threads 1 --out artifacts/checkpoints/g113.pt     # 2h15m for 18k epochs
python demo/cli.py verify --checkpoint artifacts/checkpoints/g113.pt \
    --train-fraction 0.7 --json artifacts/verification_p113.json   # 663 s
python scripts/truncation_sweep.py artifacts/checkpoints/g113.pt \
    --train-fraction 0.7 --budget 8 --no-verify
```

Two things this instance shows. First, certification is not tied to a lucky
seed or a small modulus: the same enclosure engine proves every one of the
12,769 inputs, with the *logit-level* statement as well, because a faithful
circuit is linear in the certified enclosures. Second, the seven-frequency
picture does not transfer for free: on `p = 113` a `|K|`-frequency truncation
needs the whole 56-frequency basis to be exact (`|K| = 1, 2, 3, 4, 5, 6, 7, 8`
give 14.6%, 29.6%, 49.9%, 64.5%, 73.5%, 77.9%, 92.5%, 96.1%), whereas on
`p = 59` a grokked network is already exactly reproduced at `|K| = 7`. The
circuit statistic tracks *what the network actually learned*: an almost learned
algorithm needs almost all of its frequencies.

### 9.4 A physical measurement of the emitted code

For `add59_cos.pt` (a faithful p=59 circuit, 5,763 IR nodes), `circuit2code
export` writes a 2.10 MB C file and a 2.22 MB Python module. The C file

```bash
gcc -std=c99 -O3 -Wall -Wextra -pedantic decompiled_circuit.c -o prog -lm   # 2.8 s, no warnings
```

compiles with an empty diagnostic list and the executable reproduces the Python
module's value **bit-exactly on all 3481 inputs** (max |Δ| = 0.0). Neither file
allocates memory, and the C file depends only on `math.h`.

### 9.5 The one-minute version

The same five stages on a 1,200-parameter model, end to end in about a second:

```
$ circuit2code train --p 5 --d-model 16 --n-heads 2 --init-std 0.5 \
      --epochs 800 --lr 5e-3 --weight-decay 1e-3 --train-fraction 1.0 --out /tmp/add5.pt
$ circuit2code decompile --checkpoint /tmp/add5.pt --train-fraction 1.0 --out /tmp/out5
  identified algorithm: add_mod      (a + b) % 5
$ circuit2code verify --checkpoint /tmp/add5.pt --train-fraction 1.0
  [PROVED] spec: the program implements (a + b) mod 5 for all inputs
  [PROVED] model_equivalence: for all 25 inputs, the network's argmax equals the program's output
  verified = True
```

The emitted C is ten lines, allocation-free, and compiles under `-Werror`.
(`circuit2code demo` runs exactly this pipeline on a fresh model.)

### 9.6 Circuit size versus fidelity

How much of the network does a `|K|`-frequency circuit capture? Building the
faithful circuit from `g70.pt` with `max_freqs = K` and comparing its argmax
with the network's over all 3,481 inputs:

| `|K|` | agreement | logit truncation error | `model_equivalence` (Z3) |
|---|---|---|---|
| 1 | 15.97% | 9.1e+01 | – (agreement < 1 refutes it) |
| 2 | 73.80% | 5.2e+01 | – |
| 3 | 82.07% | 2.6e+01 | – |
| 4 | 98.82% | 1.6e+01 | – |
| 5 | 99.83% | 1.7e+01 | – |
| 6 | 99.94% | 1.3e+01 | – |
| 7 | **100.00%** | 1.2e+01 | **proved** (3481/3481 inputs) |
| 8 | 100.00% | 1.1e+01 | proved |
| 10 | 100.00% | 7.6e+00 | not queried |
| 12 | 100.00% | 7.0e+00 | not queried |
| 29 (full basis) | 100.00% | 1.7e-12 | proved |

| `|K|` | `add59_cos.pt` (87% held-out) agreement | `add59.pt` (memoriser) agreement |
|---|---|---|
| 1 | 10.3% | – |
| 2 | 26.3% | – |
| 3 | 64.2% | – |
| 4 | 80.6% | – |
| 5 | 86.2% | – |
| 6 | 90.6% | – |
| 7 | 91.1% | – |
| 29 | 100.0% | 100.0% |

So the grokked network's *function* is captured by seven of the 29 available
frequencies, and Z3 proves it; the partially trained network needs the whole
basis (at `|K|=7` it still mispredicts 309 of the 3,481 inputs). The count is
not a universal constant: the replication run of §9.1 needs `|K|=10` for an
exact argmax (proved there too, 48 s), and the curve is not even monotone:
`|K| = 7` leaves 4 of the 3,481 inputs wrong, `|K| = 8` leaves 1, `|K| = 9`
leaves 2, `|K| = 10` is exact and `|K| = 11` is wrong on 2 again (all rows in
`artifacts/truncation_sweep_p59_replication.json`).
What replicates is the *shape* of the result ---
one small band of frequencies carries the function, the logit-level statement
stays false at that truncation, and the argmax-level statement is provable ---
not the integer 7. Note that the logit-level statement is much stronger than what holds:
with `|K|=7` the logits still differ by up to 12 absolute, yet the argmax — the
thing the program is required to reproduce — is identical on every input, and
the prover establishes exactly that statement, not the stronger one. Agreement
below 100% refutes equivalence by counting, so the prover is never asked about
the approximate rows.

### 9.7 Robustness, and what the solver actually does

Certified enclosures are so tight that, for a trained network, every input is
decided by exact rational comparison and Z3 is only asked to confirm
unsatisfiability (0 SMT-decided inputs in the table above). To exercise the
solver we widen the enclosures with `--slack`. On `cos13_60k.pt`:

| attention slack δ | verdict | inputs decided by Z3 |
|---|---|---|
| 0 | proved | 0 / 169 |
| 1e-3 | **proved** | 29 / 169 |
| 1e-2 | refuted (witness found) | 169 / 169 |

At δ = 1e-3 the prover establishes the stronger statement that the argmax
survives a ±0.001 perturbation of every attention probability; at δ = 1e-2 it
finds a genuine witness and reports `refuted`, not `unknown`. Both are honest
answers to different questions.

### 9.8 What is *not* claimed

* The grokked checkpoint `g70.pt` was trained with **70%** of the domain, so its
  held-out set is the remaining 30% (1,044 inputs). It is exact on both, which
  is a genuine generalisation claim, but the *grokking onset* is only visible in
  the 30%-data runs, and those stall at 87% (best we found) rather than
  generalising exactly. We swept training fractions from 0.2 to 1.0, learning
  rates from `3e-4` to `1e-2`, and schedules of 10,000–60,000 epochs; the
  30%-data runs consistently stopped short. We report this instead of selecting
  the fraction that works and hiding the rest.
* Only modular arithmetic is implemented (see §10); Dyck-1 is absent for a
  structural reason documented there.
* The `p = 113` network of §9.3 was trained on one CPU core for 2h15m and
  stopped at epoch 18,000 of 30,000, so it is an *almost* grokked model
  (99.6% held-out). We report it as such; the certification of its faithful
  circuit is nevertheless exact over the full domain, and we do not claim a
  grokked `p = 113` network or a small-`|K|` certificate there.

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
  inputs) are inside that envelope, at 1.5 s and 663 s of proof time
  respectively. Domains grow as `p²`, so p=113 is roughly the practical ceiling
  on one CPU core.

## 11. Repository layout

```
circuit2code/
├── README.md  LICENSE  Makefile  pyproject.toml  requirements.txt  .gitignore
├── paper/main.tex         NeurIPS-style write-up (self-contained, 8 pages)
├── src/models             transformer.py, train.py
├── src/interpretability   probing.py, circuit_finder.py
├── src/synthesis          ast_nodes.py, symbolic_reg.py, program_synthesizer.py
├── src/verification       z3_translator.py, verifier.py
├── src/codegen            emit_python.py, emit_c.py, common.py
├── tests/                 test_transformer, test_circuit_finder, test_synthesis,
│                          test_verifier, test_codegen, test_cli, test_scripts
├── scripts/               paper_tables.py, frequency_budget.py, truncation_sweep.py
├── artifacts/             paper_tables.json, frequency_budget.json,
│                          truncation_sweep_p59.json, truncation_sweep_p113.json
├── .github/workflows      test.yml (pytest on 3.10 and 3.12, plus the CLI demo)
└── demo/cli.py            train | extract-circuit | decompile | verify | export | demo
```

## 12. License

MIT. See `pyproject.toml`.
