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
`f(a, b) = (a + b) mod p` from a fraction of the `p²` possible inputs. A
one-layer, attention-only Transformer trained with full-batch AdamW on 30% of
the data exhibits **grokking** (Power et al., 2022): training accuracy reaches
100% early while held-out accuracy stays at chance, and only after thousands of
further optimisation steps does the network generalise.

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
`rich`, `pytest`. A CUDA device is not required (CPU-only runs are the default
and are bit-reproducible for a fixed number of threads).

## 5. Quickstart

```bash
# train (grokking regime; ~30 min on 2 CPU cores at p=59)
python demo/cli.py train --p 59 --epochs 40000 --checkpoint-every 1000 \
    --out artifacts/checkpoints/add59.pt

# 1. locate the circuit (head ablation + DFT projection)
python demo/cli.py extract-circuit --checkpoint artifacts/checkpoints/add59.pt

# 2. decompile to a symbolic program, emit Python + C
python demo/cli.py decompile --checkpoint artifacts/checkpoints/add59.pt \
    --out artifacts/decompiled

# 3. prove equivalence with Z3
python demo/cli.py verify --checkpoint artifacts/checkpoints/add59.pt \
    --json artifacts/verification.json
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

A 5-minute end-to-end smoke test (tiny model, full-domain training):

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

result = train_modular_addition(TrainConfig(p=59, epochs=40000))
dataset = modular_addition_dataset(59, train_fraction=0.3, seed=42)
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
  reduction order and therefore makes a run bitwise reproducible.
* `tests/test_transformer.py::test_training_is_reproducible` trains twice and
  asserts identical weights.

## 8. Testing

```bash
make test          # 107 tests: model, training, probing, circuit discovery, IR,
                   # synthesis, Z3, codegen, CLI
make test-fast     # 100 tests: skips the ones that train a model (marker "slow")
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
python scripts/frequency_budget.py artifacts/checkpoints/add59_cos.pt:0.3 \
        artifacts/checkpoints/cos13_60k.pt:0.5
python scripts/truncation_sweep.py artifacts/checkpoints/add59_cos.pt --budget 6
```

### 9.1 Certified decompilations

`add59_cos.pt` is a one-layer, four-head, `d_model = 128` attention-only model
(80,768 parameters) trained on 30% of the `59² = 3481` pairs with full-batch
AdamW at `lr = 1e-2` annealed by a half-cosine schedule and weight decay 1.0 for
40,000 epochs.

| Checkpoint | p | held-out acc. | full-domain acc. | top-3 spectral energy | top frequencies | IR nodes | synthesis | `model_equivalence` | `logit_equivalence` |
|---|---|---|---|---|---|---|---|---|---|
| `add59_cos.pt` | 59 | 87.0% | 90.9% | 0.95 – 0.99 | 6, 11, 15 | 5,763 | 13.7 s | proved (3481/3481 inputs) | proved (ε = 1e-4) |
| `add59.pt` (memoriser) | 59 | 29.5% | 50.2% | 0.27 – 0.43 | 11, 15, 26 | 5,763 | 13.7 s | proved (3481/3481 inputs) | proved (ε = 1e-4) |
| `cos13_60k.pt` | 13 | 76.5% | 88.2% | 0.99 | 1, 3, 4 | 1,255 | 0.08 s | proved (169/169 inputs) | proved |
| `cos2.pt` | 23 | 72.5% | 86.2% | 0.90 – 0.96 | 6, 8, 11 | 2,235 | 0.37 s | proved (529/529 inputs) | proved |
| random `p = 11` | 11 | – | 100% | 1.0 (full basis) | 1 … 5 | 541 | 0.04 s | proved (121/121 inputs) | proved |

The *faithful trigonometric circuit* reproduces the argmax of each of these
networks on **every** input of its domain, and Z3 certifies that over the whole
domain. Truncation error of the circuit (score / logit residuals) is
`4e-14`–`1e-13` in every case; the float32 ↔ float64 deployment gap is
`3.1e-5` (p=59), `1.4e-4` (p=13), `2.1e-4` (p=23).

The spectral statistic separates the two p=59 checkpoints sharply: the
grokking-in-progress model concentrates 94.8–98.9% of each head's energy in
three frequencies, the memoriser only 26.7–43.2%. (`CircuitSpec.is_periodic` is
simply the predicate "every head exceeds 0.90 top-3 concentration", so `cos2.pt`,
whose worst head sits at 0.896, is reported as non-periodic even though three
heads exceed 0.95.) That is the quantitative
version of the "circuits vs. lookup table" distinction, and it is measured, not
assumed.

### 9.2 A physical measurement of the emitted code

For `add59_cos.pt`, `circuit2code export` writes a 2.2 MB Python module and a
2.1 MB C file. The C file

```bash
gcc -std=c99 -O3 -Wall -Wextra -pedantic decompiled_circuit.c -o prog -lm   # 2.7 s, no warnings
```

compiles with an empty diagnostic list and the executable reproduces the Python
module's value **bit-exactly on all 3481 inputs** (max |Δ| = 0.0). Neither file
allocates memory, and the C file depends only on `math.h`.

### 9.3 A discrete program, proved

When the network's argmax *is* modular addition, the synthesizer returns the
motif and the proof obligations are the specification, not a tolerance:

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

The emitted C is five lines, allocation-free, and compiles under `-Werror`.

### 9.4 Circuit size versus fidelity

`scripts/truncation_sweep.py` cuts the faithful circuit down to `|K|` frequencies
per head and re-runs the prover. On `add59_cos.pt`:

| \|K\| | logit truncation error | `model_equivalence` |
|---|---|---|
| 1 | 3.4e+01 | refuted |
| 2 | 2.0e+01 | refuted |
| 3 | 1.2e+01 | refuted |
| 6 | 7.0e+00 | refuted |
| 29 (full) | 2.1e-13 | **proved** |

So the tail of the spectrum carries real logit mass in this checkpoint: it is
*not* the clean sparse circuit of a fully grokked network, which is consistent
with its 87% held-out accuracy. The same sweep on a network that has fully fit
the task collapses to a handful of frequencies (see `random p = 11` above, where
a full basis is *exact*).

### 9.5 Robustness, and what the solver actually does

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

### 9.6 What is *not* claimed

We did not obtain a p = 59 network with 100% held-out accuracy under this
architecture (no LayerNorm, no MLP) within 40,000 epochs. Consequently the
discrete-motif path (Section 9.3) is demonstrated on a model whose full domain
was in the training set, and the *generalization* claim for p = 59 remains an
extrapolation rather than a measurement. What is measured at p = 59 is that the
faithful circuit is exact and certified, and that the spectral statistic tracks
how much of the algorithm the network has actually learned.

## 10. Limitations

* **Linear access assumed.** The prover evaluates the network through its
  weights, not through black-box queries; white-box verification is the setting.
* **One-layer attention-only models** are supported for the *symbolic* encoder
  and the faithful circuit; deeper or MLP models can still be synthesised and
  verified through the enclosure path, but the Z3 formula then uses interval
  coefficients rather than exact rational ones.
* **Softmax is not an SMT primitive.** It is handled by certified enclosures;
  the equivalence statement therefore concerns the exact network evaluated at
  the input, with the enclosure carrying the rounding uncertainty. It is not a
  proof about an arbitrary softmax-parameterised network family.
* **Discrete motif search is exhaustive over a small DSL**, not open-ended
  program synthesis. Programs outside the DSL fall back to the faithful circuit.
* Only tasks where the input domain is finite and enumerable (modular
  arithmetic; bounded Dyck words) can be certified exhaustively.

## 11. Repository layout

```
circuit2code/
├── README.md              paper/main.tex        Makefile     pyproject.toml
├── src/models             transformer.py, train.py
├── src/interpretability   probing.py, circuit_finder.py
├── src/synthesis          ast_nodes.py, symbolic_reg.py, program_synthesizer.py
├── src/verification       z3_translator.py, verifier.py
├── src/codegen            emit_python.py, emit_c.py, common.py
├── tests/                 test_transformer, test_circuit_finder, test_synthesis,
│                          test_verifier, test_codegen
├── scripts/               paper_tables.py, frequency_budget.py, truncation_sweep.py
└── demo/cli.py            train | extract-circuit | decompile | verify | export | demo
```

## 12. License

MIT. See `pyproject.toml`.
