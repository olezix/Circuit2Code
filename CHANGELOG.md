# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-11-01

First public release: the complete decompiler, prover, code generator, test
suite and paper.

### Added

* **Transparent Transformer** (`src/models/transformer.py`): one-layer,
  attention-only, no LayerNorm, explicit weight matrices, full activation trace.
* **Deterministic training** (`src/models/train.py`): full-batch AdamW with
  half-cosine learning-rate annealing, seeded initialisation and data splits,
  pinned BLAS threads, early stopping after grokking, rate-limited intermediate
  checkpoints, and a `<checkpoint>.history.json` trajectory written next to every
  checkpoint (also for interrupted runs).
* **Circuit discovery** (`src/interpretability/`): activation cache, denoising
  and attribution patching, zero-ablation and direct logit attribution, the
  effective matrices `W_QK = W_Q W_K^T` and `W_OV = W_V W_O^T`, and a discrete
  Fourier projection with a top-3 spectral-energy statistic.
* **Typed IR** (`src/synthesis/ast_nodes.py`): `Const, Var, Neg, Add, Sub, Mul,
  Div, Mod, Sin, Cos, Linear, Sum, Softmax, Threshold, ArgMax, IfElse` with NumPy
  semantics, interval semantics, SymPy simplification and source rendering.
* **Synthesis** (`src/synthesis/`): circuit-grounded motif search over a small
  algorithm DSL, accepted only when it reproduces the network's argmax on every
  input, with a faithful trigonometric-circuit fallback that is exact by
  construction of its Fourier coefficients.
* **Verification** (`src/verification/`): certified outward-rounded enclosures of
  the float64 forward pass (Higham `γ_n` budgets, interval softmax), an exact
  rational Z3 encoding of the network and the IR, and three proof obligations —
  quantified `spec`, full-domain `model_equivalence` (argmax), and bounded
  `logit_equivalence`, with counterexample replay and refinement.
* **Code generation** (`src/codegen/`): dependency-free Python and ISO C99 (C89
  constructs, `math.h` only, no allocation), both compiling clean under
  `-Wall -Wextra -pedantic -Werror`.
* **CLI** (`demo/cli.py`): `train`, `extract-circuit`, `decompile`, `verify`,
  `export`, `demo`, with `rich` tables and JSON output for every stage.
* **Measurement scripts** (`scripts/`): `paper_tables.py`, `frequency_budget.py`,
  `truncation_sweep.py`, `measure_export.py` (compiles the emitted C and compares
  it with the emitted Python over the whole domain) and `environment.py`
  (records the exact software stack). Every number in the README and the paper is
  produced by them and committed under `artifacts/`.
* **Generated prose** (`scripts/render_report.py`): the results section of the
  README and the experiments section of the paper are generated from the JSON
  artefacts, and `tests/test_docs.py` fails if either is stale.
* **Tests** (`tests/`): 118 tests — model, training, circuit finding, synthesis,
  verifier, codegen (the emitted C is compiled and executed on the full domain),
  CLI end-to-end, the measurement scripts, and the generated documentation.
* **Paper** (`paper/main.tex`): self-contained NeurIPS-style write-up.
* **CI** (`.github/workflows/test.yml`): pytest on Python 3.10 and 3.12 plus the
  CLI demo.

### Results at release

* Grokked `p = 59` network (70 % of the domain): 100 % train, 100 % held-out,
  100 % full-domain accuracy; decompiles to `((a + b) % 59)` (4 IR nodes) and is
  **proved equivalent over all 3 481 inputs**; `spec` proved in 0.02 s.
* The same weights yield a trigonometric circuit whose 10-frequency truncation
  reproduces the argmax on every input and is **proved equivalent**; the
  partially trained and memorising networks need the full 29-frequency basis.
* A 99.98 %-correct `p = 113` network is certified over all 12 769 inputs, at both
  the argmax and the logit level.
* Emitted C from a 5 763-node circuit: 2.1 MB, compiles in ~3 s with no
  diagnostics, bit-identical to the emitted Python on every input.

### Known limitations

* Modular arithmetic only; Dyck-1 is not implemented (the architecture has no
  positional embeddings and is therefore permutation-invariant).
* The verified object is the float64 forward pass; the float32 deployment gap is
  measured and reported, never assumed away.
* Domains must be finite and enumerable: `p = 113` (12 769 inputs) already costs
  ~8 minutes of proof time on one core.

[1.0.0]: https://github.com/olezix/Circuit2Code/releases/tag/v1.0.0
