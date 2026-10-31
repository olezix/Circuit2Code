"""Circuit2Code: a neuro-symbolic decompiler for algorithmic Transformers.

The package is organised as a pipeline:

``src.models``
    Transparent, weight-level-accessible Transformers and a deterministic
    training loop (grokking regime).
``src.interpretability``
    Activation caching, activation/edge patching, circuit discovery and
    Fourier-basis projection of the effective attention circuits.
``src.synthesis``
    A symbolic IR (``ast_nodes``), Fourier/modular basis extraction
    (``symbolic_reg``) and the IR synthesizer that turns weights into
    symbolic algorithms (``program_synthesizer``).
``src.verification``
    Translation of PyTorch layers and of the IR into quantifier-free /
    quantified Z3 constraints (``z3_translator``) and the equivalence
    prover (``verifier``).
``src.codegen``
    Emission of verified, dependency-free Python and ANSI C code.

Every module is deterministic: no hidden global state, no randomness outside
of explicitly seeded generators.
"""

from __future__ import annotations

__version__ = "0.1.0"

DEFAULT_SEED = 42
#: The canonical toy domain used throughout the paper and the test-suite.
DEFAULT_MODULUS = 59

__all__ = ["__version__", "DEFAULT_SEED", "DEFAULT_MODULUS"]
