#!/usr/bin/env python
"""Record the software and hardware environment that produced the artefacts.

Grokking trajectories depend on the numerical details of the BLAS and of the
PyTorch release (see README section 9.1: the same recipe grokks at epoch 1,250
or at 16,750 depending on the environment), so the environment is part of the
experimental record and is written to a JSON file next to the results:

    python scripts/environment.py --out artifacts/environment.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time


def _version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception:  # pragma: no cover - optional dependency
        return "missing"
    return str(getattr(module, "__version__", "unknown"))


def collect() -> dict:
    """Return a JSON-serialisable description of the current environment."""
    try:
        import z3

        z3_version = z3.get_version_string()
    except Exception:  # pragma: no cover - optional dependency
        z3_version = "missing"

    cpu = platform.processor() or platform.machine()
    cpu_count = os.cpu_count()
    record = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu,
        "cpu_count": cpu_count,
        "torch": _version("torch"),
        "torch_cuda": bool(getattr(__import__("torch"), "cuda", None) and __import__("torch").cuda.is_available()),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "sympy": _version("sympy"),
        "z3": z3_version,
        "rich": _version("rich"),
        "gcc": _gcc_version(),
        "notes": (
            "All reported runs used CPU-only training with torch.set_num_threads(1) "
            "and full-batch gradients; no GPU was involved."
        ),
    }
    return record


def _gcc_version() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["gcc", "--version"], capture_output=True, text=True, check=False
        ).stdout
    except FileNotFoundError:  # pragma: no cover - gcc is optional
        return "missing"
    return out.splitlines()[0].strip() if out else "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "artifacts",
            "environment.json",
        ),
    )
    args = parser.parse_args(argv)
    record = collect()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
