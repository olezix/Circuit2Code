"""Compilation of verified IR programs to Python and ANSI C."""

from __future__ import annotations

from src.codegen.emit_c import emit_c
from src.codegen.emit_python import emit_python

__all__ = ["emit_python", "emit_c"]
