"""Emit dependency-free ANSI C (C99) from a verified IR program.

The generated translation unit

* includes only ``<stdio.h>`` and ``<math.h>`` (both ISO C99),
* allocates nothing on the heap (zero memory leaks by construction),
* compiles cleanly with ``gcc -std=c99 -O3 -Wall -Wextra -pedantic``,
* uses integer types whenever the program is integer-valued, so the emitted
  code is *bit-exact* with respect to the IR semantics.
"""

from __future__ import annotations

import textwrap
from typing import Dict, List, Optional

from src.codegen.common import (
    PRECEDENCE,
    fmt_number,
    is_integer_program,
    plan_hoisting,
    program_docstring,
)
from src.synthesis.ast_nodes import (
    ArgMax,
    BinOp,
    Const,
    Cos,
    Expr,
    IfElse,
    Linear,
    Mod,
    Neg,
    Not,
    Sin,
    Softmax,
    SoftmaxComponent,
    Sum,
    Threshold,
    Var,
    VectorIndex,
    walk,
)

__all__ = ["emit_c", "write_c"]


class CEmitter:
    """Turns an IR expression into a C expression."""

    def __init__(self, program, *, indent: str = "    ") -> None:
        self.program = program
        self.indent = indent
        self.hoisted, self.lookup = plan_hoisting(program.body)
        self.integer = is_integer_program(program.body)
        self.scalar_type = "int" if self.integer else "double"

    # -- numbers -----------------------------------------------------------
    def number(self, value: float) -> str:
        return fmt_number(value, float_context=not self.integer)

    # -- expressions -------------------------------------------------------
    def expr(self, node: Expr, parent_prec: int = 0) -> str:
        if isinstance(node, Const):
            return self.number(node.value)
        if isinstance(node, Var):
            return node.name
        if isinstance(node, Neg):
            return f"(-{self.expr(node.operand, PRECEDENCE['neg'])})"
        if isinstance(node, BinOp):
            if node.op == "%":
                return (
                    f"c2c_imod({self.expr(node.left, PRECEDENCE['%'])}, "
                    f"{self.expr(node.right, PRECEDENCE['%'])})"
                )
            prec = PRECEDENCE[node.op]
            left = self.expr(node.left, prec)
            right_prec = prec + (1 if node.op in ("-", "/") else 0)
            right = self.expr(node.right, right_prec)
            text = f"{left} {node.op} {right}"
            return f"({text})" if prec < parent_prec else text
        if isinstance(node, Mod):
            inner = self.expr(node.operand)
            return f"c2c_imod({inner}, {int(node.modulus)})"
        if isinstance(node, Sum):
            if not node.terms:
                return "0.0" if not self.integer else "0"
            text = " + ".join(self.expr(t, PRECEDENCE["+"]) for t in node.terms)
            return f"({text})" if PRECEDENCE["+"] < parent_prec else text
        if isinstance(node, Sin):
            return f"sin({self.expr(node.operand)})"
        if isinstance(node, Cos):
            return f"cos({self.expr(node.operand)})"
        if isinstance(node, Softmax):
            return self.lookup[id(node)].name
        if isinstance(node, SoftmaxComponent):
            parent = self.lookup[id(node.parent)]
            return f"{parent.name}[{node.index}]"
        if isinstance(node, Threshold):
            inner = self.expr(node.operand)
            return f"(({inner}) >= {self.number(node.threshold)} ? 1.0 : 0.0)"
        if isinstance(node, Linear):
            return self.lookup[id(node)].name
        if isinstance(node, VectorIndex):
            if isinstance(node.vector, Linear):
                return f"{self.lookup[id(node.vector)].name}[{node.index}]"
            return f"({self.expr(node.vector)})[{node.index}]"
        if isinstance(node, ArgMax):
            # Emitted as a helper call on a compound literal array.
            args = ", ".join(self.expr(a) for a in node.args)
            return f"c2c_argmax((double[]){{{args}}}, {len(node.args)})"
        if isinstance(node, IfElse):
            return (
                f"({self.expr(node.condition)} ? {self.expr(node.then)} "
                f": {self.expr(node.otherwise)})"
            )
        raise NotImplementedError(f"cannot emit C for {type(node).__name__}")

    # -- source ------------------------------------------------------------
    def _helpers(self) -> List[str]:
        nodes = list(walk(self.program.body))
        helpers: List[str] = []
        if any(isinstance(n, (Mod,)) or (isinstance(n, BinOp) and n.op == "%") for n in nodes):
            helpers.append(
                textwrap.dedent(
                    """\
                    /* Euclidean remainder: result carries the sign of the modulus. */
                    static int c2c_imod(int value, int modulus)
                    {
                        int r = value % modulus;
                        if (r != 0 && ((r < 0) != (modulus < 0)))
                            r += modulus;
                        return r;
                    }
                    """
                ).rstrip("\n")
            )
        if any(isinstance(n, Softmax) for n in nodes):
            helpers.append(
                textwrap.dedent(
                    """\
                    /* Numerically stable softmax over n values (in-place safe). */
                    static void c2c_softmax(const double *x, int n, double *out)
                    {
                        double peak = x[0];
                        double total = 0.0;
                        int i;
                        for (i = 1; i < n; ++i)
                            if (x[i] > peak)
                                peak = x[i];
                        for (i = 0; i < n; ++i)
                            out[i] = exp(x[i] - peak);
                        for (i = 0; i < n; ++i)
                            total += out[i];
                        for (i = 0; i < n; ++i)
                            out[i] /= total;
                    }
                    """
                ).rstrip("\n")
            )
        if any(isinstance(n, ArgMax) for n in nodes):
            helpers.append(
                textwrap.dedent(
                    """\
                    /* Index of the first maximal entry (deterministic tie-break). */
                    static int c2c_argmax(const double *x, int n)
                    {
                        int best = 0;
                        int i;
                        for (i = 1; i < n; ++i)
                            if (x[i] > x[best])
                                best = i;
                        return best;
                    }
                    """
                ).rstrip("\n")
            )
        return helpers

    def emit(self, *, include_main: bool = True) -> str:
        program = self.program
        nodes = list(walk(program.body))
        needs_math = any(isinstance(n, (Sin, Cos, Softmax)) for n in nodes)

        lines: List[str] = []
        lines.append("/*")
        for line in program_docstring(program, header="Circuit2Code").splitlines():
            lines.append(f" * {line}".rstrip())
        lines.append(" */")
        lines.append("")
        if include_main:
            lines.append("#include <stdio.h>")
        if needs_math:
            lines.append("#include <math.h>")
        lines.append("")
        lines.append(f"#define C2C_MODULUS {program.p}")
        lines.append("")

        for helper in self._helpers():
            lines.extend(helper.splitlines())
            lines.append("")

        # static tables for linear maps
        for entry in self.hoisted:
            if isinstance(entry.node, Linear):
                weights = entry.node.weights
                rows = int(weights.shape[0])
                cols = int(weights.shape[1])
                flat = ", ".join(self.number(v) for v in weights.reshape(-1))
                lines.append(f"#define C2C_{entry.name.upper()}_ROWS {rows}")
                lines.append(f"#define C2C_{entry.name.upper()}_COLS {cols}")
                lines.append(
                    f"static const double C2C_{entry.name.upper()}_W[{rows} * {cols}] = {{"
                )
                lines.append(f"    {flat}")
                lines.append("};")
                lines.append("")

        params = ", ".join(f"{self.scalar_type} {name}" for name in program.params)
        lines.append(f"static {self.scalar_type} {program.name}({params})")
        lines.append("{")
        for entry in self.hoisted:
            if isinstance(entry.node, Softmax):
                args = ", ".join(self.expr(a) for a in entry.node.args)
                lines.append(f"{self.indent}double {entry.name}_in[{len(entry.node.args)}] = {{{args}}};")
                lines.append(
                    f"{self.indent}double {entry.name}[{len(entry.node.args)}];"
                )
                lines.append(
                    f"{self.indent}c2c_softmax({entry.name}_in, {len(entry.node.args)}, {entry.name});"
                )
            elif isinstance(entry.node, Linear):
                args = ", ".join(self.expr(a) for a in entry.node.args)
                rows = int(entry.node.weights.shape[0])
                cols = int(entry.node.weights.shape[1])
                lines.append(f"{self.indent}double {entry.name}_in[{cols}] = {{{args}}};")
                lines.append(f"{self.indent}double {entry.name}[{rows}];")
                lines.append(f"{self.indent}for (int i = 0; i < {rows}; ++i) {{")
                lines.append(f"{self.indent}{self.indent}{entry.name}[i] = 0.0;")
                lines.append(f"{self.indent}{self.indent}for (int j = 0; j < {cols}; ++j)")
                lines.append(
                    f"{self.indent}{self.indent}{self.indent}{entry.name}[i] += "
                    f"C2C_{entry.name.upper()}_W[i * {cols} + j] * {entry.name}_in[j];"
                )
                lines.append(f"{self.indent}}}")
        if self.hoisted:
            lines.append("")
        lines.append(f"{self.indent}return {self.expr(program.body)};")
        lines.append("}")
        lines.append("")

        if include_main:
            lines.append("int main(void)")
            lines.append("{")
            if len(program.params) == 2:
                examples = [(1, 2), (program.p - 1, 1), (program.p - 2, program.p - 2)]
            else:
                examples = [(1,) * len(program.params)]
            spec = "%d" if self.integer else "%.6f"
            cast = "" if self.integer else ""
            for values in examples:
                args = ", ".join(str(v) for v in values)
                labels = ", ".join(str(v) for v in values)
                lines.append(
                    f'{self.indent}printf("{program.name}({labels}) = {spec}\\n", '
                    f"{cast}{program.name}({args}));"
                )
            lines.append(f"{self.indent}return 0;")
            lines.append("}")
        return "\n".join(lines).rstrip() + "\n"


def emit_c(program, *, include_main: bool = True) -> str:
    """Compile ``program`` into ANSI C source text."""
    return CEmitter(program).emit(include_main=include_main)


def write_c(program, path: str, *, include_main: bool = True) -> str:
    """Compile and write the program; returns the emitted source."""
    source = emit_c(program, include_main=include_main)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)
    return source
