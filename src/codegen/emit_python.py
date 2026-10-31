"""Emit idiomatic, dependency-free Python from a verified IR program.

The generated module has no imports beyond the standard library and exposes a
single function whose semantics are exactly the IR's reference semantics.
"""

from __future__ import annotations

import textwrap
from typing import Dict, List, Optional

from src.codegen.common import (
    PRECEDENCE,
    HoistedNode,
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

__all__ = ["emit_python", "write_python"]



class PythonEmitter:
    """Turns an IR expression into a Python expression string."""

    def __init__(self, program, *, indent: str = "    ") -> None:
        self.program = program
        self.indent = indent
        self.hoisted, self.lookup = plan_hoisting(program.body)
        self.helpers: List[str] = []
        self.float_context = not is_integer_program(program.body)

    # -- numbers -----------------------------------------------------------
    def number(self, value: float) -> str:
        return fmt_number(value, float_context=self.float_context)

    # -- expressions -------------------------------------------------------
    def expr(self, node: Expr, parent_prec: int = 0) -> str:
        if isinstance(node, Const):
            return self.number(node.value)
        if isinstance(node, Var):
            return node.name
        if isinstance(node, Neg):
            inner = self.expr(node.operand, PRECEDENCE["neg"])
            return f"(-{inner})"
        if isinstance(node, BinOp):
            prec = PRECEDENCE[node.op]
            left = self.expr(node.left, prec)
            right_prec = prec + (1 if node.op in ("-", "/", "%") else 0)
            right = self.expr(node.right, right_prec)
            text = f"{left} {node.op} {right}"
            return f"({text})" if prec < parent_prec else text
        if isinstance(node, Mod):
            inner = self.expr(node.operand, PRECEDENCE["%"])
            text = f"{inner} % {int(node.modulus)}"
            return f"({text})" if PRECEDENCE["%"] < parent_prec else text
        if isinstance(node, Sum):
            if not node.terms:
                return "0"
            text = " + ".join(self.expr(t, PRECEDENCE["+"]) for t in node.terms)
            return f"({text})" if PRECEDENCE["+"] < parent_prec else text
        if isinstance(node, Sin):
            return f"math.sin({self.expr(node.operand)})"
        if isinstance(node, Cos):
            return f"math.cos({self.expr(node.operand)})"
        if isinstance(node, Softmax):
            return self.lookup[id(node)].name
        if isinstance(node, SoftmaxComponent):
            parent = self.lookup[id(node.parent)]
            return f"{parent.name}[{node.index}]"
        if isinstance(node, Threshold):
            inner = self.expr(node.operand)
            return f"(1.0 if {inner} >= {self.number(node.threshold)} else 0.0)"
        if isinstance(node, Linear):
            return self.lookup[id(node)].name
        if isinstance(node, VectorIndex):
            if isinstance(node.vector, Linear):
                return f"{self.lookup[id(node.vector)].name}[{node.index}]"
            return f"({self.expr(node.vector)})[{node.index}]"
        if isinstance(node, ArgMax):
            args = ", ".join(self.expr(a) for a in node.args)
            return f"_argmax([{args}])"
        if isinstance(node, IfElse):
            return (
                f"({self.expr(node.then)} if {self.expr(node.condition)} "
                f"else {self.expr(node.otherwise)})"
            )
        raise NotImplementedError(f"cannot emit Python for {type(node).__name__}")

    # -- module ------------------------------------------------------------
    def _needs_math(self) -> bool:
        return any(isinstance(n, (Sin, Cos)) for n in walk(self.program.body)) or any(
            isinstance(n, Softmax) for n in walk(self.program.body)
        )

    def _helper_source(self) -> List[str]:
        nodes = list(walk(self.program.body))
        helpers: List[str] = []
        if any(isinstance(n, Softmax) for n in nodes):
            helpers.append(
                textwrap.dedent(
                    '''\
                    def _softmax(values):
                        """Numerically stable softmax over a short list of scores."""
                        peak = max(values)
                        exps = [math.exp(v - peak) for v in values]
                        total = sum(exps)
                        return [e / total for e in exps]
                    '''
                ).rstrip("\n")
            )
        if any(isinstance(n, ArgMax) for n in nodes):
            helpers.append(
                textwrap.dedent(
                    '''\
                    def _argmax(values):
                        """Index of the first maximal entry (deterministic tie-break)."""
                        best = 0
                        for i in range(1, len(values)):
                            if values[i] > values[best]:
                                best = i
                        return best
                    '''
                ).rstrip("\n")
            )
        return helpers

    def emit(self) -> str:
        program = self.program
        params = ", ".join(program.params)
        body = self.expr(program.body)

        lines: List[str] = []
        doc = program_docstring(program, header="Circuit2Code").splitlines()
        lines.append('"""')
        lines.extend(doc)
        lines.append('"""')
        lines.append("")
        lines.append("from __future__ import annotations")
        if self._needs_math():
            lines.append("")
            lines.append("import math")
        lines.append("")
        lines.append(f"MODULUS = {program.p}")
        lines.append("")

        helpers = self._helper_source()
        for helper in helpers:
            lines.extend(helper.splitlines())
            lines.append("")

        # constants for linear maps
        for entry in self.hoisted:
            if isinstance(entry.node, Linear):
                weights = entry.node.weights
                rows = ",\n".join(
                    "    (" + ", ".join(self.number(v) for v in row) + ")" for row in weights
                )
                lines.append(f"_WEIGHTS_{entry.name.upper()} = (")
                lines.append(rows)
                lines.append(")")
                lines.append("")
                lines.append(f"_OUT_{entry.name.upper()} = {weights.shape[0]}")
                lines.append("")

        signature = f"def {program.name}({params}):"
        lines.append(signature)
        docstring = (program.docstring or f"Decompiled program over Z_{program.p}.").splitlines()
        lines.append(f'{self.indent}"""')
        for line in docstring:
            lines.append(f"{self.indent}{line}".rstrip())
        lines.append(f'{self.indent}"""')

        # hoisted assignments
        for entry in self.hoisted:
            if isinstance(entry.node, Softmax):
                args = ", ".join(self.expr(a) for a in entry.node.args)
                lines.append(f"{self.indent}{entry.name} = _softmax([{args}])")
            elif isinstance(entry.node, Linear):
                table = f"_WEIGHTS_{entry.name.upper()}"
                size = f"_OUT_{entry.name.upper()}"
                args = ", ".join(self.expr(a) for a in entry.node.args)
                values = ", ".join([args])
                lines.append(f"{self.indent}_inputs = [{values}]")
                lines.append(f"{self.indent}{entry.name} = [")
                lines.append(f"{self.indent}{self.indent}sum(w * x for w, x in zip(row, _inputs))")
                lines.append(f"{self.indent}{self.indent}for row in {table}")
                lines.append(f"{self.indent}]")
                lines.append(f"{self.indent}assert len({entry.name}) == {size}")
        if self.hoisted:
            lines.append("")
        lines.append(f"{self.indent}return {body}")
        lines.append("")

        # self-test block
        lines.append("")
        lines.append('if __name__ == "__main__":  # pragma: no cover')
        if len(program.params) == 2:
            calls = [(1, 2), (program.p - 1, 1), (program.p - 2, program.p - 2)]
            for a, b in calls:
                lines.append(f"{self.indent}print({program.name}({a}, {b}))")
        else:
            lines.append(f"{self.indent}print({program.name}(*([1] * {len(program.params)})))")
        return "\n".join(lines).rstrip() + "\n"


def emit_python(program) -> str:
    """Compile ``program`` into a self-contained Python module (source text)."""
    return PythonEmitter(program).emit()


def write_python(program, path: str) -> str:
    """Compile and write the program; returns the emitted source."""
    source = emit_python(program)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)
    return source
