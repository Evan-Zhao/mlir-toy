#!/usr/bin/env python3
"""Parser for MLIR generic op form.

Converts MLIR text (in --mlir-print-op-generic format) into a simple Python
op-tree made of Op / Block / Region dataclasses.  No mlir Python bindings are
required; the input is plain text produced by mlir-opt.

Standalone usage (pretty-prints the parsed tree):
    python mlir_generic_parser.py <input.mlir> [--plugin <plugin.dylib>]
"""

import re
import subprocess
import sys
from dataclasses import dataclass

MLIR_OPT = "mlir-opt"

# ---------------------------------------------------------------------------
# IR data structures
# ---------------------------------------------------------------------------


@dataclass
class Op:
    results: list[str]  # SSA names: ["%9"] or ["%13#0", "%13#1", "%13#2"]
    name: str  # "htile.load", "arith.mulf", ...
    operands: list[str]  # SSA names of operands
    props: dict  # from <{...}> (inherent attributes / properties)
    regions: list  # list[Region]
    result_types: list[str]  # type strings, parallel to results
    operand_types: list[str]  # type strings, parallel to operands


@dataclass
class Block:
    args: list[tuple[str, str]]  # [(ssa_name, type_str), ...]
    ops: list[Op]


@dataclass
class Region:
    blocks: list[Block]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class Tokenizer:
    def __init__(self, text: str):
        self._toks = self._tokenize(text)
        self._i = 0

    def _tokenize(self, text):
        toks = []
        i = 0
        n = len(text)
        while i < n:
            c = text[i]
            if c in " \t\n\r":
                i += 1
                continue
            if text[i : i + 2] == "//":
                while i < n and text[i] != "\n":
                    i += 1
                continue
            if c == "%":
                j = i + 1
                while j < n and (text[j].isalnum() or text[j] == "_"):
                    j += 1
                toks.append(("SSA", text[i:j]))
                i = j
                continue
            if c == "^":
                j = i + 1
                while j < n and (text[j].isalnum() or text[j] == "_"):
                    j += 1
                toks.append(("BLOCK", text[i:j]))
                i = j
                continue
            if c == "#":
                # Attribute alias: #htile.encoding, #arith.overflow, ...
                j = i + 1
                while j < n and (text[j].isalnum() or text[j] in "._"):
                    j += 1
                toks.append(("ATREF", text[i:j]))
                i = j
                continue
            if c == '"':
                j = i + 1
                while j < n and text[j] != '"':
                    if text[j] == "\\":
                        j += 1
                    j += 1
                toks.append(("STR", text[i + 1 : j]))
                i = j + 1
                continue
            if text[i : i + 2] == "->":
                toks.append(("ARROW", "->"))
                i += 2
                continue
            if c == "-" and i + 1 < n and text[i + 1].isdigit():
                j, is_float = self._scan_number(text, i + 1)
                toks.append(("FLOAT" if is_float else "INT", text[i:j]))
                i = j
                continue
            if c.isdigit():
                j, is_float = self._scan_number(text, i)
                toks.append(("FLOAT" if is_float else "INT", text[i:j]))
                i = j
                continue
            if c in "(){}[]<>:,=":
                toks.append(("PUNCT", c))
                i += 1
                continue
            if c.isalpha() or c == "_":
                j = i
                while j < n and (text[j].isalnum() or text[j] in "_."):
                    j += 1
                toks.append(("IDENT", text[i:j]))
                i = j
                continue
            i += 1  # skip unknown characters
        toks.append(("EOF", ""))
        return toks

    @staticmethod
    def _scan_number(text, i):
        n = len(text)
        j = i
        while j < n and text[j].isdigit():
            j += 1
        is_float = j < n and text[j] == "."
        if is_float:
            j += 1
            while j < n and text[j].isdigit():
                j += 1
        if j < n and text[j] in "eE":
            j += 1
            if j < n and text[j] in "+-":
                j += 1
            while j < n and text[j].isdigit():
                j += 1
            is_float = True
        return j, is_float

    def peek(self, offset=0):
        i = self._i + offset
        return self._toks[i] if i < len(self._toks) else ("EOF", "")

    def consume(self):
        t = self._toks[self._i]
        self._i += 1
        return t

    def expect(self, kind, value=None):
        t = self.consume()
        if t[0] != kind or (value is not None and t[1] != value):
            raise SyntaxError(f"Expected ({kind},{value!r}) got {t} at token {self._i}")
        return t[1]

    def match(self, kind, value=None):
        t = self.peek()
        if t[0] == kind and (value is None or t[1] == value):
            return self.consume()[1]
        return None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class Parser:
    """Recursive-descent parser for MLIR generic op form.

    Produces a Region (the module body) containing Block and Op nodes.
    Attribute values are extracted for keys that backends care about
    (axis, kind, permutation, dimensions, value); everything else is skipped.
    """

    def __init__(self, text: str):
        self.t = Tokenizer(text)

    # --- top-level ---

    def parse(self) -> Region:
        """Parse a builtin.module and return its body Region."""
        self.t.expect("STR", "builtin.module")
        self.t.expect("PUNCT", "(")
        self.t.expect("PUNCT", ")")
        self._skip_props()
        self._skip_plain_attrs()
        self.t.expect("PUNCT", "(")
        region = self._region()
        self.t.expect("PUNCT", ")")
        self._skip_type_annotation()
        return region

    # --- regions and blocks ---

    def _region(self) -> Region:
        self.t.expect("PUNCT", "{")
        blocks = []
        while self.t.peek() != ("PUNCT", "}"):
            if self.t.peek()[0] == "EOF":
                break
            blocks.append(self._block())
        self.t.expect("PUNCT", "}")
        return Region(blocks=blocks)

    def _block(self) -> Block:
        args = []
        if self.t.peek()[0] == "BLOCK":
            self.t.consume()  # ^bb0
            if self.t.match("PUNCT", "("):
                while self.t.peek() != ("PUNCT", ")"):
                    name = self.t.expect("SSA")
                    self.t.expect("PUNCT", ":")
                    typ = self._type_str()
                    args.append((name, typ))
                    self.t.match("PUNCT", ",")
                self.t.expect("PUNCT", ")")
            self.t.expect("PUNCT", ":")
        ops = []
        while (
            self.t.peek() not in [("PUNCT", "}"), ("EOF", "")]
            and self.t.peek()[0] != "BLOCK"
        ):
            ops.append(self._op())
        return Block(args=args, ops=ops)

    # --- ops ---

    def _op(self) -> Op:
        # optional result list: %r = ... or %r:N = ...
        results = []
        if self.t.peek()[0] == "SSA":
            ssa = self.t.consume()[1]
            if self.t.match("PUNCT", ":"):
                count = int(self.t.expect("INT"))
                results = [f"{ssa}#{i}" for i in range(count)]
            else:
                results = [ssa]
            self.t.expect("PUNCT", "=")

        name = self.t.expect("STR")

        self.t.expect("PUNCT", "(")
        operands = []
        while self.t.peek() != ("PUNCT", ")"):
            operands.append(self._ssa_use())
            self.t.match("PUNCT", ",")
        self.t.expect("PUNCT", ")")

        props = self._skip_props()
        self._skip_plain_attrs()

        regions = []
        while self.t.match("PUNCT", "("):
            regions.append(self._region())
            self.t.expect("PUNCT", ")")

        self._skip_plain_attrs()  # trailing attrs after regions (e.g. workgroup_attributions)

        operand_types, result_types = self._type_annotation()

        # Multi-result ops emit a single tuple type; replicate for indexing convenience.
        if len(results) > 1 and len(result_types) == 1:
            result_types = result_types * len(results)

        return Op(
            results=results,
            name=name,
            operands=operands,
            props=props,
            regions=regions,
            result_types=result_types,
            operand_types=operand_types,
        )

    def _ssa_use(self) -> str:
        name = self.t.expect("SSA")
        # %13#0: the '#0' part is tokenized as ATREF('#0') not PUNCT + INT
        if self.t.peek()[0] == "ATREF" and self.t.peek()[1].startswith("#"):
            idx = self.t.consume()[1][1:]
            return f"{name}#{idx}"
        return name

    # --- properties and plain attributes ---

    def _skip_props(self) -> dict:
        """Parse <{key = val, ...}> and return a value dict."""
        if not self.t.match("PUNCT", "<"):
            return {}
        self.t.expect("PUNCT", "{")
        attrs = self._parse_attr_dict()
        self.t.expect("PUNCT", "}")
        self.t.expect("PUNCT", ">")
        return attrs

    def _skip_plain_attrs(self) -> dict:
        if self.t.peek() != ("PUNCT", "{"):
            return {}
        self.t.consume()
        attrs = self._parse_attr_dict()
        self.t.expect("PUNCT", "}")
        return attrs

    def _parse_attr_dict(self) -> dict:
        attrs = {}
        while self.t.peek() not in [("PUNCT", "}"), ("EOF", "")]:
            key = self._attr_key()
            self.t.expect("PUNCT", "=")
            val = self._attr_value()
            attrs[key] = val
            self.t.match("PUNCT", ",")
        return attrs

    def _attr_key(self) -> str:
        tok = self.t.peek()
        if tok[0] in ("IDENT", "STR"):
            return self.t.consume()[1]
        return self.t.consume()[1]

    def _attr_value(self):
        tok = self.t.peek()
        if tok[0] == "STR":
            return self.t.consume()[1]
        if tok[0] == "INT":
            v = int(self.t.consume()[1])
            if self.t.match("PUNCT", ":"):
                self._consume_balanced_type()
            return v
        if tok[0] == "FLOAT":
            v = float(self.t.consume()[1])
            if self.t.match("PUNCT", ":"):
                self._consume_balanced_type()
            return v
        if tok[0] == "IDENT" and tok[1] == "array":
            return self._parse_dense_array()
        if tok[0] == "IDENT" and tok[1] in ("true", "false"):
            return self.t.consume()[1] == "true"
        if tok[0] == "IDENT" and tok[1] == "unit":
            self.t.consume()
            return None
        if tok[0] == "ATREF":
            self.t.consume()
            if self.t.peek() == ("PUNCT", "<"):
                self._consume_balanced("<", ">")
            return None
        if tok[0] == "PUNCT" and tok[1] == "{":
            self.t.consume()
            d = self._parse_attr_dict()
            self.t.expect("PUNCT", "}")
            return d
        if tok[0] == "PUNCT" and tok[1] == "(":
            # Function type or grouped expression: (T, T) -> T
            self._consume_balanced("(", ")")
            if self.t.peek()[0] == "ARROW":
                self.t.consume()
                self._consume_type_tokens()
            return None
        if tok[0] == "PUNCT" and tok[1] == "<":
            self._consume_balanced("<", ">")
            return None
        return self.t.consume()[1]

    def _parse_dense_array(self):
        self.t.expect("IDENT", "array")
        self.t.expect("PUNCT", "<")
        self._consume_balanced_type()  # element type (e.g. i64, i32)
        self.t.expect("PUNCT", ":")
        vals = []
        while self.t.peek() != ("PUNCT", ">"):
            tok = self.t.consume()
            if tok[0] == "INT":
                vals.append(int(tok[1]))
            elif tok[0] == "FLOAT":
                vals.append(float(tok[1]))
            self.t.match("PUNCT", ",")
        self.t.expect("PUNCT", ">")
        return vals

    # --- types ---

    def _type_annotation(self):
        if not self.t.match("PUNCT", ":"):
            return [], []
        operand_types = self._type_list()
        result_types = []
        if self.t.match("ARROW"):
            result_types = self._type_list()
        return operand_types, result_types

    def _type_list(self):
        if self.t.peek() == ("PUNCT", "("):
            self.t.consume()
            types = []
            while self.t.peek() != ("PUNCT", ")"):
                types.append(self._type_str())
                self.t.match("PUNCT", ",")
            self.t.expect("PUNCT", ")")
            return types
        return [self._type_str()]

    def _type_str(self) -> str:
        """Consume one type token sequence and return it as a compact string."""
        start = self.t._i
        self._consume_type_tokens()
        end = self.t._i
        return "".join(v for _, v in self.t._toks[start:end])

    def _consume_type_tokens(self):
        tok = self.t.peek()
        if tok[0] in ("IDENT", "STR"):
            self.t.consume()
            if self.t.peek() == ("PUNCT", "<"):
                self._consume_balanced("<", ">")
        elif tok[0] == "ATREF":
            self.t.consume()
            if self.t.peek() == ("PUNCT", "<"):
                self._consume_balanced("<", ">")
        elif tok[0] == "PUNCT" and tok[1] == "(":
            self._consume_balanced("(", ")")
            if self.t.peek()[0] == "ARROW":
                self.t.consume()
                self._consume_type_tokens()

    def _consume_balanced_type(self):
        tok = self.t.peek()
        if tok[0] in ("IDENT", "STR", "ATREF"):
            self.t.consume()
            if self.t.peek() == ("PUNCT", "<"):
                self._consume_balanced("<", ">")

    def _consume_balanced(self, open_="<", close_=">"):
        self.t.expect("PUNCT", open_)
        depth = 1
        while depth > 0:
            tok = self.t.consume()
            if tok == ("PUNCT", open_):
                depth += 1
            elif tok == ("PUNCT", close_):
                depth -= 1
            elif tok[0] == "EOF":
                break

    def _skip_type_annotation(self):
        if self.t.match("PUNCT", ":"):
            self._type_list()
            if self.t.match("ARROW"):
                self._type_list()


# ---------------------------------------------------------------------------
# Type string helpers
# ---------------------------------------------------------------------------


def parse_tensor_shape(type_str: str) -> tuple[list[int], str]:
    """Return (shape, element_dtype) from a tensor<...> type string."""
    m = re.match(r"tensor<([\dx]+)(x\w+)", type_str)
    if not m:
        return [], "f32"
    parts = (m.group(1) + m.group(2)).split("x")
    *dims, dtype = parts
    return [int(d) for d in dims], dtype


def parse_memref_shape(type_str: str) -> tuple[list[int], str]:
    """Return (shape, element_dtype) from a memref<...> type string."""
    m = re.match(r"memref<([\dx]+)(x\w+)", type_str)
    if not m:
        return [], "f16"
    parts = (m.group(1) + m.group(2)).split("x")
    *dims, dtype = parts
    return [int(d) for d in dims], dtype


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_generic_mlir(text: str) -> Region:
    """Parse a string of generic-form MLIR and return the module body Region."""
    return Parser(text).parse()


def parse_mlir_file(path: str, plugin: str | None = None) -> Region:
    """Run mlir-opt to get generic form, then parse and return the module body Region."""
    cmd = [MLIR_OPT]
    if plugin:
        cmd.append(f"--load-dialect-plugin={plugin}")
    cmd += ["--mlir-print-op-generic", path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return parse_generic_mlir(result.stdout)


# ---------------------------------------------------------------------------
# Standalone: pretty-print the parsed op tree
# ---------------------------------------------------------------------------


def _dump(node, indent=0):
    pad = "  " * indent
    if isinstance(node, Region):
        print(f"{pad}Region({len(node.blocks)} block(s))")
        for b in node.blocks:
            _dump(b, indent + 1)
    elif isinstance(node, Block):
        args = ", ".join(f"{n}: {t}" for n, t in node.args)
        print(f"{pad}Block(args=[{args}])")
        for op in node.ops:
            _dump(op, indent + 1)
    elif isinstance(node, Op):
        res = ", ".join(node.results) or "-"
        print(f"{pad}Op {node.name!r}  results={res}  operands={node.operands}")
        if node.props:
            print(f"{pad}  props={node.props}")
        for r in node.regions:
            _dump(r, indent + 1)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("mlir_file")
    ap.add_argument("--plugin", default=None)
    args = ap.parse_args()
    region = parse_mlir_file(args.mlir_file, args.plugin)
    _dump(region)


if __name__ == "__main__":
    main()
