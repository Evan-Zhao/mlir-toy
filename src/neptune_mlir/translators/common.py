"""Shared helpers for HTile MLIR Python translators."""

import ast
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from mlir import ir

# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _name(s: str, ctx: ast.expr_context | None = None) -> ast.Name:
    return ast.Name(id=s, ctx=ctx or ast.Load())


def _const(v) -> ast.Constant:
    return ast.Constant(value=v)


def _attr(value: ast.expr, attr: str) -> ast.Attribute:
    return ast.Attribute(value=value, attr=attr, ctx=ast.Load())


def _tl(attr: str) -> ast.Attribute:
    return _attr(_name("tl"), attr)


def _T(attr: str) -> ast.Attribute:
    return _attr(_name("T"), attr)


def _tl_call(fn: str, *args: ast.expr, **kwargs: ast.expr) -> ast.Call:
    return _call(_tl(fn), *args, **kwargs)


def _T_call(fn: str, *args: ast.expr, **kwargs: ast.expr) -> ast.Call:
    return _call(_T(fn), *args, **kwargs)


def _call(func: ast.expr, *args: ast.expr, **kwargs: ast.expr) -> ast.Call:
    return ast.Call(
        func=func,
        args=list(args),
        keywords=[ast.keyword(arg=k, value=v) for k, v in kwargs.items()],
    )


def _assign(name: str, value: ast.expr) -> ast.Assign:
    return ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value, lineno=0)


def _expr_stmt(value: ast.expr) -> ast.Expr:
    return ast.Expr(value=value)


def _list(*elts: ast.expr) -> ast.List:
    return ast.List(elts=list(elts), ctx=ast.Load())


def _tuple(*elts: ast.expr, ctx: ast.expr_context | None = None) -> ast.Tuple:
    return ast.Tuple(elts=list(elts), ctx=ctx or ast.Load())


def _subscript(value: ast.expr, indices: list[ast.expr]) -> ast.Subscript:
    idx: ast.expr = indices[0] if len(indices) == 1 else _tuple(*indices)
    return ast.Subscript(value=value, slice=idx, ctx=ast.Load())


def _store_subscript(value: ast.expr, indices: list[ast.expr]) -> ast.Subscript:
    sub = _subscript(value, indices)
    sub.ctx = ast.Store()
    return sub


_MLIR_DTYPE_NAMES = {
    "index": "int64",
    "f8E4M3FN": "float8_e4m3fn",
    "f16": "float16",
    "f32": "float32",
    "f64": "float64",
    "i1": "bool",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
}


def _mlir_dtype_name(dtype: str) -> str:
    if dtype not in _MLIR_DTYPE_NAMES:
        raise NotImplementedError(f"unsupported MLIR dtype: {dtype}")
    return _MLIR_DTYPE_NAMES[dtype]


def _mlir_dtype_to_tl_str(dtype: str) -> str:
    return _mlir_dtype_name(dtype)


def _mlir_dtype_to_tl(dtype: str) -> ast.expr:
    return _tl(_mlir_dtype_name(dtype))


# ---------------------------------------------------------------------------
# MLIR helpers
# ---------------------------------------------------------------------------


def _tensor_shape(mlir_type) -> tuple[list[int], str]:
    """Return (shape, element_dtype_str) from an MLIR RankedTensorType."""
    if not isinstance(mlir_type, ir.RankedTensorType):
        raise NotImplementedError(f"expected RankedTensorType, got {mlir_type}")
    return list(mlir_type.shape), str(mlir_type.element_type)


def _memref_shape(mlir_type) -> tuple[list[int], str]:
    """Return (shape, element_dtype_str) from an MLIR MemRefType."""
    if not isinstance(mlir_type, ir.MemRefType):
        raise NotImplementedError(f"expected MemRefType, got {mlir_type}")
    return list(mlir_type.shape), str(mlir_type.element_type)


def _is_ranked_tensor_type(mlir_type) -> bool:
    return isinstance(mlir_type, ir.RankedTensorType)


def _op_type_name(op) -> str:
    """Return the operation type name, e.g. 'func.func'."""
    if isinstance(op, ir.OpView):
        return op.operation.name
    return op.name


def _module_top_ops(module: ir.Module):
    """Yield true top-level ops, unwrapping generic-form nested modules."""
    ops = list(module.body.operations)
    if len(ops) == 1 and _op_type_name(ops[0]) == "builtin.module":
        yield from ops[0].regions[0].blocks[0].operations
    else:
        yield from ops


def _func_sym_name(op: ir.OpView) -> str:
    raw_name = op.name
    if isinstance(raw_name, ir.StringAttr):
        return raw_name.value
    sym_attr = op.attributes.get("sym_name")
    return ir.StringAttr(sym_attr).value if sym_attr else "kernel"


def _parse_attr_int(attr) -> int:
    return ir.IntegerAttr(attr).value


def _parse_attr_str(attr) -> str:
    return ir.StringAttr(attr).value


def _parse_dense_i64_array(attr) -> list[int]:
    return list(ir.DenseI64ArrayAttr(attr))


def _dimension_order(op: ir.OpView, rank: int) -> list[int]:
    attr = op.attributes.get("dimension_order")
    if attr is None:
        return list(range(rank))
    order = _parse_dense_i64_array(attr)
    if len(order) != rank:
        raise NotImplementedError(f"dimension_order rank mismatch: got {order}, rank {rank}")
    return order


def _dot_transpose_attrs(op: ir.OpView) -> list[str]:
    return [name for name in ("transpose_a", "transpose_b") if op.attributes.get(name) is not None]


def _reject_dot_transpose_attrs(op: ir.OpView, backend: str) -> None:
    transpose_attrs = _dot_transpose_attrs(op)
    if not transpose_attrs:
        return
    joined = ", ".join(transpose_attrs)
    raise NotImplementedError(
        f"{backend} translation requires htile.dot transpose attributes "
        f"to be fissioned before emission; found {joined}. "
        "Run the htile-dot-transpose-to-load-order pass, or inspect "
        "why that pass did not rewrite this dot."
    )


@dataclass(frozen=True)
class ReductionSpec:
    value: ir.Value
    axis: int
    kind: str


@dataclass(frozen=True)
class DotSpec:
    lhs: ir.Value
    rhs: ir.Value
    accumulator: ir.Value | None
    result_shape: list[int]
    result_dtype: str
    transpose_a: bool
    transpose_b: bool


@dataclass(frozen=True)
class BroadcastSpec:
    source: ir.Value
    dimensions: list[int]
    result_shape: list[int]


@dataclass(frozen=True)
class LoadSpec:
    memref: ir.Value
    offsets: list[ir.Value]
    memref_shape: list[int]
    tile_shape: list[int]
    dimension_order: list[int]


@dataclass(frozen=True)
class ForSpec:
    lower_bound: ir.Value
    upper_bound: ir.Value
    step: ir.Value
    body: ir.Block
    induction_variable: ir.Value
    iter_arguments: list[ir.Value]
    iter_initializers: list[ir.Value]


def _decode_constant(op: ir.OpView) -> int | float:
    attr = op.attributes.get("value")
    if isinstance(attr, (ir.IntegerAttr, ir.FloatAttr)):
        return attr.value
    raise NotImplementedError(f"unsupported arith.constant value attr: {attr}")


def _decode_cmp_predicate(op: ir.OpView) -> ast.cmpop:
    attr = op.attributes.get("predicate")
    if attr is None:
        raise ValueError("arith.cmpi missing 'predicate' attribute")
    predicate_to_op: dict[int, type[ast.cmpop]] = {
        0: ast.Eq,
        1: ast.NotEq,
        2: ast.Lt,
        3: ast.LtE,
        4: ast.Gt,
        5: ast.GtE,
        6: ast.Lt,
        7: ast.LtE,
        8: ast.Gt,
        9: ast.GtE,
    }
    predicate = ir.IntegerAttr(attr).value
    cmp_op = predicate_to_op.get(predicate)
    if cmp_op is None:
        raise NotImplementedError(f"unsupported arith.cmpi predicate: {predicate}")
    return cmp_op()


def _decode_reduction(op: ir.OpView) -> ReductionSpec:
    axis_attr = op.attributes.get("axis")
    kind_attr = op.attributes.get("kind")
    return ReductionSpec(
        value=op.operands[0],
        axis=ir.IntegerAttr(axis_attr).value if axis_attr else 1,
        kind=ir.StringAttr(kind_attr).value if kind_attr else "sum",
    )


def _decode_dot(op: ir.OpView) -> DotSpec:
    shape, dtype = _tensor_shape(op.results[0].type)
    return DotSpec(
        lhs=op.operands[0],
        rhs=op.operands[1],
        accumulator=op.operands[2] if len(op.operands) > 2 else None,
        result_shape=shape,
        result_dtype=dtype,
        transpose_a=op.attributes.get("transpose_a") is not None,
        transpose_b=op.attributes.get("transpose_b") is not None,
    )


def _decode_broadcast(op: ir.OpView) -> BroadcastSpec:
    attr = op.attributes.get("dimensions")
    dimensions = _parse_dense_i64_array(attr) if attr else [1]
    shape, _ = _tensor_shape(op.results[0].type)
    return BroadcastSpec(op.operands[0], dimensions, shape)


def _decode_permutation(op: ir.OpView) -> list[int]:
    attr = op.attributes.get("permutation")
    return _parse_dense_i64_array(attr) if attr else [1, 0]


def _decode_load(op: ir.OpView) -> LoadSpec:
    memref_shape, _ = _memref_shape(op.operands[0].type)
    tile_shape, _ = _tensor_shape(op.results[0].type)
    return LoadSpec(
        memref=op.operands[0],
        offsets=list(op.operands[1:]),
        memref_shape=memref_shape,
        tile_shape=tile_shape,
        dimension_order=_dimension_order(op, len(tile_shape)),
    )


def _decode_for(op: ir.OpView) -> ForSpec:
    body = op.regions[0].blocks[0]
    return ForSpec(
        lower_bound=op.operands[0],
        upper_bound=op.operands[1],
        step=op.operands[2],
        body=body,
        induction_variable=body.arguments[0],
        iter_arguments=list(body.arguments[1:]),
        iter_initializers=list(op.operands[3:]),
    )


class BaseTranslator(ABC):
    """Shared SSA bookkeeping, traversal, and operation dispatch."""

    _BINARY_OPS: ClassVar[dict[str, type[ast.operator]]] = {
        "arith.muli": ast.Mult,
        "arith.addi": ast.Add,
        "arith.subi": ast.Sub,
        "arith.divui": ast.FloorDiv,
        "arith.remui": ast.Mod,
        "arith.andi": ast.BitAnd,
        "arith.addf": ast.Add,
        "arith.mulf": ast.Mult,
        "arith.subf": ast.Sub,
        "arith.divf": ast.Div,
    }
    _OP_METHODS: ClassVar[dict[str, str]] = {
        "arith.constant": "_arith_constant",
        "arith.maxsi": "_arith_maxsi",
        "arith.minsi": "_arith_minsi",
        "arith.maximumf": "_arith_maximumf",
        "arith.index_cast": "_arith_index_cast",
        "arith.cmpi": "_arith_cmpi",
        "arith.select": "_arith_select",
        "arith.sitofp": "_arith_cast",
        "arith.extf": "_arith_cast",
        "arith.truncf": "_arith_cast",
        "math.exp2": "_math_exp2",
        "htile.program_id": "_htile_program_id",
        "htile.load": "_htile_load",
        "htile.store": "_htile_store",
        "htile.full": "_htile_full",
        "htile.arange": "_htile_arange",
        "htile.dot": "_htile_dot",
        "htile.reduce": "_htile_reduce",
        "htile.permute": "_htile_permute",
        "htile.copy": "_htile_copy",
        "htile.broadcast": "_htile_broadcast",
        "scf.for": "_scf_for",
        "scf.yield": "_scf_yield",
    }
    _IGNORED_OPS: ClassVar[set[str]] = {"tensor.empty", "htile.return", "linalg.yield"}

    def __init__(self):
        self._names: dict[ir.Value, str] = {}
        self._counter = 0

    def _fresh(self, hint: str = "v") -> str:
        name = f"{hint}_{self._counter}"
        self._counter += 1
        return name

    def _bind(self, value: ir.Value, hint: str = "v") -> str:
        name = self._fresh(hint)
        self._names[value] = name
        return name

    def _get(self, value: ir.Value) -> str:
        if value not in self._names:
            raise KeyError(f"Unbound SSA value: {value}")
        return self._names[value]

    def _expr(self, value: ir.Value) -> ast.expr:
        return _name(self._get(value))

    def translate(self, module: ir.Module) -> ast.Module:
        body = self._module_prelude()
        for op in _module_top_ops(module):
            if _op_type_name(op) == "htile.kernel":
                body.append(self._htile_kernel(op))
        result = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(result)
        return result

    @abstractmethod
    def _module_prelude(self) -> list[ast.stmt]:
        """Return imports and other statements emitted before translated kernels."""

    @abstractmethod
    def _htile_kernel(self, op: ir.OpView) -> ast.FunctionDef:
        """Translate one top-level htile.kernel operation."""

    @abstractmethod
    def _binary_op(self, op: ir.OpView, py_op: ast.operator) -> list[ast.stmt]:
        """Translate an arithmetic operation represented by a Python binary operator."""

    def _block_ops(self, block: ir.Block) -> list[ast.stmt]:
        statements: list[ast.stmt] = []
        for op in block.operations:
            statements.extend(self._op(op))
        return statements

    def _op(self, op: ir.OpView) -> list[ast.stmt]:
        op_name = _op_type_name(op)
        binary_op = self._BINARY_OPS.get(op_name)
        if binary_op is not None:
            return self._binary_op(op, binary_op())
        if op_name in self._IGNORED_OPS:
            return []
        method_name = self._OP_METHODS.get(op_name)
        if method_name is None:
            raise NotImplementedError(f"unsupported op: {op_name}")
        return getattr(self, method_name)(op)


def parse_mlir_module(path: str, pass_pipeline: str | None = None) -> ir.Module:
    """Run neptune-opt on *path* and parse the generic-form MLIR module."""
    from ..dist import find_neptune_opt

    executable = find_neptune_opt()
    if executable is None:
        raise RuntimeError(
            "failed to locate neptune-opt; set NEPTUNE_MLIR_OPT to the executable path"
        )
    cmd = [str(executable)]
    if pass_pipeline:
        cmd.append(f"--pass-pipeline={pass_pipeline}")
        cmd += ["--mlir-print-op-generic", path]
    else:
        cmd += ["--mlir-print-op-generic", "--cse", "--canonicalize", path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or "neptune-opt failed without output"
        raise RuntimeError(f"neptune-opt failed with exit code {result.returncode}: {details}")

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    _register_neptune_dialects(ctx)
    with ctx:
        return ir.Module.parse(result.stdout)


def parse_mlir_module_from_text(text: str) -> ir.Module:
    """Parse MLIR module text with Neptune dialects loaded when available."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    _register_neptune_dialects(ctx)
    with ctx:
        return ir.Module.parse(text)


def _register_neptune_dialects(ctx: ir.Context) -> None:
    from ..dist import register_dialects

    register_dialects(ctx)


def translate_file_with(
    path: str,
    translator_cls: type[BaseTranslator],
    pass_pipeline: str | None = None,
) -> ast.Module:
    module = parse_mlir_module(path, pass_pipeline)
    return translator_cls().translate(module)


def translate_text_with(text: str, translator_cls: type[BaseTranslator]) -> ast.Module:
    return translator_cls().translate(parse_mlir_module_from_text(text))
