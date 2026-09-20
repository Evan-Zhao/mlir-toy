"""Shared helpers for HTile MLIR Python translators."""

import ast
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
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


def _add_optional_accumulator(value: ast.expr, accumulator: ast.expr | None) -> ast.expr:
    if accumulator is None:
        return value
    return ast.BinOp(left=value, op=ast.Add(), right=accumulator)


_MLIR_DTYPE_NAMES = {
    "index": "int64",
    "bf16": "bfloat16",
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


def _row_major_strides(shape: list[int]) -> list[int]:
    strides = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]
    return strides


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


def _dimensions(op: ir.OpView, memory_rank: int, tile_rank: int) -> list[int]:
    attr = op.attributes.get("dimensions")
    if attr is None:
        if tile_rank == 0:
            return []
        if memory_rank != tile_rank:
            raise ValueError(
                f"{_op_type_name(op)} requires dimensions when tile rank {tile_rank} "
                f"differs from memory rank {memory_rank}"
            )
        return list(range(tile_rank))

    dimensions = _parse_dense_i64_array(attr)
    if len(dimensions) != tile_rank:
        raise ValueError(
            f"dimensions rank mismatch: got {dimensions}, tile rank {tile_rank}"
        )
    if len(set(dimensions)) != len(dimensions) or any(
        dimension < 0 or dimension >= memory_rank for dimension in dimensions
    ):
        raise ValueError(f"invalid dimensions {dimensions} for memory rank {memory_rank}")
    return dimensions


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


class DotKind(Enum):
    MATRIX_MATRIX = "matrix-matrix"
    MATRIX_VECTOR = "matrix-vector"
    VECTOR_MATRIX = "vector-matrix"


@dataclass(frozen=True)
class DotSpec:
    lhs: ir.Value
    rhs: ir.Value
    accumulator: ir.Value | None
    kind: DotKind
    result_shape: list[int]
    lhs_dtype: str
    rhs_dtype: str
    result_dtype: str
    reduction_size: int
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
    mask: ir.Value | None
    other: ir.Value | None
    memref_shape: list[int]
    tile_shape: list[int]
    dimensions: list[int]


@dataclass(frozen=True)
class StoreSpec:
    value: ir.Value
    memref: ir.Value
    offsets: list[ir.Value]
    mask: ir.Value | None
    memref_shape: list[int]
    tile_shape: list[int]
    dimensions: list[int]


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
    op_name = _op_type_name(op)
    attr = op.attributes.get("predicate")
    if attr is None:
        raise ValueError(f"{op_name} missing 'predicate' attribute")

    if op_name == "arith.cmpi":
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
    elif op_name == "arith.cmpf":
        # Native comparisons preserve the ordered predicates and unordered
        # not-equal semantics needed by isnan(x) = cmpf une, x, x.
        predicate_to_op = {
            1: ast.Eq,  # oeq
            2: ast.Gt,  # ogt
            3: ast.GtE,  # oge
            4: ast.Lt,  # olt
            5: ast.LtE,  # ole
            13: ast.NotEq,  # une
        }
    else:
        raise NotImplementedError(f"unsupported comparison op: {op_name}")

    predicate = ir.IntegerAttr(attr).value
    cmp_op = predicate_to_op.get(predicate)
    if cmp_op is None:
        raise NotImplementedError(f"unsupported {op_name} predicate: {predicate}")
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
    lhs, rhs = op.operands[:2]
    lhs_shape, lhs_dtype = _tensor_shape(lhs.type)
    rhs_shape, rhs_dtype = _tensor_shape(rhs.type)
    result_shape, result_dtype = _tensor_shape(op.results[0].type)
    transpose_a = op.attributes.get("transpose_a") is not None
    transpose_b = op.attributes.get("transpose_b") is not None

    kind = {
        (2, 2): DotKind.MATRIX_MATRIX,
        (2, 1): DotKind.MATRIX_VECTOR,
        (1, 2): DotKind.VECTOR_MATRIX,
    }.get((len(lhs_shape), len(rhs_shape)))
    if kind is None:
        raise NotImplementedError(
            f"unsupported htile.dot operand shapes: {lhs_shape} x {rhs_shape}"
        )

    effective_lhs = lhs_shape[::-1] if transpose_a else lhs_shape
    effective_rhs = rhs_shape[::-1] if transpose_b else rhs_shape
    lhs_reduction, rhs_reduction = effective_lhs[-1], effective_rhs[0]
    expected_shape = effective_lhs[:-1] + effective_rhs[1:]
    if lhs_reduction != rhs_reduction or result_shape != expected_shape:
        raise ValueError(
            f"invalid {kind.value} htile.dot shapes: {lhs_shape} x {rhs_shape} "
            f"-> {result_shape}"
        )

    return DotSpec(
        lhs=lhs,
        rhs=rhs,
        accumulator=op.operands[2] if len(op.operands) > 2 else None,
        kind=kind,
        result_shape=result_shape,
        lhs_dtype=lhs_dtype,
        rhs_dtype=rhs_dtype,
        result_dtype=result_dtype,
        reduction_size=lhs_reduction,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )


def _decode_broadcast(op: ir.OpView) -> BroadcastSpec:
    attr = op.attributes.get("dimensions")
    dimensions = _parse_dense_i64_array(attr) if attr else [1]
    shape, _ = _tensor_shape(op.results[0].type)
    return BroadcastSpec(op.operands[0], dimensions, shape)


def _decode_permutation(op: ir.OpView) -> list[int]:
    attr = op.attributes.get("permutation")
    return _parse_dense_i64_array(attr) if attr else [1, 0]


def _operand_segment_sizes(op: ir.OpView) -> list[int]:
    attr = op.attributes.get("operandSegmentSizes")
    if attr is None:
        raise ValueError(f"{_op_type_name(op)} is missing operandSegmentSizes")
    return list(ir.DenseI32ArrayAttr(attr))


def _decode_load(op: ir.OpView) -> LoadSpec:
    segment_sizes = _operand_segment_sizes(op)
    if len(segment_sizes) != 4 or segment_sizes[0] != 1:
        raise ValueError(f"invalid htile.load operand segments: {segment_sizes}")
    num_offsets, num_masks, num_others = segment_sizes[1:]
    if num_masks not in (0, 1) or num_others not in (0, 1):
        raise ValueError(f"invalid htile.load optional operand segments: {segment_sizes}")

    offset_end = 1 + num_offsets
    mask = op.operands[offset_end] if num_masks else None
    other = op.operands[offset_end + num_masks] if num_others else None
    memref_shape, _ = _memref_shape(op.operands[0].type)
    result_type = op.results[0].type
    tile_shape = _tensor_shape(result_type)[0] if _is_ranked_tensor_type(result_type) else []
    return LoadSpec(
        memref=op.operands[0],
        offsets=list(op.operands[1:offset_end]),
        mask=mask,
        other=other,
        memref_shape=memref_shape,
        tile_shape=tile_shape,
        dimensions=_dimensions(op, len(memref_shape), len(tile_shape)),
    )


def _decode_store(op: ir.OpView) -> StoreSpec:
    segment_sizes = _operand_segment_sizes(op)
    if len(segment_sizes) != 4 or segment_sizes[:2] != [1, 1]:
        raise ValueError(f"invalid htile.store operand segments: {segment_sizes}")
    num_offsets, num_masks = segment_sizes[2:]
    if num_masks not in (0, 1):
        raise ValueError(f"invalid htile.store optional operand segment: {segment_sizes}")

    offset_end = 2 + num_offsets
    mask = op.operands[offset_end] if num_masks else None
    memref_shape, _ = _memref_shape(op.operands[1].type)
    tile_shape, _ = _tensor_shape(op.operands[0].type)
    return StoreSpec(
        value=op.operands[0],
        memref=op.operands[1],
        offsets=list(op.operands[2:offset_end]),
        mask=mask,
        memref_shape=memref_shape,
        tile_shape=tile_shape,
        dimensions=_dimensions(op, len(memref_shape), len(tile_shape)),
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
        "arith.divsi": ast.FloorDiv,
        "arith.divui": ast.FloorDiv,
        "arith.remui": ast.Mod,
        "arith.andi": ast.BitAnd,
        "arith.addf": ast.Add,
        "arith.mulf": ast.Mult,
        "arith.subf": ast.Sub,
        "arith.divf": ast.Div,
    }
    _UNARY_OPS: ClassVar[dict[str, type[ast.unaryop]]] = {
        "arith.negf": ast.USub,
    }
    _MATH_OPS: ClassVar[dict[str, str]] = {
        "math.absf": "abs",
        "math.exp": "exp",
        "math.exp2": "exp2",
        "math.log1p": "log1p",
    }
    _OP_METHODS: ClassVar[dict[str, str]] = {
        "arith.constant": "_arith_constant",
        "arith.maxsi": "_arith_maxsi",
        "arith.minsi": "_arith_minsi",
        "arith.maximumf": "_arith_maximumf",
        "arith.index_cast": "_arith_index_cast",
        "arith.cmpi": "_arith_cmp",
        "arith.cmpf": "_arith_cmp",
        "arith.select": "_arith_select",
        "arith.sitofp": "_arith_cast",
        "arith.extf": "_arith_cast",
        "arith.truncf": "_arith_cast",
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
        "htile.unsqueeze": "_htile_unsqueeze",
        "htile.squeeze": "_htile_squeeze",
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
        kernels = []
        for op in _module_top_ops(module):
            if _op_type_name(op) == "htile.kernel":
                kernels.append(self._htile_kernel(op))
        result = ast.Module(body=self._module_prelude() + kernels, type_ignores=[])
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

    @abstractmethod
    def _unary_op(self, op: ir.OpView, py_op: ast.unaryop) -> list[ast.stmt]:
        """Translate an arithmetic operation represented by a Python unary operator."""

    @abstractmethod
    def _math_op(self, op: ir.OpView, function: str) -> list[ast.stmt]:
        """Translate an elementwise math operation."""

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
        unary_op = self._UNARY_OPS.get(op_name)
        if unary_op is not None:
            return self._unary_op(op, unary_op())
        math_function = self._MATH_OPS.get(op_name)
        if math_function is not None:
            return self._math_op(op, math_function)
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
