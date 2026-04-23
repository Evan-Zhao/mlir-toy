#!/usr/bin/env python3
"""Shared helpers for HTile MLIR Python translators."""

from __future__ import annotations

import ast
import subprocess
import sys
from typing import Type

import mlir.ir as ir

from ..plugin import find_plugin_path

DEFAULT_PLUGIN = find_plugin_path()
DEFAULT_PLUGIN = None if DEFAULT_PLUGIN is None else DEFAULT_PLUGIN.as_posix()
HTILE_DOT_TRANSPOSE_TO_LOAD_ORDER_PIPELINE = (
    "builtin.module(htile-dot-transpose-to-load-order,cse,canonicalize)"
)


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
    return ast.Assign(
        targets=[ast.Name(id=name, ctx=ast.Store())], value=value, lineno=0
    )


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


def _mlir_dtype_to_tl_str(dtype: str) -> str:
    mapping = {
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
    }
    return mapping.get(dtype, dtype)


def _mlir_dtype_to_tl(dtype: str) -> ast.expr:
    return _tl(_mlir_dtype_to_tl_str(dtype))


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
        raise NotImplementedError(
            f"dimension_order rank mismatch: got {order}, rank {rank}"
        )
    return order


def _dot_transpose_attrs(op: ir.OpView) -> list[str]:
    return [
        name
        for name in ("transpose_a", "transpose_b")
        if op.attributes.get(name) is not None
    ]


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


def parse_mlir_module(
    path: str,
    plugin: str | None = None,
    pass_pipeline: str | None = None,
    pass_plugin: str | None = None,
) -> ir.Module:
    """Run mlir-opt on *path* and parse the generic-form MLIR module."""
    cmd = ["mlir-opt"]
    if plugin:
        cmd.append(f"--load-dialect-plugin={plugin}")
    if pass_plugin:
        cmd.append(f"--load-pass-plugin={pass_plugin}")
    if pass_pipeline:
        cmd.append(f"--pass-pipeline={pass_pipeline}")
        cmd += ["--mlir-print-op-generic", path]
    else:
        cmd += ["--mlir-print-op-generic", "--cse", "--canonicalize", path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        return ir.Module.parse(result.stdout)


def translate_file_with(
    path: str,
    translator_cls: Type,
    plugin: str | None = None,
    pass_pipeline: str | None = None,
    pass_plugin: str | None = None,
) -> ast.Module:
    module = parse_mlir_module(path, plugin, pass_pipeline, pass_plugin)
    translator = translator_cls()
    return translator.translate(module)
