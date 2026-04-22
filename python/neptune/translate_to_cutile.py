#!/usr/bin/env python3
"""HTile MLIR -> NVIDIA cuTile Python translator using MLIR Python bindings.

Usage:
    python translate_to_cutile.py <input.mlir> [--plugin <plugin.dylib>]

This backend is intentionally value-based: cuTile tiles are immutable values, so
the lowering is closer to the Triton translator than to TileLang.
"""

from __future__ import annotations

import ast

import mlir.ir as ir

from translate_common import (
    DEFAULT_PLUGIN,
    HTILE_DOT_TRANSPOSE_TO_LOAD_ORDER_PIPELINE,
    _assign,
    _attr,
    _call,
    _const,
    _dimension_order,
    _expr_stmt,
    _func_sym_name,
    _memref_shape,
    _module_top_ops,
    _name,
    _op_type_name,
    _parse_dense_i64_array,
    _reject_dot_transpose_attrs,
    _tensor_shape,
    _tuple,
    translate_file_with,
)


def _ct(attr: str) -> ast.Attribute:
    return _attr(_name("ct"), attr)


def _ct_call(fn: str, *args: ast.expr, **kwargs: ast.expr) -> ast.Call:
    return _call(_ct(fn), *args, **kwargs)


def _mlir_dtype_to_ct(dtype: str) -> ast.expr:
    mapping = {
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
    }
    return _ct(mapping.get(dtype, dtype))


class Translator:
    def __init__(self):
        # Keys are mlir.ir.Value objects (hash by underlying C++ pointer).
        self._names: dict[ir.Value, str] = {}
        self._counter = 0
        self._for_output_names: list[list[str]] = []

    def _fresh(self, hint="v") -> str:
        n = f"{hint}_{self._counter}"
        self._counter += 1
        return n

    def _bind(self, value: ir.Value, hint="v") -> str:
        name = self._fresh(hint)
        self._names[value] = name
        return name

    def _get(self, value: ir.Value) -> str:
        if value not in self._names:
            raise KeyError(f"Unbound SSA value: {value}")
        return self._names[value]

    def _expr(self, value: ir.Value) -> ast.expr:
        return _name(self._get(value))

    # --- module entry ---

    def translate(self, module: ir.Module) -> ast.Module:
        body: list[ast.stmt] = [
            ast.Import(names=[ast.alias(name="cuda.tile", asname="ct")]),
        ]
        for op in _module_top_ops(module):
            if _op_type_name(op) == "func.func":
                body.append(self._func(op))
        mod = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(mod)
        return mod

    def _func(self, op: ir.OpView) -> ast.FunctionDef:
        entry = op.regions[0].blocks[0]
        kernel_name = _func_sym_name(op)

        params: list[ast.arg] = []
        for arg in entry.arguments:
            params.append(ast.arg(arg=self._bind(arg, "arr"), annotation=None))

        pre_stmts: list[ast.stmt] = []
        kernel_stmts: list[ast.stmt] = []
        for child_op in entry.operations:
            if _op_type_name(child_op) == "arith.constant":
                pre_stmts.extend(self._arith_constant(child_op))
            elif _op_type_name(child_op) == "gpu.launch":
                kernel_stmts = self._gpu_launch(child_op)

        body = pre_stmts + kernel_stmts or [ast.Pass()]
        arguments = ast.arguments(
            posonlyargs=[],
            args=params,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )
        return ast.FunctionDef(  # type: ignore
            name=kernel_name,
            args=arguments,
            body=body,
            decorator_list=[_ct("kernel")],
            returns=None,
            lineno=0,
            col_offset=0,
        )

    # --- gpu.launch -> kernel body ---

    def _gpu_launch(self, op: ir.OpView) -> list[ast.stmt]:
        body_block = op.regions[0].blocks[0]
        args = list(body_block.arguments)
        stmts: list[ast.stmt] = []

        bid_names = ["bid_m", "bid_h", "bid_b"]
        for arg, bid, axis in zip(args[:3], bid_names, [0, 1, 2]):
            self._names[arg] = bid
            stmts.append(_assign(bid, _ct_call("bid", _const(axis))))
        for arg in args[3:]:
            self._names[arg] = "_"

        stmts.extend(self._block_ops(body_block))
        return stmts

    # --- block and op dispatch ---

    def _block_ops(self, block: ir.Block) -> list[ast.stmt]:
        stmts = []
        for op in block.operations:
            stmts.extend(self._op(op))
        return stmts

    def _op(self, op: ir.OpView) -> list[ast.stmt]:
        dispatch = {
            "arith.constant": self._arith_constant,
            "arith.muli": lambda o: self._binop(o, ast.Mult()),
            "arith.addi": lambda o: self._binop(o, ast.Add()),
            "arith.addf": lambda o: self._binop(o, ast.Add()),
            "arith.mulf": lambda o: self._binop(o, ast.Mult()),
            "arith.subf": lambda o: self._binop(o, ast.Sub()),
            "arith.divf": lambda o: self._binop(o, ast.Div()),
            "arith.maximumf": lambda o: self._ct_binop(o, "maximum"),
            "arith.truncf": self._arith_truncf,
            "math.exp2": lambda o: self._ct_unary(o, "exp2"),
            "htile.load": self._htile_load,
            "htile.store": self._htile_store,
            "htile.full": self._htile_full,
            "htile.dot": self._htile_dot,
            "htile.reduce": self._htile_reduce,
            "htile.permute": self._htile_permute,
            "htile.copy": self._htile_copy,
            "linalg.broadcast": self._linalg_broadcast,
            "scf.for": self._scf_for,
            "scf.yield": self._scf_yield,
            "tensor.empty": lambda o: [],
            "gpu.terminator": lambda o: [],
            "func.return": lambda o: [],
            "linalg.yield": lambda o: [],
        }
        handler = dispatch.get(_op_type_name(op))
        if handler is None:
            return [_expr_stmt(_const(f"# TODO: {_op_type_name(op)}"))]
        return handler(op)

    # --- arithmetic ops ---

    def _arith_constant(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "c")
        attr = op.attributes.get("value")
        if isinstance(attr, ir.IntegerAttr):
            val = attr.value
        elif isinstance(attr, ir.FloatAttr):
            val = attr.value
        else:
            raise NotImplementedError(f"unsupported arith.constant value attr: {attr}")
        return [_assign(name, _const(val))]

    def _binop(self, op: ir.OpView, py_op: ast.operator) -> list[ast.stmt]:
        name = self._bind(op.results[0], "v")
        return [
            _assign(
                name,
                ast.BinOp(
                    left=self._expr(op.operands[0]),
                    op=py_op,
                    right=self._expr(op.operands[1]),
                ),
            )
        ]

    def _ct_binop(self, op: ir.OpView, fn: str) -> list[ast.stmt]:
        name = self._bind(op.results[0], "v")
        return [
            _assign(
                name,
                _ct_call(fn, self._expr(op.operands[0]), self._expr(op.operands[1])),
            )
        ]

    def _ct_unary(self, op: ir.OpView, fn: str) -> list[ast.stmt]:
        name = self._bind(op.results[0], "v")
        return [_assign(name, _ct_call(fn, self._expr(op.operands[0])))]

    def _arith_truncf(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "v")
        _, dtype = _tensor_shape(op.results[0].type)
        return [
            _assign(name, _ct_call("astype", self._expr(op.operands[0]), _mlir_dtype_to_ct(dtype)))
        ]

    # --- htile ops ---

    def _htile_load(self, op: ir.OpView) -> list[ast.stmt]:
        mem_shape, _ = _memref_shape(op.operands[0].type)
        tile_shape, _ = _tensor_shape(op.results[0].type)
        offsets = list(op.operands[1:])
        if len(offsets) != len(mem_shape):
            raise NotImplementedError("cuTile load expects one offset per memref dimension")

        batch_dims = len(mem_shape) - len(tile_shape)
        if batch_dims < 0:
            raise NotImplementedError("cuTile load rank mismatch")

        dimension_order = _dimension_order(op, len(tile_shape))
        full_order = list(range(batch_dims)) + [
            batch_dims + dim for dim in dimension_order
        ]
        full_tile_shape = [1] * batch_dims + [
            tile_shape[dim] for dim in range(len(tile_shape))
        ]
        index = self._tile_space_index(offsets, full_order, full_tile_shape)

        loaded = self._fresh("load")
        stmts = [
            _assign(
                loaded,
                _ct_call(
                    "load",
                    self._expr(op.operands[0]),
                    _tuple(*index),
                    _tuple(*[_const(s) for s in full_tile_shape]),
                    order=_tuple(*[_const(i) for i in full_order]),
                ),
            )
        ]

        result = self._bind(op.results[0], "tile")
        stmts.append(
            _assign(
                result,
                _ct_call("reshape", _name(loaded), _tuple(*[_const(s) for s in tile_shape])),
            )
        )
        return stmts

    def _htile_store(self, op: ir.OpView) -> list[ast.stmt]:
        mem_shape, _ = _memref_shape(op.operands[1].type)
        tile_shape, _ = _tensor_shape(op.operands[0].type)
        offsets = list(op.operands[2:])
        if len(offsets) != len(mem_shape):
            raise NotImplementedError("cuTile store expects one offset per memref dimension")

        batch_dims = len(mem_shape) - len(tile_shape)
        full_order = list(range(len(mem_shape)))
        full_tile_shape = [1] * batch_dims + tile_shape
        index = self._tile_space_index(offsets, full_order, full_tile_shape)
        tile = _ct_call(
            "reshape",
            self._expr(op.operands[0]),
            _tuple(*[_const(s) for s in full_tile_shape]),
        )
        return [
            _expr_stmt(
                _ct_call("store", self._expr(op.operands[1]), _tuple(*index), tile)
            )
        ]

    def _tile_space_index(
        self,
        offsets: list[ir.Value],
        full_order: list[int],
        full_tile_shape: list[int],
    ) -> list[ast.expr]:
        index: list[ast.expr] = []
        # cuTile indexes the logical tile-space order, but offsets are still
        # expressed in memory-axis order.
        for logical_axis, mem_axis in enumerate(full_order):
            extent = full_tile_shape[logical_axis]
            index.append(self._index_div(self._expr(offsets[mem_axis]), extent))
        return index

    def _index_div(self, offset: ast.expr, extent: int) -> ast.expr:
        if extent == 1:
            return offset
        if isinstance(offset, ast.Constant) and isinstance(offset.value, int):
            return _const(offset.value // extent)
        return ast.BinOp(left=offset, op=ast.FloorDiv(), right=_const(extent))

    def _htile_full(self, op: ir.OpView) -> list[ast.stmt]:
        shape, dtype = _tensor_shape(op.results[0].type)
        name = self._bind(op.results[0], "tile")
        return [
            _assign(
                name,
                _ct_call(
                    "full",
                    _tuple(*[_const(s) for s in shape]),
                    self._expr(op.operands[0]),
                    dtype=_mlir_dtype_to_ct(dtype),
                ),
            )
        ]

    def _htile_dot(self, op: ir.OpView) -> list[ast.stmt]:
        _reject_dot_transpose_attrs(op, "cuTile")

        name = self._bind(op.results[0], "tile")
        shape, dtype = _tensor_shape(op.results[0].type)
        acc = (
            self._expr(op.operands[2])
            if len(op.operands) > 2
            else _ct_call(
                "full",
                _tuple(*[_const(s) for s in shape]),
                _const(0),
                dtype=_mlir_dtype_to_ct(dtype),
            )
        )
        return [
            _assign(
                name,
                _ct_call(
                    "mma",
                    self._expr(op.operands[0]),
                    self._expr(op.operands[1]),
                    acc,
                ),
            )
        ]

    def _htile_reduce(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "red")
        axis_attr = op.attributes.get("axis")
        kind_attr = op.attributes.get("kind")
        axis = ir.IntegerAttr(axis_attr).value if axis_attr else 1
        kind = ir.StringAttr(kind_attr).value if kind_attr else "sum"
        fn = "max" if kind == "max" else "sum"
        return [
            _assign(
                name,
                _ct_call(
                    fn,
                    self._expr(op.operands[0]),
                    _const(axis),
                    keepdims=_const(False),
                ),
            )
        ]

    def _htile_permute(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "tile")
        perm_attr = op.attributes.get("permutation")
        perm = _parse_dense_i64_array(perm_attr) if perm_attr else [1, 0]
        return [
            _assign(
                name,
                _ct_call(
                    "permute",
                    self._expr(op.operands[0]),
                    _tuple(*[_const(p) for p in perm]),
                ),
            )
        ]

    def _htile_copy(self, op: ir.OpView) -> list[ast.stmt]:
        # Placement change only; cuTile has no explicit shared/local placement.
        self._names[op.results[0]] = self._get(op.operands[0])
        return []

    # --- linalg.broadcast -> expand dims ---

    def _linalg_broadcast(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "bcast")
        src = self._expr(op.operands[0])
        dims_attr = op.attributes.get("dimensions")
        dims = _parse_dense_i64_array(dims_attr) if dims_attr else [1]
        shape, _ = _tensor_shape(op.results[0].type)

        expr = src
        for dim in dims:
            expr = _ct_call("expand_dims", expr, axis=_const(dim))
        return [
            _assign(
                name,
                _ct_call("broadcast_to", expr, _tuple(*[_const(s) for s in shape])),
            )
        ]

    # --- scf.for ---

    def _scf_for(self, op: ir.OpView) -> list[ast.stmt]:
        lb = self._expr(op.operands[0])
        ub = self._expr(op.operands[1])
        step = self._expr(op.operands[2])

        body_block = op.regions[0].blocks[0]
        loop_var = body_block.arguments[0]
        iter_bargs = list(body_block.arguments[1:])
        iter_inits = list(op.operands[3:])

        pre: list[ast.stmt] = []
        out_names: list[str] = []
        for init_val, barg in zip(iter_inits, iter_bargs):
            out = self._fresh("acc")
            out_names.append(out)
            self._names[barg] = out
            pre.append(_assign(out, self._expr(init_val)))

        for res, out in zip(op.results, out_names):
            self._names[res] = out

        lv = self._fresh("j")
        self._names[loop_var] = lv

        self._for_output_names.append(out_names)
        body_stmts = self._block_ops(body_block) or [ast.Pass()]
        self._for_output_names.pop()

        for_stmt = ast.For(
            target=ast.Name(id=lv, ctx=ast.Store()),
            iter=_call(_name("range"), lb, ub, step),
            body=body_stmts,  # type: ignore
            orelse=[],
            lineno=0,
            col_offset=0,
        )
        return pre + [for_stmt]

    def _scf_yield(self, op: ir.OpView) -> list[ast.stmt]:
        if not self._for_output_names:
            return []
        out_names = self._for_output_names[-1]
        stmts = []
        for out, val in zip(out_names, op.operands):
            val_name = self._get(val)
            if val_name != out:
                stmts.append(_assign(out, _name(val_name)))
        return stmts


def translate_file(path: str, plugin: str | None = None) -> ast.Module:
    return translate_file_with(
        path,
        Translator,
        plugin,
        pass_pipeline=(
            HTILE_DOT_TRANSPOSE_TO_LOAD_ORDER_PIPELINE if plugin else None
        ),
        pass_plugin=plugin,
    )


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("mlir_file")
    ap.add_argument("--plugin", default=DEFAULT_PLUGIN)
    args = ap.parse_args()

    py_module = translate_file(args.mlir_file, args.plugin)
    print(ast.unparse(py_module))


if __name__ == "__main__":
    main()
