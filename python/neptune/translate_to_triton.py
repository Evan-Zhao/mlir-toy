#!/usr/bin/env python3
"""HTile MLIR -> Triton Python translator using MLIR Python bindings.

Usage:
    python translate_to_triton.py <input.mlir> [--plugin <plugin.dylib>]

mlir-opt is called as a subprocess to convert HTile dialect ops to generic form.
"""

import ast

import mlir.ir as ir
from translate_common import (
    DEFAULT_PLUGIN,
    _assign,
    _call,
    _const,
    _func_sym_name,
    _list,
    _memref_shape,
    _mlir_dtype_to_tl,
    _module_top_ops,
    _name,
    _op_type_name,
    _parse_dense_i64_array,
    _tensor_shape,
    _tl,
    _tl_call,
    translate_file_with,
)

# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------

HTILE_DOT_TRANSPOSE_TO_LOAD_ORDER_PIPELINE = (
    "builtin.module(htile-dot-transpose-to-load-order,cse,canonicalize)"
)


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
            ast.Import(names=[ast.alias(name="triton")]),
            ast.ImportFrom(
                module="triton",
                names=[ast.alias(name="language", asname="tl")],
                level=0,
            ),
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
            pname = self._bind(arg, "ptr")
            params.append(ast.arg(arg=pname, annotation=None))

        pre_stmts: list[ast.stmt] = []
        kernel_stmts: list[ast.stmt] = []
        for child_op in entry.operations:
            if _op_type_name(child_op) == "arith.constant":
                pre_stmts.extend(self._arith_constant(child_op))
            elif _op_type_name(child_op) == "gpu.launch":
                kernel_stmts = self._gpu_launch(child_op)

        body = pre_stmts + kernel_stmts or [ast.Pass()]
        decorator = ast.Attribute(value=_name("triton"), attr="jit", ctx=ast.Load())
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
            decorator_list=[decorator],
            returns=None,
            lineno=0,
            col_offset=0,
        )

    # --- gpu.launch -> kernel body ---

    def _gpu_launch(self, op: ir.OpView) -> list[ast.stmt]:
        body_block = op.regions[0].blocks[0]
        args = list(body_block.arguments)
        stmts: list[ast.stmt] = []

        pid_names = ["pid_m", "pid_h", "pid_b"]
        for arg, pid, axis in zip(args[:3], pid_names, [0, 1, 2]):
            self._names[arg] = pid
            stmts.append(_assign(pid, _tl_call("program_id", _const(axis))))
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
            "arith.maximumf": lambda o: self._tl_binop(o, "maximum"),
            "arith.truncf": self._arith_truncf,
            "math.exp2": lambda o: self._tl_unary(o, "exp2"),
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
            return [ast.Expr(value=_const(f"# TODO: {_op_type_name(op)}"))]
        return handler(op)

    # --- arith ops ---

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
        lhs = self._expr(op.operands[0])
        rhs = self._expr(op.operands[1])
        return [_assign(name, ast.BinOp(left=lhs, op=py_op, right=rhs))]

    def _tl_binop(self, op: ir.OpView, fn: str) -> list[ast.stmt]:
        name = self._bind(op.results[0], "v")
        return [
            _assign(
                name,
                _tl_call(fn, self._expr(op.operands[0]), self._expr(op.operands[1])),
            )
        ]

    def _tl_unary(self, op: ir.OpView, fn: str) -> list[ast.stmt]:
        name = self._bind(op.results[0], "v")
        return [_assign(name, _tl_call(fn, self._expr(op.operands[0])))]

    def _arith_truncf(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "v")
        _, dtype = _tensor_shape(op.results[0].type)
        if not dtype:
            dtype = str(op.results[0].type)
        return [
            _assign(
                name,
                _tl_call("cast", self._expr(op.operands[0]), _mlir_dtype_to_tl(dtype)),
            )
        ]

    # --- htile ops ---

    def _htile_load(self, op: ir.OpView) -> list[ast.stmt]:
        mem_shape, _ = _memref_shape(op.operands[0].type)
        tile_shape, _ = _tensor_shape(op.results[0].type)
        indices = list(op.operands[1:])

        stmts, base_ptr, tile_indices, tile_strides = self._fold_batch_dims(
            op.operands[0], indices, mem_shape, tile_shape
        )
        dimension_order = self._dimension_order(op, len(tile_shape))
        mem_tile_shape = mem_shape[-len(tile_shape) :]
        logical_shape = [mem_tile_shape[i] for i in dimension_order]
        logical_strides = [tile_strides[i] for i in dimension_order]
        logical_offsets = [tile_indices[i] for i in dimension_order]
        block_ptr_order = sorted(  # type: ignore
            range(len(logical_strides)), key=logical_strides.__getitem__
        )

        bp = self._fresh("bp")
        stmts.append(
            _assign(
                bp,
                _call(
                    _tl("make_block_ptr"),
                    base=base_ptr,
                    shape=_list(*[_const(s) for s in logical_shape]),
                    strides=_list(*[_const(s) for s in logical_strides]),
                    offsets=_list(
                        *[self._expr(i) for i in logical_offsets[: len(tile_shape)]]
                    ),
                    block_shape=_list(*[_const(s) for s in tile_shape]),
                    order=_list(*[_const(i) for i in block_ptr_order]),
                ),
            )
        )
        tile = self._bind(op.results[0], "tile")
        stmts.append(
            _assign(
                tile,
                _tl_call(
                    "load",
                    _name(bp),
                    boundary_check=_list(*[_const(i) for i in range(len(tile_shape))]),
                ),
            )
        )
        return stmts

    def _dimension_order(self, op: ir.OpView, rank: int) -> list[int]:
        attr = op.attributes.get("dimension_order")
        if attr is None:
            return list(range(rank))
        order = _parse_dense_i64_array(attr)
        if len(order) != rank:
            raise NotImplementedError(
                f"dimension_order rank mismatch: got {order}, rank {rank}"
            )
        return order

    def _htile_store(self, op: ir.OpView) -> list[ast.stmt]:
        tile_val = op.operands[0]
        mem_shape, _ = _memref_shape(op.operands[1].type)
        tile_shape, _ = _tensor_shape(op.operands[0].type)
        indices = list(op.operands[2:])

        stmts, base_ptr, tile_indices, tile_strides = self._fold_batch_dims(
            op.operands[1], indices, mem_shape, tile_shape
        )
        bp = self._fresh("bp")
        stmts.append(
            _assign(
                bp,
                _call(
                    _tl("make_block_ptr"),
                    base=base_ptr,
                    shape=_list(*[_const(s) for s in mem_shape[-2:]]),
                    strides=_list(*[_const(s) for s in tile_strides]),
                    offsets=_list(
                        *[self._expr(i) for i in tile_indices[: len(tile_shape)]]
                    ),
                    block_shape=_list(*[_const(s) for s in tile_shape]),
                    order=_list(*[_const(i) for i in reversed(range(len(tile_shape)))]),
                ),
            )
        )
        stmts.append(
            ast.Expr(
                value=_tl_call(
                    "store",
                    _name(bp),
                    self._expr(tile_val),
                    boundary_check=_list(*[_const(i) for i in range(len(tile_shape))]),
                )
            )
        )
        return stmts

    def _fold_batch_dims(
        self,
        memref_val: ir.Value,
        indices: list[ir.Value],
        mem_shape: list[int],
        tile_shape: list[int],
    ):
        """Fold batch dimensions into a pointer offset.

        Returns (stmts, base_ptr_expr, tile_indices, tile_strides).
        """
        stmts: list[ast.stmt] = []
        base_ptr = self._expr(memref_val)
        if len(mem_shape) > 2 and len(indices) >= len(mem_shape):
            strides = [1] * len(mem_shape)
            for i in range(len(mem_shape) - 2, -1, -1):
                strides[i] = strides[i + 1] * mem_shape[i + 1]
            batch_dims = len(mem_shape) - 2
            offset: ast.expr = _const(0)
            for i in range(batch_dims):
                term = ast.BinOp(
                    left=self._expr(indices[i]), op=ast.Mult(), right=_const(strides[i])
                )
                offset = ast.BinOp(left=offset, op=ast.Add(), right=term)
            ptr = self._fresh("ptr")
            stmts.append(
                _assign(ptr, ast.BinOp(left=base_ptr, op=ast.Add(), right=offset))
            )
            base_ptr = _name(ptr)
            return stmts, base_ptr, indices[batch_dims:], strides[batch_dims:]
        return stmts, base_ptr, indices, [1] * len(tile_shape)

    def _htile_full(self, op: ir.OpView) -> list[ast.stmt]:
        shape, dtype = _tensor_shape(op.results[0].type)
        name = self._bind(op.results[0], "tile")
        return [
            _assign(
                name,
                _tl_call(
                    "full",
                    _list(*[_const(s) for s in shape]),
                    self._expr(op.operands[0]),
                    _mlir_dtype_to_tl(dtype),
                ),
            )
        ]

    def _htile_dot(self, op: ir.OpView) -> list[ast.stmt]:
        transpose_attrs = [
            name
            for name in ("transpose_a", "transpose_b")
            if op.attributes.get(name) is not None
        ]
        if transpose_attrs:
            joined = ", ".join(transpose_attrs)
            raise NotImplementedError(
                "Triton translation requires htile.dot transpose attributes "
                f"to be fissioned before emission; found {joined}. "
                "Run the htile-dot-transpose-to-load-order pass, or inspect "
                "why that pass did not rewrite this dot."
            )
        name = self._bind(op.results[0], "tile")
        lhs = self._expr(op.operands[0])
        rhs = self._expr(op.operands[1])
        acc = self._expr(op.operands[2]) if len(op.operands) > 2 else None
        call = (
            _tl_call("dot", lhs, rhs) if acc is None else _tl_call("dot", lhs, rhs, acc)
        )
        return [_assign(name, call)]

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
                _tl_call(
                    fn,
                    self._expr(op.operands[0]),
                    _const(axis),
                    keep_dims=_const(False),
                ),
            )
        ]

    def _htile_permute(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "tile")
        perm_attr = op.attributes.get("permutation")
        perm = list(ir.DenseI64ArrayAttr(perm_attr)) if perm_attr else [1, 0]
        return [
            _assign(
                name,
                _tl_call(
                    "permute",
                    self._expr(op.operands[0]),
                    _list(*[_const(p) for p in perm]),
                ),
            )
        ]

    def _htile_copy(self, op: ir.OpView) -> list[ast.stmt]:
        # Placement change only — alias the SSA value, emit nothing.
        self._names[op.results[0]] = self._get(op.operands[0])
        return []

    # --- linalg.broadcast -> unsqueeze ---

    def _linalg_broadcast(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "bcast")
        src = self._expr(op.operands[0])
        dims_attr = op.attributes.get("dimensions")
        dims = list(ir.DenseI64ArrayAttr(dims_attr)) if dims_attr else [1]
        shape, _ = _tensor_shape(op.results[0].type)
        rank_out = len(shape)

        indices: list[ast.expr] = []
        for out_dim in range(rank_out):
            indices.append(_const(None) if out_dim in dims else ast.Slice())

        idx = (
            ast.Tuple(elts=indices, ctx=ast.Load()) if len(indices) > 1 else indices[0]
        )
        return [_assign(name, ast.Subscript(value=src, slice=idx, ctx=ast.Load()))]

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def translate_file(path: str, plugin: str | None = None) -> ast.Module:
    """Run mlir-opt on *path*, parse with mlir.ir, and return a Python ast.Module."""
    return translate_file_with(
        path,
        Translator,
        plugin,
        pass_pipeline=HTILE_DOT_TRANSPOSE_TO_LOAD_ORDER_PIPELINE,
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
