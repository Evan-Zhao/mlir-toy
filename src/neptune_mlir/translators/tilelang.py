#!/usr/bin/env python3
"""HTile MLIR -> TileLang Python translator using MLIR Python bindings.

Usage:
    python -m neptune_mlir.translators.tilelang <input.mlir> [--plugin <plugin.dylib>]
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import mlir.ir as ir

from .common import (
    _T,
    DEFAULT_PLUGIN,
    _assign,
    _attr,
    _const,
    _expr_stmt,
    _func_sym_name,
    _is_ranked_tensor_type,
    _list,
    _memref_shape,
    _mlir_dtype_to_tl_str,
    _module_top_ops,
    _name,
    _op_type_name,
    _parse_attr_int,
    _parse_attr_str,
    _parse_dense_i64_array,
    _store_subscript,
    _subscript,
    _T_call,
    _tensor_shape,
    _tuple,
    translate_file_with,
)


@dataclass
class BroadcastInfo:
    source: ir.Value
    dimensions: list[int]


class Translator:
    def __init__(self):
        # Keys are mlir.ir.Value objects (hash by underlying C++ pointer).
        self._names: dict[ir.Value, str] = {}
        self._counter = 0
        self._broadcasts: dict[ir.Value, BroadcastInfo] = {}
        self._yield_dests: list[list[str]] = []

    def _fresh(self, hint="v") -> str:
        n = f"{hint}_{self._counter}"
        self._counter += 1
        return n

    def _bind(self, value: ir.Value, hint="v") -> str:
        if value in self._names:
            return self._names[value]
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
            ast.Import(names=[ast.alias(name="tilelang.language", asname="T")]),
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
            pname = self._bind(arg, "buf")
            shape, dtype = _memref_shape(arg.type)
            annotation = _T_call(
                "Buffer",
                _tuple(*[_const(s) for s in shape]),
                _const(_mlir_dtype_to_tl_str(dtype)),
            )
            params.append(ast.arg(arg=pname, annotation=annotation))

        body: list[ast.stmt] = []
        for child_op in entry.operations:
            if _op_type_name(child_op) == "arith.constant":
                body.extend(self._arith_constant(child_op))
            elif _op_type_name(child_op) == "gpu.launch":
                body.extend(self._gpu_launch(child_op))

        if not body:
            body = [ast.Pass()]

        arguments = ast.arguments(
            posonlyargs=[],
            args=params,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )
        return ast.FunctionDef(
            name=kernel_name,
            args=arguments,
            body=body,
            decorator_list=[_T("prim_func")],
            returns=None,
            lineno=0,
            col_offset=0,
        )

    # --- gpu.launch ---

    def _gpu_launch(self, op: ir.OpView) -> list[ast.stmt]:
        body_block = op.regions[0].blocks[0]
        args = list(body_block.arguments)

        block_names = ["bx", "by", "bz"]
        for arg, name in zip(args[:3], block_names):
            self._names[arg] = name
        for arg in args[3:]:
            self._names[arg] = "_"

        allocs = self._collect_allocs(body_block)
        body = allocs + self._block_ops(body_block)
        if not body:
            body = [ast.Pass()]

        grid = [self._expr(v) for v in op.operands[:3]]
        block = [self._expr(v) for v in op.operands[3:6]]
        threads: ast.expr
        if (
            self._const_int(op.operands[4]) == 1
            and self._const_int(op.operands[5]) == 1
        ):
            threads = block[0]
        else:
            threads = _list(*block)

        with_item = ast.withitem(
            context_expr=_T_call("Kernel", *grid, threads=threads),
            optional_vars=_tuple(
                *[_name(name, ast.Store()) for name in block_names], ctx=ast.Store()
            ),
        )
        return [ast.With(items=[with_item], body=body, lineno=0, col_offset=0)]  # type: ignore

    # --- allocation pass ---

    def _collect_allocs(self, block: ir.Block) -> list[ast.stmt]:
        allocs: list[ast.stmt] = []
        for op in block.operations:
            allocs.extend(self._collect_op_allocs(op))
        return allocs

    def _collect_op_allocs(self, op: ir.OpView) -> list[ast.stmt]:
        op_name = _op_type_name(op)
        allocs: list[ast.stmt] = []

        if op_name == "scf.for":
            allocs.extend(self._collect_allocs(op.regions[0].blocks[0]))
            return allocs
        if op_name in {"linalg.broadcast", "tensor.empty", "htile.copy"}:
            return allocs

        for result in op.results:
            if _is_ranked_tensor_type(result.type):
                name = self._bind(result, self._result_hint(op_name))
                shape, dtype = _tensor_shape(result.type)
                alloc_fn = (
                    "alloc_shared" if self._is_shared(result.type) else "alloc_fragment"
                )
                allocs.append(
                    _assign(
                        name,
                        _T_call(
                            alloc_fn,
                            _list(*[_const(s) for s in shape]),
                            _const(_mlir_dtype_to_tl_str(dtype)),
                        ),
                    )
                )
        return allocs

    def _result_hint(self, op_name: str) -> str:
        return {
            "htile.load": "shared",
            "htile.full": "frag",
            "htile.dot": "dot",
            "htile.reduce": "red",
            "arith.truncf": "cast",
            "math.exp2": "exp",
        }.get(op_name, "frag")

    # --- block and dispatch ---

    def _block_ops(self, block: ir.Block) -> list[ast.stmt]:
        stmts: list[ast.stmt] = []
        for op in block.operations:
            stmts.extend(self._op(op))
        return stmts

    def _op(self, op: ir.OpView) -> list[ast.stmt]:
        dispatch = {
            "arith.constant": self._arith_constant,
            "arith.muli": lambda o: self._scalar_binop(o, ast.Mult()),
            "arith.addi": lambda o: self._scalar_binop(o, ast.Add()),
            "arith.addf": lambda o: self._elementwise_binop(o, ast.Add()),
            "arith.mulf": lambda o: self._elementwise_binop(o, ast.Mult()),
            "arith.subf": lambda o: self._elementwise_binop(o, ast.Sub()),
            "arith.divf": lambda o: self._elementwise_binop(o, ast.Div()),
            "arith.maximumf": self._elementwise_maximum,
            "arith.truncf": self._arith_truncf,
            "math.exp2": self._math_exp2,
            "htile.load": self._htile_load,
            "htile.store": self._htile_store,
            "htile.full": self._htile_full,
            "htile.dot": self._htile_dot,
            "htile.reduce": self._htile_reduce,
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

    # --- scalar and elementwise ops ---

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

    def _scalar_binop(self, op: ir.OpView, py_op: ast.operator) -> list[ast.stmt]:
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

    def _elementwise_binop(self, op: ir.OpView, py_op: ast.operator) -> list[ast.stmt]:
        if not _is_ranked_tensor_type(op.results[0].type):
            return self._scalar_binop(op, py_op)

        def build(indices: list[ast.expr]) -> ast.expr:
            return ast.BinOp(
                left=self._value_at(op.operands[0], indices),
                op=py_op,
                right=self._value_at(op.operands[1], indices),
            )

        return self._parallel_store(op.results[0], build)

    def _elementwise_maximum(self, op: ir.OpView) -> list[ast.stmt]:
        if not _is_ranked_tensor_type(op.results[0].type):
            name = self._bind(op.results[0], "v")
            return [
                _assign(
                    name,
                    _T_call(
                        "max", self._expr(op.operands[0]), self._expr(op.operands[1])
                    ),
                )
            ]

        def build(indices: list[ast.expr]) -> ast.expr:
            return _T_call(
                "max",
                self._value_at(op.operands[0], indices),
                self._value_at(op.operands[1], indices),
            )

        return self._parallel_store(op.results[0], build)

    def _math_exp2(self, op: ir.OpView) -> list[ast.stmt]:
        return self._parallel_store(
            op.results[0],
            lambda indices: _T_call("exp2", self._value_at(op.operands[0], indices)),
        )

    def _arith_truncf(self, op: ir.OpView) -> list[ast.stmt]:
        _, dtype = _tensor_shape(op.results[0].type)
        dtype_str = _mlir_dtype_to_tl_str(dtype)
        return self._parallel_store(
            op.results[0],
            lambda indices: _T_call(
                "cast", self._value_at(op.operands[0], indices), _const(dtype_str)
            ),
        )

    def _parallel_store(self, result: ir.Value, expr_builder) -> list[ast.stmt]:
        shape, _ = _tensor_shape(result.type)
        out = self._get(result)
        index_names = [self._fresh(f"i{dim}") for dim in range(len(shape))]
        indices: list[ast.expr] = [_name(n) for n in index_names]
        target: ast.expr
        if len(index_names) == 1:
            target = _name(index_names[0], ast.Store())
        else:
            target = _tuple(
                *[_name(n, ast.Store()) for n in index_names], ctx=ast.Store()
            )
        body: list[ast.stmt] = [
            ast.Assign(
                targets=[_store_subscript(_name(out), indices)],
                value=expr_builder(indices),
                lineno=0,
            )
        ]
        return [
            ast.For(
                target=target,
                iter=_T_call("Parallel", *[_const(s) for s in shape]),
                body=body,
                orelse=[],
                lineno=0,
                col_offset=0,
            )
        ]

    def _value_at(self, value: ir.Value, indices: list[ast.expr]) -> ast.expr:
        if value in self._broadcasts:
            info = self._broadcasts[value]
            src_indices = [
                idx for dim, idx in enumerate(indices) if dim not in info.dimensions
            ]
            return self._value_at(info.source, src_indices)
        if _is_ranked_tensor_type(value.type):
            return _subscript(self._expr(value), indices)
        return self._expr(value)

    # --- htile ops ---

    def _htile_load(self, op: ir.OpView) -> list[ast.stmt]:
        src = self._mem_region(op.operands[0], list(op.operands[1:]), op.results[0])
        return [_expr_stmt(_T_call("copy", src, self._expr(op.results[0])))]

    def _htile_store(self, op: ir.OpView) -> list[ast.stmt]:
        dst = self._mem_region(op.operands[1], list(op.operands[2:]), op.operands[0])
        return [_expr_stmt(_T_call("copy", self._expr(op.operands[0]), dst))]

    def _htile_full(self, op: ir.OpView) -> list[ast.stmt]:
        return [
            _expr_stmt(
                _T_call("fill", self._expr(op.results[0]), self._expr(op.operands[0]))
            )
        ]

    def _htile_dot(self, op: ir.OpView) -> list[ast.stmt]:
        dst = self._expr(op.results[0])
        stmts: list[ast.stmt] = []
        clear_accum = len(op.operands) <= 2
        if len(op.operands) > 2:
            stmts.append(_expr_stmt(_T_call("copy", self._expr(op.operands[2]), dst)))

        kwargs: dict[str, ast.expr] = {"clear_accum": _const(clear_accum)}
        if op.attributes.get("transpose_a") is not None:
            kwargs["transpose_A"] = _const(True)
        if op.attributes.get("transpose_b") is not None:
            kwargs["transpose_B"] = _const(True)
        policy_attr = op.attributes.get("warp_policy")
        if policy_attr is not None:
            kwargs["policy"] = self._warp_policy_expr(_parse_attr_str(policy_attr))

        stmts.append(
            _expr_stmt(
                _T_call(
                    "gemm",
                    self._expr(op.operands[0]),
                    self._expr(op.operands[1]),
                    dst,
                    **kwargs,
                )
            )
        )
        return stmts

    def _htile_reduce(self, op: ir.OpView) -> list[ast.stmt]:
        kind = (
            _parse_attr_str(op.attributes.get("kind"))
            if op.attributes.get("kind")
            else "sum"
        )
        axis = (
            _parse_attr_int(op.attributes.get("axis"))
            if op.attributes.get("axis")
            else 1
        )
        fn = "reduce_max" if kind == "max" else "reduce_sum"
        return [
            _expr_stmt(
                _T_call(
                    fn,
                    self._expr(op.operands[0]),
                    self._expr(op.results[0]),
                    dim=_const(axis),
                )
            )
        ]

    def _htile_copy(self, op: ir.OpView) -> list[ast.stmt]:
        self._names[op.results[0]] = self._get(op.operands[0])
        return []

    # --- linalg.broadcast ---

    def _linalg_broadcast(self, op: ir.OpView) -> list[ast.stmt]:
        dims_attr = op.attributes.get("dimensions")
        dims = _parse_dense_i64_array(dims_attr) if dims_attr else [1]
        self._broadcasts[op.results[0]] = BroadcastInfo(op.operands[0], dims)
        return []

    # --- scf.for ---

    def _scf_for(self, op: ir.OpView) -> list[ast.stmt]:
        lb = self._expr(op.operands[0])
        ub = self._expr(op.operands[1])
        step = self._expr(op.operands[2])
        body_block = op.regions[0].blocks[0]
        loop_var = body_block.arguments[0]
        iter_bargs = list(body_block.arguments[1:])
        iter_inits = list(op.operands[3:])

        dest_names: list[str] = []
        for init, barg in zip(iter_inits, iter_bargs):
            dest = self._get(init)
            self._names[barg] = dest
            dest_names.append(dest)
        for res, init in zip(op.results, iter_inits):
            self._names[res] = self._get(init)

        loop_name = self._fresh("j")
        self._names[loop_var] = loop_name

        self._yield_dests.append(dest_names)
        body = self._block_ops(body_block) or [ast.Pass()]
        self._yield_dests.pop()

        iter_expr: ast.expr
        stages_attr = op.attributes.get("htile.pipeline_stages")
        if stages_attr is not None:
            iter_expr = _T_call(
                "Pipelined", lb, ub, num_stages=_const(_parse_attr_int(stages_attr))
            )
        else:
            iter_expr = _T_call("serial", lb, ub, step)

        if stages_attr is not None and self._const_int(op.operands[2]) != 1:
            raise NotImplementedError("T.Pipelined emission currently expects step = 1")

        return [
            ast.For(
                target=_name(loop_name, ast.Store()),
                iter=iter_expr,
                body=body,  # type: ignore
                orelse=[],
                lineno=0,
                col_offset=0,
            )
        ]

    def _scf_yield(self, op: ir.OpView) -> list[ast.stmt]:
        if not self._yield_dests:
            return []
        stmts: list[ast.stmt] = []
        for dest, value in zip(self._yield_dests[-1], op.operands):
            src = self._get(value)
            if src != dest:
                stmts.append(_expr_stmt(_T_call("copy", _name(src), _name(dest))))
        return stmts

    # --- helpers ---

    def _mem_region(
        self, memref: ir.Value, offsets: list[ir.Value], tile_value: ir.Value
    ) -> ast.Subscript:
        tile_shape, _ = _tensor_shape(tile_value.type)
        mem_shape, _ = _memref_shape(memref.type)
        batch_dims = len(mem_shape) - len(tile_shape)
        indices: list[ast.expr] = []
        for dim, offset in enumerate(offsets):
            if dim < batch_dims:
                indices.append(self._expr(offset))
            else:
                extent = tile_shape[dim - batch_dims]
                indices.append(
                    ast.Slice(
                        lower=self._expr(offset),
                        upper=ast.BinOp(
                            left=self._expr(offset), op=ast.Add(), right=_const(extent)
                        ),
                        step=None,
                    )
                )
        return _subscript(self._expr(memref), indices)

    def _warp_policy_expr(self, policy: str) -> ast.expr:
        mapping = {
            "square": "Square",
            "full_row": "FullRow",
            "full_col": "FullCol",
        }
        if policy not in mapping:
            raise NotImplementedError(f"unsupported warp_policy: {policy}")
        return _attr(_T("GemmWarpPolicy"), mapping[policy])

    def _is_shared(self, tensor_type) -> bool:
        return "placement = shared" in str(tensor_type)

    def _const_int(self, value: ir.Value) -> int | None:
        owner = value.owner
        if _op_type_name(owner) != "arith.constant":
            return None
        attr = owner.attributes.get("value")
        if isinstance(attr, ir.IntegerAttr):
            return attr.value
        return None


def translate_file(path: str, plugin: str | None = None) -> ast.Module:
    return translate_file_with(path, Translator, plugin)


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
