#!/usr/bin/env python3
"""HTile MLIR -> Triton Python translator.

Usage:
    python translate_to_triton.py <input.mlir> [--plugin <plugin.dylib>]

Imports the parsed op-tree from mlir_generic_parser and walks it to emit
Python ast nodes, then unparses them to produce a Triton kernel source string.
"""

import ast

from mlir_generic_parser import (
    Block,
    Op,
    Region,
    parse_memref_shape,
    parse_mlir_file,
    parse_tensor_shape,
)

DEFAULT_PLUGIN = "build/libHTileDialectPlugin.dylib"

# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _name(s: str) -> ast.Name:
    return ast.Name(id=s, ctx=ast.Load())


def _const(v) -> ast.Constant:
    return ast.Constant(value=v)


def _tl(attr: str) -> ast.Attribute:
    return ast.Attribute(value=_name("tl"), attr=attr, ctx=ast.Load())


def _tl_call(fn: str, *args: ast.expr, **kwargs: ast.expr) -> ast.Call:
    return ast.Call(
        func=_tl(fn),
        args=list(args),
        keywords=[ast.keyword(arg=k, value=v) for k, v in kwargs.items()],
    )


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


def _list(*elts: ast.expr) -> ast.List:
    return ast.List(elts=list(elts), ctx=ast.Load())


def _mlir_dtype_to_tl(dtype: str) -> ast.expr:
    mapping = {
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
    }
    return _tl(mapping.get(dtype, dtype))


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------


class Translator:
    def __init__(self):
        self._names: dict[str, str] = {}  # SSA name -> Python identifier
        self._counter = 0
        self._for_output_names: list[list[str]] = []  # stack for scf.for iter args

    def _fresh(self, hint="v") -> str:
        n = f"{hint}_{self._counter}"
        self._counter += 1
        return n

    def _bind(self, ssa: str, hint="v") -> str:
        name = self._fresh(hint)
        self._names[ssa] = name
        return name

    def _get(self, ssa: str) -> str:
        if ssa not in self._names:
            raise KeyError(f"Unbound SSA value: {ssa}")
        return self._names[ssa]

    def _expr(self, ssa: str) -> ast.expr:
        return _name(self._get(ssa))

    # --- module entry ---

    def translate(self, module_region: Region) -> ast.Module:
        body: list[ast.stmt] = [
            ast.Import(names=[ast.alias(name="triton")]),
            ast.ImportFrom(
                module="triton",
                names=[ast.alias(name="language", asname="tl")],
                level=0,
            ),
        ]
        for op in module_region.blocks[0].ops:
            if op.name == "func.func":
                body.append(self._func(op))
        mod = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(mod)
        return mod

    def _func(self, op: Op) -> ast.FunctionDef:
        entry = op.regions[0].blocks[0]

        # Bind memref args to pointer parameter names
        params: list[ast.arg] = []
        for arg_name, _ in entry.args:
            pname = self._bind(arg_name, "ptr")
            params.append(ast.arg(arg=pname, annotation=None))

        # Separate constants defined before gpu.launch from the kernel body
        pre_stmts: list[ast.stmt] = []
        kernel_stmts: list[ast.stmt] = []
        for child_op in entry.ops:
            if child_op.name == "arith.constant":
                pre_stmts.extend(self._arith_constant(child_op))
            elif child_op.name == "gpu.launch":
                kernel_stmts = self._gpu_launch(child_op)

        body = pre_stmts + kernel_stmts or [ast.Pass()]
        decorator = ast.Attribute(value=_name("triton"), attr="jit", ctx=ast.Load())
        # Derive kernel name from the func sym_name if available (set during translation)
        kernel_name = getattr(self, "_kernel_name", "kernel")
        return ast.FunctionDef(
            name=kernel_name,
            args=ast.arguments(
                posonlyargs=[],
                args=params,
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=body,
            decorator_list=[decorator],
            returns=None,
            lineno=0,
            col_offset=0,
        )

    # --- gpu.launch -> kernel body ---

    def _gpu_launch(self, op: Op) -> list[ast.stmt]:
        body_block = op.regions[0].blocks[0]
        stmts: list[ast.stmt] = []

        # First 3 block args are the block indices (bx, by, bz) -> program IDs
        pid_names = ["pid_m", "pid_h", "pid_b"]
        for (arg_name, _), pid, axis in zip(body_block.args[:3], pid_names, [0, 1, 2]):
            self._names[arg_name] = pid
            stmts.append(_assign(pid, _tl_call("program_id", _const(axis))))
        # Remaining args (tx/ty/tz and grid/block sizes) are unused in Triton
        for arg_name, _ in body_block.args[3:]:
            self._names[arg_name] = "_"

        stmts.extend(self._block_ops(body_block))
        return stmts

    # --- block and op dispatch ---

    def _block_ops(self, block: Block) -> list[ast.stmt]:
        stmts = []
        for op in block.ops:
            stmts.extend(self._op(op))
        return stmts

    def _op(self, op: Op) -> list[ast.stmt]:
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
            # ops that produce no Triton code
            "tensor.empty": lambda o: [],
            "gpu.terminator": lambda o: [],
            "func.return": lambda o: [],
            "linalg.yield": lambda o: [],
        }
        handler = dispatch.get(op.name)
        if handler is None:
            return [ast.Expr(value=_const(f"# TODO: {op.name}"))]
        return handler(op)

    # --- arith ops ---

    def _arith_constant(self, op: Op) -> list[ast.stmt]:
        name = self._bind(op.results[0], "c")
        return [_assign(name, _const(op.props.get("value")))]

    def _binop(self, op: Op, py_op: ast.operator) -> list[ast.stmt]:
        name = self._bind(op.results[0], "v")
        lhs, rhs = self._expr(op.operands[0]), self._expr(op.operands[1])
        return [_assign(name, ast.BinOp(left=lhs, op=py_op, right=rhs))]

    def _tl_binop(self, op: Op, fn: str) -> list[ast.stmt]:
        name = self._bind(op.results[0], "v")
        return [
            _assign(
                name,
                _tl_call(fn, self._expr(op.operands[0]), self._expr(op.operands[1])),
            )
        ]

    def _tl_unary(self, op: Op, fn: str) -> list[ast.stmt]:
        name = self._bind(op.results[0], "v")
        return [_assign(name, _tl_call(fn, self._expr(op.operands[0])))]

    def _arith_truncf(self, op: Op) -> list[ast.stmt]:
        name = self._bind(op.results[0], "v")
        _, dtype = (
            parse_tensor_shape(op.result_types[0]) if op.result_types else ([], "f16")
        )
        return [
            _assign(
                name,
                _tl_call("cast", self._expr(op.operands[0]), _mlir_dtype_to_tl(dtype)),
            )
        ]

    # --- htile ops ---

    def _htile_load(self, op: Op) -> list[ast.stmt]:
        memref_type = op.operand_types[0] if op.operand_types else ""
        result_type = op.result_types[0] if op.result_types else ""
        mem_shape, _ = parse_memref_shape(memref_type)
        tile_shape, _ = parse_tensor_shape(result_type)

        indices = op.operands[1:]
        stmts, base_ptr, tile_indices, tile_strides = self._fold_batch_dims(
            op.operands[0], indices, mem_shape, tile_shape
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

    def _htile_store(self, op: Op) -> list[ast.stmt]:
        tile_ssa = op.operands[0]
        memref_type = op.operand_types[1] if len(op.operand_types) > 1 else ""
        tile_type = op.operand_types[0] if op.operand_types else ""
        mem_shape, _ = parse_memref_shape(memref_type)
        tile_shape, _ = parse_tensor_shape(tile_type)

        indices = op.operands[2:]
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
                    self._expr(tile_ssa),
                    boundary_check=_list(*[_const(i) for i in range(len(tile_shape))]),
                )
            )
        )
        return stmts

    def _fold_batch_dims(self, memref_ssa, indices, mem_shape, tile_shape):
        """Fold batch dimensions into a pointer offset; return (stmts, base_ptr, tile_indices, tile_strides)."""
        stmts: list[ast.stmt] = []
        base_ptr = self._expr(memref_ssa)
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

    def _htile_full(self, op: Op) -> list[ast.stmt]:
        shape, dtype = (
            parse_tensor_shape(op.result_types[0]) if op.result_types else ([1], "f32")
        )
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

    def _htile_dot(self, op: Op) -> list[ast.stmt]:
        name = self._bind(op.results[0], "tile")
        lhs, rhs = self._expr(op.operands[0]), self._expr(op.operands[1])
        acc = self._expr(op.operands[2]) if len(op.operands) > 2 else None
        call = (
            _tl_call("dot", lhs, rhs) if acc is None else _tl_call("dot", lhs, rhs, acc)
        )
        return [_assign(name, call)]

    def _htile_reduce(self, op: Op) -> list[ast.stmt]:
        name = self._bind(op.results[0], "red")
        axis = op.props.get("axis", 1)
        kind = op.props.get("kind", "sum")
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

    def _htile_permute(self, op: Op) -> list[ast.stmt]:
        name = self._bind(op.results[0], "tile")
        perm = op.props.get("permutation", [1, 0])
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

    def _htile_copy(self, op: Op) -> list[ast.stmt]:
        # Placement change only — alias the name, emit nothing
        self._names[op.results[0]] = self._get(op.operands[0])
        return []

    # --- linalg.broadcast -> unsqueeze ---

    def _linalg_broadcast(self, op: Op) -> list[ast.stmt]:
        name = self._bind(op.results[0], "bcast")
        src = self._expr(op.operands[0])
        dims = op.props.get("dimensions", [1])
        rank_out = (
            len(parse_tensor_shape(op.result_types[0])[0]) if op.result_types else 2
        )

        indices: list[ast.expr] = []
        for out_dim in range(rank_out):
            indices.append(_const(None) if out_dim in dims else ast.Slice())

        idx = (
            ast.Tuple(elts=indices, ctx=ast.Load()) if len(indices) > 1 else indices[0]
        )
        return [_assign(name, ast.Subscript(value=src, slice=idx, ctx=ast.Load()))]

    # --- scf.for ---

    def _scf_for(self, op: Op) -> list[ast.stmt]:
        lb = self._expr(op.operands[0])
        ub = self._expr(op.operands[1])
        step = self._expr(op.operands[2])

        body_block = op.regions[0].blocks[0]
        loop_var_arg = body_block.args[0][0]
        iter_bargs = [a[0] for a in body_block.args[1:]]

        # Emit init assignments before the loop; bind iter block args to the same names
        pre: list[ast.stmt] = []
        out_names: list[str] = []
        for init_ssa, barg in zip(op.operands[3:], iter_bargs):
            out = self._fresh("acc")
            out_names.append(out)
            self._names[barg] = out
            pre.append(_assign(out, self._expr(init_ssa)))

        # Bind the scf.for results to the same names (they hold the post-loop values)
        for res, out in zip(op.results, out_names):
            self._names[res] = out

        lv = self._fresh("j")
        self._names[loop_var_arg] = lv

        self._for_output_names.append(out_names)
        body_stmts = self._block_ops(body_block) or [ast.Pass()]
        self._for_output_names.pop()

        for_stmt = ast.For(
            target=ast.Name(id=lv, ctx=ast.Store()),
            iter=_call(_name("range"), lb, ub, step),
            body=body_stmts,
            orelse=[],
            lineno=0,
            col_offset=0,
        )
        return pre + [for_stmt]

    def _scf_yield(self, op: Op) -> list[ast.stmt]:
        if not self._for_output_names:
            return []
        out_names = self._for_output_names[-1]
        stmts = []
        for out, val_ssa in zip(out_names, op.operands):
            val_name = self._get(val_ssa)
            if val_name != out:
                stmts.append(_assign(out, _name(val_name)))
        return stmts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("mlir_file")
    ap.add_argument("--plugin", default=DEFAULT_PLUGIN)
    args = ap.parse_args()

    region = parse_mlir_file(args.mlir_file, args.plugin)

    translator = Translator()
    py_module = translator.translate(region)
    print(ast.unparse(py_module))


if __name__ == "__main__":
    main()
