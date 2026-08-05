"""HTile MLIR -> NVIDIA cuTile Python translator using MLIR Python bindings.

This backend is intentionally value-based: cuTile tiles are immutable values, so
the lowering is closer to the Triton translator than to TileLang.
"""

import ast

from mlir import ir

from . import common as shared


def _ct(attr: str) -> ast.Attribute:
    return shared._attr(shared._name("ct"), attr)


def _ct_call(fn: str, *args: ast.expr, **kwargs: ast.expr) -> ast.Call:
    return shared._call(_ct(fn), *args, **kwargs)


def _mlir_dtype_to_ct(dtype: str) -> ast.expr:
    return _ct(shared._mlir_dtype_name(dtype))


class Translator(shared.BaseTranslator):
    def __init__(self):
        super().__init__()
        self._for_output_names: list[list[str]] = []

    def _module_prelude(self) -> list[ast.stmt]:
        return [ast.Import(names=[ast.alias(name="cuda.tile", asname="ct")])]

    def _htile_kernel(self, op: ir.OpView) -> ast.FunctionDef:
        entry = op.regions[0].blocks[0]
        kernel_name = shared._func_sym_name(op)

        params: list[ast.arg] = []
        for arg in entry.arguments:
            params.append(ast.arg(arg=self._bind(arg, "arr"), annotation=None))

        body = self._block_ops(entry) or [ast.Pass()]
        arguments = ast.arguments(
            posonlyargs=[],
            args=params,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )
        attr_kernel: list[ast.expr] = [_ct("kernel")]
        return ast.FunctionDef(
            name=kernel_name, args=arguments, body=body, decorator_list=attr_kernel, type_params=[]
        )

    # --- arithmetic ops ---

    def _arith_constant(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "c")
        val = shared._decode_constant(op)
        return [shared._assign(name, shared._const(val))]

    def _binary_op(self, op: ir.OpView, py_op: ast.operator) -> list[ast.stmt]:
        name = self._bind(op.results[0], "v")
        return [
            shared._assign(
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
            shared._assign(
                name,
                _ct_call(fn, self._expr(op.operands[0]), self._expr(op.operands[1])),
            )
        ]

    def _ct_unary(self, op: ir.OpView, fn: str) -> list[ast.stmt]:
        name = self._bind(op.results[0], "v")
        return [shared._assign(name, _ct_call(fn, self._expr(op.operands[0])))]

    def _arith_maxsi(self, op: ir.OpView) -> list[ast.stmt]:
        return self._ct_binop(op, "maximum")

    def _arith_minsi(self, op: ir.OpView) -> list[ast.stmt]:
        return self._ct_binop(op, "minimum")

    def _arith_maximumf(self, op: ir.OpView) -> list[ast.stmt]:
        return self._ct_binop(op, "maximum")

    def _math_exp2(self, op: ir.OpView) -> list[ast.stmt]:
        return self._ct_unary(op, "exp2")

    def _arith_index_cast(self, op: ir.OpView) -> list[ast.stmt]:
        self._names[op.results[0]] = self._get(op.operands[0])
        return []

    def _arith_cmpi(self, op: ir.OpView) -> list[ast.stmt]:
        cmp_op = shared._decode_cmp_predicate(op)
        name = self._bind(op.results[0], "cmp")
        return [
            shared._assign(
                name,
                ast.Compare(
                    left=self._expr(op.operands[0]),
                    ops=[cmp_op],
                    comparators=[self._expr(op.operands[1])],
                ),
            )
        ]

    def _arith_select(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "sel")
        return [
            shared._assign(
                name,
                _ct_call(
                    "where",
                    self._expr(op.operands[0]),
                    self._expr(op.operands[1]),
                    self._expr(op.operands[2]),
                ),
            )
        ]

    def _arith_cast(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "v")
        _, dtype = shared._tensor_shape(op.results[0].type)
        return [
            shared._assign(
                name,
                _ct_call("astype", self._expr(op.operands[0]), _mlir_dtype_to_ct(dtype)),
            )
        ]

    # --- htile ops ---

    def _htile_program_id(self, op: ir.OpView) -> list[ast.stmt]:
        dimension = ir.IntegerAttr(op.attributes["dimension"]).value
        name = self._bind(op.results[0], "bid")
        return [shared._assign(name, _ct_call("bid", shared._const(dimension)))]

    def _htile_load(self, op: ir.OpView) -> list[ast.stmt]:
        spec = shared._decode_load(op)
        mem_shape = spec.memref_shape
        tile_shape = spec.tile_shape
        offsets = spec.offsets
        if len(offsets) != len(mem_shape):
            raise NotImplementedError("cuTile load expects one offset per memref dimension")

        batch_dims = len(mem_shape) - len(tile_shape)
        if batch_dims < 0:
            raise NotImplementedError("cuTile load rank mismatch")

        dimension_order = spec.dimension_order
        full_order = list(range(batch_dims)) + [batch_dims + dim for dim in dimension_order]
        full_tile_shape = [1] * batch_dims + [tile_shape[dim] for dim in range(len(tile_shape))]
        index = self._tile_space_index(offsets, full_order, full_tile_shape)

        loaded = self._fresh("load")
        stmts: list[ast.stmt] = [
            shared._assign(
                loaded,
                _ct_call(
                    "load",
                    self._expr(op.operands[0]),
                    shared._tuple(*index),
                    shared._tuple(*[shared._const(s) for s in full_tile_shape]),
                    order=shared._tuple(*[shared._const(i) for i in full_order]),
                ),
            )
        ]

        result = self._bind(op.results[0], "tile")
        stmts.append(
            shared._assign(
                result,
                _ct_call("reshape", shared._name(loaded), shared._tuple(*[shared._const(s) for s in tile_shape])),
            )
        )
        return stmts

    def _htile_store(self, op: ir.OpView) -> list[ast.stmt]:
        mem_shape, _ = shared._memref_shape(op.operands[1].type)
        tile_shape, _ = shared._tensor_shape(op.operands[0].type)
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
            shared._tuple(*[shared._const(s) for s in full_tile_shape]),
        )
        return [shared._expr_stmt(_ct_call("store", self._expr(op.operands[1]), shared._tuple(*index), tile))]

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
            return shared._const(offset.value // extent)
        return ast.BinOp(left=offset, op=ast.FloorDiv(), right=shared._const(extent))

    def _htile_full(self, op: ir.OpView) -> list[ast.stmt]:
        shape, dtype = shared._tensor_shape(op.results[0].type)
        name = self._bind(op.results[0], "tile")
        return [
            shared._assign(
                name,
                _ct_call(
                    "full",
                    shared._tuple(*[shared._const(s) for s in shape]),
                    self._expr(op.operands[0]),
                    dtype=_mlir_dtype_to_ct(dtype),
                ),
            )
        ]

    def _htile_arange(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "range")
        start = self._expr(op.operands[0])
        size = ast.BinOp(left=self._expr(op.operands[1]), op=ast.Sub(), right=start)
        return [
            shared._assign(
                name,
                ast.BinOp(
                    left=_ct_call("arange", size, dtype=_ct("int64")),
                    op=ast.Add(),
                    right=start,
                ),
            )
        ]

    def _htile_dot(self, op: ir.OpView) -> list[ast.stmt]:
        shared._reject_dot_transpose_attrs(op, "cuTile")

        spec = shared._decode_dot(op)
        name = self._bind(op.results[0], "tile")
        acc = (
            self._expr(spec.accumulator)
            if spec.accumulator is not None
            else _ct_call(
                "full",
                shared._tuple(*[shared._const(s) for s in spec.result_shape]),
                shared._const(0),
                dtype=_mlir_dtype_to_ct(spec.result_dtype),
            )
        )
        return [
            shared._assign(
                name,
                _ct_call(
                    "mma",
                    self._expr(spec.lhs),
                    self._expr(spec.rhs),
                    acc,
                ),
            )
        ]

    def _htile_reduce(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "red")
        spec = shared._decode_reduction(op)
        fn = "max" if spec.kind == "max" else "sum"
        return [
            shared._assign(
                name,
                _ct_call(
                    fn,
                    self._expr(spec.value),
                    shared._const(spec.axis),
                    keepdims=shared._const(False),
                ),
            )
        ]

    def _htile_permute(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "tile")
        perm = shared._decode_permutation(op)
        return [
            shared._assign(
                name,
                _ct_call(
                    "permute",
                    self._expr(op.operands[0]),
                    shared._tuple(*[shared._const(p) for p in perm]),
                ),
            )
        ]

    def _htile_copy(self, op: ir.OpView) -> list[ast.stmt]:
        # Placement change only; cuTile has no explicit shared/local placement.
        self._names[op.results[0]] = self._get(op.operands[0])
        return []

    # --- htile.broadcast -> expand dims ---

    def _htile_broadcast(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "bcast")
        spec = shared._decode_broadcast(op)
        expr = self._expr(spec.source)
        for dim in spec.dimensions:
            expr = _ct_call("expand_dims", expr, axis=shared._const(dim))
        return [
            shared._assign(
                name,
                _ct_call("broadcast_to", expr, shared._tuple(*[shared._const(s) for s in spec.result_shape])),
            )
        ]

    # --- scf.for ---

    def _scf_for(self, op: ir.OpView) -> list[ast.stmt]:
        spec = shared._decode_for(op)
        lb = self._expr(spec.lower_bound)
        ub = self._expr(spec.upper_bound)
        step = self._expr(spec.step)
        body_block = spec.body
        loop_var = spec.induction_variable
        iter_bargs = spec.iter_arguments
        iter_inits = spec.iter_initializers

        pre: list[ast.stmt] = []
        out_names: list[str] = []
        for init_val, barg in zip(iter_inits, iter_bargs):
            out = self._fresh("acc")
            out_names.append(out)
            self._names[barg] = out
            pre.append(shared._assign(out, self._expr(init_val)))

        for res, out in zip(op.results, out_names):
            self._names[res] = out

        lv = self._fresh("j")
        self._names[loop_var] = lv

        self._for_output_names.append(out_names)
        body_stmts = self._block_ops(body_block) or [ast.Pass()]
        self._for_output_names.pop()

        for_stmt = ast.For(
            target=ast.Name(id=lv, ctx=ast.Store()),
            iter=shared._call(shared._name("range"), lb, ub, step),
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
                stmts.append(shared._assign(out, shared._name(val_name)))
        return stmts


def translate_mlir_text(text: str) -> ast.Module:
    """Parse translator-ready MLIR text and return a Python ast.Module."""
    return shared.translate_text_with(text, Translator)
