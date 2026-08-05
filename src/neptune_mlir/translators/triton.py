"""HTile kernel MLIR -> Triton Python translator using MLIR Python bindings."""

import ast

from mlir import ir

from . import common as shared


class Translator(shared.BaseTranslator):
    def __init__(self):
        super().__init__()
        self._for_output_names: list[list[str]] = []

    def _module_prelude(self) -> list[ast.stmt]:
        return [
            ast.Import(names=[ast.alias(name="triton")]),
            ast.ImportFrom(
                module="triton",
                names=[ast.alias(name="language", asname="tl")],
                level=0,
            ),
        ]

    def _htile_kernel(self, op: ir.OpView) -> ast.FunctionDef:
        entry = op.regions[0].blocks[0]
        kernel_name = shared._func_sym_name(op)

        params: list[ast.arg] = []
        for arg in entry.arguments:
            pname = self._bind(arg, "ptr")
            params.append(ast.arg(arg=pname, annotation=None))

        body = self._block_ops(entry) or [ast.Pass()]
        decorator = ast.Attribute(value=shared._name("triton"), attr="jit", ctx=ast.Load())
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
            name=kernel_name, args=arguments, body=body, decorator_list=[decorator], type_params=[]
        )

    # --- arith ops ---

    def _arith_constant(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "c")
        val = shared._decode_constant(op)
        return [
            ast.AnnAssign(
                target=shared._name(name, ast.Store()),
                annotation=shared._tl("constexpr"),
                value=shared._const(val),
                simple=1,
            )
        ]

    def _binary_op(self, op: ir.OpView, py_op: ast.operator) -> list[ast.stmt]:
        name = self._bind(op.results[0], "v")
        lhs = self._expr(op.operands[0])
        rhs = self._expr(op.operands[1])
        return [shared._assign(name, ast.BinOp(left=lhs, op=py_op, right=rhs))]

    def _tl_binop(self, op: ir.OpView, fn: str) -> list[ast.stmt]:
        name = self._bind(op.results[0], "v")
        return [
            shared._assign(
                name, shared._tl_call(fn, self._expr(op.operands[0]), self._expr(op.operands[1]))
            )
        ]

    def _scalar_call_binop(self, op: ir.OpView, fn: str) -> list[ast.stmt]:
        if shared._is_ranked_tensor_type(op.results[0].type) or any(
            shared._is_ranked_tensor_type(operand.type) for operand in op.operands
        ):
            raise NotImplementedError(f"unsupported tensor {shared._op_type_name(op)}")
        name = self._bind(op.results[0], "v")
        return [
            shared._assign(
                name,
                shared._call(
                    shared._name(fn), self._expr(op.operands[0]), self._expr(op.operands[1])
                ),
            )
        ]

    def _tl_unary(self, op: ir.OpView, fn: str) -> list[ast.stmt]:
        name = self._bind(op.results[0], "v")
        return [shared._assign(name, shared._tl_call(fn, self._expr(op.operands[0])))]

    def _arith_maxsi(self, op: ir.OpView) -> list[ast.stmt]:
        return self._scalar_call_binop(op, "max")

    def _arith_minsi(self, op: ir.OpView) -> list[ast.stmt]:
        return self._scalar_call_binop(op, "min")

    def _arith_maximumf(self, op: ir.OpView) -> list[ast.stmt]:
        return self._tl_binop(op, "maximum")

    def _math_exp2(self, op: ir.OpView) -> list[ast.stmt]:
        return self._tl_unary(op, "exp2")

    def _arith_index_cast(self, op: ir.OpView) -> list[ast.stmt]:
        self._names[op.results[0]] = self._get(op.operands[0])
        return []

    def _arith_cmpi(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "cmp")
        cmp_op = shared._decode_cmp_predicate(op)
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
        cond, true_val, false_val = map(self._expr, op.operands)
        return [shared._assign(name, shared._tl_call("where", cond, true_val, false_val))]

    def _arith_cast(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "v")
        _, dtype = shared._tensor_shape(op.results[0].type)
        if not dtype:
            dtype = str(op.results[0].type)
        return [
            shared._assign(
                name,
                shared._tl_call(
                    "cast", self._expr(op.operands[0]), shared._mlir_dtype_to_tl(dtype)
                ),
            )
        ]

    # --- htile ops ---

    def _htile_program_id(self, op: ir.OpView) -> list[ast.stmt]:
        dimension = ir.IntegerAttr(op.attributes["dimension"]).value
        name = self._bind(op.results[0], "pid")
        return [shared._assign(name, shared._tl_call("program_id", shared._const(dimension)))]

    def _htile_load(self, op: ir.OpView) -> list[ast.stmt]:
        spec = shared._decode_load(op)
        mem_shape = spec.memref_shape
        tile_shape = spec.tile_shape
        indices = spec.offsets

        # Scalar (0D) loads cannot use block pointer. Load directly.
        if not tile_shape:
            stmts, ptr = self._scalar_memref_ptr(op.operands[0], indices, mem_shape)
            tile = self._bind(op.results[0], "tile")
            stmts.append(shared._assign(tile, shared._tl_call("load", ptr)))
            return stmts

        stmts, base_ptr, tile_indices, tile_strides = self._fold_batch_dims(
            op.operands[0], indices, mem_shape, tile_shape
        )
        dimension_order = spec.dimension_order
        mem_tile_shape = mem_shape[-len(tile_shape) :]
        logical_shape = [mem_tile_shape[i] for i in dimension_order]
        logical_strides = [tile_strides[i] for i in dimension_order]
        logical_offsets = [tile_indices[i] for i in dimension_order]
        block_ptr_order = sorted(range(len(logical_strides)), key=logical_strides.__getitem__)

        bp = self._fresh("bp")
        stmts.append(
            shared._assign(
                bp,
                shared._call(
                    shared._tl("make_block_ptr"),
                    base=base_ptr,
                    shape=shared._list(*[shared._const(s) for s in logical_shape]),
                    strides=shared._list(*[shared._const(s) for s in logical_strides]),
                    offsets=shared._list(
                        *[self._expr(i) for i in logical_offsets[: len(tile_shape)]]
                    ),
                    block_shape=shared._list(*[shared._const(s) for s in tile_shape]),
                    order=shared._list(*[shared._const(i) for i in block_ptr_order]),
                ),
            )
        )
        tile = self._bind(op.results[0], "tile")
        stmts.append(shared._assign(tile, shared._tl_call("load", shared._name(bp))))
        return stmts

    def _htile_store(self, op: ir.OpView) -> list[ast.stmt]:
        tile_val = op.operands[0]
        mem_shape, _ = shared._memref_shape(op.operands[1].type)
        tile_shape, _ = shared._tensor_shape(op.operands[0].type)
        indices = list(op.operands[2:])

        if not tile_shape:
            stmts, ptr = self._scalar_memref_ptr(op.operands[1], indices, mem_shape)
            stmts.append(ast.Expr(value=shared._tl_call("store", ptr, self._expr(tile_val))))
            return stmts

        stmts, base_ptr, tile_indices, tile_strides = self._fold_batch_dims(
            op.operands[1], indices, mem_shape, tile_shape
        )
        bp = self._fresh("bp")
        stmts.append(
            shared._assign(
                bp,
                shared._call(
                    shared._tl("make_block_ptr"),
                    base=base_ptr,
                    shape=shared._list(*[shared._const(s) for s in mem_shape[-2:]]),
                    strides=shared._list(*[shared._const(s) for s in tile_strides]),
                    offsets=shared._list(*[self._expr(i) for i in tile_indices[: len(tile_shape)]]),
                    block_shape=shared._list(*[shared._const(s) for s in tile_shape]),
                    order=shared._list(
                        *[shared._const(i) for i in reversed(range(len(tile_shape)))]
                    ),
                ),
            )
        )
        stmts.append(
            ast.Expr(value=shared._tl_call("store", shared._name(bp), self._expr(tile_val)))
        )
        return stmts

    def _scalar_memref_ptr(
        self, memref_val: ir.Value, indices: list[ir.Value], mem_shape: list[int]
    ) -> tuple[list[ast.stmt], ast.expr]:
        if len(indices) != len(mem_shape):
            raise NotImplementedError(
                f"scalar memref access rank mismatch: {len(indices)} indices for rank {len(mem_shape)}"
            )

        strides = [1] * len(mem_shape)
        for i in range(len(mem_shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * mem_shape[i + 1]

        offset: ast.expr = shared._const(0)
        for index, stride in zip(indices, strides):
            term = ast.BinOp(left=self._expr(index), op=ast.Mult(), right=shared._const(stride))
            offset = ast.BinOp(left=offset, op=ast.Add(), right=term)

        ptr = self._fresh("ptr")
        stmt = shared._assign(
            ptr, ast.BinOp(left=self._expr(memref_val), op=ast.Add(), right=offset)
        )
        return [stmt], shared._name(ptr)

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
            offset: ast.expr = shared._const(0)
            for i in range(batch_dims):
                term = ast.BinOp(
                    left=self._expr(indices[i]), op=ast.Mult(), right=shared._const(strides[i])
                )
                offset = ast.BinOp(left=offset, op=ast.Add(), right=term)
            ptr = self._fresh("ptr")
            stmts.append(shared._assign(ptr, ast.BinOp(left=base_ptr, op=ast.Add(), right=offset)))
            base_ptr = shared._name(ptr)
            return stmts, base_ptr, indices[batch_dims:], strides[batch_dims:]
        return stmts, base_ptr, indices, [1] * len(tile_shape)

    def _htile_full(self, op: ir.OpView) -> list[ast.stmt]:
        shape, dtype = shared._tensor_shape(op.results[0].type)
        name = self._bind(op.results[0], "tile")
        return [
            shared._assign(
                name,
                shared._tl_call(
                    "full",
                    shared._tuple(*[shared._const(extent) for extent in shape]),
                    self._expr(op.operands[0]),
                    shared._mlir_dtype_to_tl(dtype),
                ),
            )
        ]

    def _htile_arange(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "range")
        return [
            shared._assign(
                name,
                shared._tl_call("arange", self._expr(op.operands[0]), self._expr(op.operands[1])),
            )
        ]

    def _htile_dot(self, op: ir.OpView) -> list[ast.stmt]:
        shared._reject_dot_transpose_attrs(op, "Triton")
        spec = shared._decode_dot(op)
        name = self._bind(op.results[0], "tile")
        lhs = self._expr(spec.lhs)
        rhs = self._expr(spec.rhs)
        _, lhs_dtype = shared._tensor_shape(spec.lhs.type)
        _, rhs_dtype = shared._tensor_shape(spec.rhs.type)
        lhs_is_fp8, rhs_is_fp8 = lhs_dtype == "f8", rhs_dtype == "f8"
        if lhs_is_fp8 != rhs_is_fp8:
            raise NotImplementedError(
                "Triton does not support mixed FP8/non-FP8 htile.dot operands; "
                f"got {lhs_dtype} x {rhs_dtype}. Dequantize FP8 operands before htile.dot."
            )
        acc = self._expr(spec.accumulator) if spec.accumulator is not None else None
        call = (
            shared._tl_call("dot", lhs, rhs)
            if acc is None
            else shared._tl_call("dot", lhs, rhs, acc)
        )
        return [shared._assign(name, call)]

    def _htile_reduce(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "red")
        spec = shared._decode_reduction(op)
        fn = "max" if spec.kind == "max" else "sum"
        return [
            shared._assign(
                name,
                shared._tl_call(
                    fn,
                    self._expr(spec.value),
                    shared._const(spec.axis),
                    keep_dims=shared._const(False),
                ),
            )
        ]

    def _htile_permute(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "tile")
        perm = shared._decode_permutation(op)
        return [
            shared._assign(
                name,
                shared._tl_call(
                    "permute",
                    self._expr(op.operands[0]),
                    shared._list(*[shared._const(p) for p in perm]),
                ),
            )
        ]

    def _htile_copy(self, op: ir.OpView) -> list[ast.stmt]:
        # Placement change only — alias the SSA value, emit nothing.
        self._names[op.results[0]] = self._get(op.operands[0])
        return []

    # --- htile.broadcast -> unsqueeze ---

    def _htile_broadcast(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "bcast")
        spec = shared._decode_broadcast(op)
        src = self._expr(spec.source)
        rank_out = len(spec.result_shape)

        indices: list[ast.expr] = []
        for out_dim in range(rank_out):
            indices.append(shared._const(None) if out_dim in spec.dimensions else ast.Slice())

        idx = ast.Tuple(elts=indices, ctx=ast.Load()) if len(indices) > 1 else indices[0]
        return [shared._assign(name, ast.Subscript(value=src, slice=idx, ctx=ast.Load()))]

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
