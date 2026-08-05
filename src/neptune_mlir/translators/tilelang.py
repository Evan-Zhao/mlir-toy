"""HTile MLIR -> TileLang Python translator using MLIR Python bindings."""

import ast

from mlir import ir

from . import common as shared


class Translator(shared.BaseTranslator):
    def __init__(self):
        super().__init__()
        self._broadcasts: dict[ir.Value, shared.BroadcastSpec] = {}
        self._transposed_tiles: set[ir.Value] = set()
        self._scalar_tiles: set[ir.Value] = set()
        self._yield_dests: list[list[str]] = []

    def _bind(self, value: ir.Value, hint="v") -> str:
        if value in self._names:
            return self._names[value]
        name = self._fresh(hint)
        self._names[value] = name
        return name

    def _module_prelude(self) -> list[ast.stmt]:
        return [ast.Import(names=[ast.alias(name="tilelang.language", asname="T")])]

    def _htile_kernel(self, op: ir.OpView) -> ast.FunctionDef:
        entry = op.regions[0].blocks[0]
        kernel_name = shared._func_sym_name(op)

        params: list[ast.arg] = []
        for arg in entry.arguments:
            pname = self._bind(arg, "buf")
            shape, dtype = shared._memref_shape(arg.type)
            annotation = shared._T_call(
                "Tensor",
                shared._tuple(*[shared._const(s) for s in shape]),
                shared._const(shared._mlir_dtype_to_tl_str(dtype)),
            )
            params.append(ast.arg(arg=pname, annotation=annotation))

        bounds_attr = op.attributes.get("program_bounds")
        if bounds_attr is None:
            raise ValueError("htile.kernel requires program_bounds for TileLang translation")
        program_bounds = shared._parse_dense_i64_array(bounds_attr)
        pid_names = [self._fresh("bid") for _ in program_bounds]
        for child_op in entry.operations:
            if shared._op_type_name(child_op) != "htile.program_id":
                continue
            dimension = ir.IntegerAttr(child_op.attributes["dimension"]).value
            if dimension >= len(pid_names):
                raise ValueError(f"program ID dimension {dimension} is outside program_bounds")
            self._names[child_op.results[0]] = pid_names[dimension]

        kernel_body = self._collect_allocs(entry) + self._block_ops(entry)
        with_item = ast.withitem(
            context_expr=shared._T_call(
                "Kernel",
                *[shared._const(bound) for bound in program_bounds],
                threads=shared._const(128),
            ),
            optional_vars=shared._tuple(
                *[shared._name(name, ast.Store()) for name in pid_names], ctx=ast.Store()
            ),
        )
        body: list[ast.stmt] = [ast.With(items=[with_item], body=kernel_body or [ast.Pass()])]
        arguments = ast.arguments(
            posonlyargs=[],
            args=params,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )
        attr_kernel: list[ast.expr] = [shared._T("prim_func")]
        return ast.FunctionDef(
            name=kernel_name, args=arguments, body=body, decorator_list=attr_kernel, type_params=[]
        )

    # --- allocation pass ---

    def _collect_allocs(self, block: ir.Block) -> list[ast.stmt]:
        allocs: list[ast.stmt] = []
        for op in block.operations:
            allocs.extend(self._collect_op_allocs(op))
        return allocs

    def _collect_op_allocs(self, op: ir.OpView) -> list[ast.stmt]:
        op_name = shared._op_type_name(op)
        allocs: list[ast.stmt] = []

        if op_name == "scf.for":
            allocs.extend(self._collect_allocs(op.regions[0].blocks[0]))
            return allocs
        if op_name in {"htile.broadcast", "htile.permute", "tensor.empty", "htile.copy"}:
            return allocs

        for result in op.results:
            if shared._is_ranked_tensor_type(result.type):
                shape, dtype = shared._tensor_shape(result.type)
                if op_name == "htile.load" and not shape:
                    continue
                if op_name == "htile.load":
                    dimension_order = shared._dimension_order(op, len(shape))
                    shape = [shape[dim] for dim in dimension_order]
                name = self._bind(result, self._result_hint(op_name))
                alloc_fn = (
                    "alloc_shared"
                    if op_name == "htile.load" or self._is_shared(result.type)
                    else "alloc_fragment"
                )
                allocs.append(
                    shared._assign(
                        name,
                        shared._T_call(
                            alloc_fn,
                            shared._list(*[shared._const(s) for s in shape]),
                            shared._const(shared._mlir_dtype_to_tl_str(dtype)),
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
            "htile.broadcast": "bcast",
            "arith.truncf": "cast",
            "math.exp2": "exp",
        }.get(op_name, "frag")

    # --- scalar and elementwise ops ---

    def _arith_constant(self, op: ir.OpView) -> list[ast.stmt]:
        name = self._bind(op.results[0], "c")
        val = shared._decode_constant(op)
        return [shared._assign(name, shared._const(val))]

    def _scalar_binop(self, op: ir.OpView, py_op: ast.operator) -> list[ast.stmt]:
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

    def _binary_op(self, op: ir.OpView, py_op: ast.operator) -> list[ast.stmt]:
        if not shared._is_ranked_tensor_type(op.results[0].type):
            return self._scalar_binop(op, py_op)

        def build(indices: list[ast.expr]) -> ast.expr:
            return ast.BinOp(
                left=self._value_at(op.operands[0], indices),
                op=py_op,
                right=self._value_at(op.operands[1], indices),
            )

        return self._parallel_store(op.results[0], build)

    def _scalar_call_binop(self, op: ir.OpView, fn: str) -> list[ast.stmt]:
        name = self._bind(op.results[0], "v")
        return [
            shared._assign(
                name,
                shared._T_call(fn, self._expr(op.operands[0]), self._expr(op.operands[1])),
            )
        ]

    def _arith_maxsi(self, op: ir.OpView) -> list[ast.stmt]:
        return self._scalar_call_binop(op, "max")

    def _arith_minsi(self, op: ir.OpView) -> list[ast.stmt]:
        return self._scalar_call_binop(op, "min")

    def _arith_maximumf(self, op: ir.OpView) -> list[ast.stmt]:
        if not shared._is_ranked_tensor_type(op.results[0].type):
            name = self._bind(op.results[0], "v")
            return [
                shared._assign(
                    name,
                    shared._T_call("max", self._expr(op.operands[0]), self._expr(op.operands[1])),
                )
            ]

        def build(indices: list[ast.expr]) -> ast.expr:
            return shared._T_call(
                "max",
                self._value_at(op.operands[0], indices),
                self._value_at(op.operands[1], indices),
            )

        return self._parallel_store(op.results[0], build)

    def _math_exp2(self, op: ir.OpView) -> list[ast.stmt]:
        return self._parallel_store(
            op.results[0],
            lambda indices: shared._T_call("exp2", self._value_at(op.operands[0], indices)),
        )

    def _arith_index_cast(self, op: ir.OpView) -> list[ast.stmt]:
        _, dtype = shared._tensor_shape(op.results[0].type)
        dtype_str = shared._mlir_dtype_to_tl_str(dtype)
        return self._parallel_store(
            op.results[0],
            lambda indices: shared._T_call(
                "cast", self._value_at(op.operands[0], indices), shared._const(dtype_str)
            ),
        )

    def _arith_cmpi(self, op: ir.OpView) -> list[ast.stmt]:
        cmp_op = shared._decode_cmp_predicate(op)
        return self._parallel_store(
            op.results[0],
            lambda indices: ast.Compare(
                left=self._value_at(op.operands[0], indices),
                ops=[cmp_op],
                comparators=[self._value_at(op.operands[1], indices)],
            ),
        )

    def _arith_select(self, op: ir.OpView) -> list[ast.stmt]:
        return self._parallel_store(
            op.results[0],
            lambda indices: shared._T_call(
                "if_then_else",
                self._value_at(op.operands[0], indices),
                self._value_at(op.operands[1], indices),
                self._value_at(op.operands[2], indices),
            ),
        )

    def _arith_cast(self, op: ir.OpView) -> list[ast.stmt]:
        _, dtype = shared._tensor_shape(op.results[0].type)
        dtype_str = shared._mlir_dtype_to_tl_str(dtype)
        return self._parallel_store(
            op.results[0],
            lambda indices: shared._T_call(
                "cast", self._value_at(op.operands[0], indices), shared._const(dtype_str)
            ),
        )

    def _parallel_store(self, result: ir.Value, expr_builder) -> list[ast.stmt]:
        shape, _ = shared._tensor_shape(result.type)
        out = self._get(result)
        index_names = [self._fresh(f"i{dim}") for dim in range(len(shape))]
        indices: list[ast.expr] = [shared._name(n) for n in index_names]
        target: ast.expr
        if len(index_names) == 1:
            target = shared._name(index_names[0], ast.Store())
        else:
            target = shared._tuple(*[shared._name(n, ast.Store()) for n in index_names], ctx=ast.Store())
        body: list[ast.stmt] = [
            ast.Assign(
                targets=[shared._store_subscript(shared._name(out), indices)],
                value=expr_builder(indices),
                lineno=0,
            )
        ]
        return [
            ast.For(
                target=target,
                iter=shared._T_call("Parallel", *[shared._const(s) for s in shape]),
                body=body,
                orelse=[],
                lineno=0,
                col_offset=0,
            )
        ]

    def _value_at(self, value: ir.Value, indices: list[ast.expr]) -> ast.expr:
        if value in self._broadcasts:
            info = self._broadcasts[value]
            src_indices = [idx for dim, idx in enumerate(indices) if dim not in info.dimensions]
            return self._value_at(info.source, src_indices)
        if value in self._scalar_tiles:
            return self._expr(value)
        if shared._is_ranked_tensor_type(value.type):
            return shared._subscript(self._expr(value), indices)
        return self._expr(value)

    # --- htile ops ---

    def _htile_program_id(self, op: ir.OpView) -> list[ast.stmt]:
        return []

    def _htile_load(self, op: ir.OpView) -> list[ast.stmt]:
        spec = shared._decode_load(op)
        shape = spec.tile_shape
        if not shape:
            self._scalar_tiles.add(op.results[0])
            name = self._bind(op.results[0], "scalar")
            src = self._mem_region(op.operands[0], list(op.operands[1:]), op.results[0])
            return [shared._assign(name, src)]
        dimension_order = spec.dimension_order
        physical_shape = [shape[dim] for dim in dimension_order]
        if dimension_order != list(range(len(shape))):
            if dimension_order != [1, 0]:
                raise NotImplementedError(
                    f"unsupported TileLang load dimension_order: {dimension_order}"
                )
            self._transposed_tiles.add(op.results[0])
        src = self._mem_region(
            op.operands[0], list(op.operands[1:]), op.results[0], tile_shape=physical_shape
        )
        return [shared._expr_stmt(shared._T_call("copy", src, self._expr(op.results[0])))]

    def _htile_store(self, op: ir.OpView) -> list[ast.stmt]:
        dst = self._mem_region(op.operands[1], list(op.operands[2:]), op.operands[0])
        return [shared._expr_stmt(shared._T_call("copy", self._expr(op.operands[0]), dst))]

    def _htile_full(self, op: ir.OpView) -> list[ast.stmt]:
        return [shared._expr_stmt(shared._T_call("fill", self._expr(op.results[0]), self._expr(op.operands[0])))]

    def _htile_arange(self, op: ir.OpView) -> list[ast.stmt]:
        start = self._expr(op.operands[0])
        return self._parallel_store(
            op.results[0],
            lambda indices: ast.BinOp(left=start, op=ast.Add(), right=indices[0]),
        )

    def _htile_dot(self, op: ir.OpView) -> list[ast.stmt]:
        spec = shared._decode_dot(op)
        dst = self._expr(op.results[0])
        stmts: list[ast.stmt] = []
        accumulator = spec.accumulator
        clear_accum = accumulator is None or shared._op_type_name(accumulator.owner) == "htile.full"
        if accumulator is not None and not clear_accum:
            stmts.append(shared._expr_stmt(shared._T_call("copy", self._expr(accumulator), dst)))

        kwargs: dict[str, ast.expr] = {"clear_accum": shared._const(clear_accum)}
        if spec.transpose_a or spec.lhs in self._transposed_tiles:
            kwargs["transpose_A"] = shared._const(True)
        if spec.transpose_b or spec.rhs in self._transposed_tiles:
            kwargs["transpose_B"] = shared._const(True)
        policy_attr = op.attributes.get("warp_policy")
        policy = shared._parse_attr_str(policy_attr) if policy_attr is not None else "full_row"
        kwargs["policy"] = self._warp_policy_expr(policy)

        stmts.append(
            shared._expr_stmt(
                shared._T_call(
                    "gemm",
                    self._expr(spec.lhs),
                    self._expr(spec.rhs),
                    dst,
                    **kwargs,
                )
            )
        )
        return stmts

    def _htile_reduce(self, op: ir.OpView) -> list[ast.stmt]:
        spec = shared._decode_reduction(op)
        fn = "reduce_max" if spec.kind == "max" else "reduce_sum"
        return [
            shared._expr_stmt(
                shared._T_call(
                    fn,
                    self._expr(spec.value),
                    self._expr(op.results[0]),
                    dim=shared._const(spec.axis),
                )
            )
        ]

    def _htile_permute(self, op: ir.OpView) -> list[ast.stmt]:
        permutation = shared._decode_permutation(op)
        if permutation != [1, 0]:
            raise NotImplementedError(f"unsupported TileLang permutation: {permutation}")
        self._names[op.results[0]] = self._get(op.operands[0])
        self._transposed_tiles.add(op.results[0])
        return []

    def _htile_copy(self, op: ir.OpView) -> list[ast.stmt]:
        self._names[op.results[0]] = self._get(op.operands[0])
        return []

    # --- htile.broadcast ---

    def _htile_broadcast(self, op: ir.OpView) -> list[ast.stmt]:
        self._broadcasts[op.results[0]] = shared._decode_broadcast(op)
        return []

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
            iter_expr = shared._T_call(
                "Pipelined", lb, ub, num_stages=shared._const(shared._parse_attr_int(stages_attr))
            )
        else:
            iter_expr = shared._T_call("serial", lb, ub, step)

        if stages_attr is not None and self._const_int(op.operands[2]) != 1:
            raise NotImplementedError("T.Pipelined emission currently expects step = 1")

        return [
            ast.For(
                target=shared._name(loop_name, ast.Store()),
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
                stmts.append(shared._expr_stmt(shared._T_call("copy", shared._name(src), shared._name(dest))))
        return stmts

    # --- helpers ---

    def _mem_region(
        self,
        memref: ir.Value,
        offsets: list[ir.Value],
        tile_value: ir.Value,
        tile_shape: list[int] | None = None,
    ) -> ast.Subscript:
        if tile_shape is None:
            tile_shape, _ = shared._tensor_shape(tile_value.type)
        mem_shape, _ = shared._memref_shape(memref.type)
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
                            left=self._expr(offset), op=ast.Add(), right=shared._const(extent)
                        ),
                        step=None,
                    )
                )
        return shared._subscript(self._expr(memref), indices)

    def _warp_policy_expr(self, policy: str) -> ast.expr:
        mapping = {
            "square": "Square",
            "full_row": "FullRow",
            "full_col": "FullCol",
        }
        if policy not in mapping:
            raise NotImplementedError(f"unsupported warp_policy: {policy}")
        return shared._attr(shared._T("GemmWarpPolicy"), mapping[policy])

    def _is_shared(self, tensor_type) -> bool:
        return "placement = shared" in str(tensor_type)

    def _const_int(self, value: ir.Value) -> int | None:
        owner = value.owner
        if shared._op_type_name(owner) != "arith.constant":
            return None
        assert not isinstance(owner, ir.Block)
        attr = owner.attributes.get("value")
        if isinstance(attr, ir.IntegerAttr):
            return attr.value
        return None


def translate_mlir_text(text: str) -> ast.Module:
    """Parse translator-ready MLIR text and return a Python ast.Module."""
    return shared.translate_text_with(text, Translator)
