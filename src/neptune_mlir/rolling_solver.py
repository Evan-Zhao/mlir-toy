from typing import Any, Iterable, Mapping, cast

import sympy as sp
from sympy import Expr, Symbol

JsonExpr = dict[str, Any]
Subs = dict[Symbol, Expr]


def _prime_symbol(symbol: Symbol) -> Symbol:
    return sp.Symbol(f"{symbol.name}'", real=True)


def _free_vars_symbols(expr: sp.Basic) -> set[Symbol]:
    def assert_symbol(sym: sp.Basic) -> Symbol:
        if not isinstance(sym, Symbol):
            raise ValueError(f"expected symbol, got {sym} in {expr}")
        return sym

    return {assert_symbol(sym) for sym in expr.free_symbols}


def solve_rolling_updater_json(
    f_expr: JsonExpr, g_expr: JsonExpr, r_var_names: list[str], acc_var_name: str
) -> JsonExpr:
    """Typed expression interface for the MLIR integration path."""
    symtab: dict[str, Symbol] = {}
    f = _json_expr_to_sympy(f_expr, symtab)
    g = _json_expr_to_sympy(g_expr, symtab)
    acc = {v.name: v for v in _free_vars_symbols(f)}[acc_var_name]
    r_var_name_set = set(r_var_names)
    g_free_vars = _free_vars_symbols(g)
    r_vars = [sym for sym in g_free_vars if sym.name in r_var_name_set]
    r_vars = sorted(r_vars, key=lambda sym: sym.name)
    c_vars = sorted(g_free_vars - set(r_vars), key=lambda sym: sym.name)
    r_to_rp = {r: _prime_symbol(r) for r in r_vars}

    h_expr = sympy_solve_rolling_updater(f, g, r_to_rp, c_vars, acc)
    return _sympy_to_json_expr(h_expr, _expr_type(g_expr))


def sympy_solve_rolling_updater(
    f_expr: sp.Expr,
    g_expr: sp.Expr,
    r_to_rp: Mapping[Symbol, Symbol],
    c_vars: list[Symbol],
    t: Symbol,
) -> sp.Expr:
    """Solve for H(t, R, R') such that reduce(f, g(R', C)) can repair reduce(f, g(R, C)).

    This implements the paper's Eq. 5 directly:
      1. Invert t = g(R, C) with respect to C.
      2. Calculate and simplify g(R', g_c^{-1}(R, t)).
      3. Prove that the resulting H distributes over f.

    C can represent multiple variables. In that case, our algorithm inverts g over one each `c in C`,
    finding partial inverses. E.g. for `t = r + c0 + c1`, it can find `c0 = t - r - c1`.
    This may still work, because when we use this solution, it may also cancel c1 out.
    In our example, replacing c0 in `g(R', C) = r' + c0 + c1` gives
    `t - r + r'`, which does not contain any C variables.

    If raw-C inversion cannot produce a useful repair, we also try the paper's variable-substitution
    trick. We discover large subexpressions that depend only on C, such as `Max(0, c)**2` in
    `exp(r - Max(0, c)**2)`, replace one with a fresh `c'`, and solve the simpler
    `t = g(R, c')` problem.
    """
    if not c_vars:
        raise ValueError(f"Cannot invert {g_expr} with respect to an empty C-variable set")

    cand_substs = list(_partial_inverse_by_substitution(g_expr, c_vars, t))
    c_var_set = set(c_vars)
    valid_candidates: set[sp.Expr] = set()
    for c_subst in cand_substs:
        h_expr = cast(Expr, g_expr.subs(r_to_rp).xreplace(c_subst))  # type: ignore
        h_expr = sp.simplify(h_expr)
        if c_var_set.intersection(h_expr.free_symbols):
            continue
        if _prove_repair_term_distribute(h_expr, f_expr, t):
            valid_candidates.add(h_expr)
    if not valid_candidates:
        raise ValueError(f"Cannot find a valid solution for the rolling-update updater ({g_expr=})")
    if len(valid_candidates) > 1:
        raise ValueError(
            f"Multiple valid solutions found for the rolling-update updater ({g_expr=}): {valid_candidates}"
        )
    return valid_candidates.pop()


def _prove_repair_term_distribute(h_expr: Expr, f_expr: Expr, acc: Symbol) -> bool:
    f_free_vars = sorted(_free_vars_symbols(f_expr), key=lambda sym: sym.name)
    other_vars = [sym for sym in f_free_vars if sym != acc]
    if len(other_vars) != 1:
        raise ValueError(
            "Expected `f_expr` to have exactly two free variables, one being the `acc` variable provided; "
            f"got {f_expr=} ({acc=}), free vars: {f_free_vars})"
        )
    reduce_var = other_vars[0]
    y1 = sp.Symbol("y1", real=True)
    y2 = sp.Symbol("y2", real=True)
    reducer_expr = sp.simplify(f_expr.subs({acc: y1, reduce_var: y2}))

    def h_subst(acc_value: Expr) -> Expr:
        return sp.simplify(h_expr.subs({acc: acc_value}))

    lhs = sp.simplify(h_subst(reducer_expr))
    rhs = sp.simplify(f_expr.subs({acc: h_subst(y1), reduce_var: h_subst(y2)}))
    return _prove_eq(lhs, rhs)


def _partial_inverse_by_substitution(
    g_expr: Expr, c_vars: list[Symbol], t: Symbol
) -> Iterable[Subs]:
    """Invert `t = g(R, C)` by finding change-of-variable candidates: each candidate is a
    subexpression `e(C)` that depends only on C variables, which we can replace with a fresh variable `c'`.

    `c'` is not required to capture all C-dependence in `g`, so this substitution produces a
    `t = g'(R, c', C)`, but it can make the inversion problem easier, because `g'` is now a simpler
    expression with fewer c-terms.
    """
    for c_expr in _candidate_c_subexpressions(g_expr, c_vars):
        synthetic_c = sp.Dummy("c_sub", real=True)
        substituted_g = cast(Expr, g_expr.xreplace({c_expr: synthetic_c}))
        try:
            solutions = sp.solve(sp.Eq(substituted_g, t), synthetic_c)
        except NotImplementedError:
            continue
        for solution in solutions:
            yield {c_expr: solution}


def _candidate_c_subexpressions(g_expr: Expr, c_vars: list[Symbol]) -> list[Expr]:
    c_var_set = set(c_vars)
    candidates: list[Expr] = []
    seen: set[Expr] = set()
    for subexpr in sp.preorder_traversal(g_expr):
        assert isinstance(subexpr, Expr)
        if subexpr in seen:
            continue
        seen.add(subexpr)
        free_vars = _free_vars_symbols(subexpr)
        if not free_vars or not free_vars <= c_var_set:
            continue
        candidates.append(subexpr)

    def size(expr: Expr) -> tuple[int, int, int]:
        return (len(_free_vars_symbols(expr)), sp.count_ops(expr), len(str(expr)))

    return sorted(candidates, key=size, reverse=True)


def _is_homogeneous_scaling_in_t(expr: Expr, t: Symbol) -> bool:
    return t in expr.free_symbols and t not in sp.simplify(expr / t).free_symbols


def _prove_eq(lhs: sp.Expr, rhs: sp.Expr) -> bool:
    diff = sp.simplify(cast(Expr, lhs - rhs))
    if diff == 0 or diff.is_zero is True:
        return True
    if diff.is_zero is False:
        return False
    return False


def _expr_type(expr: JsonExpr) -> str:
    type_value = expr.get("type")
    if not isinstance(type_value, str):
        raise ValueError(f"expected expression node to have string type: {expr}")
    return type_value


def _expr_args(expr: JsonExpr, expected: int | None = None) -> list[JsonExpr]:
    args = expr.get("args")
    if not isinstance(args, list):
        raise ValueError(f"expected expression node to have list args: {expr}")
    if expected is not None and len(args) != expected:
        raise ValueError(f"expected {expected} args for {expr.get('op')}, got {len(args)}: {expr}")
    return [cast(JsonExpr, arg) for arg in args]


def _json_expr_to_sympy(expr: JsonExpr, symtab: dict[str, Symbol]) -> Expr:
    op = expr.get("op")
    if not isinstance(op, str):
        raise ValueError(f"expected expression node to have string op: {expr}")

    if op == "var":
        name = expr.get("name")
        if not isinstance(name, str):
            raise ValueError(f"expected var node to have string name: {expr}")
        return symtab.setdefault(name, sp.Symbol(name, real=True))
    if op == "const":
        value = expr.get("value")
        if isinstance(value, int):
            return cast(Expr, sp.Integer(value))
        if isinstance(value, str):
            return cast(Expr, sp.sympify(value))
        if isinstance(value, float):
            return cast(Expr, sp.Float(value))
        raise ValueError(f"unsupported const value: {expr}")

    if op in {"add", "sub", "mul", "div"}:
        lhs_expr, rhs_expr = _expr_args(expr, 2)
        lhs = _json_expr_to_sympy(lhs_expr, symtab)
        rhs = _json_expr_to_sympy(rhs_expr, symtab)
        if op == "add":
            return cast(Expr, lhs + rhs)
        if op == "sub":
            return cast(Expr, lhs - rhs)
        if op == "mul":
            return cast(Expr, lhs * rhs)
        return cast(Expr, lhs / rhs)

    if op in {"exp", "exp2", "log", "sqrt", "rsqrt", "abs"}:
        (arg_expr,) = _expr_args(expr, 1)
        arg = _json_expr_to_sympy(arg_expr, symtab)
        if op == "exp":
            return cast(Expr, sp.exp(arg))
        if op == "exp2":
            return cast(Expr, 2**arg)
        if op == "log":
            return cast(Expr, sp.log(arg))
        if op == "sqrt":
            return cast(Expr, sp.sqrt(arg))
        if op == "rsqrt":
            return cast(Expr, 1 / sp.sqrt(arg))
        return cast(Expr, sp.Abs(arg))

    if op == "pow":
        base_expr, exponent_expr = _expr_args(expr, 2)
        base = _json_expr_to_sympy(base_expr, symtab)
        exponent = _json_expr_to_sympy(exponent_expr, symtab)
        return cast(Expr, sp.Pow(base, exponent))
    if op == "max":
        args = [_json_expr_to_sympy(arg, symtab) for arg in _expr_args(expr)]
        return cast(Expr, sp.Max(*args))

    raise ValueError(f"unsupported JSON expression op: {op}")


def _sympy_to_json_expr(expr: sp.Expr, dtype: str) -> JsonExpr:
    expr = sp.simplify(expr)
    if isinstance(expr, Symbol):
        return {"op": "var", "name": expr.name, "type": dtype}
    if isinstance(expr, sp.Integer):
        return {"op": "const", "value": int(expr), "type": dtype}
    if isinstance(expr, sp.Rational):
        return {"op": "const", "value": str(expr), "type": dtype}
    if isinstance(expr, sp.Float):
        return {"op": "const", "value": str(expr), "type": dtype}

    if isinstance(expr, sp.Add):
        args = list(expr.args)
        result = _sympy_to_json_expr(args[0], dtype)
        for arg in args[1:]:
            is_negative, positive_arg = _split_negative_term(arg)
            result = {
                "op": "sub" if is_negative else "add",
                "type": dtype,
                "args": [result, _sympy_to_json_expr(positive_arg, dtype)],
            }
        return result

    if isinstance(expr, sp.Mul):
        coeff, factors = expr.as_coeff_mul()
        factor_exprs = list(factors)
        if coeff != 1:
            factor_exprs.insert(0, coeff)
        result = _sympy_to_json_expr(cast(Expr, factor_exprs[0]), dtype)
        for factor in factor_exprs[1:]:
            result = {
                "op": "mul",
                "type": dtype,
                "args": [result, _sympy_to_json_expr(cast(Expr, factor), dtype)],
            }
        return result

    if isinstance(expr, sp.Pow):
        base, exponent = expr.args
        if base == 2:
            return {
                "op": "exp2",
                "type": dtype,
                "args": [_sympy_to_json_expr(exponent, dtype)],
            }
        if exponent == sp.Rational(1, 2):
            return {
                "op": "sqrt",
                "type": dtype,
                "args": [_sympy_to_json_expr(base, dtype)],
            }
        if exponent == -sp.Rational(1, 2):
            return {
                "op": "rsqrt",
                "type": dtype,
                "args": [_sympy_to_json_expr(base, dtype)],
            }
        if exponent == -1:
            return {
                "op": "div",
                "type": dtype,
                "args": [
                    {"op": "const", "value": 1, "type": dtype},
                    _sympy_to_json_expr(base, dtype),
                ],
            }
        return {
            "op": "pow",
            "type": dtype,
            "args": [
                _sympy_to_json_expr(base, dtype),
                _sympy_to_json_expr(exponent, dtype),
            ],
        }

    if expr.func == sp.exp:
        return {
            "op": "exp",
            "type": dtype,
            "args": [_sympy_to_json_expr(cast(Expr, expr.args[0]), dtype)],
        }
    if expr.func == sp.log:
        return {
            "op": "log",
            "type": dtype,
            "args": [_sympy_to_json_expr(cast(Expr, expr.args[0]), dtype)],
        }
    if expr.func == sp.Abs:
        return {
            "op": "abs",
            "type": dtype,
            "args": [_sympy_to_json_expr(cast(Expr, expr.args[0]), dtype)],
        }
    if isinstance(expr, sp.Max):
        return {
            "op": "max",
            "type": dtype,
            "args": [_sympy_to_json_expr(cast(Expr, arg), dtype) for arg in expr.args],
        }

    raise ValueError(f"unsupported SymPy expression in JSON conversion: {expr} ({type(expr)})")


def _split_negative_term(expr: Expr) -> tuple[bool, Expr]:
    coeff, terms = expr.as_coeff_mul()
    if coeff == -1:
        return True, sp.Mul(*terms)
    if coeff < 0:
        return True, sp.Mul(-coeff, *terms)
    return False, expr
