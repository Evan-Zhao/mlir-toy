from collections import Counter, defaultdict
from typing import Any, Mapping, cast

import sympy as sp
from sympy import Expr, Symbol

JsonExpr = dict[str, Any]


def solve_rolling_updater(g_expr: str, old_var: str, new_var: str, acc_var: str) -> str:
    """Derive a rolling-update repair expression.

    The interface is intentionally string-based so the MLIR side can serialize a
    scalar expression without exposing MLIR objects to Python. The current public
    wrapper handles the common one-old-var API, while the implementation below
    mirrors TVM's more general SymPy solver.
    """
    old = sp.Symbol(old_var, real=True)
    new = sp.Symbol(new_var, real=True)
    acc = sp.Symbol(acc_var, real=True, nonzero=True)

    locals_ = {
        old_var: old,
        new_var: new,
        acc_var: acc,
        "c": sp.Symbol("c", real=True),
        "exp": sp.exp,
        "exp2": lambda x: 2**x,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "rsqrt": lambda x: 1 / sp.sqrt(x),
        "pow": sp.Pow,
        "abs": sp.Abs,
        "Max": sp.Max,
        "max": sp.Max,
    }
    g = cast(Expr, sp.sympify(g_expr, locals=locals_))
    c_vars = sorted(g.free_symbols - {old}, key=lambda sym: sym.name)

    h_expr = sympy_solve_rolling_updater(g, {old: new}, c_vars, acc)
    c_variables = set(c_vars).intersection(h_expr.free_symbols)
    if c_variables:
        raise ValueError(
            "Cannot find a valid solution for the rolling-update updater function "
            f"`H`: attempt to solve produced expression {h_expr} with remaining "
            f"`c`-variables: {c_variables}"
        )
    return str(h_expr)


def solve_rolling_updater_json(
    g_expr: JsonExpr,
    old_var: str,
    new_var: str,
    acc_var: str,
) -> JsonExpr:
    """String-free typed expression interface for the MLIR integration path."""
    symtab: dict[str, Symbol] = {}
    g = _json_expr_to_sympy(g_expr, symtab)
    old = symtab.setdefault(old_var, sp.Symbol(old_var, real=True))
    new = symtab.setdefault(new_var, sp.Symbol(new_var, real=True))
    acc = symtab.setdefault(acc_var, sp.Symbol(acc_var, real=True, nonzero=True))
    c_vars = sorted(g.free_symbols - {old}, key=lambda sym: sym.name)

    h_expr = sympy_solve_rolling_updater(g, {old: new}, c_vars, acc)
    c_variables = set(c_vars).intersection(h_expr.free_symbols)
    if c_variables:
        raise ValueError(
            "Cannot find a valid solution for the rolling-update updater function "
            f"`H`: attempt to solve produced expression {h_expr} with remaining "
            f"`c`-variables: {c_variables}"
        )
    return _sympy_to_json_expr(h_expr, _expr_type(g_expr))


def sympy_solve_rolling_updater(
    g_expr: sp.Expr, r_to_rp: Mapping[Symbol, Symbol], c_vars: list[Symbol], t: Symbol
) -> sp.Expr:
    """Solve for H(t, R, R') such that reduce(g(R', C)) can repair reduce(g(R, C)).

    This is the TVM algorithm with the TIR conversion layer removed:
      1. Separate g(R, C) into gcomb(g1(R), g2(C)).
      2. Invert t = gcomb(r, c) with respect to c.
      3. Return gcomb(g1(R'), gcomb^{-1}(g1(R), t)).
    """
    r, c = Symbol("r"), Symbol("c")
    r_vars = list(r_to_rp.keys())
    g1, _, gcomb = separate_vars(g_expr, r_vars, c_vars, r, c)

    gcomb_inv = invert_binary_function(gcomb, c, t)
    term1 = g1.subs(r_to_rp)
    term2 = gcomb_inv.subs({r: g1})
    h_expr = cast(Expr, gcomb.subs({r: term1, c: term2}))

    h_expr = sp.simplify(_simplify_abs_sqrt(h_expr))
    return _prefer_quotient_power_form(h_expr)


def prove(lhs: str, rhs: str, cmp: str = "eq") -> bool:
    """Conservatively prove a relation between two string expressions."""
    locals_ = _default_sympify_locals(lhs, rhs)
    lhs_expr = _simplify_for_proof(cast(Expr, sp.sympify(lhs, locals=locals_)))
    rhs_expr = _simplify_for_proof(cast(Expr, sp.sympify(rhs, locals=locals_)))
    return _prove_expr(lhs_expr, rhs_expr, cmp)


def separate_vars(
    f: Expr, xs: list[Symbol], us: list[Symbol], s: Symbol, t: Symbol
) -> tuple[Expr, Expr, Expr]:
    """Separate f(xs, us) as fcomb(fx(xs), fu(us)).

    SymPy's separatevars handles the useful multiplicative cases. As in TVM, if
    separation fails for a single x and a single u, fall back to the direct
    binary combiner fcomb(s, t) = f(s, t).
    """
    d = sp.separatevars(f, symbols=xs + us, dict=True, force=True)
    if d is None:
        if len(xs) == 1 and len(us) == 1:
            return xs[0], us[0], f.subs({xs[0]: s, us[0]: t})
        raise ValueError(f"Failed to separate variables for {f} over {xs} and {us}")

    d = cast(dict[Symbol | str, sp.Expr], d)
    fx = sp.simplify(sp.Mul(*[d.get(v, 1) for v in xs]))
    fu = sp.simplify(sp.Mul(*[d.get(v, 1) for v in us]))
    coeff = d.get("coeff", 1)
    fcomb = cast(Expr, coeff * s * t)
    return fx, fu, fcomb


def invert_binary_function(expr: sp.Expr, y: Symbol, z: Symbol) -> sp.Expr:
    """Find y = f^{-1}(z, ...) for z = expr."""
    solutions = sp.solve(sp.Eq(expr, z), y)
    if not solutions:
        raise ValueError(f"No solution found for {expr} == {z} with respect to {y}")
    return cast(Expr, solutions[0])


def _prove_expr(lhs: sp.Expr, rhs: sp.Expr, cmp: str = "eq") -> bool:
    cmp_builders = {
        "eq": sp.Eq,
        "ne": sp.Ne,
        "lt": sp.Lt,
        "le": sp.Le,
        "gt": sp.Gt,
        "ge": sp.Ge,
    }
    if cmp not in cmp_builders:
        raise ValueError(f"unsupported comparison: {cmp}")

    if cmp == "eq":
        diff = _simplify_for_proof(cast(Expr, lhs - rhs))
        if diff == 0 or diff.is_zero is True:
            return True
        if diff.is_zero is False:
            return False
        return False

    try:
        simplified = sp.simplify(cmp_builders[cmp](lhs, rhs))
    except ValueError:
        return False
    return simplified is sp.true


def _prefer_quotient_power_form(expr: sp.Expr) -> sp.Expr:
    """Rewrite a**k / b**k-style products into (a / b)**k."""

    def _rewrite_mul(mul_expr: sp.Expr) -> sp.Expr:
        if not isinstance(mul_expr, sp.Mul):
            return mul_expr

        num_by_exp: dict[sp.Rational, list[sp.Expr]] = defaultdict(list)
        den_by_exp: dict[sp.Rational, list[sp.Expr]] = defaultdict(list)
        other_factors: list[sp.Expr] = []

        for factor in sp.Mul.make_args(mul_expr):
            if isinstance(factor, sp.Pow) and isinstance(factor.exp, sp.Rational):
                exp = factor.exp
                if exp > 0 and exp != 1:
                    num_by_exp[exp].append(factor.base)
                    continue
                if exp < 0 and exp != -1:
                    den_by_exp[cast(sp.Rational, -exp)].append(factor.base)
                    continue
            other_factors.append(factor)

        changed = False
        for exp in sorted(set(num_by_exp) & set(den_by_exp), key=float):
            num = sp.Mul(*num_by_exp.pop(exp), evaluate=False)
            den = sp.Mul(*den_by_exp.pop(exp), evaluate=False)
            ratio = sp.Mul(num, sp.Pow(den, -1, evaluate=False), evaluate=False)
            other_factors.append(cast(Expr, sp.Pow(ratio, exp, evaluate=False)))
            changed = True

        for exp in sorted(num_by_exp, key=float):
            other_factors.extend(
                [cast(Expr, sp.Pow(base, exp, evaluate=False)) for base in num_by_exp[exp]]
            )
        for exp in sorted(den_by_exp, key=float):
            other_factors.extend(
                [cast(Expr, sp.Pow(base, -exp, evaluate=False)) for base in den_by_exp[exp]]
            )

        if not changed:
            return mul_expr
        return sp.Mul(*other_factors, evaluate=False)

    return cast(Expr, expr.replace(lambda e: isinstance(e, sp.Mul), _rewrite_mul))


def _simplify_abs_sqrt(expr: sp.Expr) -> sp.Expr:
    sqrt_exprs = {
        cast(Expr, subexpr)
        for subexpr in sp.preorder_traversal(expr)
        if isinstance(subexpr, sp.Pow) and subexpr.exp == sp.Rational(1, 2)
    }
    if not sqrt_exprs:
        return expr
    assumptions = [sp.Q.real(sqrt_expr) & sp.Q.nonnegative(sqrt_expr) for sqrt_expr in sqrt_exprs]
    return cast(Expr, sp.refine(expr, sp.And(*assumptions)))


def _is_nonnegative_by_construction(expr: Expr) -> bool:
    if expr.is_nonnegative is True or expr.is_positive is True:
        return True
    if isinstance(expr, sp.Number):
        return bool(expr >= 0)
    if isinstance(expr, sp.Pow):
        if expr.exp == sp.Rational(1, 2):
            return True
        if isinstance(expr.exp, sp.Rational) and expr.exp.q == 2:
            return True
        if expr.exp == -1:
            return _is_nonnegative_by_construction(expr.base)
    if isinstance(expr, sp.Mul):
        return all(_is_nonnegative_by_construction(factor) for factor in expr.args)
    return False


def _pull_common_nonnegative_max_factor(expr: sp.Expr) -> sp.Expr:
    def _rewrite_max(max_expr: sp.Expr) -> sp.Expr:
        if not isinstance(max_expr, sp.Max) or not max_expr.args:
            return max_expr

        counters = [Counter(sp.Mul.make_args(cast(Expr, arg))) for arg in max_expr.args]
        common = counters[0].copy()
        for counter in counters[1:]:
            for factor in list(common):
                common[factor] = min(common[factor], counter.get(factor, 0))
                if common[factor] == 0:
                    del common[factor]
        if not common:
            return max_expr

        pulled: list[Expr] = []
        for factor, count in common.items():
            if _is_nonnegative_by_construction(factor):
                pulled.extend([factor] * count)
        if not pulled:
            return max_expr

        factor = sp.Mul(*pulled)
        if factor == 1:
            return max_expr
        reduced_args = [sp.simplify(arg / factor) for arg in max_expr.args]
        return cast(Expr, sp.simplify(factor * sp.Max(*reduced_args)))

    return cast(Expr, expr.replace(lambda e: isinstance(e, sp.Max), _rewrite_max))


def _simplify_for_proof(expr: sp.Expr) -> sp.Expr:
    expr = _simplify_abs_sqrt(expr)
    expr = _pull_common_nonnegative_max_factor(expr)
    return sp.simplify(expr)


def _default_sympify_locals(*exprs: str) -> dict[str, object]:
    names = set()
    for expr in exprs:
        names.update(str(sym) for sym in sp.sympify(expr).free_symbols)
    locals_: dict[str, object] = {name: sp.Symbol(name, real=True) for name in names}
    locals_.update(
        {
            "exp": sp.exp,
            "exp2": lambda x: 2**x,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "rsqrt": lambda x: 1 / sp.sqrt(x),
            "pow": sp.Pow,
            "abs": sp.Abs,
            "Max": sp.Max,
            "max": sp.Max,
        }
    )
    return locals_


def _expr_type(expr: JsonExpr) -> str:
    type_value = expr.get("type")
    if not isinstance(type_value, str):
        raise ValueError(f"expected expression node to have string type: {expr}")
    return type_value


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
        lhs = _json_expr_to_sympy(cast(JsonExpr, expr["lhs"]), symtab)
        rhs = _json_expr_to_sympy(cast(JsonExpr, expr["rhs"]), symtab)
        if op == "add":
            return cast(Expr, lhs + rhs)
        if op == "sub":
            return cast(Expr, lhs - rhs)
        if op == "mul":
            return cast(Expr, lhs * rhs)
        return cast(Expr, lhs / rhs)

    if op in {"exp", "exp2", "log", "sqrt", "rsqrt", "abs"}:
        arg = _json_expr_to_sympy(cast(JsonExpr, expr["arg"]), symtab)
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
        base = _json_expr_to_sympy(cast(JsonExpr, expr["base"]), symtab)
        exponent = _json_expr_to_sympy(cast(JsonExpr, expr["exponent"]), symtab)
        return cast(Expr, sp.Pow(base, exponent))
    if op == "max":
        args = [_json_expr_to_sympy(cast(JsonExpr, arg), symtab) for arg in expr["args"]]
        return cast(Expr, sp.Max(*args))

    raise ValueError(f"unsupported JSON expression op: {op}")


def _sympy_to_json_expr(expr: sp.Expr, dtype: str) -> JsonExpr:
    expr = cast(Expr, sp.simplify(expr))
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
        result = _sympy_to_json_expr(cast(Expr, args[0]), dtype)
        for arg in args[1:]:
            is_negative, positive_arg = _split_negative_term(cast(Expr, arg))
            result = {
                "op": "sub" if is_negative else "add",
                "type": dtype,
                "lhs": result,
                "rhs": _sympy_to_json_expr(positive_arg, dtype),
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
                "lhs": result,
                "rhs": _sympy_to_json_expr(cast(Expr, factor), dtype),
            }
        return result

    if isinstance(expr, sp.Pow):
        base, exponent = expr.args
        if base == 2:
            return {"op": "exp2", "type": dtype, "arg": _sympy_to_json_expr(cast(Expr, exponent), dtype)}
        if exponent == sp.Rational(1, 2):
            return {"op": "sqrt", "type": dtype, "arg": _sympy_to_json_expr(cast(Expr, base), dtype)}
        if exponent == -sp.Rational(1, 2):
            return {"op": "rsqrt", "type": dtype, "arg": _sympy_to_json_expr(cast(Expr, base), dtype)}
        if exponent == -1:
            return {
                "op": "div",
                "type": dtype,
                "lhs": {"op": "const", "value": 1, "type": dtype},
                "rhs": _sympy_to_json_expr(cast(Expr, base), dtype),
            }
        return {
            "op": "pow",
            "type": dtype,
            "base": _sympy_to_json_expr(cast(Expr, base), dtype),
            "exponent": _sympy_to_json_expr(cast(Expr, exponent), dtype),
        }

    if expr.func == sp.exp:
        return {"op": "exp", "type": dtype, "arg": _sympy_to_json_expr(cast(Expr, expr.args[0]), dtype)}
    if expr.func == sp.log:
        return {"op": "log", "type": dtype, "arg": _sympy_to_json_expr(cast(Expr, expr.args[0]), dtype)}
    if expr.func == sp.Abs:
        return {"op": "abs", "type": dtype, "arg": _sympy_to_json_expr(cast(Expr, expr.args[0]), dtype)}
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
        return True, cast(Expr, sp.Mul(*terms))
    if coeff < 0:
        return True, cast(Expr, sp.Mul(-coeff, *terms))
    return False, expr
