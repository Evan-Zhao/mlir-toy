import pytest
import sympy as sp

from neptune_mlir.rolling_solver import sympy_solve_rolling_updater

T = sp.Symbol("t", real=True)
FADD = T + sp.Symbol("x", real=True)


def _assert_equiv(actual: sp.Expr, expected: sp.Expr) -> None:
    assert sp.simplify(actual - expected) == 0


def test_attention_row_sum_repair() -> None:
    score, row_max = sp.symbols("score row_max", real=True)
    row_max_next = sp.Symbol("row_max'", real=True)

    h_expr = sympy_solve_rolling_updater(
        FADD, sp.exp(score - row_max), {row_max: row_max_next}, [score], T
    )
    _assert_equiv(h_expr, T * sp.exp(row_max - row_max_next))


def test_attention_output_repair_from_row_sum() -> None:
    prob_num, value, row_sum = sp.symbols("prob_num value row_sum", real=True)
    row_sum_next = sp.Symbol("row_sum'", real=True)

    h_expr = sympy_solve_rolling_updater(
        FADD, prob_num * value / row_sum, {row_sum: row_sum_next}, [prob_num, value], T
    )
    _assert_equiv(h_expr, T * row_sum / row_sum_next)


def test_attention_output_repair_from_row_max_and_row_sum() -> None:
    score, value = sp.symbols("score value", real=True)
    row_max, row_sum = sp.symbols("row_max row_sum", real=True)
    row_max_next = sp.Symbol("row_max'", real=True)
    row_sum_next = sp.Symbol("row_sum'", real=True)

    h_expr = sympy_solve_rolling_updater(
        FADD,
        sp.exp(score - row_max) * value / row_sum,
        {row_max: row_max_next, row_sum: row_sum_next},
        [score, value],
        T,
    )
    _assert_equiv(h_expr, T * sp.exp(row_max - row_max_next) * row_sum / row_sum_next)


def test_paper_exponential_repair_term() -> None:
    m, x = sp.symbols("m x", real=True)
    m_next = sp.Symbol("m'", real=True)

    h_expr = sympy_solve_rolling_updater(FADD, sp.exp(x - m), {m: m_next}, [x], T)
    _assert_equiv(h_expr, T * sp.exp(m - m_next))


def test_paper_noninvertible_raw_c_example_uses_variable_substitution() -> None:
    r, c = sp.symbols("r c", real=True)
    r_next = sp.Symbol("r'", real=True)

    h_expr = sympy_solve_rolling_updater(FADD, sp.exp(r - sp.Max(c, 0) ** 2), {r: r_next}, [c], T)
    _assert_equiv(h_expr, T * sp.exp(-r + r_next))


def test_variable_substitution_succeeds_when_raw_c_solve_fails() -> None:
    r, c = sp.symbols("r c", real=True)
    r_next = sp.Symbol("r'", real=True)

    h_expr = sympy_solve_rolling_updater(FADD, sp.exp(r - sp.floor(c)), {r: r_next}, [c], T)
    _assert_equiv(h_expr, T * sp.exp(-r + r_next))


@pytest.mark.parametrize(
    ("g_expr", "expected"),
    [
        (lambda r, c0, c1: r * c0 * c1, lambda t, r, rp: t * rp / r),
        (lambda r, c0, c1: sp.exp(c0 + c1 - r), lambda t, r, rp: t * sp.exp(r - rp)),
    ],
)
def test_multi_c_inverses_that_cancel_and_distribute(g_expr, expected) -> None:
    r = sp.Symbol("r", real=True)
    r_next = sp.Symbol("r'", real=True)
    c0, c1 = sp.symbols("c0 c1", real=True)

    h_expr = sympy_solve_rolling_updater(FADD, g_expr(r, c0, c1), {r: r_next}, [c0, c1], T)
    _assert_equiv(h_expr, expected(T, r, r_next))


def test_rejects_empty_c_variable_set() -> None:
    r = sp.Symbol("r", real=True)
    r_next = sp.Symbol("r'", real=True)

    with pytest.raises(ValueError, match="empty C-variable set"):
        sympy_solve_rolling_updater(FADD, r, {r: r_next}, [], T)


def test_multi_c_non_distributive() -> None:
    r = sp.Symbol("r", real=True)
    r_next = sp.Symbol("r'", real=True)
    c0, c1 = sp.symbols("c0 c1", real=True)

    with pytest.raises(ValueError, match="Cannot find a valid solution"):
        sympy_solve_rolling_updater(FADD, r + c0 + c1, {r: r_next}, [c0, c1], T)


def test_rejects_noninvertible_max() -> None:
    r, c = sp.symbols("r c", real=True)
    r_next = sp.Symbol("r'", real=True)

    with pytest.raises(ValueError, match="Cannot find a valid solution"):
        sympy_solve_rolling_updater(FADD, sp.Max(r, c), {r: r_next}, [c], T)


@pytest.mark.parametrize(
    "g_expr",
    [
        lambda r, c0, c1: r * c0 + c1,
        lambda r, c0, c1: c0 * c1 + r * c0,
        lambda r, c0, c1: r * c0 + c1**2,
    ],
)
def test_fail_to_cancel_all_c_variables(g_expr) -> None:
    r = sp.Symbol("r", real=True)
    r_next = sp.Symbol("r'", real=True)
    c0, c1 = sp.symbols("c0 c1", real=True)

    with pytest.raises(ValueError, match="Cannot find a valid solution"):
        sympy_solve_rolling_updater(FADD, g_expr(r, c0, c1), {r: r_next}, [c0, c1], T)
