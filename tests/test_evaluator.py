"""Tests for the safe AST expression evaluator."""
from pycalc_pro.core.calculator import CalculatorEngine
from pycalc_pro.utils.evaluator import SafeEvaluator


def _engine():
    return CalculatorEngine()


def test_evaluate_addition():
    ev = SafeEvaluator(_engine())
    assert ev.safe_eval("2 + 3") == 5


def test_evaluate_precedence():
    ev = SafeEvaluator(_engine())
    assert ev.safe_eval("2 + 3 * 4") == 14


def test_evaluate_parentheses():
    ev = SafeEvaluator(_engine())
    assert ev.safe_eval("(2 + 3) * 4") == 20


def test_evaluate_power():
    ev = SafeEvaluator(_engine())
    assert ev.safe_eval("2 ** 10") == 1024


def test_evaluate_functions_degrees():
    ev = SafeEvaluator(_engine())
    assert ev.safe_eval("sin(90)") == 1.0


def test_evaluate_constant_ans():
    engine = _engine()
    engine.set_last_result(10)
    ev = SafeEvaluator(engine)
    assert ev.safe_eval("ans + 5") == 15


def test_evaluate_unknown_variable_is_error():
    ev = SafeEvaluator(_engine())
    result = ev.safe_eval("foo + 1")
    assert isinstance(result, str) and result.startswith("Error:")
