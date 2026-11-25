"""Tests for the calculator engine."""
from pycalc_pro.core.calculator import CalculatorEngine


def test_add_subtract_multiply_divide():
    calc = CalculatorEngine()
    assert calc.add([1, 2, 3]) == 6.0
    assert calc.subtract([10, 4, 2]) == 4.0
    assert calc.multiply([2, 3, 4]) == 24.0
    assert calc.divide([100, 2, 5]) == 10.0


def test_divide_by_zero_is_error():
    calc = CalculatorEngine()
    assert calc.divide([10, 0]).startswith("Error:")


def test_add_need_at_least_two():
    calc = CalculatorEngine()
    assert calc.add([5]).startswith("Error:")


def test_vector_operations():
    calc = CalculatorEngine()
    a = calc.add_vector([1, 2, 3], [4, 5, 6])
    assert a.tolist() == [5.0, 7.0, 9.0]
    m = calc.multiply_vector([1, 2, 3], [2, 2, 2])
    assert m.tolist() == [2.0, 4.0, 6.0]


def test_batch_operations():
    calc = CalculatorEngine()
    res = calc.batch_operations("add", [[1, 2], [3, 4]])
    assert res.tolist() == [3.0, 7.0]


def test_memory_roundtrip():
    calc = CalculatorEngine()
    calc.store_memory(42.0)
    assert calc.recall_memory() == 42.0
    calc.clear_memory()
    assert calc.recall_memory() == 0.0
