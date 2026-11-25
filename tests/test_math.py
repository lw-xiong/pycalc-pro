"""Tests for the mathematical operations engine."""
import math

import pytest

from pycalc_pro.core.math_ops import MathOperations
from pycalc_pro.utils.cache import MathCache


@pytest.fixture
def math_engine():
    return MathOperations(MathCache())


def test_power_integer(math_engine):
    assert math_engine.power(2, 10) == 1024.0


def test_power_fractional_exponent(math_engine):
    assert math_engine.power(4, 0.5) == 2.0


def test_power_negative_base_fractional_exponent_is_error(math_engine):
    result = math_engine.power(-2, 0.5)
    assert isinstance(result, str) and result.startswith("Error:")


def test_power_zero_exponent(math_engine):
    assert math_engine.power(7, 0) == 1.0


def test_sqrt_positive(math_engine):
    assert math_engine.sqrt(16) == 4.0


def test_sqrt_zero(math_engine):
    assert math_engine.sqrt(0) == 0.0


def test_sqrt_negative_is_error(math_engine):
    result = math_engine.sqrt(-4)
    assert isinstance(result, str) and result.startswith("Error:")


def test_exp(math_engine):
    assert math_engine.exp(1) == pytest.approx(math.e)


def test_logarithm(math_engine):
    assert math_engine.logarithm(100, 10) == 2.0


def test_logarithm_invalid_base_is_error(math_engine):
    result = math_engine.logarithm(100, 1)
    assert isinstance(result, str) and result.startswith("Error:")


def test_factorial(math_engine):
    assert math_engine.factorial(5) == 120


def test_factorial_zero_and_one(math_engine):
    assert math_engine.factorial(0) == 1
    assert math_engine.factorial(1) == 1


def test_factorial_negative_is_error(math_engine):
    result = math_engine.factorial(-3)
    assert isinstance(result, str) and result.startswith("Error:")


def test_nth_root(math_engine):
    assert math_engine.nth_root(27, 3) == pytest.approx(3.0)


def test_nth_root_zero_root_is_error(math_engine):
    result = math_engine.nth_root(8, 0)
    assert isinstance(result, str) and result.startswith("Error:")


def test_trig_degrees(math_engine):
    assert math_engine.sin(90) == pytest.approx(1.0)
    assert math_engine.cos(60) == pytest.approx(0.5)
    assert math_engine.tan(45) == pytest.approx(1.0, rel=1e-9)


def test_mod(math_engine):
    assert math_engine.mod(10, 3) == 1.0


def test_mod_division_by_zero_is_error(math_engine):
    result = math_engine.mod(10, 0)
    assert isinstance(result, str) and result.startswith("Error:")


def test_abs(math_engine):
    assert math_engine.abs(-7) == 7.0


def test_batch_power(math_engine):
    assert math_engine.batch_power([2, 3, 4], [2, 3, 2]) == [4.0, 27.0, 16.0]


def test_batch_sqrt(math_engine):
    assert math_engine.batch_sqrt([4, 9, 16]) == [2.0, 3.0, 4.0]


def test_invalid_input_type_is_error(math_engine):
    result = math_engine.power("a", 2)
    assert isinstance(result, str) and result.startswith("Error:")
