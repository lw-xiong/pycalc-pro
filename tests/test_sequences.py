"""Tests for mathematical sequence operations."""
import pytest

from pycalc_pro.core.sequences import SequenceOperations
from pycalc_pro.utils.cache import MathCache


@pytest.fixture
def sequences():
    return SequenceOperations(MathCache())


def test_fibonacci(sequences):
    assert sequences.fibonacci(8) == [0, 1, 1, 2, 3, 5, 8, 13]


def test_fibonacci_zero(sequences):
    assert sequences.fibonacci(0) == []


def test_fibonacci_negative(sequences):
    assert sequences.fibonacci(-3) == []


def test_arithmetic_sequence(sequences):
    assert sequences.arithmetic_sequence(0, 2, 5) == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_geometric_sequence(sequences):
    assert sequences.geometric_sequence(1, 2, 4) == [1.0, 2.0, 4.0, 8.0]


def test_prime_sequence(sequences):
    assert sequences.prime_sequence(6) == [2, 3, 5, 7, 11, 13]


def test_is_prime(sequences):
    assert sequences.is_prime(7) is True
    assert sequences.is_prime(8) is False
    assert sequences.is_prime(2) is True
    assert sequences.is_prime(1) is False


def test_batch_is_prime(sequences):
    assert sequences.batch_is_prime([2, 3, 4, 5]) == [True, True, False, True]
