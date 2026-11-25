"""Tests for unit conversion operations."""
import pytest

from pycalc_pro.core.unit_ops import UnitOperations


@pytest.fixture
def units():
    return UnitOperations()


def test_convert_length(units):
    # 1 meter ~ 3.28084 feet
    result = units.convert(1.0, "meter", "foot")
    assert result == pytest.approx(3.28084, rel=1e-4)


def test_convert_temperature_c_to_f(units):
    assert units.convert(0.0, "celsius", "fahrenheit") == 32.0
    assert units.convert(100.0, "celsius", "fahrenheit") == 212.0


def test_convert_temperature_c_to_k(units):
    assert units.convert(0.0, "celsius", "kelvin") == 273.15


def test_convert_unknown_unit_is_error(units):
    result = units.convert(1.0, "m", "nonsense")
    assert isinstance(result, str) and result.startswith("Error:")


def test_convert_incompatible_units_is_error(units):
    result = units.convert(1.0, "second", "meter")
    assert isinstance(result, str) and result.startswith("Error:")


def test_batch_convert(units):
    out = units.batch_convert([1.0, 2.0], "meter", "foot")
    assert out[0] == pytest.approx(3.28084, rel=1e-4)
    assert out[1] == pytest.approx(6.56168, rel=1e-4)


def test_get_unit_conversions_has_categories(units):
    cats = units.get_unit_conversions()
    assert "length" in cats
    assert "temperature" in cats


def test_validate_unit(units):
    assert units.validate_unit("meter") is True
    assert units.validate_unit("not_a_unit") is False
