"""Tests for the physics calculations engine."""
import math

import pytest

from pycalc_pro.core.physics_ops import PhysicsOperations


C = 299_792_458.0


@pytest.fixture
def physics():
    return PhysicsOperations()


def test_kinetic_energy(physics):
    assert physics.kinetic_energy(10.0, 5.0) == 125.0


def test_kinetic_energy_negative_input_is_error(physics):
    assert physics.kinetic_energy(-1, 5).startswith("Error:")
    assert physics.kinetic_energy(1, -5).startswith("Error:")


def test_potential_energy(physics):
    assert physics.potential_energy(10.0, 5.0) == pytest.approx(490.3325)


def test_potential_energy_custom_gravity(physics):
    assert physics.potential_energy(1.0, 1.0, 1.0) == 1.0


def test_relativistic_gamma(physics):
    assert physics.relativistic_gamma(0.5 * C) == pytest.approx(1.154700538, rel=1e-6)


def test_relativistic_gamma_at_light_speed_is_error(physics):
    assert physics.relativistic_gamma(C).startswith("Error:")
    assert physics.relativistic_gamma(-1.0).startswith("Error:")


def test_time_dilation(physics):
    assert physics.time_dilation(1.0, 0.5 * C) > 1.0


def test_length_contraction(physics):
    assert physics.length_contraction(10.0, 0.5 * C) < 10.0


def test_schwarzschild_radius_positive(physics):
    assert physics.schwarzschild_radius(5.9722e24) > 0


def test_schwarzschild_radius_negative_is_error(physics):
    assert physics.schwarzschild_radius(-1.0).startswith("Error:")


def test_de_broglie_wavelength(physics):
    assert physics.de_broglie_wavelength(1.0) > 0


def test_de_broglie_wavelength_zero_is_error(physics):
    assert physics.de_broglie_wavelength(0).startswith("Error:")


def test_centripetal_force(physics):
    assert physics.centripetal_force(1.0, 10.0, 2.0) == 50.0


def test_ideal_gas_law_pressure(physics):
    # PV = nRT -> P = nRT/V, 1 mol at 273.15K in 22.4 L -> ~1 atm (101325 Pa)
    # V must be in m^3 for SI: 22.4 L = 0.0224 m^3 -> P ~ 101325 Pa
    result = physics.ideal_gas_law(volume=0.0224, temperature=273.15, moles=1.0)
    assert result == pytest.approx(101325.0, rel=0.01)


def test_ideal_gas_law_moles(physics):
    result = physics.ideal_gas_law(pressure=101325.0, volume=0.0224, temperature=273.15)
    assert result == pytest.approx(1.0, rel=0.01)


def test_ideal_gas_law_needs_three_inputs(physics):
    assert physics.ideal_gas_law(pressure=1.0, volume=1.0).startswith("Error:")


def test_projectile_range(physics):
    # v=50, angle=45, g=9.80665 -> R = 2500*1/9.80665 ~= 254.93
    assert physics.projectile_range(50.0, 45.0) == pytest.approx(254.93, rel=0.01)


def test_batch_kinetic_energy(physics):
    out = physics.batch_kinetic_energy([1.0, 2.0], [10.0, 20.0])
    assert out == [50.0, 400.0]


def test_batch_relativistic_gamma(physics):
    out = physics.batch_relativistic_gamma([0.0, 0.5 * C])
    assert out[0] == 1.0
    assert out[1] == pytest.approx(1.154700538, rel=1e-6)
