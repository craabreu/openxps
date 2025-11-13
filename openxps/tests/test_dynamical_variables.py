"""
Unit tests for the DynamicalVariable class.
"""

import cvpack
import pytest
import yaml
from openmm import unit as mmunit

from openxps.bounds import NoBounds, PeriodicBounds, ReflectiveBounds
from openxps.dynamical_variable import DynamicalVariable


def test_dv_initialization():
    """Test successful DynamicalVariable initialization."""
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    bounds = PeriodicBounds(-180, 180, mmunit.degree)
    dv = DynamicalVariable("phi", mmunit.radian, mass, bounds)

    assert dv.name == "phi"
    assert dv.unit == mmunit.radian
    assert dv.mass.unit == mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    assert isinstance(dv.bounds, PeriodicBounds)


def test_dv_invalid_unit():
    """Test DynamicalVariable initialization with an invalid unit."""
    with pytest.raises(ValueError) as excinfo:
        DynamicalVariable(
            "phi",
            "not_a_unit",
            3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2,
            NoBounds(0, 1, mmunit.dimensionless),
        )
    assert "The unit must be a valid OpenMM unit." in str(excinfo.value)


def test_dv_mass_without_unit():
    """Test DynamicalVariable initialization with mass missing a unit."""
    with pytest.raises(TypeError) as excinfo:
        DynamicalVariable("phi", mmunit.radian, 3, NoBounds(0, 1, mmunit.dimensionless))
    assert "Mass must be have units of measurement." in str(excinfo.value)


def test_dv_incompatible_mass_unit():
    """Test DynamicalVariable initialization with mass having incompatible units."""
    with pytest.raises(TypeError):
        DynamicalVariable(
            "phi", mmunit.radian, 3 * mmunit.meter, NoBounds(0, 1, mmunit.dimensionless)
        )


def test_dv_bounds_incompatible_units():
    """Test DynamicalVariable initialization with bounds having incompatible units."""
    bounds = ReflectiveBounds(0, 10, mmunit.meter)  # Incompatible with radian
    with pytest.raises(ValueError):
        DynamicalVariable(
            "phi",
            mmunit.radian,
            3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2,
            bounds,
        )


def test_dv_serialization():
    """Test YAML serialization and deserialization of DynamicalVariable."""
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    bounds = PeriodicBounds(-180, 180, mmunit.degree)
    dv = DynamicalVariable("phi", mmunit.radian, mass, bounds)

    serialized = yaml.safe_dump(dv)
    deserialized = yaml.safe_load(serialized)

    assert deserialized == dv


def test_dv_without_bounds():
    """Test DynamicalVariable initialization without bounds."""
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    with pytest.raises(TypeError) as excinfo:
        DynamicalVariable("phi", mmunit.radian, mass, None)
    assert "The bounds must be an instance of Bounds." in str(excinfo.value)


def test_dv_bounds_type_error():
    """Test DynamicalVariable initialization with incorrect bounds type."""
    with pytest.raises(TypeError):
        DynamicalVariable(
            "phi",
            mmunit.radian,
            3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2,
            "not_bounds",
        )


def test_dv_distance_method():
    """
    Test the distanceTo method of DynamicalVariable.
    """
    psi0 = DynamicalVariable(
        "psi0",
        mmunit.radian,
        3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2,
        PeriodicBounds(-180, 180, mmunit.degree),
    )
    phi0 = DynamicalVariable(
        "phi0",
        mmunit.radian,
        3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2,
        PeriodicBounds(-180, 180, mmunit.degree),
    )

    assert psi0.distanceTo(cvpack.Torsion(6, 8, 14, 16, name="psi")) == (
        "(psi-psi0-6.283185307179586*floor(0.5+(psi-psi0)/6.283185307179586))"
    )

    assert (
        psi0.distanceTo(phi0)
        == "(phi0-psi0-6.283185307179586*floor(0.5+(phi0-psi0)/6.283185307179586))"
    )

    with pytest.raises(TypeError) as excinfo:
        psi0.distanceTo("not_cv")
    assert "Method distanceTo not implemented for type" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        psi0.distanceTo(cvpack.Distance(0, 1))
    assert "Incompatible boundary conditions." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        psi0.distanceTo(
            DynamicalVariable(
                "x",
                mmunit.nanometer,
                3 * mmunit.dalton,
                NoBounds(0, 1, mmunit.dimensionless),
            )
        )
    assert "Incompatible boundary conditions." in str(excinfo.value)

    distance0 = DynamicalVariable(
        "distance0",
        mmunit.nanometer,
        3 * mmunit.dalton,
        ReflectiveBounds(0, 1, mmunit.nanometer),
    )
    assert distance0.distanceTo(cvpack.Distance(0, 1)) == "(distance-distance0)"

    distance1 = DynamicalVariable(
        "distance1",
        mmunit.nanometer,
        3 * mmunit.dalton,
        NoBounds(0, 1, mmunit.dimensionless),
    )
    assert distance1.distanceTo(distance0) == "(distance0-distance1)"
