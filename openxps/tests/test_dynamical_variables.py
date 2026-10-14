"""
Unit tests for the DynamicalVariable class.
"""

import cvpack
import pytest
import yaml
from openmm import unit as mmunit

from openxps.bounds import CircularBounds, NoBounds, PeriodicBounds, ReflectiveBounds
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


@pytest.mark.parametrize(
    "name",
    ["my-var", "1phi", "phi s", "f(x)", "lambda.vdw", "x'", "a<b", "a=b", "x²", "", 3],
)
def test_dv_invalid_name(name):
    """Names must be identifiers, as in Python."""
    with pytest.raises(ValueError, match="Invalid name"):
        DynamicalVariable(
            name,
            mmunit.radian,
            3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2,
            NoBounds(0, 1, mmunit.radian),
        )


@pytest.mark.parametrize("name", ["phi", "_x1", "λ", "λ_vdw", "θ1"])
def test_dv_valid_name(name):
    """Identifiers may contain non-ASCII letters."""
    dv = DynamicalVariable(
        name,
        mmunit.radian,
        3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2,
        NoBounds(0, 1, mmunit.radian),
    )
    assert yaml.safe_load(yaml.safe_dump(dv)).name == name


def test_dv_invalid_unit():
    """Test DynamicalVariable initialization with an invalid unit."""
    with pytest.raises(TypeError) as excinfo:
        DynamicalVariable(
            "phi",
            "not_a_unit",
            3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2,
            NoBounds(0, 1, mmunit.radian),
        )
    assert "The unit must be a valid OpenMM unit." in str(excinfo.value)


def test_dv_mass_without_unit():
    """Test DynamicalVariable initialization with mass missing a unit."""
    with pytest.raises(TypeError) as excinfo:
        DynamicalVariable("phi", mmunit.radian, 3, NoBounds(0, 1, mmunit.radian))
    assert "Mass must have units of measurement." in str(excinfo.value)


def test_dv_incompatible_mass_unit():
    """Test DynamicalVariable initialization with mass having incompatible units."""
    with pytest.raises(TypeError):
        DynamicalVariable(
            "phi", mmunit.radian, 3 * mmunit.meter, NoBounds(0, 1, mmunit.radian)
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
    assert "Incompatible units." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        psi0.distanceTo(
            DynamicalVariable(
                "x",
                mmunit.radian,
                3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2,
                NoBounds(0, 1, mmunit.radian),
            )
        )
    assert "Incompatible boundary conditions" in str(excinfo.value)

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
        NoBounds(0, 1, mmunit.nanometer),
    )
    assert distance1.distanceTo(distance0) == "(distance0-distance1)"


def test_dv_no_bounds_units_are_checked_and_converted():
    """NoBounds values are typical values, so they must be in the DV's unit."""
    mass = 1 * mmunit.dalton
    with pytest.raises(ValueError, match="incompatible units"):
        DynamicalVariable(
            "x", mmunit.nanometer, mass, NoBounds(-180, 180, mmunit.degree)
        )
    dv = DynamicalVariable(
        "x", mmunit.nanometer, mass, NoBounds(1, 10, mmunit.angstrom)
    )
    assert dv.bounds.unit == mmunit.nanometer
    assert dv.bounds.lower == pytest.approx(0.1)
    assert dv.bounds.upper == pytest.approx(1.0)


def test_distance_to_cv_uses_md_units_for_degree_dv():
    """A degree DV coupled to a radian torsion must wrap with period 2*pi."""
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.degree) ** 2
    dv = DynamicalVariable(
        "phi0", mmunit.degree, mass, PeriodicBounds(-180, 180, mmunit.degree)
    )
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    expression = dv.distanceTo(phi)
    assert "360" not in expression
    assert "6.28318" in expression


def test_distance_to_dv_requires_matching_bounds():
    """Periodic DVs with different periods cannot be compared."""
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    a = DynamicalVariable(
        "a", mmunit.radian, mass, PeriodicBounds(-1, 1, mmunit.radian)
    )
    b = DynamicalVariable("b", mmunit.radian, mass, CircularBounds())
    with pytest.raises(ValueError, match="Incompatible boundary conditions"):
        a.distanceTo(b)
    with pytest.raises(ValueError, match="Incompatible boundary conditions"):
        b.distanceTo(a)
    c = DynamicalVariable(
        "c",
        mmunit.degree,
        3 * mmunit.dalton * (mmunit.nanometer / mmunit.degree) ** 2,
        PeriodicBounds(-180, 180, mmunit.degree),
    )
    assert "6.28318" in b.distanceTo(c)
    x = DynamicalVariable(
        "x", mmunit.nanometer, 1 * mmunit.dalton, NoBounds(0, 1, mmunit.nanometer)
    )
    with pytest.raises(ValueError, match="Incompatible units"):
        x.distanceTo(a)


def test_dv_is_hashable_and_hash_matches_equality():
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    a = DynamicalVariable(
        "phi", mmunit.radian, mass, PeriodicBounds(-180, 180, mmunit.degree)
    )
    b = DynamicalVariable("phi", mmunit.radian, mass, CircularBounds())
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
