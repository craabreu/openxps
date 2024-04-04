import pytest
from openmm import unit as mmunit
from openxps.extra_dof import ExtraDOF
from openxps.bounds import Periodic, Reflective
import yaml


def test_extra_dof_initialization():
    """Test successful ExtraDOF initialization."""
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    bounds = Periodic(-180, 180, mmunit.degree)
    extra_dof = ExtraDOF("phi", mmunit.radian, mass, bounds)

    assert extra_dof.name == "phi"
    assert extra_dof.unit == mmunit.radian
    assert (
        extra_dof.mass.unit == mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    )
    assert isinstance(extra_dof.bounds, Periodic)


def test_extra_dof_invalid_unit():
    """Test ExtraDOF initialization with an invalid unit."""
    with pytest.raises(ValueError) as excinfo:
        ExtraDOF(
            "phi",
            "not_a_unit",
            3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2,
            None,
        )
    assert "The unit must be a valid OpenMM unit." in str(excinfo.value)


def test_extra_dof_not_md_unit_system():
    """Test ExtraDOF initialization with an incompatible unit."""
    with pytest.raises(ValueError) as excinfo:
        ExtraDOF("phi", mmunit.degree, 3 * mmunit.dalton, None)
    assert "Unit degree is incompatible with OpenMM's MD unit system." in str(
        excinfo.value
    )


def test_extra_dof_mass_without_unit():
    """Test ExtraDOF initialization with mass missing a unit."""
    with pytest.raises(TypeError) as excinfo:
        ExtraDOF("phi", mmunit.radian, 3, None)
    assert "Mass must be have units of measurement." in str(excinfo.value)


def test_extra_dof_incompatible_mass_unit():
    """Test ExtraDOF initialization with mass having incompatible units."""
    with pytest.raises(TypeError):
        ExtraDOF("phi", mmunit.radian, 3 * mmunit.meter, None)


def test_extra_dof_bounds_incompatible_units():
    """Test ExtraDOF initialization with bounds having incompatible units."""
    bounds = Reflective(0, 10, mmunit.meter)  # Incompatible with radian
    with pytest.raises(ValueError):
        ExtraDOF(
            "phi",
            mmunit.radian,
            3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2,
            bounds,
        )


def test_extra_dof_serialization():
    """Test YAML serialization and deserialization of ExtraDOF."""
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    bounds = Periodic(-180, 180, mmunit.degree)
    extra_dof = ExtraDOF("phi", mmunit.radian, mass, bounds)

    serialized = yaml.safe_dump(extra_dof)
    deserialized = yaml.safe_load(serialized)

    assert deserialized == extra_dof


def test_extra_dof_without_bounds():
    """Test ExtraDOF initialization without bounds."""
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    extra_dof = ExtraDOF("phi", mmunit.radian, mass, None)

    assert extra_dof.bounds is None


def test_extra_dof_bounds_type_error():
    """Test ExtraDOF initialization with incorrect bounds type."""
    with pytest.raises(TypeError):
        ExtraDOF(
            "phi",
            mmunit.radian,
            3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2,
            "not_bounds",
        )
