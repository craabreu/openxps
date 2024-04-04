import pytest
import numpy as np
from openmm import unit as mmunit
from openxps.bounds import Bounds, Periodic, Reflective, CIRCULAR

def test_bounds_initialization():
    # Test normal initialization
    bounds = Bounds(-1, 1, mmunit.meters)
    assert bounds.lower == -1
    assert bounds.upper == 1
    assert bounds.unit == mmunit.meters

    # Test initialization with invalid lower and upper bounds
    with pytest.raises(ValueError):
        Bounds(1, -1, mmunit.meters)

    # Test initialization with invalid unit
    with pytest.raises(TypeError):
        Bounds(-1, 1, "not_a_unit")

    # Test initialization with non-numeric bounds
    with pytest.raises(TypeError):
        Bounds("low", "high", mmunit.meters)

def test_bounds_equality():
    # Test equality with non-Bounds object
    assert not (Bounds(-1, 1, mmunit.meters) == "not_a_bounds")

    # Test equality between Bounds objects
    bounds1 = Bounds(-1, 1, mmunit.meters)
    bounds2 = Bounds(-1000, 1000, mmunit.millimeters)
    assert bounds1 == bounds2

def test_bounds_conversion():
    bounds = Bounds(-1, 1, mmunit.meters)
    converted = bounds.convert(mmunit.centimeters)
    assert converted.lower == -100
    assert converted.upper == 100
    assert converted.unit == mmunit.centimeters

    # Test conversion with incompatible unit
    with pytest.raises(ValueError):
        bounds.convert(mmunit.seconds)

def test_periodic_wrap():
    bounds = Periodic(-180, 180, mmunit.degrees)
    wrapped_value, wrapped_rate = bounds.wrap(190, 10)
    assert wrapped_value == -170
    assert wrapped_rate == 10

def test_reflective_wrap():
    bounds = Reflective(0, 10, mmunit.dimensionless)
    wrapped_value, wrapped_rate = bounds.wrap(12, 1)
    assert wrapped_value == 8
    assert wrapped_rate == -1

    wrapped_value, wrapped_rate = bounds.wrap(-2, -1)
    assert wrapped_value == 2
    assert wrapped_rate == 1

    # Test wrapping at the upper bound edge
    wrapped_value, wrapped_rate = bounds.wrap(20, 1)
    assert wrapped_value == 0  # This ensures the missed line is covered
    assert wrapped_rate == 1

    # Additional case for coverage: directly at the lower bound, moving into the bounds
    wrapped_value, wrapped_rate = bounds.wrap(0, -1)
    assert wrapped_value == 0
    assert wrapped_rate == -1


def test_circular_constant():
    assert CIRCULAR.lower == -np.pi
    assert CIRCULAR.upper == np.pi
    assert CIRCULAR.unit == mmunit.radians

def test_serialization():
    # Test YAML serialization and deserialization for Periodic
    import yaml
    periodic = Periodic(-np.pi, np.pi, mmunit.radians)
    serialized = yaml.safe_dump(periodic)
    deserialized = yaml.safe_load(serialized)
    assert deserialized == periodic

    # Test YAML serialization and deserialization for Reflective
    reflective = Reflective(0, 10, mmunit.dimensionless)
    serialized = yaml.safe_dump(reflective)
    deserialized = yaml.safe_load(serialized)
    assert deserialized == reflective

def test_bounds_wrap_not_implemented():
    bounds = Bounds(-1, 1, mmunit.meters)
    with pytest.raises(NotImplementedError):
        bounds.wrap(0, 0)
