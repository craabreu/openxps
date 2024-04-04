"""
Unit tests for the openxps.bounds module.
"""

import numpy as np
import pytest
import yaml
from openmm import unit as mmunit

from openxps.bounds import CIRCULAR, Bounds, Periodic, Reflective


def test_bounds_initialization():
    """
    Test the initialization of Bounds.
    """
    bounds = Bounds(-1, 1, mmunit.meters)
    assert bounds.lower == -1
    assert bounds.upper == 1
    assert bounds.unit == mmunit.meters

    with pytest.raises(ValueError):
        Bounds(1, -1, mmunit.meters)

    with pytest.raises(TypeError):
        Bounds(-1, 1, "not_a_unit")

    with pytest.raises(TypeError):
        Bounds("low", "high", mmunit.meters)


def test_bounds_equality():
    """
    Test the equality of Bounds objects.
    """
    assert Bounds(-1, 1, mmunit.meters) != "not_a_bounds"

    bounds1 = Bounds(-1, 1, mmunit.meters)
    bounds2 = Bounds(-1000, 1000, mmunit.millimeters)
    assert bounds1 == bounds2


def test_bounds_conversion():
    """
    Test the conversion of Bounds units.
    """
    bounds = Bounds(-1, 1, mmunit.meters)
    converted = bounds.convert(mmunit.centimeters)
    assert converted.lower == -100
    assert converted.upper == 100
    assert converted.unit == mmunit.centimeters

    with pytest.raises(ValueError):
        bounds.convert(mmunit.seconds)


def test_periodic_wrap():
    """
    Test the wrapping behavior of Periodic bounds.
    """
    bounds = Periodic(-180, 180, mmunit.degrees)
    wrapped_value, wrapped_rate = bounds.wrap(190, 10)
    assert wrapped_value == -170
    assert wrapped_rate == 10


def test_reflective_wrap():
    """
    Test the wrapping behavior of Reflective bounds.
    """
    bounds = Reflective(0, 10, mmunit.dimensionless)
    wrapped_value, wrapped_rate = bounds.wrap(12, 1)
    assert wrapped_value == 8
    assert wrapped_rate == -1

    wrapped_value, wrapped_rate = bounds.wrap(-2, -1)
    assert wrapped_value == 2
    assert wrapped_rate == 1

    wrapped_value, wrapped_rate = bounds.wrap(20, 1)
    assert wrapped_value == 0
    assert wrapped_rate == 1

    wrapped_value, wrapped_rate = bounds.wrap(0, -1)
    assert wrapped_value == 0
    assert wrapped_rate == -1


def test_circular_constant():
    """
    Test the CIRCULAR constant.
    """
    assert CIRCULAR.lower == -np.pi
    assert CIRCULAR.upper == np.pi
    assert CIRCULAR.unit == mmunit.radians


def test_serialization():
    """
    Test the serialization and deserialization of Bounds objects using YAML.
    """

    periodic = Periodic(-np.pi, np.pi, mmunit.radians)
    serialized = yaml.safe_dump(periodic)
    deserialized = yaml.safe_load(serialized)
    assert deserialized == periodic

    reflective = Reflective(0, 10, mmunit.dimensionless)
    serialized = yaml.safe_dump(reflective)
    deserialized = yaml.safe_load(serialized)
    assert deserialized == reflective


def test_bounds_wrap_not_implemented():
    """
    Test the NotImplementedError when calling the wrap method of Bounds.
    """
    bounds = Bounds(-1, 1, mmunit.meters)
    with pytest.raises(NotImplementedError):
        bounds.wrap(0, 0)
