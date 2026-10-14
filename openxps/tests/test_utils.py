"""
Unit tests for openxps.utils.Function.
"""

from collections import namedtuple

import openmm as mm
import pytest
from cvpack import units
from openmm import unit as mmunit

from openxps.utils import Function, preprocess_args


def test_function_ignores_lepton_definitions():
    """Names defined after ';' are local definitions, not variables."""
    fn = Function("s", "(1-cos(u))/2; u=alpha*lam", alpha=3.0)
    assert fn.getVariables() == {"lam"}
    assert set(fn.getParameters()) == {"alpha"}


def test_function_collective_variable_uses_index_mapping():
    """Particle indices come from the mapping, not from list positions."""
    fn = Function("g", "x*y")
    cv = fn.createCollectiveVariable({"a": 0, "x": 2, "y": 1})
    system = mm.System()
    for _ in range(3):
        system.addParticle(1.0)
    cv.addToSystem(system)
    context = mm.Context(system, mm.VerletIntegrator(0.0))
    context.setPositions([mm.Vec3(9, 0, 0), mm.Vec3(3, 0, 0), mm.Vec3(2, 0, 0)])
    energy = context.getState(getEnergy=True).getPotentialEnergy()._value
    assert energy == pytest.approx(6.0)


def test_function_collective_variable_unknown_variable():
    fn = Function("g", "x*typo")
    with pytest.raises(ValueError, match="unknown variables"):
        fn.createCollectiveVariable({"x": 0})


@pytest.mark.parametrize("name", ["square", "cube", "recip"])
def test_function_recognizes_lepton_builtins(name):
    """Lepton built-ins are not mistaken for variables."""
    fn = Function("g", f"{name}(x)")
    assert fn.getVariables() == {"x"}
    fn.createCollectiveVariable({"x": 0})


def _energy(fn, mapping, xs):
    cv = fn.createCollectiveVariable(mapping)
    system = mm.System()
    for _ in xs:
        system.addParticle(1.0)
    cv.addToSystem(system)
    context = mm.Context(
        system, mm.VerletIntegrator(0.0), mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions([mm.Vec3(x, 0, 0) for x in xs])
    return context.getState(getEnergy=True).getPotentialEnergy()._value


@pytest.mark.parametrize(
    ("expression", "mapping", "expected"),
    [
        ("1000*s + x1", {"s": 0, "x1": 1}, 1000 * 1.0 + 2.0),
        ("x2 + 10*x1", {"x1": 0, "x2": 1}, 2.0 + 10 * 1.0),
        ("x1 + 10*x2 + 100*x3", {"x1": 2, "x2": 0, "x3": 1}, 3.0 + 10 + 200),
        ("10*u + s; u=3*x1", {"s": 0, "x1": 1}, 10 * 6.0 + 1.0),
        ("openxps_var0*s", {"s": 0, "openxps_var0": 1}, 2.0),
    ],
)
def test_function_variables_named_like_coordinates(expression, mapping, expected):
    """Variables named like Lepton coordinates still read their own particle."""
    xs = [1.0, 2.0, 3.0][: len(mapping)]
    assert _energy(Function("f", expression), mapping, xs) == pytest.approx(expected)


@pytest.mark.parametrize("name", ["λ", "λ_vdw", "θ1", "a·b"])
def test_function_non_ascii_identifiers(name):
    """Names are identifiers, as in Python, including non-ASCII ones."""
    fn = Function("f", f"2*{name}^2 + k", k=1.0)
    assert fn.getVariables() == {name}
    assert set(fn.getParameters()) == {"k"}
    assert _energy(fn, {name: 0}, [3.0]) == pytest.approx(2 * 9 + 1)


def test_function_names_are_lepton_tokens():
    """A name is a whole Lepton token, so parts of other tokens are not names."""
    fn = Function("f", "lambda.vdw + x²*y + 1e-3*z")
    assert fn.getVariables() == {"y", "z"}


def test_preprocess_args_handles_named_tuples_and_other_sequences():
    @preprocess_args
    def identity(data):
        return data

    vec = mm.Vec3(1.0, 2.0, 3.0)
    assert identity(vec) is vec
    assert identity(range(3)) == range(3)
    assert identity("abc") == "abc"

    Pair = namedtuple("Pair", ["a", "b"])
    converted = identity(Pair(1 * mmunit.angstrom, 2))
    assert type(converted) is Pair
    assert isinstance(converted.a, units.Quantity)
    assert converted.b == 2

    nested = identity([(1 * mmunit.nanometer, mmunit.kelvin), {"t": 2 * mmunit.second}])
    assert isinstance(nested[0], tuple)
    assert isinstance(nested[0][0], units.Quantity)
    assert isinstance(nested[0][1], units.Unit)
    assert isinstance(nested[1]["t"], units.Quantity)
