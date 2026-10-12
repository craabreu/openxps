"""
Unit tests for openxps.utils.Function.
"""

import openmm as mm
import pytest

from openxps.utils import Function


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
