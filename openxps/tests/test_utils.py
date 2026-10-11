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
