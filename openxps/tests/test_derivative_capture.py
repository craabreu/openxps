"""
Tests for capturing coupling derivatives inside openxps custom integrators.
"""

import cvpack
import openmm as mm
import pytest
from openmm import unit as mmunit

import openxps as xps
from openxps.bounds import NoBounds, ReflectiveBounds

REFERENCE = mm.Platform.getPlatformByName("Reference")
DIMENSIONLESS = mmunit.dimensionless
POSITIONS = [mm.Vec3(0, 0, 0), mm.Vec3(0.2, 0, 0)] * mmunit.nanometer
MOVED = [mm.Vec3(0, 0, 0), mm.Vec3(0.25, 0, 0)] * mmunit.nanometer
VELOCITIES = (
    [mm.Vec3(0, 0, 0), mm.Vec3(0.3, 0.1, 0)] * mmunit.nanometer / mmunit.picosecond
)


def _dv(name="lam"):
    return xps.DynamicalVariable(
        name,
        DIMENSIONLESS,
        1 * mmunit.dalton * mmunit.nanometer**2,
        ReflectiveBounds(0, 1, DIMENSIONLESS),
    )


def _bond_force(parameter="scaling"):
    force = mm.CustomBondForce(f"{parameter}*500*(r-0.15)^2")
    force.addGlobalParameter(parameter, 1.0)
    force.addEnergyParameterDerivative(parameter)
    force.addBond(0, 1, [])
    return force


def _inner(name="lam"):
    """An inner product coupling whose parameter is a smoothstep of the variable."""
    return xps.InnerProductCoupling(
        [_bond_force()], [_dv(name)], functions={"scaling": f"{name}^2*(3-2*{name})"}
    )


def _harmonic():
    dv = xps.DynamicalVariable(
        "b", mmunit.nanometer, 1 * mmunit.dalton, NoBounds(0, 1, mmunit.nanometer)
    )
    return xps.HarmonicCoupling(
        cvpack.Distance(0, 1, name="d"),
        dv,
        1000 * mmunit.kilojoule_per_mole / mmunit.nanometer**2,
    )


def _system():
    system = mm.System()
    system.addParticle(12.0)
    system.addParticle(1.0)
    return system


def _verlet_context(coupling):
    context = xps.ExtendedSpaceContext(
        xps.ExtendedSpaceSystem(_system(), coupling),
        xps.LockstepIntegrator(mm.VerletIntegrator(0.001)),
        REFERENCE,
    )
    context.setPositions(POSITIONS)
    return context


def test_energy_derivative_names():
    assert _inner().getEnergyDerivativeNames() == ("scaling",)
    assert _harmonic().getEnergyDerivativeNames() == ()
    assert (_inner() + _harmonic()).getEnergyDerivativeNames() == ("scaling",)


def test_inner_product_sends_only_derivatives():
    context = _verlet_context(_inner())
    extension = context.getExtensionContext()
    coupling = context.getSystem().getCoupling()
    before = extension.getParameter("derivative_with_respect_to_scaling")
    mm.Context.setPositions(context, MOVED)  # bypasses the automatic update
    coupling.updateExtensionCollectiveVariables(context, extension)
    assert extension.getParameter("derivative_with_respect_to_scaling") == before
    coupling.updateExtensionDerivatives(extension, {"scaling": 12.5})
    assert extension.getParameter("derivative_with_respect_to_scaling") == 12.5


def test_harmonic_sends_only_collective_variables():
    context = _verlet_context(_harmonic())
    extension = context.getExtensionContext()
    coupling = context.getSystem().getCoupling()
    mm.Context.setPositions(context, MOVED)
    coupling.updateExtensionDerivatives(extension, {})
    assert extension.getParameter("d") == pytest.approx(0.2)
    coupling.updateExtensionCollectiveVariables(context, extension)
    assert extension.getParameter("d") == pytest.approx(0.25)


def test_coupling_sum_dispatches_both_parts():
    context = _verlet_context(_inner() + _harmonic())
    extension = context.getExtensionContext()
    coupling = context.getSystem().getCoupling()
    mm.Context.setPositions(context, MOVED)
    coupling.updateExtensionDerivatives(extension, {"scaling": 7.0})
    assert extension.getParameter("derivative_with_respect_to_scaling") == 7.0
    assert extension.getParameter("d") == pytest.approx(0.2)
    coupling.updateExtensionCollectiveVariables(context, extension)
    assert extension.getParameter("derivative_with_respect_to_scaling") == 7.0
    assert extension.getParameter("d") == pytest.approx(0.25)


def test_both_parts_equal_update_extension_context():
    context = _verlet_context(_inner() + _harmonic())
    extension = context.getExtensionContext()
    coupling = context.getSystem().getCoupling()
    mm.Context.setPositions(context, MOVED)
    coupling.updateExtensionContext(context, extension)
    expected = {
        name: extension.getParameter(name) for name in extension.getParameters()
    }
    mm.Context.setPositions(context, POSITIONS)
    coupling.updateExtensionContext(context, extension)
    mm.Context.setPositions(context, MOVED)
    derivative = context.getState(
        getEnergy=True, getParameterDerivatives=True
    ).getEnergyParameterDerivatives()["scaling"]
    coupling.updateExtensionCollectiveVariables(context, extension)
    coupling.updateExtensionDerivatives(extension, {"scaling": derivative})
    for name, value in expected.items():
        assert extension.getParameter(name) == pytest.approx(value, rel=1e-12)
