"""
Tests for capturing coupling derivatives inside openxps custom integrators.
"""

import pickle
import re
from copy import deepcopy

import cvpack
import openmm as mm
import pytest
from openmm import unit as mmunit

import openxps as xps
from openxps.bounds import NoBounds, ReflectiveBounds
from openxps.integrators.utils import CAPTURE_PREFIX

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


FEMTOSECOND = 1 * mmunit.femtosecond

INTEGRATORS = {
    "SymmetricVerlet": lambda names=(): xps.integrators.SymmetricVerletIntegrator(
        FEMTOSECOND, energyDerivatives=names
    ),
    "SymmetricLangevin": lambda names=(): xps.integrators.SymmetricLangevinIntegrator(
        300 * mmunit.kelvin, 1 / mmunit.picosecond, FEMTOSECOND, energyDerivatives=names
    ),
}
for _first in (True, False):
    _suffix = "first" if _first else "symmetric"
    INTEGRATORS[f"CSVR-{_suffix}"] = lambda names=(), first=_first: (
        xps.integrators.CSVRIntegrator(
            300 * mmunit.kelvin,
            10 / mmunit.picosecond,
            FEMTOSECOND,
            forceFirst=first,
            energyDerivatives=names,
        )
    )
    INTEGRATORS[f"MassiveGGMT-{_suffix}"] = lambda names=(), first=_first: (
        xps.integrators.MassiveGGMTIntegrator(
            300 * mmunit.kelvin,
            100 * FEMTOSECOND,
            FEMTOSECOND,
            forceFirst=first,
            energyDerivatives=names,
        )
    )
    INTEGRATORS[f"RegulatedNHL-{_suffix}"] = lambda names=(), first=_first: (
        xps.integrators.RegulatedNHLIntegrator(
            300 * mmunit.kelvin,
            100 * FEMTOSECOND,
            FEMTOSECOND,
            forceFirst=first,
            energyDerivatives=names,
        )
    )


def _steps(integrator):
    return [
        integrator.getComputationStep(i) for i in range(integrator.getNumComputations())
    ]


def _capture_indices(integrator):
    return [
        i
        for i, (kind, _, expression) in enumerate(_steps(integrator))
        if kind == mm.CustomIntegrator.ComputeGlobal
        and expression.startswith("deriv(energy,")
    ]


def _moves_positions(kind, target):
    return kind == mm.CustomIntegrator.ConstrainPositions or (
        kind == mm.CustomIntegrator.ComputePerDof and target == "x"
    )


def _uses_forces(expression):
    return re.search(r"\bf\b", expression) is not None


def _assert_capture_point(integrator):
    """Capture steps must share a force evaluation the integrator already does."""
    steps = _steps(integrator)
    first = _capture_indices(integrator)[0]
    if integrator.getCapturePoint() == "start":
        assert first == 1
        assert steps[0][0] == mm.CustomIntegrator.UpdateContextState
    else:
        later = steps[first:]
        assert not any(_moves_positions(kind, target) for kind, target, _ in later)
        assert any(_uses_forces(expression) for _, _, expression in later)


@pytest.mark.parametrize("name", list(INTEGRATORS))
def test_without_option_nothing_changes(name):
    integrator = INTEGRATORS[name]()
    assert integrator.getEnergyDerivativeNames() == ()
    assert _capture_indices(integrator) == []
    assert not any(
        integrator.getGlobalVariableName(i).startswith(CAPTURE_PREFIX)
        for i in range(integrator.getNumGlobalVariables())
    )


@pytest.mark.parametrize("name", list(INTEGRATORS))
def test_capture_steps_share_a_force_evaluation(name):
    plain = INTEGRATORS[name]()
    integrator = INTEGRATORS[name](["scaling", "other"])
    assert integrator.getEnergyDerivativeNames() == ("scaling", "other")
    assert integrator.getNumComputations() == plain.getNumComputations() + 2
    assert [_steps(integrator)[i][2] for i in _capture_indices(integrator)] == [
        "deriv(energy, scaling)",
        "deriv(energy, other)",
    ]
    assert integrator.getCapturePoint() == (
        "start" if integrator.isForceFirst() else "end"
    )
    _assert_capture_point(integrator)


@pytest.mark.parametrize("name", list(INTEGRATORS))
def test_capture_matches_get_state(name):
    """Start capture sees the positions before the step, end capture those after."""
    system = _system()
    system.addForce(_bond_force())
    integrator = INTEGRATORS[name](["scaling"])
    integrator.setRandomNumberSeed(1)
    context = mm.Context(system, integrator, REFERENCE)
    integrator.registerWithSystem(system, False)  # as ExtendedSpaceIntegrator.configure
    context.setPositions(POSITIONS)
    context.setVelocities(VELOCITIES)
    integrator.step(3)

    def derivative():
        state = context.getState(getEnergy=True, getParameterDerivatives=True)
        return state.getEnergyParameterDerivatives()["scaling"]

    before = derivative()
    integrator.step(1)
    after = derivative()
    expected = before if integrator.getCapturePoint() == "start" else after
    assert integrator.getCapturedDerivatives() == pytest.approx({"scaling": expected})
    assert before != pytest.approx(after)


@pytest.mark.parametrize(
    "duplicate",
    [deepcopy, lambda x: pickle.loads(pickle.dumps(x))],
    ids=["deepcopy", "pickle"],
)
def test_capture_survives_copy_and_pickle(duplicate):
    # Pickled through LockstepIntegrator, which keeps the integrator attributes;
    # OpenMM's own pickling of an integrator keeps only its XML.
    original = xps.LockstepIntegrator(INTEGRATORS["CSVR-first"](["scaling"]))
    integrator = duplicate(original).getPhysicalIntegrator()
    assert integrator.getEnergyDerivativeNames() == ("scaling",)
    assert len(_capture_indices(integrator)) == 1
    assert integrator.getCapturePoint() == "start"
