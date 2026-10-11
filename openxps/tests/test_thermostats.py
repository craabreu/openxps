"""
Unit tests for the integrators in openxps.integrators.
"""

import copy
import pickle

import numpy as np
import openmm as mm
import pytest
from openmm import unit as mmunit

from openxps import LockstepIntegrator, SplitIntegrator, integrators
from openxps.tests.test_integrator import create_non_force_first_integrator

KELVIN = mmunit.kelvin
PS = mmunit.picosecond
FS = mmunit.femtosecond


def _ideal_gas(num_particles=100, mass=10.0):
    system = mm.System()
    for _ in range(num_particles):
        system.addParticle(mass)
    return system


def _context(system, integrator, seed=1):
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(np.zeros((system.getNumParticles(), 3)))
    context.setVelocitiesToTemperature(300 * KELVIN, seed)
    return context


@pytest.mark.parametrize(
    "factory",
    [
        lambda: integrators.SymmetricLangevinIntegrator(300 * KELVIN, 1 / PS, 1 * FS),
        lambda: integrators.CSVRIntegrator(300 * KELVIN, 10 / PS, 1 * FS),
        lambda: integrators.MassiveGGMTIntegrator(300 * KELVIN, 0.1 * PS, 1 * FS),
        lambda: integrators.RegulatedNHLIntegrator(300 * KELVIN, 0.1 * PS, 1 * FS),
    ],
)
def test_property_setters_update_global_variables(factory):
    """set<Property> must not crash and must refresh the integrator globals."""
    integrator = factory()
    integrator.setTemperature(310 * KELVIN)
    assert integrator.getTemperature() == 310 * KELVIN
    kt = (mmunit.MOLAR_GAS_CONSTANT_R * 310 * KELVIN).value_in_unit(
        mmunit.kilojoule_per_mole
    )
    assert integrator.getGlobalVariableByName("kT") == pytest.approx(kt)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: integrators.SymmetricLangevinIntegrator(300 * KELVIN, 10 / PS, 4 * FS),
        lambda: integrators.MassiveGGMTIntegrator(
            300 * KELVIN, 0.1 * PS, 4 * FS, stochastic=True
        ),
        lambda: integrators.RegulatedNHLIntegrator(300 * KELVIN, 0.1 * PS, 4 * FS),
    ],
)
def test_set_step_size_refreshes_friction_coefficients(factory):
    """a = exp(-gamma dt) must follow the step size."""
    integrator = factory()
    gamma = integrator.getFrictionCoefficient().value_in_unit(PS**-1)
    integrator.setStepSize(1 * FS)
    assert integrator.getGlobalVariableByName("a") == pytest.approx(
        np.exp(-gamma * 0.001)
    )


def test_repr_has_no_blank_line_without_per_dof_variables():
    integrator = integrators.RegulatedNHLIntegrator(300 * KELVIN, 0.1 * PS, 1 * FS)
    text = repr(integrator)
    assert "Per-dof variables:\n  v1" in text
    integrator = mm.CustomIntegrator(0.001)
    integrator.addComputePerDof("v", "v")
    text = integrators.utils.IntegratorMixin.__repr__(integrator)
    assert not text.startswith("  \n")


def test_pickled_extended_integrator_keeps_subclasses():
    """CSVR's Python-level step() must survive a pickle round trip."""
    csvr = integrators.CSVRIntegrator(300 * KELVIN, 10 / PS, 1 * FS, forceFirst=True)
    original = LockstepIntegrator(mm.VerletIntegrator(1 * FS), csvr)
    restored = pickle.loads(pickle.dumps(original))
    assert isinstance(restored.getExtensionIntegrator(), integrators.CSVRIntegrator)
    assert isinstance(restored.getPhysicalIntegrator(), mm.VerletIntegrator)
    extension = restored.getExtensionIntegrator()
    extension.registerWithSystem(_ideal_gas(4), True)
    context = _context(_ideal_gas(4), extension)
    before = extension.getGlobalVariableByName("sumRsq")
    extension.step(1)
    assert extension.getGlobalVariableByName("sumRsq") != before
    assert context.getStepCount() == 1


def test_deepcopy_keeps_assume_flags():
    custom = create_non_force_first_integrator(1 * FS)
    lockstep = LockstepIntegrator(custom, assume_force_first=True)
    assert isinstance(copy.deepcopy(lockstep), LockstepIntegrator)
    split = SplitIntegrator(
        mm.LangevinIntegrator(300 * KELVIN, 1 / PS, 2 * FS), assume_symmetric=True
    )
    assert isinstance(copy.copy(split), SplitIntegrator)
