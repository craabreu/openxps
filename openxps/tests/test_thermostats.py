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


def test_default_extension_integrator_uses_a_different_seed():
    """A copied extension integrator must not replay the physical noise."""
    langevin = mm.LangevinMiddleIntegrator(300 * KELVIN, 1 / PS, 1 * FS)
    langevin.setRandomNumberSeed(42)
    lockstep = LockstepIntegrator(langevin)
    assert lockstep.getExtensionIntegrator().getRandomNumberSeed() == 43
    csvr = integrators.CSVRIntegrator(300 * KELVIN, 10 / PS, 1 * FS, forceFirst=True)
    lockstep = LockstepIntegrator(csvr)
    assert csvr._rng.random() != lockstep.getExtensionIntegrator()._rng.random()


def test_csvr_seed_follows_openmm_conventions():
    """getRandomNumberSeed returns what was set; 0 means random."""
    a = integrators.CSVRIntegrator(300 * KELVIN, 10 / PS, 1 * FS)
    b = integrators.CSVRIntegrator(300 * KELVIN, 10 / PS, 1 * FS)
    for integrator in (a, b):
        integrator.registerWithSystem(_ideal_gas(), False)
        integrator.setRandomNumberSeed(5)
    assert a.getRandomNumberSeed() == 5
    np.testing.assert_allclose(
        a._sums_of_squared_gaussians(3), b._sums_of_squared_gaussians(3)
    )
    a.setRandomNumberSeed(0)
    b.setRandomNumberSeed(0)
    assert not np.allclose(
        a._sums_of_squared_gaussians(3), b._sums_of_squared_gaussians(3)
    )


def test_csvr_step_before_register_raises_clear_error():
    integrator = integrators.CSVRIntegrator(300 * KELVIN, 10 / PS, 1 * FS)
    context = _context(_ideal_gas(2), integrator)
    with pytest.raises(ValueError, match="registerWithSystem"):
        integrator.step(1)
    assert context.getStepCount() == 0


def test_ggmt_time_constant_unit():
    integrator = integrators.MassiveGGMTIntegrator(300 * KELVIN, 0.1, 1 * FS)
    assert integrator.getTimeConstant() == 0.1 * PS


def test_csvr_at_rest_does_not_produce_nan():
    """A system at rest under zero force stays at rest."""
    integrator = integrators.CSVRIntegrator(
        300 * KELVIN, 10 / PS, 1 * FS, forceFirst=True
    )
    system = _ideal_gas(3)
    integrator.registerWithSystem(system, True)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(np.zeros((3, 3)))
    context.setVelocities(np.zeros((3, 3)))
    integrator.step(5)
    state = context.getState(getVelocities=True)
    velocities = state.getVelocities(asNumpy=True)._value
    assert np.all(np.isfinite(velocities))
    assert np.all(velocities == 0)


def _mean_kinetic_energy_per_dof(integrator, num_particles=100, equil=500, prod=1500):
    system = _ideal_gas(num_particles)
    if isinstance(integrator, integrators.CSVRIntegrator):
        integrator.registerWithSystem(system, False)
    context = _context(system, integrator)
    integrator.step(equil)
    total = 0.0
    for _ in range(prod // 10):
        integrator.step(10)
        total += context.getState(getEnergy=True).getKineticEnergy()._value
    return total / (prod // 10) / (3 * num_particles)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: integrators.SymmetricLangevinIntegrator(300 * KELVIN, 10 / PS, 2 * FS),
        lambda: integrators.CSVRIntegrator(300 * KELVIN, 10 / PS, 2 * FS),
        lambda: integrators.MassiveGGMTIntegrator(300 * KELVIN, 0.1 * PS, 2 * FS),
        lambda: integrators.MassiveGGMTIntegrator(
            300 * KELVIN, 0.1 * PS, 2 * FS, stochastic=True
        ),
        lambda: integrators.RegulatedNHLIntegrator(300 * KELVIN, 0.1 * PS, 2 * FS),
    ],
)
def test_thermostats_reproduce_target_temperature(factory):
    """Each thermostat gives <K>/dof = kT/2 within 5% on an ideal gas."""
    integrator = factory()
    integrator.setRandomNumberSeed(7)
    kt = (mmunit.MOLAR_GAS_CONSTANT_R * 300 * KELVIN).value_in_unit(
        mmunit.kilojoule_per_mole
    )
    assert _mean_kinetic_energy_per_dof(integrator) == pytest.approx(0.5 * kt, rel=0.05)
