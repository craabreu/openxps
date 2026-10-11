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


def _mean_kinetic_energy_per_dof(integrator, num_particles=100, equil=2000, prod=6000):
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
    """Each thermostat gives <K>/dof = kT/2 within 3% on an ideal gas."""
    integrator = factory()
    integrator.setRandomNumberSeed(7)
    kt = (mmunit.MOLAR_GAS_CONSTANT_R * 300 * KELVIN).value_in_unit(
        mmunit.kilojoule_per_mole
    )
    assert _mean_kinetic_energy_per_dof(integrator) == pytest.approx(0.5 * kt, rel=0.03)


@pytest.mark.parametrize(
    "build",
    [
        lambda: SplitIntegrator(
            integrators.SymmetricLangevinIntegrator(300 * KELVIN, 10 / PS, 4 * FS)
        ),
        lambda: LockstepIntegrator(
            integrators.CSVRIntegrator(300 * KELVIN, 10 / PS, 1 * FS, forceFirst=True)
        ),
        lambda: SplitIntegrator(
            integrators.RegulatedNHLIntegrator(300 * KELVIN, 0.1 * PS, 4 * FS)
        ),
    ],
)
def test_pickle_round_trip_with_default_extension_integrator(build):
    original = build()
    restored = pickle.loads(pickle.dumps(original))
    for old, new in (
        (original.getPhysicalIntegrator(), restored.getPhysicalIntegrator()),
        (original.getExtensionIntegrator(), restored.getExtensionIntegrator()),
    ):
        assert type(new) is type(old)
        assert new.getGlobalVariableByName("kT") == old.getGlobalVariableByName("kT")
        assert new.getTemperature() == old.getTemperature()


def test_copies_do_not_share_random_generators():
    csvr = integrators.CSVRIntegrator(300 * KELVIN, 10 / PS, 1 * FS, forceFirst=True)
    original = LockstepIntegrator(csvr)
    for clone in (copy.copy(original), copy.deepcopy(original)):
        assert clone.getPhysicalIntegrator()._rng is not csvr._rng
        assert (
            clone.getExtensionIntegrator()._rng
            is not original.getExtensionIntegrator()._rng
        )


@pytest.mark.parametrize("seed", [42, -1, 2**31 - 1])
def test_default_extension_seed_is_valid_and_different(seed):
    langevin = mm.LangevinMiddleIntegrator(300 * KELVIN, 1 / PS, 1 * FS)
    langevin.setRandomNumberSeed(seed)
    extension_seed = LockstepIntegrator(langevin).getExtensionIntegrator()
    extension_seed = extension_seed.getRandomNumberSeed()
    assert extension_seed not in (0, seed)


def test_csvr_accepts_negative_seeds():
    integrator = integrators.CSVRIntegrator(300 * KELVIN, 10 / PS, 1 * FS)
    integrator.setRandomNumberSeed(-3)
    assert integrator.getRandomNumberSeed() == -3


def _harmonic_oscillators(num_particles, mass=1.0, k=100.0):
    system = mm.System()
    force = mm.CustomExternalForce("0.5*k*(x^2+y^2+z^2)")
    force.addGlobalParameter("k", k)
    for i in range(num_particles):
        system.addParticle(mass)
        force.addParticle(i, [])
    system.addForce(force)
    return system


def _oscillator_context(integrator, num_particles):
    platform = mm.Platform.getPlatformByName("Reference")
    context = mm.Context(_harmonic_oscillators(num_particles), integrator, platform)
    kt = (mmunit.MOLAR_GAS_CONSTANT_R * 300 * KELVIN).value_in_unit(
        mmunit.kilojoule_per_mole
    )
    rng = np.random.default_rng(0)
    context.setPositions(rng.normal(0, np.sqrt(kt / 100.0), (num_particles, 3)))
    context.setVelocitiesToTemperature(300 * KELVIN, 1)
    return context, kt


def test_ggmt_has_no_square_root_problem_at_large_steps():
    """A step of tau/5 must not produce NaN."""
    integrator = integrators.MassiveGGMTIntegrator(300 * KELVIN, 0.1 * PS, 20 * FS)
    context, _ = _oscillator_context(integrator, 200)
    integrator.step(5000)
    state = context.getState(getVelocities=True)
    assert np.all(np.isfinite(state.getVelocities(asNumpy=True)._value))


def test_ggmt_fourth_moment_at_one_tenth_of_tau():
    """<(m v^2)^2>/(kT)^2 = 3 within 5% at a step of tau/10."""
    integrator = integrators.MassiveGGMTIntegrator(300 * KELVIN, 0.1 * PS, 10 * FS)
    integrator.setRandomNumberSeed(3)
    context, kt = _oscillator_context(integrator, 300)
    integrator.step(2000)
    second = fourth = 0.0
    samples = 400
    for _ in range(samples):
        integrator.step(10)
        state = context.getState(getVelocities=True)
        kinetic = state.getVelocities(asNumpy=True)._value ** 2 / kt
        second += kinetic.mean()
        fourth += (kinetic**2).mean()
    assert second / samples == pytest.approx(1.0, rel=0.03)
    assert fourth / samples == pytest.approx(3.0, rel=0.05)


def _reference_ggmt_step(state, constants, dt, substeps=200):
    v, v1, v2 = state
    kt, inv_q, inv_q2 = constants
    c = (2 / np.sqrt(3)) * kt**3 * inv_q2

    def zw_flow(v, v2, h):
        z = np.sqrt(3) * kt / (v * v)
        w = (2 / np.sqrt(3)) * kt * v2

        def f(z, w):
            return w, c * (1 / z**2 - 1)

        hh = h / substeps
        for _ in range(substeps):
            k1 = f(z, w)
            k2 = f(z + 0.5 * hh * k1[0], w + 0.5 * hh * k1[1])
            k3 = f(z + 0.5 * hh * k2[0], w + 0.5 * hh * k2[1])
            k4 = f(z + hh * k3[0], w + hh * k3[1])
            z += hh / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            w += hh / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return np.sign(v) * np.sqrt(np.sqrt(3) * kt / z), (np.sqrt(3) / 2) * w / kt

    v1 += 0.5 * dt * (v * v - kt) * inv_q
    v, v2 = zw_flow(v, v2, 0.5 * dt)
    v *= np.exp(-dt * (v1 + kt * v2))
    v, v2 = zw_flow(v, v2, 0.5 * dt)
    v1 += 0.5 * dt * (v * v - kt) * inv_q
    return v, v1, v2


def test_ggmt_step_matches_reference_sub_dynamics():
    """The thermostat step reproduces the exact (z, w) flow for a small step."""
    dt = 0.0002
    integrator = integrators.MassiveGGMTIntegrator(300 * KELVIN, 0.1 * PS, dt * PS)
    system = _ideal_gas(1, mass=1.0)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions([mm.Vec3(0, 0, 0)])
    v0, v1_0, v2_0 = 2.5, 0.3, -1.7
    context.setVelocities([mm.Vec3(v0, -v0, 0.5 * v0)])
    integrator.setPerDofVariableByName("v1", [mm.Vec3(v1_0, v1_0, v1_0)])
    integrator.setPerDofVariableByName("v2", [mm.Vec3(v2_0, v2_0, v2_0)])
    kt = integrator.getGlobalVariableByName("kT")
    inv_q = integrator.getGlobalVariableByName("invQ")
    inv_q2 = integrator.getGlobalVariableByName("invQ2")
    integrator.step(1)
    v = context.getState(getVelocities=True).getVelocities(asNumpy=True)._value[0]
    v2 = integrator.getPerDofVariableByName("v2")[0]
    for component, start in ((0, v0), (1, -v0), (2, 0.5 * v0)):
        v_ref, _, v2_ref = _reference_ggmt_step(
            (start, v1_0, v2_0), (kt, inv_q, inv_q2), dt
        )
        assert v[component] == pytest.approx(v_ref, rel=1e-6)
        assert v2[component] == pytest.approx(v2_ref, rel=1e-6)
