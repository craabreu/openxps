"""Tests for the extended-space OPES samplers."""

import cvpack
import numpy as np
import openmm
import openmm_opes
import pytest
from openmm import app, unit
from openmmtools import testsystems

import openxps as xps
from openxps.opes import _biasVariables, _extensionContext, _extensionSimulation

REFERENCE = openmm.Platform.getPlatformByName("Reference")
KELVIN = unit.kelvin
KJ = unit.kilojoules_per_mole
MASS = 3 * unit.dalton * (unit.nanometer / unit.radian) ** 2


def _circular_dv(name="phi0"):
    return xps.DynamicalVariable(name, unit.radian, MASS, xps.CircularBounds())


def _reflective_dv(name="s", lower=0.0, upper=1.0):
    """A length-like variable, so that it can couple to a distance."""
    return xps.DynamicalVariable(
        name,
        unit.nanometer,
        1 * unit.dalton,
        xps.ReflectiveBounds(lower, upper, unit.nanometer),
    )


def _alanine_system(dv, cv=None):
    """Alanine dipeptide with a harmonic coupling between a torsion and ``dv``."""
    model = testsystems.AlanineDipeptideVacuum()
    cv = cv or cvpack.Torsion(6, 8, 14, 16, name="phi")
    kappa = 1000 * unit.kilojoules_per_mole / dv.unit**2
    return model, xps.ExtendedSpaceSystem(
        model.system, xps.HarmonicCoupling(cv, dv, kappa)
    )


def test_bias_variables_follow_the_dynamical_variables():
    dv = _circular_dv()
    _, system = _alanine_system(dv)
    variable = xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree, grid_width=73)
    (bias_variable,), bounded = _biasVariables(system, [variable])
    assert isinstance(bias_variable, app.BiasVariable)
    assert bias_variable.periodic
    assert not bounded
    assert bias_variable.minValue == pytest.approx(-np.pi)
    assert bias_variable.maxValue == pytest.approx(np.pi)
    assert bias_variable.biasWidth == pytest.approx(np.radians(18))
    assert bias_variable.gridWidth == 73


def test_reflective_variable_sets_bounded():
    dv = _reflective_dv()
    _, system = _alanine_system(dv, cv=cvpack.Distance(0, 1, name="d"))
    variable = xps.ExtendedSpaceBiasVariable(dv, 0.05 * unit.nanometer)
    (bias_variable,), bounded = _biasVariables(system, [variable])
    assert bounded
    assert not bias_variable.periodic
    assert (bias_variable.minValue, bias_variable.maxValue) == (0.0, 1.0)


def test_unknown_and_unbounded_variables_are_rejected():
    dv = _circular_dv()
    _, system = _alanine_system(dv)
    other = xps.ExtendedSpaceBiasVariable(_circular_dv("psi0"), 18 * unit.degree)
    with pytest.raises(ValueError, match="not found"):
        _biasVariables(system, [other])
    free = xps.DynamicalVariable(
        "free", unit.nanometer, 1 * unit.dalton, xps.NoBounds(0, 1, unit.nanometer)
    )
    _, free_system = _alanine_system(free, cv=cvpack.Distance(0, 1, name="d"))
    with pytest.raises(ValueError, match="no bounds"):
        _biasVariables(
            free_system, [xps.ExtendedSpaceBiasVariable(free, 0.1 * unit.nanometer)]
        )


def test_conversions_pass_converted_objects_through():
    dv = _circular_dv()
    model, system = _alanine_system(dv)
    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        system,
        xps.LockstepIntegrator(openmm.VerletIntegrator(1 * unit.femtosecond)),
        REFERENCE,
    )
    wrapped = _extensionSimulation(simulation)
    assert wrapped.context is simulation.context.getExtensionContext()
    assert _extensionSimulation(wrapped) is wrapped
    extension = _extensionContext(simulation.context)
    assert extension is simulation.context.getExtensionContext()
    assert _extensionContext(extension) is extension


def _opes(system, variables, **kwargs):
    settings = {
        "temperature": 300 * KELVIN,
        "barrier": 20 * unit.kilojoules_per_mole,
        "frequency": 10,
        "varianceFrequency": None,
    }
    settings.update(kwargs)
    return xps.ExtendedSpaceOPES(system, variables, **settings)


def _simulation(model, system, seed=1):
    integrator = openmm.LangevinMiddleIntegrator(
        300 * KELVIN, 1 / unit.picosecond, 2 * unit.femtosecond
    )
    integrator.setRandomNumberSeed(seed)
    simulation = xps.ExtendedSpaceSimulation(
        model.topology, system, xps.LockstepIntegrator(integrator), REFERENCE
    )
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * KELVIN, seed)
    simulation.context.setDynamicalVariableValues([180 * unit.degree])
    simulation.context.setDynamicalVariableVelocitiesToTemperature(300 * KELVIN, seed)
    return simulation


def test_opes_adds_one_force_to_the_extension_system():
    dv = _circular_dv()
    _, system = _alanine_system(dv)
    before = system.getExtensionSystem().getNumForces()
    sampler = _opes(system, [xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)])
    extension = system.getExtensionSystem()
    assert extension.getNumForces() == before + 1
    assert isinstance(sampler, openmm_opes.OPES)
    assert sampler.variables[0].periodic
    assert not sampler.bounded


def test_opes_rejects_bounded_and_forwards_other_keywords():
    dv = _reflective_dv()
    _, system = _alanine_system(dv, cv=cvpack.Distance(0, 1, name="d"))
    variable = xps.ExtendedSpaceBiasVariable(dv, 0.05 * unit.nanometer)
    with pytest.raises(TypeError, match="bounded"):
        _opes(system, [variable], bounded=True)
    sampler = _opes(system, [variable], exploreMode=True, kernelShape="compact")
    assert sampler.bounded
    assert sampler.exploreMode
    with pytest.raises(ValueError, match="frequency"):
        _opes(system, [variable], frequency=0)


def test_opes_rejects_mixed_periodicity():
    phi0, s = _circular_dv(), _reflective_dv()
    model = testsystems.AlanineDipeptideVacuum()
    coupling = xps.HarmonicCoupling(
        cvpack.Torsion(6, 8, 14, 16, name="phi"),
        phi0,
        1000 * unit.kilojoules_per_mole / unit.radian**2,
    ) + xps.HarmonicCoupling(
        cvpack.Distance(0, 1, name="d"),
        s,
        1000 * unit.kilojoules_per_mole / unit.nanometer**2,
    )
    system = xps.ExtendedSpaceSystem(model.system, coupling)
    with pytest.raises(ValueError, match="mixed periodic"):
        _opes(
            system,
            [
                xps.ExtendedSpaceBiasVariable(phi0, 18 * unit.degree),
                xps.ExtendedSpaceBiasVariable(s, 0.05 * unit.nanometer),
            ],
        )


def test_opes_step_and_collective_variables():
    dv = _circular_dv()
    model, system = _alanine_system(dv)
    sampler = _opes(system, [xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)])
    simulation = _simulation(model, system)
    sampler.step(simulation, 50)
    assert simulation.currentStep == 50
    assert sampler.getNumKernels() >= 1
    (value,) = sampler.getCollectiveVariables(simulation)
    (dv_value,) = simulation.context.getDynamicalVariableValues()
    assert value == pytest.approx(dv_value.value_in_unit(unit.radian), abs=1e-6)
    assert sampler.getFreeEnergy().shape == (sampler.variables[0].gridWidth,)


def test_opes_methods_accept_openxps_and_extension_objects():
    dv = _circular_dv()
    model, system = _alanine_system(dv)
    sampler = _opes(system, [xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)])
    simulation = _simulation(model, system)
    sampler.step(simulation, 20)
    context = simulation.context
    extension = context.getExtensionContext()
    assert sampler.getBiasEnergy(context) == sampler.getBiasEnergy(extension)
    assert sampler.getCollectiveVariables(simulation) == sampler.getCollectiveVariables(
        _extensionSimulation(simulation)
    )
    sampler.updateContext(context)  # must not raise
    sampler.updateContext(extension)
    assert sampler.getBiasEnergy(context) == sampler.getBiasEnergy(extension)
    # step and stepTogether accept the converted simulation too, as openmm-opes
    # passes it around internally
    sampler.step(_extensionSimulation(simulation), 10)
    xps.ExtendedSpaceOPES.stepTogether(_extensionSimulation(simulation), [sampler], 10)
    assert simulation.currentStep == 40


class _Replay:
    """A stand-alone 'simulation' of an extension system that replays recorded values.

    openmm-opes needs only ``context``, ``currentStep`` and ``step``.
    """

    def __init__(self, context, values, frequency):
        self.context = context
        self.currentStep = 0
        self._values = values
        self._frequency = frequency

    def step(self, steps):
        self.currentStep += steps
        if self.currentStep % self._frequency == 0:
            value = self._values[self.currentStep // self._frequency - 1]
            self.context.setPositions([openmm.Vec3(value, 0, 0)])


def _plain_bias_variable(dv, sigma):
    """The BiasVariable the openxps classes would build, on a throwaway system."""
    _, system = _alanine_system(dv)
    (bias_variable,), bounded = _biasVariables(
        system, [xps.ExtendedSpaceBiasVariable(dv, sigma)]
    )
    return bias_variable, bounded


def test_opes_matches_plain_openmm_opes_on_the_extension_system():
    """The adapter deposits what a plain OPES would deposit on the extension system."""
    dv = _circular_dv()
    model, system = _alanine_system(dv)
    plain_extension = openmm.XmlSerializer.clone(system.getExtensionSystem())
    frequency = 10
    # No kernel merging and a low barrier, so that counts and kernel weights
    # (which depend on the bias energy at each deposition) both carry information
    sampler = _opes(
        system,
        [xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)],
        frequency=frequency,
        barrier=5 * unit.kilojoules_per_mole,
        compressionThreshold=0,
    )
    simulation = _simulation(model, system)
    values, energies = [], []
    for _ in range(8):
        sampler.step(simulation, frequency)
        (value,) = simulation.context.getDynamicalVariableValues()
        values.append(value.value_in_unit(unit.radian))
        energies.append(sampler.getBiasEnergy(simulation.context))

    bias_variable, bounded = _plain_bias_variable(dv, 18 * unit.degree)
    plain = openmm_opes.OPES(
        plain_extension,
        [bias_variable],
        temperature=300 * KELVIN,
        barrier=5 * unit.kilojoules_per_mole,
        frequency=frequency,
        varianceFrequency=None,
        bounded=bounded,
        compressionThreshold=0,
    )
    context = openmm.Context(
        plain_extension, openmm.VerletIntegrator(1 * unit.femtosecond), REFERENCE
    )
    context.setPositions([openmm.Vec3(0, 0, 0)])
    replay = _Replay(context, values, frequency)
    for energy in energies:
        plain.step(replay, frequency)
        assert plain.getBiasEnergy(context).value_in_unit(KJ) == pytest.approx(
            energy.value_in_unit(KJ), abs=1e-9
        )
    assert abs(energies[-1].value_in_unit(unit.kilojoules_per_mole)) > 1e-3

    assert plain.getNumKernels() == sampler.getNumKernels() == len(values)
    np.testing.assert_allclose(plain.getBias(), sampler.getBias(), rtol=1e-10)
    np.testing.assert_allclose(
        plain.getFreeEnergy(), sampler.getFreeEnergy(), rtol=1e-10
    )


def _combined(system, variables, **kwargs):
    settings = {
        "temperature": 300 * KELVIN,
        "barrier": 20 * unit.kilojoules_per_mole,
        "exploreBarrier": 10 * unit.kilojoules_per_mole,
        "frequency": 10,
        "varianceFrequency": None,
    }
    settings.update(kwargs)
    return xps.ExtendedSpaceCombinedOPES(system, variables, **settings)


def test_combined_opes_adds_one_force_and_rejects_bounded():
    dv = _circular_dv()
    _, system = _alanine_system(dv)
    variable = xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)
    before = system.getExtensionSystem().getNumForces()
    sampler = _combined(system, [variable])
    assert system.getExtensionSystem().getNumForces() == before + 1
    assert isinstance(sampler, openmm_opes.CombinedOPES)
    with pytest.raises(TypeError, match="bounded"):
        _combined(system, [variable], bounded=False)


def test_combined_opes_step_and_component_energies():
    dv = _circular_dv()
    model, system = _alanine_system(dv)
    sampler = _combined(system, [xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)])
    simulation = _simulation(model, system)
    sampler.step(simulation, 50)
    assert simulation.currentStep == 50
    assert all(n >= 1 for n in sampler.getNumKernels())
    context = simulation.context
    extension = context.getExtensionContext()
    opes, explore = sampler.getComponentBiasEnergies(context)
    assert (opes, explore) == sampler.getComponentBiasEnergies(extension)
    total = sampler.getBiasEnergy(context)
    assert (opes + explore).value_in_unit(unit.kilojoules_per_mole) == pytest.approx(
        total.value_in_unit(unit.kilojoules_per_mole), abs=1e-9
    )
    assert sampler.getCollectiveVariables(simulation) == sampler.getCollectiveVariables(
        _extensionSimulation(simulation)
    )
    sampler.updateContext(context)
    sampler.updateContext(extension)
    assert sampler.getFreeEnergy().shape == (sampler.variables[0].gridWidth,)


def test_combined_opes_matches_plain_combined_opes():
    dv = _circular_dv()
    model, system = _alanine_system(dv)
    plain_extension = openmm.XmlSerializer.clone(system.getExtensionSystem())
    frequency = 10
    sampler = _combined(
        system,
        [xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)],
        frequency=frequency,
        barrier=5 * unit.kilojoules_per_mole,
        exploreBarrier=4 * unit.kilojoules_per_mole,
        compressionThreshold=0,
    )
    simulation = _simulation(model, system)
    values, energies = [], []
    for _ in range(8):
        sampler.step(simulation, frequency)
        (value,) = simulation.context.getDynamicalVariableValues()
        values.append(value.value_in_unit(unit.radian))
        energies.append(sampler.getComponentBiasEnergies(simulation.context))
    bias_variable, bounded = _plain_bias_variable(dv, 18 * unit.degree)
    plain = openmm_opes.CombinedOPES(
        plain_extension,
        [bias_variable],
        temperature=300 * KELVIN,
        barrier=5 * unit.kilojoules_per_mole,
        exploreBarrier=4 * unit.kilojoules_per_mole,
        frequency=frequency,
        varianceFrequency=None,
        bounded=bounded,
        compressionThreshold=0,
    )
    context = openmm.Context(
        plain_extension, openmm.VerletIntegrator(1 * unit.femtosecond), REFERENCE
    )
    context.setPositions([openmm.Vec3(0, 0, 0)])
    replay = _Replay(context, values, frequency)
    for expected in energies:
        plain.step(replay, frequency)
        for actual, value in zip(plain.getComponentBiasEnergies(context), expected):
            assert actual.value_in_unit(KJ) == pytest.approx(
                value.value_in_unit(KJ), abs=1e-9
            )
    assert abs(energies[-1][0].value_in_unit(KJ)) > 1e-3
    assert plain.getNumKernels() == sampler.getNumKernels() == (8, 8)
    np.testing.assert_allclose(plain.getBias(), sampler.getBias(), rtol=1e-10)


def test_samplers_step_together_on_their_own_schedules():
    dv = _circular_dv()
    model, system = _alanine_system(dv)
    variable = xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)
    opes = _opes(system, [variable], frequency=10)
    combined = _combined(system, [variable], frequency=20)
    groups = {opes._force.getForceGroup(), combined._force.getForceGroup()}
    assert len(groups) == 2
    simulation = _simulation(model, system)
    xps.ExtendedSpaceOPES.stepTogether(simulation, [opes, combined], 40)
    xps.ExtendedSpaceCombinedOPES.stepTogether(simulation, [opes, combined], 20)
    assert simulation.currentStep == 60
    assert opes.getNumKernels() >= 1
    assert all(n >= 1 for n in combined.getNumKernels())
    # Each sampler's bias is in its own group, so they add up in the extension context
    extension = simulation.context.getExtensionContext()
    both = extension.getState(getEnergy=True, groups=groups).getPotentialEnergy()
    sum_of_parts = opes.getBiasEnergy(simulation.context) + opes.barrier
    sum_of_parts += combined.getBiasEnergy(simulation.context)
    sum_of_parts += combined.barrier + combined.exploreBarrier
    assert both.value_in_unit(unit.kilojoules_per_mole) == pytest.approx(
        sum_of_parts.value_in_unit(unit.kilojoules_per_mole), abs=1e-9
    )


def test_state_round_trip_restores_the_bias():
    dv = _circular_dv()
    model, system = _alanine_system(dv)
    variable = xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)
    sampler = _opes(system, [variable])
    sampler.step(_simulation(model, system), 40)
    state = sampler.getState()
    _, fresh_system = _alanine_system(dv)
    restored = _opes(fresh_system, [variable])
    restored.setState(state)
    assert restored.getNumKernels() == sampler.getNumKernels()
    np.testing.assert_allclose(restored.getBias(), sampler.getBias())


def test_two_walkers_share_a_bias(tmp_path):
    dv = _circular_dv()
    walkers = []
    for walker_id in (1, 2):
        model, system = _alanine_system(dv)
        sampler = _opes(
            system,
            [xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)],
            saveFrequency=20,
            biasDir=str(tmp_path),
            walkerId=walker_id,
            compressionThreshold=0,  # no merging, so kernel counts add up
        )
        walkers.append((sampler, _simulation(model, system, seed=walker_id)))
    for sampler, simulation in walkers:
        sampler.step(simulation, 40)
    for sampler, simulation in walkers:
        sampler.step(simulation, 20)  # syncs at step 60 and reads the other's file
    # Each walker deposited 6 kernels of its own; the rest came from its peer
    for sampler, _ in walkers:
        assert sampler.getNumKernels() > 6
