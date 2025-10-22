"""
Unit tests for the openxps.simulation module.
"""

import io
import os
import tempfile
from copy import deepcopy
from math import pi

import cvpack
import numpy as np
import openmm as mm
import pytest
from openmm import unit as mmunit
from openmmtools import testsystems

import openxps as xps


def create_test_system():
    """Helper function to create a test system."""
    model = testsystems.AlanineDipeptideVacuum()
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    umbrella_potential = cvpack.MetaCollectiveVariable(
        f"0.5*kappa*min(delta,{2 * pi}-delta)^2; delta=abs(phi-phi0)",
        [phi],
        mmunit.kilojoule_per_mole,
        kappa=1000 * mmunit.kilojoule_per_mole / mmunit.radian**2,
        phi0=pi * mmunit.radian,
    )
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    phi0 = xps.DynamicalVariable("phi0", mmunit.radian, mass, xps.bounds.CIRCULAR)
    integrator = mm.LangevinMiddleIntegrator(
        300 * mmunit.kelvin, 1 / mmunit.picosecond, 1 * mmunit.femtosecond
    )
    integrator.setRandomNumberSeed(1234)
    return model, [phi0], umbrella_potential, integrator


def test_basic_initialization():
    """Test basic initialization with single integrator."""
    model, dvs, coupling_potential, integrator = create_test_system()

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(dvs, coupling_potential, model.system),
        xps.LockstepIntegrator(integrator),
    )

    assert simulation is not None
    assert simulation.topology == model.topology
    assert simulation.system is not None
    assert simulation.integrator is not None
    assert simulation.currentStep == 0
    assert simulation.reporters == []


def test_with_two_integrators():
    """Test initialization with two integrators."""
    model, dvs, coupling_potential, integrator = create_test_system()

    # Create a second integrator for the extension system
    extension_integrator = mm.LangevinMiddleIntegrator(
        300 * mmunit.kelvin, 1 / mmunit.picosecond, 1 * mmunit.femtosecond
    )
    extension_integrator.setRandomNumberSeed(5678)

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(dvs, coupling_potential, model.system),
        xps.LockstepIntegrator(integrator, extension_integrator),
    )

    assert simulation is not None
    assert simulation.context is not None


def test_with_platform():
    """Test initialization with explicit platform."""
    model, dvs, coupling_potential, integrator = create_test_system()
    platform = mm.Platform.getPlatformByName("Reference")

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(dvs, coupling_potential, model.system),
        xps.LockstepIntegrator(integrator),
        platform=platform,
    )

    assert simulation is not None
    assert simulation.context.getPlatform().getName() == "Reference"


def test_with_platform_properties():
    """Test initialization with platform and properties."""
    model, dvs, coupling_potential, integrator = create_test_system()
    platform = mm.Platform.getPlatformByName("Reference")
    properties = {}

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(dvs, coupling_potential, model.system),
        xps.LockstepIntegrator(integrator),
        platform=platform,
        platformProperties=properties,
    )

    assert simulation is not None


def test_context_is_extended():
    """Confirm context is an xps.ExtendedSpaceContext instance."""
    model, dvs, coupling_potential, integrator = create_test_system()

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(dvs, coupling_potential, model.system),
        xps.LockstepIntegrator(integrator),
        mm.Platform.getPlatformByName("Reference"),
    )

    assert isinstance(simulation.context, xps.ExtendedSpaceContext)
    assert simulation.context.getSystem().getDynamicalVariables() == tuple(dvs)
    assert simulation.context.getSystem().getCouplingPotential() == coupling_potential


def test_inherited_step_method():
    """Verify step() method works correctly."""
    model, dvs, coupling_potential, integrator = create_test_system()

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(dvs, coupling_potential, model.system),
        xps.LockstepIntegrator(integrator),
        mm.Platform.getPlatformByName("Reference"),
    )

    # Set initial conditions
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * mmunit.kelvin, 1234)
    simulation.context.setDynamicalVariableValues([180 * mmunit.degree])
    simulation.context.setDynamicalVariableVelocitiesToTemperature(
        300 * mmunit.kelvin, 1234
    )

    # Run simulation
    initial_step = simulation.currentStep
    simulation.step(10)

    # Check that the simulation advanced
    assert simulation.currentStep == initial_step + 10


def test_inherited_minimize_energy():
    """Verify minimizeEnergy() method works correctly."""
    model, dvs, coupling_potential, integrator = create_test_system()

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(dvs, coupling_potential, model.system),
        xps.LockstepIntegrator(integrator),
        mm.Platform.getPlatformByName("Reference"),
    )

    # Set initial conditions
    simulation.context.setPositions(model.positions)
    simulation.context.setDynamicalVariableValues([180 * mmunit.degree])

    # Get initial energy
    initial_state = simulation.context.getState(getEnergy=True)
    initial_energy = initial_state.getPotentialEnergy()

    # Minimize energy
    simulation.minimizeEnergy()

    # Get final energy
    final_state = simulation.context.getState(getEnergy=True)
    final_energy = final_state.getPotentialEnergy()

    # Energy should be reduced (or stay the same if already at minimum)
    assert final_energy <= initial_energy


def test_reporters():
    """Verify reporters work with the extended simulation."""
    model, dvs, coupling_potential, integrator = create_test_system()

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(dvs, coupling_potential, model.system),
        xps.LockstepIntegrator(integrator),
        mm.Platform.getPlatformByName("Reference"),
    )

    # Set initial conditions
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * mmunit.kelvin, 1234)
    simulation.context.setDynamicalVariableValues([180 * mmunit.degree])
    simulation.context.setDynamicalVariableVelocitiesToTemperature(
        300 * mmunit.kelvin, 1234
    )

    # Create a reporter
    stream = io.StringIO()
    reporter = cvpack.reporting.StateDataReporter(
        stream,
        10,
        step=True,
        potentialEnergy=True,
        kineticEnergy=True,
        writers=[xps.ExtensionWriter(simulation.context, kinetic=True)],
    )
    simulation.reporters.append(reporter)

    # Run simulation
    simulation.step(50)

    # Check that data was written
    stream.seek(0)
    output = stream.read()
    assert len(output) > 0
    assert "Step" in output
    assert "Potential Energy" in output
    assert "Extension Kinetic Energy" in output


def test_save_and_load_checkpoint():
    """Test saveCheckpoint and loadCheckpoint methods."""
    model, dvs, coupling_potential, integrator = create_test_system()

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(dvs, coupling_potential, model.system),
        xps.LockstepIntegrator(integrator),
        mm.Platform.getPlatformByName("Reference"),
    )

    # Set initial conditions
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * mmunit.kelvin, 1234)
    simulation.context.setDynamicalVariableValues([180 * mmunit.degree])
    simulation.context.setDynamicalVariableVelocitiesToTemperature(
        300 * mmunit.kelvin, 1234
    )

    # Run a few steps
    simulation.step(10)

    # Save state
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".xml") as f:
        checkpoint_file = f.name
        simulation.saveCheckpoint(checkpoint_file)

    # Get current DV values
    dv_values_before = simulation.context.getDynamicalVariableValues()

    # Run more steps
    simulation.step(10)

    # DV values should have changed
    dv_values_after = simulation.context.getDynamicalVariableValues()
    assert dv_values_before[0] != dv_values_after[0]

    # Load checkpoint
    simulation.loadCheckpoint(checkpoint_file)

    # DV values should be restored
    dv_values_restored = simulation.context.getDynamicalVariableValues()
    assert (
        dv_values_restored[0]
        == pytest.approx(dv_values_before[0].value_in_unit(mmunit.radian))
        * mmunit.radian
    )

    # Clean up
    os.unlink(checkpoint_file)


def test_missing_dv_in_coupling_potential():
    """Test error when DV is not in coupling potential parameters."""
    model = testsystems.AlanineDipeptideVacuum()
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")

    # Create coupling potential without phi0 parameter
    coupling_potential = cvpack.MetaCollectiveVariable(
        "0.5*kappa*phi^2",
        [phi],
        mmunit.kilojoule_per_mole,
        kappa=1000 * mmunit.kilojoule_per_mole / mmunit.radian**2,
    )

    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    phi0 = xps.DynamicalVariable("phi0", mmunit.radian, mass, xps.bounds.CIRCULAR)

    integrator = mm.LangevinMiddleIntegrator(
        300 * mmunit.kelvin, 1 / mmunit.picosecond, 1 * mmunit.femtosecond
    )

    with pytest.raises(ValueError) as excinfo:
        xps.ExtendedSpaceSimulation(
            model.topology,
            xps.ExtendedSpaceSystem([phi0], coupling_potential, model.system),
            xps.LockstepIntegrator(integrator),
            mm.Platform.getPlatformByName("Reference"),
        )
    assert "dynamical variables are not coupling potential parameters" in str(
        excinfo.value
    )


def test_simulation_with_state():
    """Test initialization with a state parameter."""
    model, dvs, coupling_potential, integrator1 = create_test_system()

    # Create first simulation and run it
    sim1 = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(
            dvs, deepcopy(coupling_potential), deepcopy(model.system)
        ),
        xps.LockstepIntegrator(deepcopy(integrator1)),
        mm.Platform.getPlatformByName("Reference"),
    )
    sim1.context.setPositions(model.positions)
    sim1.context.setVelocitiesToTemperature(300 * mmunit.kelvin, 1234)
    sim1.context.setDynamicalVariableValues([180 * mmunit.degree])
    sim1.step(10)

    # Get state
    state = sim1.context.getState(getPositions=True, getVelocities=True)

    # Create second simulation with the saved state
    integrator2 = mm.LangevinMiddleIntegrator(
        300 * mmunit.kelvin, 1 / mmunit.picosecond, 1 * mmunit.femtosecond
    )
    integrator2.setRandomNumberSeed(1234)

    sim2 = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(
            dvs, deepcopy(coupling_potential), deepcopy(model.system)
        ),
        xps.LockstepIntegrator(integrator2),
        mm.Platform.getPlatformByName("Reference"),
        state=state,
    )

    # Verify positions were loaded
    state1_pos = state.getPositions()
    state2_pos = sim2.context.getState(getPositions=True).getPositions()

    assert len(state1_pos) == len(state2_pos)
    for p1, p2 in zip(state1_pos, state2_pos):
        assert p1.x == pytest.approx(p2.x)
        assert p1.y == pytest.approx(p2.y)
        assert p1.z == pytest.approx(p2.z)


def test_loadcheckpoint_comprehensive():
    """Test loadCheckpoint method comprehensively restores all state components."""
    model, dvs, coupling_potential, integrator = create_test_system()

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(dvs, coupling_potential, model.system),
        xps.LockstepIntegrator(integrator),
        mm.Platform.getPlatformByName("Reference"),
    )

    # Set initial conditions
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * mmunit.kelvin, 1234)
    simulation.context.setDynamicalVariableValues([180 * mmunit.degree])
    simulation.context.setDynamicalVariableVelocitiesToTemperature(
        300 * mmunit.kelvin, 5678
    )

    # Run a few steps to evolve the system
    simulation.step(10)

    # Get complete state before checkpoint
    state_before = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getEnergy=True,
        getForces=True,
    )
    positions_before = state_before.getPositions(asNumpy=True)
    velocities_before = state_before.getVelocities(asNumpy=True)
    energy_before = state_before.getPotentialEnergy()
    kinetic_before = state_before.getKineticEnergy()
    dv_values_before = simulation.context.getDynamicalVariableValues()
    dv_velocities_before = simulation.context.getDynamicalVariableVelocities()
    current_step_before = simulation.currentStep

    # Save checkpoint
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".xml") as f:
        checkpoint_file = f.name
        simulation.saveCheckpoint(checkpoint_file)

    # Significantly modify the state
    simulation.step(100)
    new_dv_values = [90 * mmunit.degree]
    simulation.context.setDynamicalVariableValues(new_dv_values)
    simulation.context.setVelocitiesToTemperature(500 * mmunit.kelvin, 9999)

    # Verify state has changed
    state_changed = simulation.context.getState(getPositions=True, getEnergy=True)
    positions_changed = state_changed.getPositions(asNumpy=True)
    dv_values_changed = simulation.context.getDynamicalVariableValues()

    # Check that positions have changed significantly
    assert not np.allclose(
        positions_before.value_in_unit(mmunit.nanometer),
        positions_changed.value_in_unit(mmunit.nanometer),
        rtol=1e-8,
        atol=1e-10,
    )
    assert dv_values_before[0] != dv_values_changed[0]

    # Load checkpoint
    simulation.loadCheckpoint(checkpoint_file)

    # Get state after loading checkpoint
    state_after = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getEnergy=True,
        getForces=True,
    )
    positions_after = state_after.getPositions(asNumpy=True)
    velocities_after = state_after.getVelocities(asNumpy=True)
    energy_after = state_after.getPotentialEnergy()
    kinetic_after = state_after.getKineticEnergy()
    dv_values_after = simulation.context.getDynamicalVariableValues()
    dv_velocities_after = simulation.context.getDynamicalVariableVelocities()
    current_step_after = simulation.currentStep

    # Assert physical system state is restored
    np.testing.assert_allclose(
        positions_before.value_in_unit(mmunit.nanometer),
        positions_after.value_in_unit(mmunit.nanometer),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        velocities_before.value_in_unit(mmunit.nanometer / mmunit.picosecond),
        velocities_after.value_in_unit(mmunit.nanometer / mmunit.picosecond),
        rtol=1e-10,
        atol=1e-12,
    )

    # Assert energies are restored
    assert energy_before.value_in_unit(mmunit.kilojoule_per_mole) == pytest.approx(
        energy_after.value_in_unit(mmunit.kilojoule_per_mole), rel=1e-9
    )
    assert kinetic_before.value_in_unit(mmunit.kilojoule_per_mole) == pytest.approx(
        kinetic_after.value_in_unit(mmunit.kilojoule_per_mole), rel=1e-9
    )

    # Assert dynamical variables are restored
    for val_before, val_after in zip(dv_values_before, dv_values_after):
        assert val_before.value_in_unit(val_before.unit) == pytest.approx(
            val_after.value_in_unit(val_after.unit), rel=1e-10
        )

    for vel_before, vel_after in zip(dv_velocities_before, dv_velocities_after):
        assert vel_before.value_in_unit(vel_before.unit) == pytest.approx(
            vel_after.value_in_unit(vel_after.unit), rel=1e-10
        )

    # Assert current step is restored
    assert current_step_after == current_step_before

    # Clean up
    os.unlink(checkpoint_file)
