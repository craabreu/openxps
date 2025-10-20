"""
Unit tests for the openxps.simulation module.
"""

import io
import os
import tempfile
from copy import deepcopy
from math import pi

import cvpack
import openmm as mm
import pytest
from openmm import unit as mmunit
from openmmtools import testsystems

import openxps as xps
from openxps.context import ExtendedSpaceContext
from openxps.simulation import ExtendedSpaceSimulation
from openxps.integrators import InTandemIntegrator


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
        300 * mmunit.kelvin, 1 / mmunit.picosecond, 4 * mmunit.femtosecond
    )
    integrator.setRandomNumberSeed(1234)
    return model, [phi0], umbrella_potential, InTandemIntegrator(integrator)


def test_basic_initialization():
    """Test basic initialization with single integrator."""
    model, dvs, coupling_potential, integrator = create_test_system()

    simulation = ExtendedSpaceSimulation(
        dvs,
        coupling_potential,
        model.topology,
        model.system,
        integrator,
    )

    assert simulation is not None
    assert simulation.topology == model.topology
    assert simulation.system == model.system
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

    simulation = ExtendedSpaceSimulation(
        dvs,
        coupling_potential,
        model.topology,
        model.system,
        InTandemIntegrator(integrator.getPhysicalIntegrator(), extension_integrator),
    )

    assert simulation is not None
    assert simulation.context is not None


def test_with_platform():
    """Test initialization with explicit platform."""
    model, dvs, coupling_potential, integrator = create_test_system()
    platform = mm.Platform.getPlatformByName("Reference")

    simulation = ExtendedSpaceSimulation(
        dvs,
        coupling_potential,
        model.topology,
        model.system,
        integrator,
        platform=platform,
    )

    assert simulation is not None
    assert simulation.context.getPlatform().getName() == "Reference"


def test_with_platform_properties():
    """Test initialization with platform and properties."""
    model, dvs, coupling_potential, integrator = create_test_system()
    platform = mm.Platform.getPlatformByName("Reference")
    properties = {}

    simulation = ExtendedSpaceSimulation(
        dvs,
        coupling_potential,
        model.topology,
        model.system,
        integrator,
        platform=platform,
        platformProperties=properties,
    )

    assert simulation is not None


def test_context_is_extended():
    """Confirm context is an ExtendedSpaceContext instance."""
    model, dvs, coupling_potential, integrator = create_test_system()

    simulation = ExtendedSpaceSimulation(
        dvs,
        coupling_potential,
        model.topology,
        model.system,
        integrator,
    )

    assert isinstance(simulation.context, ExtendedSpaceContext)
    assert simulation.context.getDynamicalVariables() == tuple(dvs)
    assert simulation.context.getCouplingPotential() == coupling_potential


def test_inherited_step_method():
    """Verify step() method works correctly."""
    model, dvs, coupling_potential, integrator = create_test_system()

    simulation = ExtendedSpaceSimulation(
        dvs,
        coupling_potential,
        model.topology,
        model.system,
        integrator,
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

    simulation = ExtendedSpaceSimulation(
        dvs,
        coupling_potential,
        model.topology,
        model.system,
        integrator,
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

    simulation = ExtendedSpaceSimulation(
        dvs,
        coupling_potential,
        model.topology,
        model.system,
        integrator,
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

    simulation = ExtendedSpaceSimulation(
        dvs,
        coupling_potential,
        model.topology,
        model.system,
        integrator,
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


def test_invalid_dynamical_variables():
    """Test error handling for invalid dynamical variables."""
    model, _, coupling_potential, integrator = create_test_system()

    with pytest.raises(TypeError):
        ExtendedSpaceSimulation(
            [None],  # Invalid DV
            coupling_potential,
            model.topology,
            model.system,
            integrator,
        )


def test_invalid_coupling_potential():
    """Test error handling for invalid coupling potential."""
    model, dvs, _, integrator = create_test_system()

    with pytest.raises(TypeError):
        ExtendedSpaceSimulation(
            dvs,
            None,  # Invalid coupling potential
            model.topology,
            model.system,
            integrator,
        )


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
        300 * mmunit.kelvin, 1 / mmunit.picosecond, 4 * mmunit.femtosecond
    )

    with pytest.raises(ValueError) as excinfo:
        ExtendedSpaceSimulation(
            [phi0],
            coupling_potential,
            model.topology,
            model.system,
            integrator,
        )
    assert "dynamical variables are not coupling potential parameters" in str(
        excinfo.value
    )


def test_simulation_with_state():
    """Test initialization with a state parameter."""
    model, dvs, coupling_potential, integrator1 = create_test_system()

    # Create first simulation and run it
    sim1 = ExtendedSpaceSimulation(
        dvs,
        deepcopy(coupling_potential),
        model.topology,
        deepcopy(model.system),
        deepcopy(integrator1),
    )
    sim1.context.setPositions(model.positions)
    sim1.context.setVelocitiesToTemperature(300 * mmunit.kelvin, 1234)
    sim1.context.setDynamicalVariableValues([180 * mmunit.degree])
    sim1.step(10)

    # Get state
    state = sim1.context.getState(getPositions=True, getVelocities=True)

    # Create second simulation with the saved state
    integrator2 = mm.LangevinMiddleIntegrator(
        300 * mmunit.kelvin, 1 / mmunit.picosecond, 4 * mmunit.femtosecond
    )
    integrator2.setRandomNumberSeed(1234)

    sim2 = ExtendedSpaceSimulation(
        dvs,
        deepcopy(coupling_potential),
        model.topology,
        deepcopy(model.system),
        InTandemIntegrator(integrator2),
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
