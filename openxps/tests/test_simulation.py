"""
Unit tests for the openxps.simulation module.
"""

import io
import tempfile
from copy import deepcopy

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
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    phi0 = xps.DynamicalVariable("phi0", mmunit.radian, mass, xps.CircularBounds())
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    kappa = 1000 * mmunit.kilojoules_per_mole / mmunit.radian**2
    harmonic_force = xps.HarmonicCoupling(phi, phi0, kappa)
    integrator = mm.LangevinMiddleIntegrator(
        300 * mmunit.kelvin, 1 / mmunit.picosecond, 1 * mmunit.femtosecond
    )
    integrator.setRandomNumberSeed(1234)
    return model, [phi0], harmonic_force, integrator


def test_basic_initialization():
    """Test basic initialization with single integrator."""
    model, dvs, coupling, integrator = create_test_system()

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling),
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
    model, dvs, coupling, integrator = create_test_system()

    # Create a second integrator for the extension system
    extension_integrator = mm.LangevinMiddleIntegrator(
        300 * mmunit.kelvin, 1 / mmunit.picosecond, 1 * mmunit.femtosecond
    )
    extension_integrator.setRandomNumberSeed(5678)

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling),
        xps.LockstepIntegrator(integrator, extension_integrator),
    )

    assert simulation is not None
    assert simulation.context is not None


def test_with_platform():
    """Test initialization with explicit platform."""
    model, dvs, coupling, integrator = create_test_system()
    platform = mm.Platform.getPlatformByName("Reference")

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling),
        xps.LockstepIntegrator(integrator),
        platform=platform,
    )

    assert simulation is not None
    assert simulation.context.getPlatform().getName() == "Reference"


def test_with_platform_properties():
    """Test initialization with platform and properties."""
    model, dvs, coupling, integrator = create_test_system()
    platform = mm.Platform.getPlatformByName("Reference")
    properties = {}

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling),
        xps.LockstepIntegrator(integrator),
        platform=platform,
        platformProperties=properties,
    )

    assert simulation is not None


def test_context_is_extended():
    """Confirm context is an xps.ExtendedSpaceContext instance."""
    model, dvs, coupling, integrator = create_test_system()

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling),
        xps.LockstepIntegrator(integrator),
        mm.Platform.getPlatformByName("Reference"),
    )

    assert isinstance(simulation.context, xps.ExtendedSpaceContext)
    assert simulation.context.getSystem().getDynamicalVariables() == tuple(dvs)
    assert simulation.context.getSystem().getCoupling() == coupling


def test_inherited_step_method():
    """Verify step() method works correctly."""
    model, dvs, coupling, integrator = create_test_system()

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling),
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
    model, dvs, coupling, integrator = create_test_system()

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling),
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
    model, dvs, coupling, integrator = create_test_system()

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling),
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
        writers=[xps.ExtensionWriter(kinetic=True)],
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


def test_save_and_load_checkpoint(tmp_path):
    """Test saveCheckpoint and loadCheckpoint methods."""
    model, dvs, coupling, integrator = create_test_system()

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling),
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
    checkpoint_file = str(tmp_path / "checkpoint.chk")
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


def test_missing_dv_in_coupling():
    """Test that simulation with coupling having no DVs fails
    (no particles in extension system)."""
    model = testsystems.AlanineDipeptideVacuum()
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")

    # Create coupling with no dynamical variables
    coupling = xps.CollectiveVariableCoupling(
        "0.5*kappa*phi^2",
        [phi],
        [],
        kappa=1000 * mmunit.kilojoule_per_mole / mmunit.radian**2,
    )

    integrator = mm.LangevinMiddleIntegrator(
        300 * mmunit.kelvin, 1 / mmunit.picosecond, 1 * mmunit.femtosecond
    )

    # Should raise an error because extension system has no particles
    with pytest.raises(Exception) as excinfo:
        xps.ExtendedSpaceSimulation(
            model.topology,
            xps.ExtendedSpaceSystem(model.system, coupling),
            xps.LockstepIntegrator(integrator),
            mm.Platform.getPlatformByName("Reference"),
        )
    assert "no particles" in str(excinfo.value).lower()


def test_simulation_with_state():
    """Test initialization with a state parameter."""
    model, dvs, coupling, integrator1 = create_test_system()

    # Create first simulation and run it
    sim1 = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(deepcopy(model.system), deepcopy(coupling)),
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
        xps.ExtendedSpaceSystem(deepcopy(model.system), deepcopy(coupling)),
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


def test_loadcheckpoint_comprehensive(tmp_path):
    """Test loadCheckpoint method comprehensively restores all state components."""
    model, dvs, coupling, integrator = create_test_system()

    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling),
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
    checkpoint_file = str(tmp_path / "checkpoint.chk")
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


def _simulation():
    model, _, coupling, integrator = create_test_system()
    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling),
        xps.LockstepIntegrator(integrator),
        mm.Platform.getPlatformByName("Reference"),
    )
    simulation.context.setPositions(model.positions)
    return simulation, coupling


def test_minimize_energy_resyncs_extension():
    """After minimization the extension context sees the new CV values."""
    simulation, coupling = _simulation()
    simulation.context.setDynamicalVariableValues([3.0])
    before = simulation.context.getExtensionContext().getParameter("phi")
    simulation.minimizeEnergy(maxIterations=50)
    expected = coupling.getCollectiveVariableValues(simulation.context)["phi"]
    actual = simulation.context.getExtensionContext().getParameter("phi")
    assert expected != pytest.approx(before)
    assert actual == pytest.approx(expected)


def test_save_and_load_state_roundtrip_dvs(tmp_path):
    """saveState/loadState carry the dynamical variables."""
    simulation, _ = _simulation()
    simulation.context.setDynamicalVariableValues([2.5])
    simulation.context.setDynamicalVariableVelocities([0.4])
    path = str(tmp_path / "state.xml")
    simulation.saveState(path)
    simulation.context.setDynamicalVariableValues([-1.0])
    simulation.context.setDynamicalVariableVelocities([0.0])
    simulation.loadState(path)
    (value,) = simulation.context.getDynamicalVariableValues()
    (rate,) = simulation.context.getDynamicalVariableVelocities()
    assert value.value_in_unit(mmunit.radian) == pytest.approx(2.5)
    assert rate.value_in_unit(mmunit.radian / mmunit.picosecond) == pytest.approx(0.4)
    assert simulation.context.getParameter("phi0") == pytest.approx(2.5)


@pytest.mark.parametrize("mode", ["path", "text", "binary"])
def test_save_and_load_state_file_kinds(tmp_path, mode):
    """saveState/loadState accept paths and text or binary file objects."""
    simulation, _ = _simulation()
    simulation.context.setDynamicalVariableValues([2.5])
    path = tmp_path / "state.xml"
    if mode == "path":
        simulation.saveState(path)
    else:
        with open(path, "w" if mode == "text" else "wb") as file:
            simulation.saveState(file)
    simulation.context.setDynamicalVariableValues([-1.0])
    if mode == "path":
        simulation.loadState(path)
    else:
        with open(path, "r" if mode == "text" else "rb") as file:
            simulation.loadState(file)
    (value,) = simulation.context.getDynamicalVariableValues()
    assert value.value_in_unit(mmunit.radian) == pytest.approx(2.5)


@pytest.mark.parametrize("mode", ["w+", "w+b"])
def test_save_and_load_state_temporary_files(mode):
    """Wrapped streams such as temporary files work in text and binary mode."""
    simulation, _ = _simulation()
    simulation.context.setDynamicalVariableValues([2.5])
    with tempfile.SpooledTemporaryFile(mode=mode) as spooled:
        with tempfile.NamedTemporaryFile(mode=mode) as named:
            for file in (spooled, named):
                simulation.saveState(file)
                file.seek(0)
                simulation.context.setDynamicalVariableValues([-1.0])
                simulation.loadState(file)
                (value,) = simulation.context.getDynamicalVariableValues()
                assert value.value_in_unit(mmunit.radian) == pytest.approx(2.5)


def test_state_argument_accepts_filename(tmp_path):
    """Like OpenMM's Simulation, state= may be a file name."""
    simulation, _ = _simulation()
    simulation.context.setDynamicalVariableValues([1.0])
    path = str(tmp_path / "state.xml")
    simulation.saveState(path)
    model, _, coupling, integrator = create_test_system()
    restored = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling),
        xps.LockstepIntegrator(integrator),
        mm.Platform.getPlatformByName("Reference"),
        state=path,
    )
    (value,) = restored.context.getDynamicalVariableValues()
    assert value.value_in_unit(mmunit.radian) == pytest.approx(1.0)


def test_state_argument_accepts_state_object():
    """A State passed at construction leaves both contexts in sync."""
    simulation, coupling = _simulation()
    state = simulation.context.getState(getPositions=True)
    model, _, coupling2, integrator = create_test_system()
    restored = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling2),
        xps.LockstepIntegrator(integrator),
        mm.Platform.getPlatformByName("Reference"),
        state=state,
    )
    expected = coupling2.getCollectiveVariableValues(restored.context)["phi"]
    actual = restored.context.getExtensionContext().getParameter("phi")
    assert actual == pytest.approx(expected)


def test_save_and_load_state_keeps_extension_thermostat():
    """The extension integrator's thermostat variables survive a state file."""

    def build():
        model, _, coupling, integrator = create_test_system()
        extension_integrator = xps.integrators.MassiveGGMTIntegrator(
            300 * mmunit.kelvin,
            0.1 * mmunit.picosecond,
            1 * mmunit.femtosecond,
            forceFirst=True,
        )
        simulation = xps.ExtendedSpaceSimulation(
            model.topology,
            xps.ExtendedSpaceSystem(model.system, coupling),
            xps.LockstepIntegrator(integrator, extension_integrator),
            mm.Platform.getPlatformByName("Reference"),
        )
        simulation.context.setPositions(model.positions)
        simulation.context.setVelocitiesToTemperature(300 * mmunit.kelvin, 1)
        simulation.context.setDynamicalVariableValues([1.0])
        simulation.context.setDynamicalVariableVelocitiesToTemperature(
            300 * mmunit.kelvin, 1
        )
        return simulation

    def thermostat(simulation):
        integrator = simulation.context.getExtensionContext().getIntegrator()
        return np.array(
            [integrator.getPerDofVariableByName(name) for name in ("v1", "v2")]
        )

    original = build()
    original.step(50)
    saved = thermostat(original)
    assert np.any(saved != 0)
    buffer = io.StringIO()
    original.saveState(buffer)
    buffer.seek(0)
    restored = build()
    restored.loadState(buffer)
    np.testing.assert_array_equal(thermostat(restored), saved)
