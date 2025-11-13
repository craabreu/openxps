"""
Unit tests for the openxps.context module.
"""

import cvpack
import numpy as np
import openmm as mm
import pytest
from openmm import unit as mmunit
from openmmtools import testsystems

from openxps import (
    CollectiveVariableCoupling,
    DynamicalVariable,
    ExtendedSpaceContext,
    ExtendedSpaceSystem,
    LockstepIntegrator,
)
from openxps.bounds import CircularBounds, NoBounds, ReflectiveBounds
from openxps.utils import BINARY_SEPARATOR


def system_integrator_platform(coupling, model):
    """Helper function to create a basic OpenMM Context object."""
    integrator = mm.VerletIntegrator(1.0 * mmunit.femtosecond)
    platform = mm.Platform.getPlatformByName("Reference")
    return (
        ExtendedSpaceSystem(model.system, coupling),
        LockstepIntegrator(integrator),
        platform,
    )


def create_dvs():
    """Helper function to create a DynamicalVariable object."""
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    kwargs = {"unit": mmunit.nanometer, "mass": 1 * mmunit.dalton}
    return [
        DynamicalVariable(
            name="phi0", unit=mmunit.radian, mass=mass, bounds=CircularBounds()
        ),
        DynamicalVariable(
            name="x0", bounds=NoBounds(0, 1, mmunit.dimensionless), **kwargs
        ),
        DynamicalVariable(
            name="y0", bounds=ReflectiveBounds(-1, 1, mmunit.nanometer), **kwargs
        ),
    ]


def create_coupling(phi0=180 * mmunit.degrees):
    """Helper function to create a CollectiveVariableCoupling object."""
    dvs = create_dvs()
    kwargs = {
        "kappa": 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2,
        "alpha": 0.01 * mmunit.kilojoule_per_mole / mmunit.nanometer**2,
    }
    if phi0 is not None:
        kwargs["phi0"] = phi0
    # Always include x0 and y0 as parameters (they're in the expression)
    kwargs["x0"] = 1 * mmunit.nanometer
    kwargs["y0"] = 1 * mmunit.nanometer
    return CollectiveVariableCoupling(
        f"0.5*kappa*min(delta_phi,{2 * np.pi}-delta_phi)^2+alpha*(x0-y0)^2"
        "; delta_phi=abs(phi-phi0)",
        [cvpack.Torsion(6, 8, 14, 16, name="phi")],
        dvs if phi0 is not None else [],
        **kwargs,
    )


def create_extended_context(model, coupling=None):
    """Helper function to create an ExtendedSpaceContext object."""
    return ExtendedSpaceContext(
        ExtendedSpaceSystem(
            model.system,
            coupling or create_coupling(),
        ),
        LockstepIntegrator(mm.VerletIntegrator(1.0 * mmunit.femtosecond)),
        mm.Platform.getPlatformByName("Reference"),
    )


def test_initialization():
    """Test the initialization of ExtendedSpaceContext with basic setup."""
    assert create_extended_context(testsystems.AlanineDipeptideVacuum()) is not None


def test_set_positions_and_velocities():
    """Test setting positions and velocities including DVs."""
    model = testsystems.AlanineDipeptideVacuum()
    context = create_extended_context(model)

    random = np.random.RandomState()
    num_atoms = context.getSystem().getNumParticles()
    positions = model.positions.value_in_unit(mmunit.nanometer)
    velocities = random.uniform(-1, 1, (num_atoms, 3))

    urad = mmunit.radian
    unm = mmunit.nanometer
    ups = mmunit.picosecond

    context.setPositions(positions)
    context.setVelocities(velocities)
    context.setDynamicalVariableValues([1 * urad, 0.1 * unm, 0.1 * unm])
    context.setDynamicalVariableVelocities(
        [1 * urad / ups, 1 * unm / ups, 1 * unm / ups]
    )

    state = context.getState(getPositions=True, getVelocities=True)
    assert state.getPositions(asNumpy=True) == pytest.approx(positions) * unm
    assert state.getVelocities() == pytest.approx(velocities) * unm / ups
    assert context.getDynamicalVariableValues() == (
        pytest.approx(1) * urad,
        pytest.approx(0.1) * unm,
        pytest.approx(0.1) * unm,
    )
    assert context.getDynamicalVariableVelocities() == (
        pytest.approx(1) * urad / ups,
        pytest.approx(1) * unm / ups,
        pytest.approx(1) * unm / ups,
    )


def test_raise_exceptions():
    """Test raising exceptions for invalid operations."""
    model = testsystems.AlanineDipeptideVacuum()
    context = create_extended_context(model)

    # with pytest.raises(mm.OpenMMException) as e:
    #     context.setVelocitiesToTemperature(300 * mmunit.kelvin)
    # assert "Particle positions have not been set" in str(e.value)

    context.setPositions(model.positions)
    context.setVelocitiesToTemperature(300 * mmunit.kelvin)

    with pytest.raises(mm.OpenMMException) as e:
        context.setDynamicalVariableVelocitiesToTemperature(300 * mmunit.kelvin)
    assert "Particle positions have not been set" in str(e.value)

    with pytest.raises(mm.OpenMMException) as e:
        context.getIntegrator().step(1)
    assert "Particle positions have not been set" in str(e.value)

    context.setDynamicalVariableValues(
        [1 * mmunit.radian, 0.1 * mmunit.nanometer, 0.1 * mmunit.nanometer]
    )
    context.setDynamicalVariableVelocitiesToTemperature(300 * mmunit.kelvin)

    state = context.getState(getVelocities=True)
    velocities = state.getVelocities()
    extra_velocities = context.getDynamicalVariableVelocities()
    assert len(velocities) == len(model.positions)
    assert len(extra_velocities) == 3


def test_validation():
    """Test the validation of extended space context."""
    model = testsystems.AlanineDipeptideVacuum()

    # Test invalid DVs by corrupting the coupling
    bad_coupling = create_coupling()
    bad_coupling._dynamical_variables = [None]
    # Will raise AttributeError when trying to access .mass on None
    with pytest.raises(AttributeError):
        ExtendedSpaceContext(*system_integrator_platform(bad_coupling, model))

    # Test invalid coupling (None)
    with pytest.raises(AttributeError) as e:
        ExtendedSpaceContext(*system_integrator_platform(None, model))
    assert "'NoneType' object has no attribute 'addToPhysicalSystem'" in str(e.value)

    # Test coupling with no DVs (will raise error about conflicting parameters
    # since x0 and y0 are parameters but not DVs)
    with pytest.raises(Exception) as e:
        ExtendedSpaceContext(
            *system_integrator_platform(create_coupling(phi0=None), model),
        )
    assert "different default values" in str(e.value).lower()


def test_consistency():
    """Test the consistency of the extended space context."""
    model = testsystems.AlanineDipeptideVacuum()
    coupling = create_coupling()
    context = create_extended_context(model, coupling)
    context.setPositions(model.positions)
    context.setVelocitiesToTemperature(300 * mmunit.kelvin)
    context.setDynamicalVariableValues(
        [1000 * mmunit.degrees, 1 * mmunit.nanometer, 1 * mmunit.nanometer]
    )
    context.setDynamicalVariableVelocitiesToTemperature(300 * mmunit.kelvin)

    for _ in range(10):
        context.getIntegrator().step(1000)

        extension_state = context.getExtensionContext().getState(
            getEnergy=True, getPositions=True, getForces=True
        )

        # Check the consistency of the potential energy
        x1 = extension_state.getPotentialEnergy() / mmunit.kilojoule_per_mole
        x2 = coupling.getForce(0).getValue(context) / mmunit.kilojoule_per_mole
        assert x1 == pytest.approx(x2)


def test_checkpoint_creation_and_loading():
    """Test creating and loading checkpoints preserves context state."""
    model = testsystems.AlanineDipeptideVacuum()
    context = create_extended_context(model)

    # Set up initial state
    context.setPositions(model.positions)
    context.setVelocitiesToTemperature(300 * mmunit.kelvin, 1234)
    initial_dv_values = [
        1 * mmunit.radian,
        0.5 * mmunit.nanometer,
        -0.3 * mmunit.nanometer,
    ]
    context.setDynamicalVariableValues(initial_dv_values)
    context.setDynamicalVariableVelocitiesToTemperature(300 * mmunit.kelvin, 5678)

    # Run a few steps to evolve the system
    context.getIntegrator().step(10)

    # Get state before checkpoint
    state_before = context.getState(
        getPositions=True, getVelocities=True, getEnergy=True
    )
    positions_before = state_before.getPositions(asNumpy=True)
    velocities_before = state_before.getVelocities(asNumpy=True)
    dv_values_before = context.getDynamicalVariableValues()
    dv_velocities_before = context.getDynamicalVariableVelocities()
    energy_before = state_before.getPotentialEnergy()

    # Create checkpoint
    checkpoint = context.createCheckpoint()
    assert isinstance(checkpoint, bytes)
    assert BINARY_SEPARATOR in checkpoint

    # Modify the state significantly
    context.getIntegrator().step(100)
    new_positions = positions_before + 0.1 * mmunit.nanometer
    context.setPositions(new_positions)
    context.setDynamicalVariableValues(
        [2 * mmunit.radian, 1.0 * mmunit.nanometer, 0.5 * mmunit.nanometer]
    )

    # Verify state has changed
    state_after_change = context.getState(getPositions=True, getEnergy=True)
    positions_after_change = state_after_change.getPositions(asNumpy=True)
    assert not np.allclose(
        positions_before.value_in_unit(mmunit.nanometer),
        positions_after_change.value_in_unit(mmunit.nanometer),
    )

    # Load checkpoint
    context.loadCheckpoint(checkpoint)

    # Get state after loading checkpoint
    state_after = context.getState(
        getPositions=True, getVelocities=True, getEnergy=True
    )
    positions_after = state_after.getPositions(asNumpy=True)
    velocities_after = state_after.getVelocities(asNumpy=True)
    dv_values_after = context.getDynamicalVariableValues()
    dv_velocities_after = context.getDynamicalVariableVelocities()
    energy_after = state_after.getPotentialEnergy()

    # Assert that state is restored
    np.testing.assert_allclose(positions_before, positions_after, rtol=1e-10)
    np.testing.assert_allclose(velocities_before, velocities_after, rtol=1e-10)
    assert energy_before.value_in_unit(mmunit.kilojoule_per_mole) == pytest.approx(
        energy_after.value_in_unit(mmunit.kilojoule_per_mole)
    )

    # Check dynamical variables
    for val_before, val_after in zip(dv_values_before, dv_values_after):
        assert val_before.value_in_unit(val_before.unit) == pytest.approx(
            val_after.value_in_unit(val_after.unit)
        )

    for vel_before, vel_after in zip(dv_velocities_before, dv_velocities_after):
        assert vel_before.value_in_unit(vel_before.unit) == pytest.approx(
            vel_after.value_in_unit(vel_after.unit)
        )


def test_checkpoint_file_save_load(tmp_path):
    """Test saving checkpoint to file and loading from file."""
    model = testsystems.AlanineDipeptideVacuum()
    context = create_extended_context(model)

    # Set up initial state
    context.setPositions(model.positions)
    context.setVelocitiesToTemperature(300 * mmunit.kelvin, 1234)
    context.setDynamicalVariableValues(
        [1 * mmunit.radian, 0.5 * mmunit.nanometer, -0.3 * mmunit.nanometer]
    )
    context.setDynamicalVariableVelocitiesToTemperature(300 * mmunit.kelvin, 5678)
    context.getIntegrator().step(10)

    # Get state before checkpoint
    state_before = context.getState(getPositions=True, getVelocities=True)
    positions_before = state_before.getPositions(asNumpy=True)
    dv_values_before = context.getDynamicalVariableValues()

    # Create checkpoint and save to file
    checkpoint = context.createCheckpoint()
    checkpoint_file = tmp_path / "test_checkpoint.chk"
    with open(checkpoint_file, "wb") as f:
        f.write(checkpoint)

    # Verify file exists and has content
    assert checkpoint_file.exists()
    assert checkpoint_file.stat().st_size > 0

    # Modify state
    context.getIntegrator().step(100)

    # Load checkpoint from file
    with open(checkpoint_file, "rb") as f:
        loaded_checkpoint = f.read()
    context.loadCheckpoint(loaded_checkpoint)

    # Verify state is restored
    state_after = context.getState(getPositions=True, getVelocities=True)
    positions_after = state_after.getPositions(asNumpy=True)
    dv_values_after = context.getDynamicalVariableValues()

    np.testing.assert_allclose(positions_before, positions_after, rtol=1e-10)
    for val_before, val_after in zip(dv_values_before, dv_values_after):
        assert val_before.value_in_unit(val_before.unit) == pytest.approx(
            val_after.value_in_unit(val_after.unit)
        )


def test_context_with_platform_properties():
    """Test ExtendedSpaceContext with platform and properties."""
    model = testsystems.AlanineDipeptideVacuum()
    platform = mm.Platform.getPlatformByName("CPU")
    properties = {"Threads": "1"}

    coupling = create_coupling()
    integrator = mm.VerletIntegrator(1.0 * mmunit.femtosecond)

    context = ExtendedSpaceContext(
        ExtendedSpaceSystem(model.system, coupling),
        LockstepIntegrator(integrator),
        platform,
        properties,
    )

    assert context is not None
    assert context.getPlatform().getName() == "CPU"


def test_invalid_integrator_type():
    """Test that non-ExtendedSpaceIntegrator raises TypeError."""
    model = testsystems.AlanineDipeptideVacuum()
    coupling = create_coupling()
    integrator = mm.VerletIntegrator(1.0 * mmunit.femtosecond)

    with pytest.raises(
        TypeError, match="The integrator must be an instance of ExtendedSpaceIntegrator"
    ):
        ExtendedSpaceContext(
            ExtendedSpaceSystem(model.system, coupling),
            integrator,  # Wrong type - not wrapped in ExtendedSpaceIntegrator
        )


def test_set_protected_parameter():
    """Test setParameter for dynamical variables."""
    model = testsystems.AlanineDipeptideVacuum()
    context = create_extended_context(model)

    # The parameter name matches a dynamical variable (i.e., it is protected)
    protected_param = "phi0"
    with pytest.raises(
        ValueError,
        match=(
            f'Cannot manually set the parameter "{protected_param}". '
            "This parameter is set automatically via setDynamicalVariableValues."
        ),
    ):
        context.setParameter(protected_param, 2.0)


def test_set_parameter_for_regular_param():
    """Test setParameter for non-DV parameters."""
    model = testsystems.AlanineDipeptideVacuum()
    context = create_extended_context(model)

    # Initialize context
    context.setPositions(model.positions)

    # Set a regular (non-DV) parameter
    original_kappa = context.getParameter("kappa")
    new_kappa = 2000.0
    context.setParameter("kappa", new_kappa)
    assert context.getParameter("kappa") == pytest.approx(new_kappa)
    assert context.getParameter("kappa") != pytest.approx(original_kappa)


def test_set_dv_values_without_units():
    """Test setDynamicalVariableValues with raw floats."""
    model = testsystems.AlanineDipeptideVacuum()
    context = create_extended_context(model)

    # Initialize context
    context.setPositions(model.positions)

    # Set DV values without units (raw floats)
    dv_values_raw = [1.5, 0.3, -0.2]
    context.setDynamicalVariableValues(dv_values_raw)

    # Retrieve and verify
    dv_values = context.getDynamicalVariableValues()
    dvs = create_dvs()

    for i, (val, dv) in enumerate(zip(dv_values, dvs)):
        assert val.value_in_unit(dv.unit) == pytest.approx(dv_values_raw[i])
