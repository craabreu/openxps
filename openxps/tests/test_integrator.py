"""
Unit tests for the openxps.integrator module.
"""

import math

import numpy as np
import openmm as mm
import pytest
from openmm import _openmm as mmswig
from openmm import unit as mmunit

import openxps as xps
from openxps import integrators
from openxps.integrator import LockstepIntegrator, SplitIntegrator

REFERENCE = mm.Platform.getPlatformByName("Reference")
POSITIONS = [mm.Vec3(0, 0, 0), mm.Vec3(0.2, 0, 0)] * mmunit.nanometer
VELOCITIES = (
    [mm.Vec3(0, 0, 0), mm.Vec3(0.3, 0.1, 0)] * mmunit.nanometer / mmunit.picosecond
)


# Helper functions
def create_force_first_integrator(step_size):
    """Create a LangevinMiddleIntegrator with given step size."""
    return mm.LangevinMiddleIntegrator(
        300 * mmunit.kelvin, 1 / mmunit.picosecond, step_size
    )


def create_symmetric_integrator(step_size):
    """Create a symmetric SymmetricVerletIntegrator with given step size."""
    return integrators.SymmetricVerletIntegrator(step_size)


def create_non_force_first_integrator(step_size):
    """Create a CustomIntegrator that is not in KNOWN_FORCE_FIRST_INTEGRATORS."""
    integrator = mm.CustomIntegrator(step_size)
    integrator.addPerDofVariable("x1", 0)
    integrator.addUpdateContextState()
    integrator.addComputePerDof("v", "v+dt*f/m")
    integrator.addComputePerDof("x", "x+dt*v")
    integrator.addConstrainPositions()
    integrator.addConstrainVelocities()
    return integrator


def create_non_symmetric_integrator(step_size):
    """Create a LangevinIntegrator that is not in KNOWN_SYMMETRIC_INTEGRATORS."""
    return mm.LangevinIntegrator(300 * mmunit.kelvin, 1 / mmunit.picosecond, step_size)


# LockstepIntegrator Tests


def test_lockstep_integrator_with_single_integrator():
    """Test LockstepIntegrator with only physical integrator."""
    step_size = 2.0 * mmunit.femtosecond
    physical_integrator = create_force_first_integrator(step_size)

    integrator = LockstepIntegrator(physical_integrator)

    # Verify both integrators exist
    assert integrator.getPhysicalIntegrator() is physical_integrator
    assert integrator.getExtensionIntegrator() is not physical_integrator

    # Verify both have same step size
    physical_dt = mmswig.Integrator_getStepSize(integrator.getPhysicalIntegrator())
    extension_dt = mmswig.Integrator_getStepSize(integrator.getExtensionIntegrator())
    assert physical_dt == pytest.approx(extension_dt)
    assert physical_dt == pytest.approx(step_size.value_in_unit(mmunit.picosecond))


def test_lockstep_integrator_with_two_integrators_same_step_size():
    """Test LockstepIntegrator with explicit physical and extension integrators."""
    step_size = 2.0 * mmunit.femtosecond
    physical_integrator = create_force_first_integrator(step_size)
    extension_integrator = create_force_first_integrator(step_size)

    integrator = LockstepIntegrator(physical_integrator, extension_integrator)

    assert integrator.getPhysicalIntegrator() is physical_integrator
    assert integrator.getExtensionIntegrator() is extension_integrator


def test_lockstep_integrator_error_different_step_sizes():
    """Test that different step sizes raise ValueError."""
    physical_integrator = create_force_first_integrator(2.0 * mmunit.femtosecond)
    extension_integrator = create_force_first_integrator(1.0 * mmunit.femtosecond)

    with pytest.raises(ValueError, match="The step sizes must be equal."):
        LockstepIntegrator(physical_integrator, extension_integrator)


def test_lockstep_integrator_error_non_force_first():
    """Test that non-force-first integrators raise ValueError."""
    step_size = 2.0 * mmunit.femtosecond
    physical_integrator = create_non_force_first_integrator(step_size)
    extension_integrator = create_non_force_first_integrator(step_size)

    with pytest.raises(ValueError, match="must follow a force-first"):
        LockstepIntegrator(physical_integrator, extension_integrator)


def test_lockstep_integrator_assume_force_first_bypass():
    """Test that assume_force_first=True bypasses validation."""
    step_size = 2.0 * mmunit.femtosecond
    physical_integrator = create_non_force_first_integrator(step_size)
    extension_integrator = create_non_force_first_integrator(step_size)

    # Should succeed with assume_force_first=True
    integrator = LockstepIntegrator(
        physical_integrator, extension_integrator, assume_force_first=True
    )

    assert integrator.getPhysicalIntegrator() is physical_integrator
    assert integrator.getExtensionIntegrator() is extension_integrator


# SplitIntegrator Tests


def test_split_integrator_with_single_integrator():
    """Test SplitIntegrator with only physical integrator."""
    step_size = 4.0 * mmunit.femtosecond
    physical_integrator = create_symmetric_integrator(step_size)

    integrator = SplitIntegrator(physical_integrator)

    # Verify extension integrator has half the step size
    physical_dt = mmswig.Integrator_getStepSize(integrator.getPhysicalIntegrator())
    extension_dt = mmswig.Integrator_getStepSize(integrator.getExtensionIntegrator())

    assert physical_dt == pytest.approx(step_size.value_in_unit(mmunit.picosecond))
    assert extension_dt == pytest.approx(physical_dt / 2)

    # Check _num_substeps calculated correctly
    # (ratio = physical_dt / (2 * extension_dt) = 1)
    assert integrator._num_substeps == 1


def test_split_integrator_with_two_integrators_valid_even_ratio():
    """Test SplitIntegrator with valid even ratio step sizes."""
    # Test ratio = 2 (physical: 4 fs, extension: 1 fs)
    physical_integrator = create_symmetric_integrator(4.0 * mmunit.femtosecond)
    extension_integrator = create_symmetric_integrator(1.0 * mmunit.femtosecond)

    integrator = SplitIntegrator(physical_integrator, extension_integrator)
    assert integrator._num_substeps == 2

    # Test ratio = 1 (physical: 2 fs, extension: 1 fs)
    physical_integrator = create_symmetric_integrator(2.0 * mmunit.femtosecond)
    extension_integrator = create_symmetric_integrator(1.0 * mmunit.femtosecond)

    integrator = SplitIntegrator(physical_integrator, extension_integrator)
    assert integrator._num_substeps == 1


def test_split_integrator_error_non_symmetric():
    """Test that non-symmetric integrators raise ValueError."""
    step_size = 4.0 * mmunit.femtosecond
    physical_integrator = create_non_symmetric_integrator(step_size)
    extension_integrator = create_non_symmetric_integrator(step_size / 2)

    with pytest.raises(ValueError, match="must be symmetric"):
        SplitIntegrator(physical_integrator, extension_integrator)


def test_split_integrator_error_invalid_step_size_ratio():
    """Test that invalid step size ratio raises ValueError."""
    physical_integrator = create_symmetric_integrator(3.0 * mmunit.femtosecond)
    extension_integrator = create_symmetric_integrator(1.0 * mmunit.femtosecond)

    with pytest.raises(ValueError, match="even integer ratio"):
        SplitIntegrator(physical_integrator, extension_integrator)


def test_split_integrator_assume_symmetric_bypass():
    """Test that assume_symmetric=True bypasses validation."""
    physical_integrator = create_non_symmetric_integrator(4.0 * mmunit.femtosecond)
    extension_integrator = create_non_symmetric_integrator(2.0 * mmunit.femtosecond)

    # Should succeed with assume_symmetric=True
    integrator = SplitIntegrator(
        physical_integrator, extension_integrator, assume_symmetric=True
    )

    assert integrator.getPhysicalIntegrator() is physical_integrator
    assert integrator.getExtensionIntegrator() is extension_integrator
    assert integrator._num_substeps == 1


# Getter/Setter Methods Tests


def test_get_step_size():
    """Test getStepSize returns Quantity in picoseconds."""
    step_size = 2.0 * mmunit.femtosecond
    physical_integrator = create_force_first_integrator(step_size)
    integrator = LockstepIntegrator(physical_integrator)

    result = integrator.getStepSize()

    assert isinstance(result, mmunit.Quantity)
    assert result.value_in_unit(mmunit.picosecond) == pytest.approx(
        step_size.value_in_unit(mmunit.picosecond)
    )


def test_set_step_size_with_quantity():
    """Test setStepSize with Quantity."""
    initial_step_size = 2.0 * mmunit.femtosecond
    new_step_size = 4.0 * mmunit.femtosecond
    physical_integrator = create_force_first_integrator(initial_step_size)
    integrator = LockstepIntegrator(physical_integrator)

    integrator.setStepSize(new_step_size)

    # Verify both integrators scaled proportionally (factor of 2)
    physical_dt = mmswig.Integrator_getStepSize(integrator.getPhysicalIntegrator())
    extension_dt = mmswig.Integrator_getStepSize(integrator.getExtensionIntegrator())

    assert physical_dt == pytest.approx(new_step_size.value_in_unit(mmunit.picosecond))
    assert extension_dt == pytest.approx(new_step_size.value_in_unit(mmunit.picosecond))
    result_step_size = integrator.getStepSize().value_in_unit(mmunit.picosecond)
    assert result_step_size == pytest.approx(
        new_step_size.value_in_unit(mmunit.picosecond)
    )


def test_set_step_size_with_float():
    """Test setStepSize with float (ps)."""
    initial_step_size = 2.0 * mmunit.femtosecond
    new_step_size_ps = 0.003  # 3 fs in ps
    physical_integrator = create_force_first_integrator(initial_step_size)
    integrator = LockstepIntegrator(physical_integrator)

    integrator.setStepSize(new_step_size_ps)

    physical_dt = mmswig.Integrator_getStepSize(integrator.getPhysicalIntegrator())
    extension_dt = mmswig.Integrator_getStepSize(integrator.getExtensionIntegrator())

    assert physical_dt == pytest.approx(new_step_size_ps)
    assert extension_dt == pytest.approx(new_step_size_ps)


def test_get_physical_integrator():
    """Test getPhysicalIntegrator returns correct instance."""
    physical_integrator = create_force_first_integrator(2.0 * mmunit.femtosecond)
    integrator = LockstepIntegrator(physical_integrator)

    result = integrator.getPhysicalIntegrator()

    assert result is physical_integrator


def test_get_extension_integrator():
    """Test getExtensionIntegrator returns correct instance."""
    physical_integrator = create_force_first_integrator(2.0 * mmunit.femtosecond)
    extension_integrator = create_force_first_integrator(2.0 * mmunit.femtosecond)
    integrator = LockstepIntegrator(physical_integrator, extension_integrator)

    result = integrator.getExtensionIntegrator()

    assert result is extension_integrator


# Serialization Tests


def test_copy_lockstep_integrator():
    """Test __copy__ for LockstepIntegrator."""
    step_size = 2.0 * mmunit.femtosecond
    original = LockstepIntegrator(create_force_first_integrator(step_size))

    copy = original.__copy__()

    # Verify copy has independent integrator instances
    assert copy.getPhysicalIntegrator() is not original.getPhysicalIntegrator()
    assert copy.getExtensionIntegrator() is not original.getExtensionIntegrator()

    # Check step sizes match
    copy_phys_dt = mmswig.Integrator_getStepSize(copy.getPhysicalIntegrator())
    orig_phys_dt = mmswig.Integrator_getStepSize(original.getPhysicalIntegrator())
    assert copy_phys_dt == pytest.approx(orig_phys_dt)

    copy_ext_dt = mmswig.Integrator_getStepSize(copy.getExtensionIntegrator())
    orig_ext_dt = mmswig.Integrator_getStepSize(original.getExtensionIntegrator())
    assert copy_ext_dt == pytest.approx(orig_ext_dt)


def test_copy_split_integrator():
    """Test __copy__ for SplitIntegrator."""
    step_size = 4.0 * mmunit.femtosecond
    original = SplitIntegrator(create_symmetric_integrator(step_size))

    copy = original.__copy__()

    # Verify copy has independent integrator instances
    assert copy.getPhysicalIntegrator() is not original.getPhysicalIntegrator()
    assert copy.getExtensionIntegrator() is not original.getExtensionIntegrator()

    # Verify _num_substeps preserved
    assert copy._num_substeps == original._num_substeps

    # Check step sizes match
    assert mmswig.Integrator_getStepSize(copy.getPhysicalIntegrator()) == pytest.approx(
        mmswig.Integrator_getStepSize(original.getPhysicalIntegrator())
    )


def test_getstate_setstate_lockstep():
    """Test __getstate__ and __setstate__ for LockstepIntegrator."""
    step_size = 2.0 * mmunit.femtosecond
    original = LockstepIntegrator(create_force_first_integrator(step_size))

    # Get state
    state = original.__getstate__()
    assert isinstance(state, dict)

    # Create new instance and set state
    new_integrator = LockstepIntegrator.__new__(LockstepIntegrator)
    new_integrator.__setstate__(state)

    # Verify step sizes preserved
    assert mmswig.Integrator_getStepSize(
        new_integrator.getPhysicalIntegrator()
    ) == pytest.approx(mmswig.Integrator_getStepSize(original.getPhysicalIntegrator()))
    assert mmswig.Integrator_getStepSize(
        new_integrator.getExtensionIntegrator()
    ) == pytest.approx(mmswig.Integrator_getStepSize(original.getExtensionIntegrator()))

    # Check that _physical_context and other None attributes reset properly
    assert new_integrator._physical_context is None
    assert new_integrator._extension_context is None
    assert new_integrator._dynamical_variables is None
    assert new_integrator._coupling is None


def test_getstate_setstate_split():
    """Test __getstate__ and __setstate__ for SplitIntegrator."""
    step_size = 4.0 * mmunit.femtosecond
    original = SplitIntegrator(create_symmetric_integrator(step_size))

    # Get state
    state = original.__getstate__()
    assert isinstance(state, dict)

    # Create new instance and set state
    new_integrator = SplitIntegrator.__new__(SplitIntegrator)
    new_integrator.__setstate__(state)

    # Verify _num_substeps recalculated correctly via _initialize()
    assert new_integrator._num_substeps == original._num_substeps

    # Verify step sizes preserved
    assert mmswig.Integrator_getStepSize(
        new_integrator.getPhysicalIntegrator()
    ) == pytest.approx(mmswig.Integrator_getStepSize(original.getPhysicalIntegrator()))
    assert mmswig.Integrator_getStepSize(
        new_integrator.getExtensionIntegrator()
    ) == pytest.approx(mmswig.Integrator_getStepSize(original.getExtensionIntegrator()))


def test_split_default_extension_integrator_has_consistent_coefficients():
    """The halved extension step comes with a recomputed friction factor."""
    physical = integrators.SymmetricLangevinIntegrator(
        300 * mmunit.kelvin, 10 / mmunit.picosecond, 4 * mmunit.femtosecond
    )
    split = SplitIntegrator(physical)
    extension = split.getExtensionIntegrator()
    assert mmswig.Integrator_getStepSize(extension) == pytest.approx(0.002)
    assert extension.getGlobalVariableByName("a") == pytest.approx(math.exp(-0.02))
    split.setStepSize(8 * mmunit.femtosecond)
    assert physical.getGlobalVariableByName("a") == pytest.approx(math.exp(-0.08))
    assert extension.getGlobalVariableByName("a") == pytest.approx(math.exp(-0.04))


# SplitIntegrator with a coupling that reads energy derivatives


def _inner_coupling():
    """A bond scaled by a smoothstep of a dynamical variable."""
    force = mm.CustomBondForce("scaling*500*(r-0.15)^2")
    force.addGlobalParameter("scaling", 1.0)
    force.addEnergyParameterDerivative("scaling")
    force.addBond(0, 1, [])
    variable = xps.DynamicalVariable(
        "lam",
        mmunit.dimensionless,
        1 * mmunit.dalton * mmunit.nanometer**2,
        xps.bounds.ReflectiveBounds(0, 1, mmunit.dimensionless),
    )
    return xps.InnerProductCoupling(
        [force], [variable], functions={"scaling": "lam^2*(3-2*lam)"}
    )


def _split_context(coupling, physical_fs, extension_fs):
    system = mm.System()
    system.addParticle(12.0)
    system.addParticle(1.0)
    integrator = SplitIntegrator(
        create_symmetric_integrator(physical_fs * mmunit.femtosecond),
        create_symmetric_integrator(extension_fs * mmunit.femtosecond),
    )
    context = xps.ExtendedSpaceContext(
        xps.ExtendedSpaceSystem(system, coupling), integrator, REFERENCE
    )
    context.setPositions(POSITIONS)
    context.setVelocities(VELOCITIES)
    context.setDynamicalVariableValues([0.3])
    context.setDynamicalVariableVelocities([0.5])
    return context


LAYOUTS = {"physical-substeps": (1, 4), "extension-substeps": (4, 1)}


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_split_updates_extension_only_when_needed(layout):
    """With the physical integrator taking the substeps, the extension context is
    updated before each extension step, and after the last step of the call."""
    coupling = _inner_coupling()
    calls = []
    update = coupling.updateExtensionContext
    coupling.updateExtensionContext = lambda *args: calls.append(update(*args))
    context = _split_context(coupling, *LAYOUTS[layout])
    calls.clear()  # the context synchronizes when created
    context.getIntegrator().step(3)
    assert len(calls) == (4 if layout == "physical-substeps" else 3)


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_split_extension_context_is_current_after_step(layout):
    context = _split_context(_inner_coupling(), *LAYOUTS[layout])
    context.getIntegrator().step(7)
    extension = context.getExtensionContext()
    after_step = extension.getParameter("derivative_with_respect_to_scaling")
    context.synchronize()
    assert extension.getParameter("derivative_with_respect_to_scaling") == after_step


def _trajectory(context, calls):
    for steps in calls:
        context.getIntegrator().step(steps)
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    values = [value / value.unit for value in context.getDynamicalVariableValues()]
    return np.concatenate([positions.value_in_unit(mmunit.nanometer).ravel(), values])


def test_split_trajectory_does_not_depend_on_call_chunking():
    """Skipping the observer-only updates leaves the dynamics unchanged."""
    one_call = _trajectory(_split_context(_inner_coupling(), 1, 4), [12])
    many_calls = _trajectory(_split_context(_inner_coupling(), 1, 4), [1] * 12)
    np.testing.assert_allclose(one_call, many_calls, rtol=1e-12, atol=1e-14)
