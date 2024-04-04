"""
Unit tests for the openxps.context module.
"""

import openmm as mm
import pytest
from openmm import unit as mmunit
from openmmtools import testsystems

from openxps.bounds import CIRCULAR
from openxps.context import ExtendedSpaceContext
from openxps.extra_dof import ExtraDOF


def create_model(request_derivative=True):
    """
    Helper function to create a fresh OpenMM system for each test,
    with a CustomTorsionForce that depends on the provided global parameter.
    """
    model = testsystems.AlanineDipeptideVacuum()
    dihedral = mm.CustomTorsionForce("0.5 * k * (theta - phi)^2")
    dihedral.addPerTorsionParameter("k")
    dihedral.addTorsion(6, 8, 14, 16, [1000.0])
    dihedral.addGlobalParameter("phi", 0.0)
    if request_derivative:
        dihedral.addEnergyParameterDerivative("phi")
    model.system.addForce(dihedral)
    return model


def create_extra_dof():
    """Helper function to create an ExtraDOF object."""
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    return ExtraDOF(name="phi", unit=mmunit.radian, mass=mass, bounds=CIRCULAR)


def create_basic_context(system):
    """Helper function to create a basic OpenMM context for testing."""
    integrator = mm.VerletIntegrator(1.0 * mmunit.femtosecond)
    platform = mm.Platform.getPlatformByName("Reference")
    context = mm.Context(system, integrator, platform)
    return context


def test_initialization():
    """Test the initialization of ExtendedSpaceContext with basic setup."""
    model = create_model(request_derivative=True)
    context = create_basic_context(model.system)
    extra_dof = create_extra_dof()
    extended_context = ExtendedSpaceContext(context, [extra_dof])
    assert extended_context is not None

    model = create_model(request_derivative=False)
    context = create_basic_context(model.system)
    extra_dof = create_extra_dof()
    extended_context = ExtendedSpaceContext(context, [extra_dof])
    assert extended_context is not None


def test_set_positions_and_velocities():
    """Test setting positions and velocities including extra DOFs."""
    extra_dof = create_extra_dof()
    model = create_model()
    basic_context = create_basic_context(model.system)
    extended_context = ExtendedSpaceContext(basic_context, [extra_dof])

    num_atoms = model.system.getNumParticles()
    positions = [mm.Vec3(1, 2, 3)] * num_atoms * mmunit.nanometer
    velocities = [mm.Vec3(2, 3, 4)] * num_atoms * mmunit.nanometer / mmunit.picosecond

    extended_context.setPositions(positions)
    extended_context.setExtraValues([1 * mmunit.radian])
    extended_context.setVelocities(velocities)
    extended_context.setExtraVelocities([1 * mmunit.radians / mmunit.picosecond])

    state = extended_context.getState(  # pylint: disable=unexpected-keyword-arg
        getPositions=True, getVelocities=True
    )
    assert state.getPositions() == positions
    assert state.getVelocities() == velocities
    assert extended_context.getExtraValues() == (1 * mmunit.radian,)
    assert extended_context.getExtraVelocities() == (
        1 * mmunit.radians / mmunit.picosecond,
    )


def test_raise_exceptions():
    """Test raising exceptions for invalid operations."""
    extra_dof = create_extra_dof()
    model = create_model()
    system = model.system
    basic_context = create_basic_context(system)
    extended_context = ExtendedSpaceContext(basic_context, [extra_dof])

    with pytest.raises(mm.OpenMMException) as excinfo:
        extended_context.setVelocitiesToTemperature(300 * mmunit.kelvin)
    assert "Particle positions have not been set" in str(excinfo.value)

    extended_context.setPositions(model.positions)
    extended_context.setVelocitiesToTemperature(300 * mmunit.kelvin)

    with pytest.raises(RuntimeError) as excinfo:
        extended_context.setExtraVelocitiesToTemperature(300 * mmunit.kelvin)
    assert "Extra degrees of freedom have not been set" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        extended_context.getIntegrator().step(1)
    assert "Extra degrees of freedom have not been set" in str(excinfo.value)

    extended_context.setExtraValues([1 * mmunit.radian])
    extended_context.setExtraVelocitiesToTemperature(300 * mmunit.kelvin)

    state = extended_context.getState(  # pylint: disable=unexpected-keyword-arg
        getVelocities=True
    )
    velocities = state.getVelocities()
    extra_velocities = extended_context.getExtraVelocities()
    assert len(velocities) == len(model.positions)
    assert len(extra_velocities) == 1
