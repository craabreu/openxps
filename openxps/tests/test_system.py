"""
Unit tests for the openxps.system module.
"""

import cvpack
import numpy as np
import openmm as mm
import pytest
from openmm import unit as mmunit
from openmmtools import testsystems

from openxps import CollectiveVariableCoupling, DynamicalVariable, ExtendedSpaceSystem
from openxps.bounds import CircularBounds, NoBounds, ReflectiveBounds


def create_dvs():
    """Helper function to create a DynamicalVariable object."""
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    kwargs = {"unit": mmunit.nanometer, "mass": 1 * mmunit.dalton}
    return [
        DynamicalVariable(
            name="phi0", unit=mmunit.radian, mass=mass, bounds=CircularBounds()
        ),
        DynamicalVariable(name="x0", bounds=NoBounds(), **kwargs),
        DynamicalVariable(
            name="y0", bounds=ReflectiveBounds(-1, 1, mmunit.nanometer), **kwargs
        ),
    ]


def create_coupling(phi0=180 * mmunit.degrees):
    """Helper function to create a MetaCollectiveVariable object."""
    dvs = create_dvs()
    kwargs = {
        "kappa": 1000 * mmunit.kilojoule_per_mole / mmunit.radians**2,
        "alpha": 0.01 * mmunit.kilojoule_per_mole / mmunit.nanometer**2,
        "x0": 1 * mmunit.nanometer,
        "y0": 1 * mmunit.nanometer,
    }
    if phi0 is not None:
        kwargs["phi0"] = phi0
    return CollectiveVariableCoupling(
        f"0.5*kappa*min(delta_phi,{2 * np.pi}-delta_phi)^2+alpha*(x0-y0)^2"
        "; delta_phi=abs(phi-phi0)",
        [cvpack.Torsion(6, 8, 14, 16, name="phi")],
        dvs if phi0 is not None else [],
        **kwargs,
    )


def test_initialization():
    """Test basic initialization of ExtendedSpaceSystem."""
    model = testsystems.AlanineDipeptideVacuum()
    coupling = create_coupling()

    # Get initial force count from the base system
    initial_force_count = model.system.getNumForces()

    # Create ExtendedSpaceSystem
    system = ExtendedSpaceSystem(model.system, coupling)

    # Verify it's an instance of mm.System (inheritance)
    assert isinstance(system, mm.System)

    # Verify the coupling was added to the system
    assert system.getNumForces() > initial_force_count

    # Verify basic properties are accessible
    assert system.getNumParticles() == model.system.getNumParticles()


def test_get_dynamical_variables():
    """Test getDynamicalVariables returns the correct DVs."""
    model = testsystems.AlanineDipeptideVacuum()
    coupling = create_coupling()
    original_dvs = coupling.getDynamicalVariables()

    system = ExtendedSpaceSystem(model.system, coupling)

    retrieved_dvs = system.getDynamicalVariables()

    # Verify it returns a tuple
    assert isinstance(retrieved_dvs, tuple)

    # Verify correct number of DVs
    assert len(retrieved_dvs) == 3

    # Verify each DV matches
    for original_dv, retrieved_dv in zip(original_dvs, retrieved_dvs):
        assert retrieved_dv == original_dv
        assert retrieved_dv.name == original_dv.name
        assert retrieved_dv.unit == original_dv.unit
        assert retrieved_dv.mass == original_dv.mass


def test_get_coupling():
    """Test getCoupling returns the correct potential."""
    model = testsystems.AlanineDipeptideVacuum()
    coupling = create_coupling()

    system = ExtendedSpaceSystem(model.system, coupling)

    retrieved_potential = system.getCoupling()
    assert retrieved_potential is coupling
    assert (
        retrieved_potential.getForce(0).getEnergyFunction()
        == coupling.getForce(0).getEnergyFunction()
    )


def test_get_extension_system():
    """Test getExtensionSystem returns a properly configured system."""
    model = testsystems.AlanineDipeptideVacuum()
    coupling = create_coupling()

    system = ExtendedSpaceSystem(model.system, coupling)
    dvs = system.getDynamicalVariables()

    extension_system = system.getExtensionSystem()

    # Verify it returns an mm.System object
    assert isinstance(extension_system, mm.System)

    # Verify the extension system has correct number of particles (one per DV)
    assert extension_system.getNumParticles() == len(dvs)

    # Verify particle masses match DV masses
    # The mass is stored in the extension system as the numerical value
    for i, dv in enumerate(dvs):
        expected_mass = dv.mass / dv.mass.unit  # Get dimensionless value
        actual_mass = extension_system.getParticleMass(i)._value  # Get raw value
        assert actual_mass == pytest.approx(expected_mass)


def test_invalid_dynamical_variable_type():
    """Test that invalid DV type raises AttributeError when accessed."""
    # Note: DVs are now validated at coupling creation time, not in ExtendedSpaceSystem
    # If someone manually corrupts the coupling's DVs, it will fail when accessed
    model = testsystems.AlanineDipeptideVacuum()
    coupling = create_coupling()

    # Manually corrupt the coupling's DVs
    coupling._dynamical_variables = [None]

    # Will raise AttributeError when trying to access .mass on None
    with pytest.raises(AttributeError):
        ExtendedSpaceSystem(model.system, coupling)


def test_invalid_coupling_type():
    """Test that invalid coupling type raises TypeError."""
    model = testsystems.AlanineDipeptideVacuum()

    with pytest.raises(
        AttributeError,
        match="'NoneType' object has no attribute 'addToPhysicalSystem'",
    ):
        ExtendedSpaceSystem(model.system, None)


def test_missing_dv_parameter():
    """Test that ExtendedSpaceSystem with coupling having no DVs works."""
    model = testsystems.AlanineDipeptideVacuum()
    # Create coupling without dynamical variables
    potential_without_dvs = create_coupling(phi0=None)

    # This should work fine - just creates a system with no extension DVs
    system = ExtendedSpaceSystem(model.system, potential_without_dvs)
    assert len(system.getDynamicalVariables()) == 0
    assert system.getExtensionSystem().getNumParticles() == 0
