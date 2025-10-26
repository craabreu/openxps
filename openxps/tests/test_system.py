"""
Unit tests for the openxps.system module.
"""

import cvpack
import numpy as np
import openmm as mm
import pytest
from openmm import unit as mmunit
from openmmtools import testsystems

from openxps import CustomCoupling, DynamicalVariable, ExtendedSpaceSystem
from openxps.bounds import CIRCULAR, NoBounds, Reflective


def create_dvs():
    """Helper function to create a DynamicalVariable object."""
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    kwargs = {"unit": mmunit.nanometer, "mass": 1 * mmunit.dalton}
    return [
        DynamicalVariable(name="phi0", unit=mmunit.radian, mass=mass, bounds=CIRCULAR),
        DynamicalVariable(name="x0", bounds=NoBounds(), **kwargs),
        DynamicalVariable(
            name="y0", bounds=Reflective(-1, 1, mmunit.nanometer), **kwargs
        ),
    ]


def create_coupling(phi0=180 * mmunit.degrees):
    """Helper function to create a MetaCollectiveVariable object."""
    kwargs = {
        "kappa": 1000 * mmunit.kilojoule_per_mole / mmunit.radians**2,
        "alpha": 0.01 * mmunit.kilojoule_per_mole / mmunit.nanometer**2,
        "x0": 1 * mmunit.nanometer,
        "y0": 1 * mmunit.nanometer,
    }
    if phi0 is not None:
        kwargs["phi0"] = phi0
    return CustomCoupling(
        f"0.5*kappa*min(delta_phi,{2 * np.pi}-delta_phi)^2+alpha*(x0-y0)^2"
        "; delta_phi=abs(phi-phi0)",
        [cvpack.Torsion(6, 8, 14, 16, name="phi")],
        **kwargs,
    )


def test_initialization():
    """Test basic initialization of ExtendedSpaceSystem."""
    model = testsystems.AlanineDipeptideVacuum()
    coupling = create_coupling()
    dvs = create_dvs()

    # Get initial force count from the base system
    initial_force_count = model.system.getNumForces()

    # Create ExtendedSpaceSystem
    system = ExtendedSpaceSystem(dvs, coupling, model.system)

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
    dvs = create_dvs()

    system = ExtendedSpaceSystem(dvs, coupling, model.system)

    retrieved_dvs = system.getDynamicalVariables()

    # Verify it returns a tuple
    assert isinstance(retrieved_dvs, tuple)

    # Verify correct number of DVs
    assert len(retrieved_dvs) == 3

    # Verify each DV matches
    for original_dv, retrieved_dv in zip(dvs, retrieved_dvs):
        assert retrieved_dv == original_dv
        assert retrieved_dv.name == original_dv.name
        assert retrieved_dv.unit == original_dv.unit
        assert retrieved_dv.mass == original_dv.mass


def test_get_coupling():
    """Test getCoupling returns the correct potential."""
    model = testsystems.AlanineDipeptideVacuum()
    coupling = create_coupling()
    dvs = create_dvs()

    system = ExtendedSpaceSystem(dvs, coupling, model.system)

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
    dvs = create_dvs()

    system = ExtendedSpaceSystem(dvs, coupling, model.system)

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
    """Test that invalid DV type raises TypeError."""
    model = testsystems.AlanineDipeptideVacuum()
    coupling = create_coupling()

    with pytest.raises(
        TypeError, match="dynamical variables must be instances of DynamicalVariable"
    ):
        ExtendedSpaceSystem([None], coupling, model.system)


def test_invalid_coupling_type():
    """Test that invalid coupling type raises TypeError."""
    model = testsystems.AlanineDipeptideVacuum()
    dvs = create_dvs()

    with pytest.raises(
        TypeError,
        match="must be an instance of Coupling",
    ):
        ExtendedSpaceSystem(dvs, None, model.system)


def test_missing_dv_parameter():
    """Test that missing DV parameter raises ValueError."""
    model = testsystems.AlanineDipeptideVacuum()
    dvs = create_dvs()
    # Create coupling without phi0 parameter
    potential_without_phi0 = create_coupling(phi0=None)

    with pytest.raises(
        ValueError, match="dynamical variables are not coupling parameters"
    ):
        ExtendedSpaceSystem(dvs, potential_without_phi0, model.system)
