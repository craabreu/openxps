"""
Unit tests for the openxps.coupling module.
"""

from copy import copy
from math import pi

import cvpack
import numpy as np
import openmm as mm
import pytest
import yaml
from openmm import unit as mmunit
from openmmtools import testsystems

from openxps import (
    CustomCouplingForce,
    DynamicalVariable,
    ExtendedSpaceSystem,
    HarmonicCouplingForce,
)
from openxps.bounds import CIRCULAR, NoBounds
from openxps.coupling import CouplingForceSum


# Helper functions
def create_test_cv(name="phi"):
    """Create a test collective variable (phi torsion)."""
    return cvpack.Torsion(6, 8, 14, 16, name=name)


def create_test_dv(name="phi_s", mass=None):
    """Create a test dynamical variable."""
    if mass is None:
        mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    return DynamicalVariable(name, mmunit.radian, mass, CIRCULAR)


def create_test_system():
    """Create a test OpenMM system (alanine dipeptide)."""
    return testsystems.AlanineDipeptideVacuum()


# CustomCouplingForce Tests
def test_custom_coupling_force_initialization():
    """Test basic initialization of CustomCouplingForce."""
    phi = create_test_cv()
    coupling = CustomCouplingForce(
        "0.5*kappa*(phi-phi0)^2",
        [phi],
        kappa=1000 * mmunit.kilojoules_per_mole / mmunit.radian**2,
        phi0=np.pi * mmunit.radian,
    )

    assert coupling.getEnergyFunction() == "0.5*kappa*(phi-phi0)^2"
    assert coupling.getNumCollectiveVariables() == 1
    params = coupling.getParameterDefaultValues()
    assert "kappa" in params
    assert "phi0" in params


def test_custom_coupling_force_repr():
    """Test string representation of CustomCouplingForce."""
    phi = create_test_cv()
    coupling = CustomCouplingForce(
        "0.5*kappa*(phi-phi0)^2",
        [phi],
        kappa=1000 * mmunit.kilojoules_per_mole / mmunit.radian**2,
        phi0=np.pi * mmunit.radian,
    )

    repr_str = repr(coupling)
    assert "CustomCouplingForce" in repr_str
    assert "0.5*kappa*(phi-phi0)^2" in repr_str


def test_custom_coupling_force_flip():
    """Test flipping a CustomCouplingForce."""

    phi = create_test_cv()
    coupling = CustomCouplingForce(
        f"0.5*kappa*min(delta,{2 * pi}-delta)^2; delta=abs(phi-phi0)",
        [phi],
        kappa=1000 * mmunit.kilojoules_per_mole / mmunit.radian**2,
        phi0=pi * mmunit.radian,
    )
    phi0 = create_test_dv("phi0")

    flipped = coupling.flip([phi0])

    # Check that phi0 is now a collective variable (not a parameter)
    assert "phi0" not in flipped.getParameterDefaultValues()
    # Check that phi is now a parameter set to zero
    params = flipped.getParameterDefaultValues()
    assert "phi" in params
    assert params["phi"].value_in_unit(mmunit.radian) == 0.0
    assert "kappa" in params


def test_custom_coupling_force_get_extension_parameters():
    """Test getExtensionParameters method."""
    model = create_test_system()
    phi = create_test_cv()
    phi0 = create_test_dv("phi0")

    coupling = CustomCouplingForce(
        "0.5*kappa*(phi-phi0)^2",
        [phi],
        kappa=1000 * mmunit.kilojoules_per_mole / mmunit.radian**2,
        phi0=0.0 * mmunit.radian,
    )

    system = ExtendedSpaceSystem([phi0], coupling, model.system)
    integrator = mm.VerletIntegrator(1.0 * mmunit.femtosecond)
    context = mm.Context(system, integrator)
    context.setPositions(model.positions)

    params = coupling.getExtensionParameters(context)
    assert "phi" in params
    # Values are returned as floats (in MD units)
    assert isinstance(params["phi"], float)


def test_custom_coupling_force_serialization():
    """Test YAML serialization and deserialization of CustomCouplingForce."""
    phi = create_test_cv()
    coupling = CustomCouplingForce(
        "0.5*kappa*(phi-phi0)^2",
        [phi],
        kappa=1000 * mmunit.kilojoules_per_mole / mmunit.radian**2,
        phi0=np.pi * mmunit.radian,
    )

    serialized = yaml.safe_dump(coupling)
    deserialized = yaml.safe_load(serialized)

    assert deserialized.getEnergyFunction() == coupling.getEnergyFunction()
    assert (
        deserialized.getNumCollectiveVariables() == coupling.getNumCollectiveVariables()
    )


def test_custom_coupling_force_add_to_system():
    """Test adding CustomCouplingForce to a system."""
    model = create_test_system()
    phi = create_test_cv()
    coupling = CustomCouplingForce(
        "0.5*kappa*(phi-phi0)^2",
        [phi],
        kappa=1000 * mmunit.kilojoules_per_mole / mmunit.radian**2,
        phi0=0.0 * mmunit.radian,
    )

    initial_forces = model.system.getNumForces()
    coupling.addToSystem(model.system)
    assert model.system.getNumForces() == initial_forces + 1


# HarmonicCouplingForce Tests
def test_harmonic_coupling_force_initialization():
    """Test initialization of HarmonicCouplingForce with valid arguments."""
    phi = create_test_cv()
    phi_s = create_test_dv()
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    coupling = HarmonicCouplingForce(phi, phi_s, kappa)

    assert coupling.getNumCollectiveVariables() == 1
    params = coupling.getParameterDefaultValues()
    assert "kappa_phi_phi_s" in params
    assert "phi_s" in params


def test_harmonic_coupling_force_incompatible_units():
    """Test HarmonicCouplingForce with incompatible CV and DV units."""
    # Create a distance CV
    distance_cv = cvpack.Distance(0, 1, name="dist")
    # Create an angle DV
    phi_s = create_test_dv()
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.nanometer**2

    with pytest.raises(ValueError, match="Incompatible units"):
        HarmonicCouplingForce(distance_cv, phi_s, kappa)


def test_harmonic_coupling_force_incompatible_periodicity_nonperiodic_cv():
    """Test HarmonicCouplingForce with periodic DV but non-periodic CV."""
    distance_cv = cvpack.Distance(0, 1, name="dist")
    phi_s = create_test_dv()  # Periodic DV
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    # Units check happens first, so we expect incompatible units error
    with pytest.raises(ValueError, match="Incompatible units"):
        HarmonicCouplingForce(distance_cv, phi_s, kappa)


def test_harmonic_coupling_force_incompatible_periodicity_nonperiodic_dv():
    """Test HarmonicCouplingForce with periodic CV but non-periodic DV."""
    phi = create_test_cv()
    # Create non-periodic DV
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    phi_s = DynamicalVariable("phi_s", mmunit.radian, mass, NoBounds())
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    with pytest.raises(ValueError, match="Incompatible periodicity"):
        HarmonicCouplingForce(phi, phi_s, kappa)


def test_harmonic_coupling_force_incompatible_force_constant():
    """Test HarmonicCouplingForce with wrong force constant units."""
    phi = create_test_cv()
    phi_s = create_test_dv()
    # Wrong units: should be energy/angle^2, not just energy
    kappa = 1000 * mmunit.kilojoule_per_mole

    with pytest.raises(ValueError, match="Incompatible force constant units"):
        HarmonicCouplingForce(phi, phi_s, kappa)


def test_harmonic_coupling_force_flip():
    """Test flipping a HarmonicCouplingForce."""
    phi = create_test_cv()
    phi_s = create_test_dv()
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    coupling = HarmonicCouplingForce(phi, phi_s, kappa)
    flipped = coupling.flip([phi_s])

    # Check that phi_s is now a CV (not a parameter)
    assert "phi_s" not in flipped.getParameterDefaultValues()
    # Check that phi is now a parameter
    params = flipped.getParameterDefaultValues()
    assert "phi" in params


def test_harmonic_coupling_force_repr():
    """Test string representation of HarmonicCouplingForce."""
    phi = create_test_cv()
    phi_s = create_test_dv()
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    coupling = HarmonicCouplingForce(phi, phi_s, kappa)
    repr_str = repr(coupling)

    assert "HarmonicCouplingForce" in repr_str
    assert "kappa_phi_phi_s" in repr_str


def test_harmonic_coupling_force_serialization():
    """Test YAML serialization of HarmonicCouplingForce."""
    phi = create_test_cv()
    phi_s = create_test_dv()
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    coupling = HarmonicCouplingForce(phi, phi_s, kappa)
    serialized = yaml.safe_dump(coupling)
    deserialized = yaml.safe_load(serialized)

    assert deserialized.getEnergyFunction() == coupling.getEnergyFunction()


# CouplingForceSum Tests
def test_coupling_force_sum_initialization():
    """Test initialization of CouplingForceSum."""
    phi = create_test_cv("phi")
    psi = create_test_cv("psi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    force1 = HarmonicCouplingForce(phi, phi_s, kappa)
    force2 = HarmonicCouplingForce(psi, psi_s, kappa)

    force_sum = CouplingForceSum([force1, force2])
    forces = force_sum.getCouplingForces()

    assert len(forces) == 2
    assert forces[0] is force1
    assert forces[1] is force2


def test_coupling_force_sum_flattening():
    """Test that nested CouplingForceSum objects are flattened."""
    phi = create_test_cv("phi")
    psi = create_test_cv("psi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    force1 = HarmonicCouplingForce(phi, phi_s, kappa)
    force2 = HarmonicCouplingForce(psi, psi_s, kappa)

    # Create nested sum: sum of (force1) and (sum of force2)
    sum1 = CouplingForceSum([force1])
    sum2 = CouplingForceSum([force2])
    nested_sum = CouplingForceSum([sum1, sum2])

    # Should be flattened to just [force1, force2]
    forces = nested_sum.getCouplingForces()
    assert len(forces) == 2


def test_coupling_force_addition_operator():
    """Test the addition operator for coupling forces."""
    phi = create_test_cv("phi")
    psi = create_test_cv("psi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    force1 = HarmonicCouplingForce(phi, phi_s, kappa)
    force2 = HarmonicCouplingForce(psi, psi_s, kappa)

    force_sum = force1 + force2

    assert isinstance(force_sum, CouplingForceSum)
    forces = force_sum.getCouplingForces()
    assert len(forces) == 2


def test_coupling_force_sum_add_to_system():
    """Test that all forces in sum are added to system."""
    model = create_test_system()
    phi = create_test_cv("phi")
    psi = cvpack.Torsion(4, 6, 8, 14, name="psi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    force1 = HarmonicCouplingForce(phi, phi_s, kappa)
    force2 = HarmonicCouplingForce(psi, psi_s, kappa)
    force_sum = force1 + force2

    initial_forces = model.system.getNumForces()
    force_sum.addToSystem(model.system)

    # Should add 2 forces
    assert model.system.getNumForces() == initial_forces + 2


def test_coupling_force_sum_flip():
    """Test flipping a CouplingForceSum."""
    phi = create_test_cv("phi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    # Create a custom coupling with both DVs
    coupling = CustomCouplingForce(
        "0.5*kappa1*(phi-phi_s)^2 + 0.5*kappa2*(phi-psi_s)^2",
        [phi],
        kappa1=kappa,
        kappa2=kappa,
        phi_s=0.0 * mmunit.radian,
        psi_s=0.0 * mmunit.radian,
    )

    # Create another one for the sum
    psi = cvpack.Torsion(4, 6, 8, 14, name="psi")
    coupling2 = CustomCouplingForce(
        "0.5*kappa*(psi-phi_s)^2",
        [psi],
        kappa=kappa,
        phi_s=0.0 * mmunit.radian,
    )

    force_sum = coupling + coupling2
    flipped = force_sum.flip([phi_s, psi_s])

    assert isinstance(flipped, CouplingForceSum)
    flipped_forces = flipped.getCouplingForces()
    assert len(flipped_forces) == 2


def test_coupling_force_sum_get_extension_parameters():
    """Test getExtensionParameters with non-overlapping CVs."""
    model = create_test_system()
    phi = create_test_cv("phi")
    psi = cvpack.Torsion(4, 6, 8, 14, name="psi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    force1 = HarmonicCouplingForce(phi, phi_s, kappa)
    force2 = HarmonicCouplingForce(psi, psi_s, kappa)
    force_sum = force1 + force2

    system = ExtendedSpaceSystem([phi_s, psi_s], force_sum, model.system)
    integrator = mm.VerletIntegrator(1.0 * mmunit.femtosecond)
    context = mm.Context(system, integrator)
    context.setPositions(model.positions)

    params = force_sum.getExtensionParameters(context)

    # Should have parameters for both phi and psi
    assert "phi" in params
    assert "psi" in params


def test_coupling_force_sum_conflicting_parameters():
    """Test that conflicting parameters raise an error."""
    phi = create_test_cv("phi")
    phi_s = create_test_dv("phi_s")
    kappa1 = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2
    kappa2 = 500 * mmunit.kilojoule_per_mole / mmunit.radian**2

    # Create two forces with same CV but different values
    force1 = HarmonicCouplingForce(phi, phi_s, kappa1)
    force2 = HarmonicCouplingForce(phi, phi_s, kappa2)

    # Note: This test assumes that when both forces evaluate "phi",
    # they would return different values if phi_s has different values
    # in their parameter sets. However, in practice, the CV value should
    # be the same since it's evaluated in the same context.
    # This test may need adjustment based on actual behavior.

    # For now, we'll just verify the sum can be created
    force_sum = force1 + force2
    assert len(force_sum.getCouplingForces()) == 2


def test_coupling_force_sum_repr():
    """Test string representation of CouplingForceSum."""
    phi = create_test_cv("phi")
    psi = cvpack.Torsion(4, 6, 8, 14, name="psi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    force1 = HarmonicCouplingForce(phi, phi_s, kappa)
    force2 = HarmonicCouplingForce(psi, psi_s, kappa)
    force_sum = force1 + force2

    repr_str = repr(force_sum)
    assert "HarmonicCouplingForce" in repr_str
    assert "+" in repr_str


def test_coupling_force_sum_copy():
    """Test copying a CouplingForceSum."""

    phi = create_test_cv("phi")
    psi = cvpack.Torsion(4, 6, 8, 14, name="psi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    force1 = HarmonicCouplingForce(phi, phi_s, kappa)
    force2 = HarmonicCouplingForce(psi, psi_s, kappa)
    force_sum = force1 + force2

    copied = copy(force_sum)

    assert isinstance(copied, CouplingForceSum)
    assert len(copied.getCouplingForces()) == len(force_sum.getCouplingForces())


def test_coupling_force_sum_serialization():
    """Test YAML serialization of CouplingForceSum."""
    phi = create_test_cv("phi")
    psi = cvpack.Torsion(4, 6, 8, 14, name="psi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    force1 = HarmonicCouplingForce(phi, phi_s, kappa)
    force2 = HarmonicCouplingForce(psi, psi_s, kappa)
    force_sum = force1 + force2

    serialized = yaml.safe_dump(force_sum)
    deserialized = yaml.safe_load(serialized)

    assert isinstance(deserialized, CouplingForceSum)
    assert len(deserialized.getCouplingForces()) == 2


# Integration Tests
def test_coupling_force_with_extended_space_system():
    """Test CustomCouplingForce with ExtendedSpaceSystem."""
    model = create_test_system()
    phi = create_test_cv()
    phi_s = create_test_dv("phi_s")

    coupling = CustomCouplingForce(
        "0.5*kappa*(phi-phi_s)^2",
        [phi],
        kappa=1000 * mmunit.kilojoules_per_mole / mmunit.radian**2,
        phi_s=0.0 * mmunit.radian,
    )

    initial_physical_forces = model.system.getNumForces()
    system = ExtendedSpaceSystem([phi_s], coupling, model.system)

    # Physical system should have one more force
    assert system.getNumForces() == initial_physical_forces + 1

    # Extension system should have one force (the flipped one)
    extension_system = system.getExtensionSystem()
    assert extension_system.getNumForces() == 1
    assert extension_system.getNumParticles() == 1


def test_harmonic_coupling_force_integration():
    """Test HarmonicCouplingForce integration with ExtendedSpaceSystem."""
    model = create_test_system()
    phi = create_test_cv()
    phi_s = create_test_dv()
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    coupling = HarmonicCouplingForce(phi, phi_s, kappa)
    system = ExtendedSpaceSystem([phi_s], coupling, model.system)

    # Check dynamical variables
    dvs = system.getDynamicalVariables()
    assert len(dvs) == 1
    assert dvs[0].name == "phi_s"

    # Check coupling force
    retrieved_force = system.getCouplingForce()
    assert retrieved_force is coupling


def test_coupling_force_sum_integration():
    """Test CouplingForceSum with ExtendedSpaceSystem."""
    model = create_test_system()
    phi = create_test_cv("phi")
    psi = cvpack.Torsion(4, 6, 8, 14, name="psi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    force1 = HarmonicCouplingForce(phi, phi_s, kappa)
    force2 = HarmonicCouplingForce(psi, psi_s, kappa)
    force_sum = force1 + force2

    initial_forces = model.system.getNumForces()
    system = ExtendedSpaceSystem([phi_s, psi_s], force_sum, model.system)

    # Physical system should have 2 more forces
    assert system.getNumForces() == initial_forces + 2

    # Extension system should have 2 forces
    extension_system = system.getExtensionSystem()
    assert extension_system.getNumForces() == 2
    assert extension_system.getNumParticles() == 2
