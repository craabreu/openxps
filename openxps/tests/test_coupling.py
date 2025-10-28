"""
Unit tests for the openxps.coupling module.
"""

from copy import copy
from math import pi

import cvpack
import openmm as mm
import pytest
import yaml
from openmm import unit as mmunit
from openmmtools import testsystems

from openxps import (
    CollectiveVariableCoupling,
    DynamicalVariable,
    ExtendedSpaceSystem,
    HarmonicCoupling,
)
from openxps.bounds import CIRCULAR, NoBounds
from openxps.coupling import CouplingSum


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


# CollectiveVariableCoupling Tests
def test_custom_coupling_initialization():
    """Test basic initialization of CollectiveVariableCoupling."""
    phi = create_test_cv()
    phi0 = create_test_dv("phi0")
    coupling = CollectiveVariableCoupling(
        "0.5*kappa*(phi-phi0)^2",
        [phi],
        [phi0],
        kappa=1000 * mmunit.kilojoules_per_mole / mmunit.radian**2,
    )

    assert coupling.getForce(0).getEnergyFunction() == "0.5*kappa*(phi-phi0)^2"
    assert coupling.getForce(0).getNumCollectiveVariables() == 1


def test_custom_coupling_repr():
    """Test string representation of CollectiveVariableCoupling."""
    phi = create_test_cv()
    phi0 = create_test_dv("phi0")
    coupling = CollectiveVariableCoupling(
        "0.5*kappa*(phi-phi0)^2",
        [phi],
        [phi0],
        kappa=1000 * mmunit.kilojoules_per_mole / mmunit.radian**2,
    )

    repr_str = repr(coupling)
    assert "CollectiveVariableCoupling" in repr_str
    assert "0.5*kappa*(phi-phi0)^2" in repr_str


def test_custom_coupling_add_to_extension_system():
    """Test adding a CollectiveVariableCoupling to the extension system."""

    phi = create_test_cv()
    phi0 = create_test_dv("phi0")
    coupling = CollectiveVariableCoupling(
        f"0.5*kappa*min(delta,{2 * pi}-delta)^2; delta=abs(phi-phi0)",
        [phi],
        [phi0],
        kappa=1000 * mmunit.kilojoules_per_mole / mmunit.radian**2,
    )

    # Create extension system and add coupling
    extension_system = mm.System()
    extension_system.addParticle(phi0.mass / phi0.mass.unit)
    coupling.addToExtensionSystem(extension_system)

    # Check that a force was added to the extension system
    assert extension_system.getNumForces() == 1
    flipped_force = extension_system.getForce(0)

    # Verify it's a CustomCVForce
    assert isinstance(flipped_force, mm.CustomCVForce)
    # Check that phi0 is now a collective variable
    assert flipped_force.getNumCollectiveVariables() == 1
    assert flipped_force.getCollectiveVariableName(0) == "phi0"
    # Check that phi and kappa are parameters
    param_names = [
        flipped_force.getGlobalParameterName(i)
        for i in range(flipped_force.getNumGlobalParameters())
    ]
    assert "phi" in param_names
    assert "kappa" in param_names


def test_custom_coupling_get_extension_parameters():
    """Test updateExtensionContext method."""
    model = create_test_system()
    phi = create_test_cv()
    phi0 = create_test_dv("phi0")

    coupling = CollectiveVariableCoupling(
        "0.5*kappa*(phi-phi0)^2",
        [phi],
        [phi0],
        kappa=1000 * mmunit.kilojoules_per_mole / mmunit.radian**2,
    )

    system = ExtendedSpaceSystem(model.system, coupling)
    physical_integrator = mm.VerletIntegrator(1.0 * mmunit.femtosecond)
    physical_context = mm.Context(system, physical_integrator)
    physical_context.setPositions(model.positions)

    # Create extension context
    extension_system = system.getExtensionSystem()
    extension_integrator = mm.VerletIntegrator(1.0 * mmunit.femtosecond)
    extension_context = mm.Context(extension_system, extension_integrator)

    # Update extension context with physical parameters
    coupling.updateExtensionContext(physical_context, extension_context)

    # Verify that phi parameter was set in extension context
    phi_value = extension_context.getParameter("phi")
    assert isinstance(phi_value, float)


def test_custom_coupling_serialization():
    """Test YAML serialization and deserialization of CollectiveVariableCoupling."""
    phi = create_test_cv()
    phi0 = create_test_dv("phi0")
    coupling = CollectiveVariableCoupling(
        "0.5*kappa*(phi-phi0)^2",
        [phi],
        [phi0],
        kappa=1000 * mmunit.kilojoules_per_mole / mmunit.radian**2,
    )

    serialized = yaml.safe_dump(coupling)
    deserialized = yaml.safe_load(serialized)

    assert (
        deserialized.getForce(0).getEnergyFunction()
        == coupling.getForce(0).getEnergyFunction()
    )
    assert (
        deserialized.getForce(0).getNumCollectiveVariables()
        == coupling.getForce(0).getNumCollectiveVariables()
    )


def test_custom_coupling_add_to_system():
    """Test adding CollectiveVariableCoupling to a system."""
    model = create_test_system()
    phi = create_test_cv()
    phi0 = create_test_dv("phi0")
    coupling = CollectiveVariableCoupling(
        "0.5*kappa*(phi-phi0)^2",
        [phi],
        [phi0],
        kappa=1000 * mmunit.kilojoules_per_mole / mmunit.radian**2,
    )

    initial_forces = model.system.getNumForces()
    coupling.addToPhysicalSystem(model.system)
    assert model.system.getNumForces() == initial_forces + 1


# HarmonicCoupling Tests
def test_harmonic_coupling_initialization():
    """Test initialization of HarmonicCoupling with valid arguments."""
    phi = create_test_cv()
    phi_s = create_test_dv()
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    coupling = HarmonicCoupling(phi, phi_s, kappa)

    assert coupling.getForce(0).getNumCollectiveVariables() == 1


def test_harmonic_coupling_incompatible_units():
    """Test HarmonicCoupling with incompatible CV and DV units."""
    # Create a distance CV
    distance_cv = cvpack.Distance(0, 1, name="dist")
    # Create an angle DV
    phi_s = create_test_dv()
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.nanometer**2

    with pytest.raises(ValueError, match="Incompatible units"):
        HarmonicCoupling(distance_cv, phi_s, kappa)


def test_harmonic_coupling_incompatible_periodicity_nonperiodic_cv():
    """Test HarmonicCoupling with periodic DV but non-periodic CV."""
    distance_cv = cvpack.Distance(0, 1, name="dist")
    phi_s = create_test_dv()  # Periodic DV
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    # Units check happens first, so we expect incompatible units error
    with pytest.raises(ValueError, match="Incompatible units"):
        HarmonicCoupling(distance_cv, phi_s, kappa)


def test_harmonic_coupling_incompatible_periodicity_nonperiodic_dv():
    """Test HarmonicCoupling with periodic CV but non-periodic DV."""
    phi = create_test_cv()
    # Create non-periodic DV
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    phi_s = DynamicalVariable("phi_s", mmunit.radian, mass, NoBounds())
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    with pytest.raises(ValueError, match="Incompatible periodicity"):
        HarmonicCoupling(phi, phi_s, kappa)


def test_harmonic_coupling_incompatible_force_constant():
    """Test HarmonicCoupling with wrong force constant units."""
    phi = create_test_cv()
    phi_s = create_test_dv()
    # Wrong units: should be energy/angle^2, not just energy
    kappa = 1000 * mmunit.kilojoule_per_mole

    with pytest.raises(ValueError, match="Incompatible force constant units"):
        HarmonicCoupling(phi, phi_s, kappa)


def test_harmonic_coupling_add_to_extension_system():
    """Test adding a HarmonicCoupling to the extension system."""
    phi = create_test_cv()
    phi_s = create_test_dv()
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    coupling = HarmonicCoupling(phi, phi_s, kappa)

    # Create extension system and add coupling
    extension_system = mm.System()
    extension_system.addParticle(phi_s.mass / phi_s.mass.unit)
    coupling.addToExtensionSystem(extension_system)

    # Check that a force was added to the extension system
    assert extension_system.getNumForces() == 1
    flipped_force = extension_system.getForce(0)

    # Verify it's a CustomCVForce
    assert isinstance(flipped_force, mm.CustomCVForce)
    # Check that phi_s is now a collective variable
    assert flipped_force.getNumCollectiveVariables() == 1
    assert flipped_force.getCollectiveVariableName(0) == "phi_s"
    # Check that phi is now a parameter
    param_names = [
        flipped_force.getGlobalParameterName(i)
        for i in range(flipped_force.getNumGlobalParameters())
    ]
    assert "phi" in param_names


def test_harmonic_coupling_repr():
    """Test string representation of HarmonicCoupling."""
    phi = create_test_cv()
    phi_s = create_test_dv()
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    coupling = HarmonicCoupling(phi, phi_s, kappa)
    repr_str = repr(coupling)

    assert "HarmonicCoupling" in repr_str
    assert "kappa_phi_phi_s" in repr_str


def test_harmonic_coupling_serialization():
    """Test YAML serialization of HarmonicCoupling."""
    phi = create_test_cv()
    phi_s = create_test_dv()
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    coupling = HarmonicCoupling(phi, phi_s, kappa)
    serialized = yaml.safe_dump(coupling)
    deserialized = yaml.safe_load(serialized)

    assert (
        deserialized.getForce(0).getEnergyFunction()
        == coupling.getForce(0).getEnergyFunction()
    )


# CouplingSum Tests
def test_coupling_sum_initialization():
    """Test initialization of CouplingSum."""
    phi = create_test_cv("phi")
    psi = create_test_cv("psi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    force1 = HarmonicCoupling(phi, phi_s, kappa)
    force2 = HarmonicCoupling(psi, psi_s, kappa)

    force_sum = CouplingSum([force1, force2])
    forces = force_sum.getCouplings()

    assert len(forces) == 2
    assert forces[0] is force1
    assert forces[1] is force2


def test_coupling_sum_flattening():
    """Test that nested CouplingSum objects are flattened."""
    phi = create_test_cv("phi")
    psi = create_test_cv("psi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    force1 = HarmonicCoupling(phi, phi_s, kappa)
    force2 = HarmonicCoupling(psi, psi_s, kappa)

    # Create nested sum: sum of (force1) and (sum of force2)
    sum1 = CouplingSum([force1])
    sum2 = CouplingSum([force2])
    nested_sum = CouplingSum([sum1, sum2])

    # Should be flattened to just [force1, force2]
    forces = nested_sum.getCouplings()
    assert len(forces) == 2


def test_coupling_addition_operator():
    """Test the addition operator for couplings."""
    phi = create_test_cv("phi")
    psi = create_test_cv("psi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    force1 = HarmonicCoupling(phi, phi_s, kappa)
    force2 = HarmonicCoupling(psi, psi_s, kappa)

    force_sum = force1 + force2

    assert isinstance(force_sum, CouplingSum)
    forces = force_sum.getCouplings()
    assert len(forces) == 2


def test_coupling_sum_add_to_system():
    """Test that all forces in sum are added to system."""
    model = create_test_system()
    phi = create_test_cv("phi")
    psi = cvpack.Torsion(4, 6, 8, 14, name="psi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    force1 = HarmonicCoupling(phi, phi_s, kappa)
    force2 = HarmonicCoupling(psi, psi_s, kappa)
    force_sum = force1 + force2

    initial_forces = model.system.getNumForces()
    force_sum.addToPhysicalSystem(model.system)

    # Should add 2 forces
    assert model.system.getNumForces() == initial_forces + 2


def test_coupling_sum_add_to_extension_system():
    """Test adding a CouplingSum to the extension system."""
    phi = create_test_cv("phi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    # Create a custom coupling with both DVs
    coupling = CollectiveVariableCoupling(
        "0.5*kappa1*(phi-phi_s)^2 + 0.5*kappa2*(phi-psi_s)^2",
        [phi],
        [phi_s, psi_s],
        kappa1=kappa,
        kappa2=kappa,
    )

    # Create another one for the sum
    psi2 = cvpack.Torsion(4, 6, 8, 14, name="psi2")
    psi_s2 = create_test_dv("psi_s2")
    coupling2 = CollectiveVariableCoupling(
        "0.5*kappa*(psi2-psi_s2)^2",
        [psi2],
        [psi_s2],
        kappa=kappa,
    )

    force_sum = coupling + coupling2

    # Create extension system and add coupling
    extension_system = mm.System()
    extension_system.addParticle(phi_s.mass / phi_s.mass.unit)
    extension_system.addParticle(psi_s.mass / psi_s.mass.unit)
    force_sum.addToExtensionSystem(extension_system)

    # Check that 2 forces were added to the extension system
    assert extension_system.getNumForces() == 2


def test_coupling_sum_get_extension_parameters():
    """Test updateExtensionContext with non-overlapping CVs."""
    model = create_test_system()
    phi = create_test_cv("phi")
    psi = cvpack.Torsion(4, 6, 8, 14, name="psi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    force1 = HarmonicCoupling(phi, phi_s, kappa)
    force2 = HarmonicCoupling(psi, psi_s, kappa)
    force_sum = force1 + force2

    system = ExtendedSpaceSystem(model.system, force_sum)
    physical_integrator = mm.VerletIntegrator(1.0 * mmunit.femtosecond)
    physical_context = mm.Context(system, physical_integrator)
    physical_context.setPositions(model.positions)

    # Create extension context
    extension_system = system.getExtensionSystem()
    extension_integrator = mm.VerletIntegrator(1.0 * mmunit.femtosecond)
    extension_context = mm.Context(extension_system, extension_integrator)

    # Update extension context with physical parameters
    force_sum.updateExtensionContext(physical_context, extension_context)

    # Should have parameters for both phi and psi set in extension context
    phi_value = extension_context.getParameter("phi")
    psi_value = extension_context.getParameter("psi")
    assert isinstance(phi_value, float)
    assert isinstance(psi_value, float)


def test_coupling_sum_conflicting_parameters():
    """Test that conflicting parameters raise an error."""
    phi = create_test_cv("phi")
    phi_s = create_test_dv("phi_s")
    kappa1 = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2
    kappa2 = 500 * mmunit.kilojoule_per_mole / mmunit.radian**2

    # Create two forces with same CV and DV but different force constants
    force1 = HarmonicCoupling(phi, phi_s, kappa1)
    force2 = HarmonicCoupling(phi, phi_s, kappa2)

    # Adding them should raise an error because they have the same parameter name
    # (kappa_phi_phi_s) but different values
    with pytest.raises(ValueError, match="conflicting default values"):
        _ = force1 + force2


def test_coupling_sum_repr():
    """Test string representation of CouplingSum."""
    phi = create_test_cv("phi")
    psi = cvpack.Torsion(4, 6, 8, 14, name="psi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    force1 = HarmonicCoupling(phi, phi_s, kappa)
    force2 = HarmonicCoupling(psi, psi_s, kappa)
    force_sum = force1 + force2

    repr_str = repr(force_sum)
    assert "HarmonicCoupling" in repr_str
    assert "+" in repr_str


def test_coupling_sum_copy():
    """Test copying a CouplingSum."""

    phi = create_test_cv("phi")
    psi = cvpack.Torsion(4, 6, 8, 14, name="psi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    force1 = HarmonicCoupling(phi, phi_s, kappa)
    force2 = HarmonicCoupling(psi, psi_s, kappa)
    force_sum = force1 + force2

    copied = copy(force_sum)

    assert isinstance(copied, CouplingSum)
    assert len(copied.getCouplings()) == len(force_sum.getCouplings())


def test_coupling_sum_serialization():
    """Test YAML serialization of CouplingSum."""
    phi = create_test_cv("phi")
    psi = cvpack.Torsion(4, 6, 8, 14, name="psi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    force1 = HarmonicCoupling(phi, phi_s, kappa)
    force2 = HarmonicCoupling(psi, psi_s, kappa)
    force_sum = force1 + force2

    serialized = yaml.safe_dump(force_sum)
    deserialized = yaml.safe_load(serialized)

    assert isinstance(deserialized, CouplingSum)
    assert len(deserialized.getCouplings()) == 2


# Integration Tests
def test_coupling_with_extended_space_system():
    """Test CollectiveVariableCoupling with ExtendedSpaceSystem."""
    model = create_test_system()
    phi = create_test_cv()
    phi_s = create_test_dv("phi_s")

    coupling = CollectiveVariableCoupling(
        "0.5*kappa*(phi-phi_s)^2",
        [phi],
        [phi_s],
        kappa=1000 * mmunit.kilojoules_per_mole / mmunit.radian**2,
    )

    initial_physical_forces = model.system.getNumForces()
    system = ExtendedSpaceSystem(model.system, coupling)

    # Physical system should have one more force
    assert system.getNumForces() == initial_physical_forces + 1

    # Extension system should have one force (the flipped one)
    extension_system = system.getExtensionSystem()
    assert extension_system.getNumForces() == 1
    assert extension_system.getNumParticles() == 1


def test_harmonic_coupling_integration():
    """Test HarmonicCoupling integration with ExtendedSpaceSystem."""
    model = create_test_system()
    phi = create_test_cv()
    phi_s = create_test_dv()
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    coupling = HarmonicCoupling(phi, phi_s, kappa)
    system = ExtendedSpaceSystem(model.system, coupling)

    # Check dynamical variables
    dvs = system.getDynamicalVariables()
    assert len(dvs) == 1
    assert dvs[0].name == "phi_s"

    # Check coupling
    retrieved_force = system.getCoupling()
    assert retrieved_force is coupling


def test_coupling_sum_integration():
    """Test CouplingSum with ExtendedSpaceSystem."""
    model = create_test_system()
    phi = create_test_cv("phi")
    psi = cvpack.Torsion(4, 6, 8, 14, name="psi")
    phi_s = create_test_dv("phi_s")
    psi_s = create_test_dv("psi_s")
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.radian**2

    force1 = HarmonicCoupling(phi, phi_s, kappa)
    force2 = HarmonicCoupling(psi, psi_s, kappa)
    force_sum = force1 + force2

    initial_forces = model.system.getNumForces()
    system = ExtendedSpaceSystem(model.system, force_sum)

    # Physical system should have 2 more forces
    assert system.getNumForces() == initial_forces + 2

    # Extension system should have 2 forces
    extension_system = system.getExtensionSystem()
    assert extension_system.getNumForces() == 2
    assert extension_system.getNumParticles() == 2
