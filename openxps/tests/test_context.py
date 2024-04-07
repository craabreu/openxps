"""
Unit tests for the openxps.context module.
"""

import cvpack
import numpy as np
import openmm as mm
import pytest
from openmm import unit as mmunit
from openmmtools import testsystems

from openxps.bounds import CIRCULAR
from openxps.context import ExtendedSpaceContext
from openxps.extra_dof import ExtraDOF


def create_basic_context(model):
    """Helper function to create a basic OpenMM Context object."""
    integrator = mm.VerletIntegrator(1.0 * mmunit.femtosecond)
    platform = mm.Platform.getPlatformByName("Reference")
    return mm.Context(model.system, integrator, platform)


def create_extra_dofs():
    """Helper function to create an ExtraDOF object."""
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    return [
        ExtraDOF(name="phi0", unit=mmunit.radian, mass=mass, bounds=CIRCULAR),
        ExtraDOF(name="x0", unit=mmunit.nanometer, mass=1 * mmunit.dalton, bounds=None),
    ]


def create_coupling_potential(
    phi0=180 * mmunit.degrees, unit=mmunit.kilojoule_per_mole
):
    """Helper function to create a MetaCollectiveVariable object."""
    kwargs = {
        "kappa": 1000 * mmunit.kilojoule_per_mole / mmunit.radians**2,
        "alpha": 0.01 * mmunit.kilojoule_per_mole / mmunit.nanometer,
        "x0": 1 * mmunit.nanometer,
    }
    if phi0 is not None:
        kwargs["phi0"] = phi0
    return cvpack.MetaCollectiveVariable(
        f"0.5*kappa*min(delta_phi,{2*np.pi}-delta_phi)^2+alpha*x0"
        "; delta_phi=abs(phi-phi0)",
        [cvpack.Torsion(6, 8, 14, 16, name="phi")],
        unit,
        **kwargs,
    )


def create_extended_context(model, coupling_potential=None):
    """Helper function to create an ExtendedSpaceContext object."""
    return ExtendedSpaceContext(
        create_basic_context(model),
        create_extra_dofs(),
        coupling_potential or create_coupling_potential(),
    )


def test_initialization():
    """Test the initialization of ExtendedSpaceContext with basic setup."""
    assert create_extended_context(testsystems.AlanineDipeptideVacuum()) is not None


def test_set_positions_and_velocities():
    """Test setting positions and velocities including extra DOFs."""
    model = testsystems.AlanineDipeptideVacuum()
    context = create_extended_context(model)

    random = np.random.RandomState()  # pylint: disable=no-member
    num_atoms = context.getSystem().getNumParticles()
    positions = model.positions.value_in_unit(mmunit.nanometer)
    velocities = random.uniform(-1, 1, (num_atoms, 3))

    urad = mmunit.radians
    unm = mmunit.nanometers
    ups = mmunit.picoseconds

    context.setPositions(positions)
    context.setVelocities(velocities)
    context.setExtraValues([1 * urad, 0.1 * unm])
    context.setExtraVelocities([1 * urad / ups, 1 * unm / ups])

    state = context.getState(  # pylint: disable=unexpected-keyword-arg
        getPositions=True, getVelocities=True
    )
    assert state.getPositions(asNumpy=True) == pytest.approx(positions) * unm
    assert state.getVelocities() == pytest.approx(velocities) * unm / ups
    assert context.getExtraValues() == (
        pytest.approx(1) * urad,
        pytest.approx(0.1) * unm,
    )
    assert context.getExtraVelocities() == (
        pytest.approx(1) * urad / ups,
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
        context.setExtraVelocitiesToTemperature(300 * mmunit.kelvin)
    assert "Particle positions have not been set" in str(e.value)

    with pytest.raises(mm.OpenMMException) as e:
        context.getIntegrator().step(1)
    assert "Particle positions have not been set" in str(e.value)

    context.setExtraValues([1 * mmunit.radian, 0.1 * mmunit.nanometer])
    context.setExtraVelocitiesToTemperature(300 * mmunit.kelvin)

    state = context.getState(  # pylint: disable=unexpected-keyword-arg
        getVelocities=True
    )
    velocities = state.getVelocities()
    extra_velocities = context.getExtraVelocities()
    assert len(velocities) == len(model.positions)
    assert len(extra_velocities) == 2


def test_validation():
    """Test the validation of extended space context."""
    model = testsystems.AlanineDipeptideVacuum()
    context = create_basic_context(model)
    extra_dofs = create_extra_dofs()
    coupling_potential = create_coupling_potential()

    with pytest.raises(TypeError) as e:
        ExtendedSpaceContext(context, [None], coupling_potential)
    assert "extra degrees of freedom must be instances of ExtraDOF" in str(e.value)

    with pytest.raises(TypeError) as e:
        ExtendedSpaceContext(context, extra_dofs, None)
    assert "must be an instance of MetaCollectiveVariable" in str(e.value)

    with pytest.raises(ValueError) as e:
        ExtendedSpaceContext(context, extra_dofs, create_coupling_potential(phi0=None))
    assert "The coupling potential parameters do not include ['phi0']." in str(e.value)

    with pytest.raises(ValueError) as e:
        ExtendedSpaceContext(
            context, extra_dofs, create_coupling_potential(phi0=1 * mmunit.kelvin)
        )
    assert "Unit mismatch for parameter 'phi0'." in str(e.value)

    with pytest.raises(ValueError) as e:
        ExtendedSpaceContext(
            context, extra_dofs, create_coupling_potential(unit=mmunit.radian)
        )
    assert "The coupling potential must have units of molar energy." in str(e.value)

    with pytest.raises(ValueError) as e:
        force = mm.CustomExternalForce("phi0*x")
        force.addGlobalParameter("phi0", 180 * mmunit.degrees)
        force.addGlobalParameter("x0", 1 * mmunit.nanometer)
        context.getSystem().addForce(force)
        context.reinitialize()
        ExtendedSpaceContext(context, extra_dofs, coupling_potential)
    assert "The context already contains ['phi0', 'x0'] among its parameters." in str(
        e.value
    )


def test_consistency():
    """Test the consistency of the extended space context."""
    model = testsystems.AlanineDipeptideVacuum()
    coupling = create_coupling_potential()
    coupling.addEnergyParameterDerivative("phi0")
    coupling.addEnergyParameterDerivative("x0")
    context = create_extended_context(model, coupling)
    context.setPositions(model.positions)
    context.setVelocitiesToTemperature(300 * mmunit.kelvin)
    context.setExtraValues([1000 * mmunit.degrees, 1 * mmunit.nanometer])
    context.setExtraVelocitiesToTemperature(300 * mmunit.kelvin)

    for _ in range(10):
        context.getIntegrator().step(1000)

        # pylint: disable=unexpected-keyword-arg
        physical_state = context.getState(getParameterDerivatives=True)
        extension_state = context.getExtensionState(getEnergy=True, getForces=True)
        # pylint: enable=unexpected-keyword-arg

        # Check the consistency of the potential energy
        x1 = extension_state.getPotentialEnergy() / mmunit.kilojoule_per_mole
        x2 = coupling.getValue(context) / mmunit.kilojoule_per_mole
        assert x1 == pytest.approx(x2)

        # Check the consistency of the energy parameter derivatives
        x1 = {
            xdof.name: -force.x
            for xdof, force in zip(context.getExtraDOFs(), extension_state.getForces())
        }
        x2 = physical_state.getEnergyParameterDerivatives()
        for xdof in context.getExtraDOFs():
            assert x1[xdof.name] == pytest.approx(x2[xdof.name])
