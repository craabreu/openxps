"""
Unit tests for the openxps.context module.
"""

import cvpack
import numpy as np
import openmm as mm
import pytest
from openmm import unit as mmunit
from openmmtools import testsystems

from openxps.bounds import CIRCULAR, Reflective
from openxps.context import ExtendedSpaceContext
from openxps.dynamical_variable import DynamicalVariable


def system_integrator_platform(model):
    """Helper function to create a basic OpenMM Context object."""
    integrator = mm.VerletIntegrator(1.0 * mmunit.femtosecond)
    platform = mm.Platform.getPlatformByName("Reference")
    return model.system, integrator, platform


def create_dvs():
    """Helper function to create a DynamicalVariable object."""
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    kwargs = {"unit": mmunit.nanometer, "mass": 1 * mmunit.dalton}
    return [
        DynamicalVariable(name="phi0", unit=mmunit.radian, mass=mass, bounds=CIRCULAR),
        DynamicalVariable(name="x0", bounds=None, **kwargs),
        DynamicalVariable(
            name="y0", bounds=Reflective(-1, 1, mmunit.nanometer), **kwargs
        ),
    ]


def create_coupling_potential(
    phi0=180 * mmunit.degrees, unit=mmunit.kilojoule_per_mole
):
    """Helper function to create a MetaCollectiveVariable object."""
    kwargs = {
        "kappa": 1000 * mmunit.kilojoule_per_mole / mmunit.radians**2,
        "alpha": 0.01 * mmunit.kilojoule_per_mole / mmunit.nanometer**2,
        "x0": 1 * mmunit.nanometer,
        "y0": 1 * mmunit.nanometer,
    }
    if phi0 is not None:
        kwargs["phi0"] = phi0
    return cvpack.MetaCollectiveVariable(
        f"0.5*kappa*min(delta_phi,{2 * np.pi}-delta_phi)^2+alpha*(x0-y0)^2"
        "; delta_phi=abs(phi-phi0)",
        [cvpack.Torsion(6, 8, 14, 16, name="phi")],
        unit,
        **kwargs,
    )


def create_extended_context(model, coupling_potential=None):
    """Helper function to create an ExtendedSpaceContext object."""
    return ExtendedSpaceContext(
        create_dvs(),
        coupling_potential or create_coupling_potential(),
        *system_integrator_platform(model),
    )


def test_initialization():
    """Test the initialization of ExtendedSpaceContext with basic setup."""
    assert create_extended_context(testsystems.AlanineDipeptideVacuum()) is not None


def test_set_positions_and_velocities():
    """Test setting positions and velocities including DVs."""
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
    context.setDynamicalVariableValues([1 * urad, 0.1 * unm, 0.1 * unm])
    context.setDynamicalVariableVelocities(
        [1 * urad / ups, 1 * unm / ups, 1 * unm / ups]
    )

    state = context.getState(  # pylint: disable=unexpected-keyword-arg
        getPositions=True, getVelocities=True
    )
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

    state = context.getState(  # pylint: disable=unexpected-keyword-arg
        getVelocities=True
    )
    velocities = state.getVelocities()
    extra_velocities = context.getDynamicalVariableVelocities()
    assert len(velocities) == len(model.positions)
    assert len(extra_velocities) == 3


def test_validation():
    """Test the validation of extended space context."""
    model = testsystems.AlanineDipeptideVacuum()
    dvs = create_dvs()
    coupling_potential = create_coupling_potential()

    with pytest.raises(TypeError) as e:
        ExtendedSpaceContext(
            [None], coupling_potential, *system_integrator_platform(model)
        )
    assert "dynamical variables must be instances of DynamicalVariable" in str(e.value)

    with pytest.raises(TypeError) as e:
        ExtendedSpaceContext(dvs, None, *system_integrator_platform(model))
    assert "must be an instance of MetaCollectiveVariable" in str(e.value)

    with pytest.raises(ValueError) as e:
        ExtendedSpaceContext(
            dvs,
            create_coupling_potential(phi0=None),
            *system_integrator_platform(model),
        )
    assert "dynamical variables are not coupling potential parameters" in str(e.value)

    with pytest.raises(ValueError) as e:
        ExtendedSpaceContext(
            dvs,
            create_coupling_potential(unit=mmunit.radian),
            *system_integrator_platform(model),
        )
    assert "The coupling potential must have units of molar energy." in str(e.value)


def test_consistency():
    """Test the consistency of the extended space context."""
    model = testsystems.AlanineDipeptideVacuum()
    coupling = create_coupling_potential()
    context = create_extended_context(model, coupling)
    context.setPositions(model.positions)
    context.setVelocitiesToTemperature(300 * mmunit.kelvin)
    context.setDynamicalVariableValues(
        [1000 * mmunit.degrees, 1 * mmunit.nanometer, 1 * mmunit.nanometer]
    )
    context.setDynamicalVariableVelocitiesToTemperature(300 * mmunit.kelvin)

    for _ in range(10):
        context.getIntegrator().step(1000)

        # pylint: disable=unexpected-keyword-arg
        extension_state = context.getExtensionContext().getState(
            getEnergy=True, getPositions=True, getForces=True
        )
        # pylint: enable=unexpected-keyword-arg

        # Check the consistency of the potential energy
        x1 = extension_state.getPotentialEnergy() / mmunit.kilojoule_per_mole
        x2 = coupling.getValue(context) / mmunit.kilojoule_per_mole
        assert x1 == pytest.approx(x2)

        # Check the consistency of the energy parameter derivatives
        positions = extension_state.getPositions()
        forces = extension_state.getForces()
        x1 = {}
        for i, dv in enumerate(context.getDynamicalVariables()):
            force = forces[i].x
            if dv.bounds is not None:
                _, force = dv.bounds.wrap(positions[i].x, force)
            x1[dv.name] = -force
        x2 = coupling.getParameterDerivatives(context)
        for dv in context.getDynamicalVariables():
            assert x1[dv.name] == pytest.approx(x2[dv.name] / x2[dv.name].unit)
