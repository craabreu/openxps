"""
Unit tests for the openxps.metadynamics module.
"""

import cvpack
import openmm
import pytest
import yaml
from openmm import unit
from openmmtools import testsystems

import openxps as xps


def _system_with(dv, cv=None):
    model = testsystems.AlanineDipeptideVacuum()
    cv = cv or cvpack.Torsion(6, 8, 14, 16, name="phi")
    kappa = 1000 * unit.kilojoules_per_mole / dv.unit**2
    coupling = xps.HarmonicCoupling(cv, dv, kappa)
    return model, xps.ExtendedSpaceSystem(model.system, coupling)


def _metadynamics(system, variable, **kwargs):
    return xps.ExtendedSpaceMetadynamics(
        system=system,
        variables=[variable],
        temperature=300 * unit.kelvin,
        biasFactor=5,
        height=2 * unit.kilojoule_per_mole,
        frequency=10,
        **kwargs,
    )


def test_bias_variable_rejects_non_positive_sigma():
    mass = 3 * unit.dalton * (unit.nanometer / unit.radian) ** 2
    dv = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    with pytest.raises(ValueError, match="sigma must be positive"):
        xps.ExtendedSpaceBiasVariable(dv, -18 * unit.degree)
    with pytest.raises(ValueError, match="grid_width must be greater than 1"):
        xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree, grid_width=1)


def test_bias_variable_yaml_roundtrip():
    mass = 3 * unit.dalton * (unit.nanometer / unit.radian) ** 2
    dv = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    variable = xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree, grid_width=50)
    assert yaml.safe_load(yaml.safe_dump(variable)) == variable


def test_metadynamics_with_degree_dv_uses_radian_grid():
    mass = 3 * unit.dalton * (unit.nanometer / unit.degree) ** 2
    dv = xps.DynamicalVariable(
        "phi0", unit.degree, mass, xps.PeriodicBounds(-180, 180, unit.degree)
    )
    _, system = _system_with(dv)
    meta = _metadynamics(system, xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree))
    (variable,) = meta.variables
    assert variable.minValue == pytest.approx(-3.14159, rel=1e-5)
    assert variable.maxValue == pytest.approx(3.14159, rel=1e-5)
    assert variable.periodic


def test_metadynamics_with_reflective_dv():
    mass = 10 * unit.dalton
    dv = xps.DynamicalVariable(
        "d0", unit.nanometer, mass, xps.ReflectiveBounds(0.1, 1.0, unit.nanometer)
    )
    _, system = _system_with(dv, cvpack.Distance(0, 1, name="d"))
    meta = _metadynamics(
        system, xps.ExtendedSpaceBiasVariable(dv, 0.05 * unit.nanometer)
    )
    (variable,) = meta.variables
    assert not variable.periodic


def test_metadynamics_rejects_unknown_and_unbounded_dvs():
    mass = 3 * unit.dalton * (unit.nanometer / unit.radian) ** 2
    dv = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    _, system = _system_with(dv)
    other = xps.DynamicalVariable("psi0", unit.radian, mass, xps.CircularBounds())
    with pytest.raises(ValueError, match="not found in system"):
        _metadynamics(system, xps.ExtendedSpaceBiasVariable(other, 18 * unit.degree))
    free = xps.DynamicalVariable(
        "x0", unit.nanometer, 1 * unit.dalton, xps.NoBounds(0, 1, unit.nanometer)
    )
    _, free_system = _system_with(free, cvpack.Distance(0, 1, name="d"))
    with pytest.raises(ValueError, match="has no bounds"):
        _metadynamics(
            free_system, xps.ExtendedSpaceBiasVariable(free, 0.1 * unit.nanometer)
        )


def test_metadynamics_step_and_collective_variables():
    model = testsystems.AlanineDipeptideVacuum()
    mass = 3 * unit.dalton * (unit.nanometer / unit.radian) ** 2
    dv = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    _, system = _system_with(dv)
    meta = _metadynamics(system, xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree))
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond
    )
    integrator.setRandomNumberSeed(1)
    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        system,
        xps.LockstepIntegrator(integrator),
        openmm.Platform.getPlatformByName("Reference"),
    )
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    simulation.context.setDynamicalVariableValues([180 * unit.degree])
    simulation.context.setDynamicalVariableVelocitiesToTemperature(300 * unit.kelvin, 1)
    meta.step(simulation, 50)
    (value,) = meta.getCollectiveVariables(simulation)
    (dv_value,) = simulation.context.getDynamicalVariableValues()
    assert value == pytest.approx(dv_value.value_in_unit(unit.radian), abs=1e-6)
    assert meta.getFreeEnergy().shape == (meta.variables[0].gridWidth,)
