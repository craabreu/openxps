"""
Unit tests for the openxps.metadynamics module.
"""

import functools
import itertools

import cvpack
import numpy as np
import openmm
import pytest
import yaml
from openmm import unit
from openmmtools import testsystems

import openxps as xps
from openxps.metadynamics import _axis_kernel


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


def test_metadynamics_rejects_unknown_dvs():
    mass = 3 * unit.dalton * (unit.nanometer / unit.radian) ** 2
    dv = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    _, system = _system_with(dv)
    other = xps.DynamicalVariable("psi0", unit.radian, mass, xps.CircularBounds())
    with pytest.raises(ValueError, match="not found in system"):
        _metadynamics(system, xps.ExtendedSpaceBiasVariable(other, 18 * unit.degree))


def test_metadynamics_rejects_unbounded_dvs():
    free = xps.DynamicalVariable(
        "x0", unit.nanometer, 1 * unit.dalton, xps.NoBounds(0, 1, unit.nanometer)
    )
    _, system = _system_with(free, cvpack.Distance(0, 1, name="d"))
    with pytest.raises(ValueError, match="has no bounds.*Use ReflectiveBounds"):
        _metadynamics(system, xps.ExtendedSpaceBiasVariable(free, 0.1 * unit.nanometer))


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


def _reflective_dv(name, lower, upper):
    return xps.DynamicalVariable(
        name,
        unit.nanometer,
        10 * unit.dalton,
        xps.ReflectiveBounds(lower, upper, unit.nanometer),
    )


def _deposit_uniformly(meta, extension_system, num_per_axis):
    """Deposit kernels on a uniform lattice of centers, mimicking flat sampling."""
    context = openmm.Context(
        extension_system,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    axes = [
        v.minValue
        + (np.arange(num_per_axis) + 0.5) * (v.maxValue - v.minValue) / num_per_axis
        for v in meta.variables
    ]
    for center in itertools.product(*axes):
        meta._addGaussian(list(center), 1 * unit.kilojoule_per_mole, context)


@pytest.mark.parametrize("num_dvs", [1, 2])
def test_uniform_sampling_gives_flat_bias_with_reflective_walls(num_dvs):
    """Without mirrored kernels, the bias sags by up to half within sigma of a wall."""
    dvs = [_reflective_dv(f"s{i}", 0.1, 1.0) for i in range(num_dvs)]
    cvs = [cvpack.Distance(0, 1 + i, name=f"d{i}") for i in range(num_dvs)]
    model = testsystems.AlanineDipeptideVacuum()
    kappa = 1000 * unit.kilojoules_per_mole / unit.nanometer**2
    coupling = xps.HarmonicCoupling(cvs[0], dvs[0], kappa)
    for cv, dv in zip(cvs[1:], dvs[1:]):
        coupling += xps.HarmonicCoupling(cv, dv, kappa)
    system = xps.ExtendedSpaceSystem(model.system, coupling)
    meta = xps.ExtendedSpaceMetadynamics(
        system,
        [xps.ExtendedSpaceBiasVariable(dv, 0.05 * unit.nanometer) for dv in dvs],
        300 * unit.kelvin,
        5,
        1 * unit.kilojoule_per_mole,
        10,
    )
    _deposit_uniformly(meta, system.getExtensionSystem(), 90 if num_dvs == 1 else 45)
    bias = meta._totalBias
    assert bias.min() == pytest.approx(bias.max(), rel=1e-3)


def test_reflective_kernel_is_mirror_symmetric_at_both_walls():
    lower, upper, sigma = 0.1, 1.0, 0.2
    offsets = np.linspace(0, 0.5, 11)
    for center in (0.1, 0.15, 0.6, 0.97):
        kernel = functools.partial(
            _axis_kernel,
            center=center,
            sigma=sigma,
            bounds="reflective",
            lower=lower,
            upper=upper,
        )
        np.testing.assert_allclose(kernel(lower - offsets), kernel(lower + offsets))
        np.testing.assert_allclose(kernel(upper - offsets), kernel(upper + offsets))


def test_kernels_match_gaussian_for_small_widths():
    """With sigma a hundredth of the period, kernels agree to 0.1% of their peak."""
    grid = np.linspace(0, 10, 1001)
    gaussian = np.exp(-0.5 * ((grid - 5) / 0.1) ** 2)
    for kind in ("periodic", "reflective"):
        kernel = _axis_kernel(grid, 5.0, 0.1, kind, 0.0, 10.0)
        np.testing.assert_allclose(kernel, gaussian, atol=1e-3)


def test_periodic_kernel_is_periodic():
    grid = np.linspace(-np.pi, np.pi, 101)
    kernel = _axis_kernel(grid, 3.0, 0.5, "periodic", -np.pi, np.pi)
    shifted = _axis_kernel(grid + 2 * np.pi, 3.0, 0.5, "periodic", -np.pi, np.pi)
    np.testing.assert_allclose(kernel, shifted, atol=1e-12)
    assert kernel[0] == pytest.approx(kernel[-1])
