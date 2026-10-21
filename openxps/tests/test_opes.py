"""Tests for the extended-space OPES samplers."""

import cvpack
import numpy as np
import openmm
import pytest
from openmm import app, unit
from openmmtools import testsystems

import openxps as xps
from openxps.opes import _biasVariables, _extensionContext, _extensionSimulation

REFERENCE = openmm.Platform.getPlatformByName("Reference")
KELVIN = unit.kelvin
MASS = 3 * unit.dalton * (unit.nanometer / unit.radian) ** 2


def _circular_dv(name="phi0"):
    return xps.DynamicalVariable(name, unit.radian, MASS, xps.CircularBounds())


def _reflective_dv(name="s", lower=0.0, upper=1.0):
    """A length-like variable, so that it can couple to a distance."""
    return xps.DynamicalVariable(
        name,
        unit.nanometer,
        1 * unit.dalton,
        xps.ReflectiveBounds(lower, upper, unit.nanometer),
    )


def _alanine_system(dv, cv=None):
    """Alanine dipeptide with a harmonic coupling between a torsion and ``dv``."""
    model = testsystems.AlanineDipeptideVacuum()
    cv = cv or cvpack.Torsion(6, 8, 14, 16, name="phi")
    kappa = 1000 * unit.kilojoules_per_mole / dv.unit**2
    return model, xps.ExtendedSpaceSystem(
        model.system, xps.HarmonicCoupling(cv, dv, kappa)
    )


def test_bias_variables_follow_the_dynamical_variables():
    dv = _circular_dv()
    _, system = _alanine_system(dv)
    variable = xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree, grid_width=73)
    (bias_variable,), bounded = _biasVariables(system, [variable])
    assert isinstance(bias_variable, app.BiasVariable)
    assert bias_variable.periodic
    assert not bounded
    assert bias_variable.minValue == pytest.approx(-np.pi)
    assert bias_variable.maxValue == pytest.approx(np.pi)
    assert bias_variable.biasWidth == pytest.approx(np.radians(18))
    assert bias_variable.gridWidth == 73


def test_reflective_variable_sets_bounded():
    dv = _reflective_dv()
    _, system = _alanine_system(dv, cv=cvpack.Distance(0, 1, name="d"))
    variable = xps.ExtendedSpaceBiasVariable(dv, 0.05 * unit.nanometer)
    (bias_variable,), bounded = _biasVariables(system, [variable])
    assert bounded
    assert not bias_variable.periodic
    assert (bias_variable.minValue, bias_variable.maxValue) == (0.0, 1.0)


def test_unknown_and_unbounded_variables_are_rejected():
    dv = _circular_dv()
    _, system = _alanine_system(dv)
    other = xps.ExtendedSpaceBiasVariable(_circular_dv("psi0"), 18 * unit.degree)
    with pytest.raises(ValueError, match="not found"):
        _biasVariables(system, [other])
    free = xps.DynamicalVariable(
        "free", unit.nanometer, 1 * unit.dalton, xps.NoBounds(0, 1, unit.nanometer)
    )
    _, free_system = _alanine_system(free, cv=cvpack.Distance(0, 1, name="d"))
    with pytest.raises(ValueError, match="no bounds"):
        _biasVariables(
            free_system, [xps.ExtendedSpaceBiasVariable(free, 0.1 * unit.nanometer)]
        )


def test_conversions_pass_converted_objects_through():
    dv = _circular_dv()
    model, system = _alanine_system(dv)
    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        system,
        xps.LockstepIntegrator(openmm.VerletIntegrator(1 * unit.femtosecond)),
        REFERENCE,
    )
    wrapped = _extensionSimulation(simulation)
    assert wrapped.context is simulation.context.getExtensionContext()
    assert _extensionSimulation(wrapped) is wrapped
    extension = _extensionContext(simulation.context)
    assert extension is simulation.context.getExtensionContext()
    assert _extensionContext(extension) is extension
