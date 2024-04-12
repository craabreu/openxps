"""
Test the extension writer
"""

import os
import tempfile
from math import pi

import cvpack
import openmm
import pytest
from openmm import app, unit
from openmmtools import testsystems

import openxps as xps


def test_extension_writer():
    """
    Test the extension writer
    """

    model = testsystems.AlanineDipeptideVacuum()
    umbrella_potential = cvpack.MetaCollectiveVariable(
        f"0.5*kappa*min(delta,{2*pi}-delta)^2; delta=abs(phi-phi0)",
        [cvpack.Torsion(6, 8, 14, 16, name="phi")],
        unit.kilojoule_per_mole,
        kappa=1000 * unit.kilojoule_per_mole / unit.radian**2,
        phi0=pi * unit.radian,
    )
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond
    )
    integrator.setRandomNumberSeed(1234)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = app.Simulation(model.topology, model.system, integrator, platform)
    mass = 3 * unit.dalton * (unit.nanometer / unit.radian) ** 2
    phi0 = xps.ExtraDOF("phi0", unit.radian, mass, xps.bounds.CIRCULAR)
    context = xps.ExtendedSpaceContext(simulation.context, [phi0], umbrella_potential)
    context.setPositions(model.positions)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    context.setExtraValues([180 * unit.degree])
    context.setExtraVelocitiesToTemperature(300 * unit.kelvin, 1234)
    with tempfile.TemporaryDirectory() as dirpath:
        with open(os.path.join(dirpath, "report.csv"), "w", encoding="utf-8") as file:
            reporter = cvpack.reporting.StateDataReporter(
                file,
                10,
                step=True,
                writers=[
                    xps.ExtensionWriter(
                        context,
                        potential=True,
                        kinetic=True,
                        total=True,
                        temperature=True,
                    )
                ],
            )
            simulation.reporters.append(reporter)
            simulation.step(100)
        with open(os.path.join(dirpath, "report.csv"), "r", encoding="utf-8") as file:
            assert file.readline() == ",".join(
                [
                    '#"Step"',
                    '"Extension Potential Energy (kJ/mole)"',
                    '"Extension Kinetic Energy (kJ/mole)"',
                    '"Extension Total Energy (kJ/mole)"',
                    '"Extension Temperature (K)"\n',
                ]
            )


def test_raise_exception():
    """
    Test the extension writer raises an exception
    """

    model = testsystems.AlanineDipeptideVacuum()
    integrator = openmm.VerletIntegrator(0.001)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, integrator, platform)
    with pytest.raises(TypeError) as excinfo:
        xps.ExtensionWriter(context, potential=True)
    assert "The context must be an instance of ExtendedSpaceContext" in str(
        excinfo.value
    )
