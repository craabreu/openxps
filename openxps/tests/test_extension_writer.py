"""
Test the extension writer
"""

import os
import tempfile

import cvpack
import openmm
from openmm import unit
from openmmtools import testsystems

import openxps as xps


def test_extension_writer():
    """
    Test the extension writer
    """

    model = testsystems.AlanineDipeptideVacuum()
    mass = 3 * unit.dalton * (unit.nanometer / unit.radian) ** 2
    phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.bounds.CIRCULAR)
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    kappa = 1000 * unit.kilojoules_per_mole / unit.radian**2
    harmonic_force = xps.HarmonicCoupling(phi, phi0, kappa)
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond
    )
    integrator.setRandomNumberSeed(1234)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, harmonic_force),
        xps.LockstepIntegrator(integrator),
        platform,
    )
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    simulation.context.setDynamicalVariableValues([180 * unit.degree])
    simulation.context.setDynamicalVariableVelocitiesToTemperature(
        300 * unit.kelvin, 1234
    )
    with tempfile.TemporaryDirectory() as dirpath:
        with open(os.path.join(dirpath, "report.csv"), "w", encoding="utf-8") as file:
            reporter = cvpack.reporting.StateDataReporter(
                file,
                10,
                step=True,
                writers=[
                    xps.ExtensionWriter(
                        potential=True,
                        kinetic=True,
                        total=True,
                        temperature=True,
                    )
                ],
            )
            simulation.reporters.append(reporter)
            simulation.step(100)
        with open(os.path.join(dirpath, "report.csv"), encoding="utf-8") as file:
            assert file.readline() == ",".join(
                [
                    '#"Step"',
                    '"Extension Potential Energy (kJ/mole)"',
                    '"Extension Kinetic Energy (kJ/mole)"',
                    '"Extension Total Energy (kJ/mole)"',
                    '"Extension Temperature (K)"\n',
                ]
            )
