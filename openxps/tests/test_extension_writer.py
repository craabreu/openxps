"""
Test the extension writer
"""

import os
import tempfile
from math import pi

import cvpack
import openmm
from openmm import unit
from openmmtools import testsystems

import openxps as xps


def create_test_simulation():
    """Helper function to create a test simulation."""
    model = testsystems.AlanineDipeptideVacuum()
    mass = 3 * unit.dalton * (unit.nanometer / unit.radian) ** 2
    phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
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
    return simulation


def test_extension_writer_kinetic_and_temperature():
    """Test the extension writer with kinetic and temperature."""
    simulation = create_test_simulation()
    with tempfile.TemporaryDirectory() as dirpath:
        with open(os.path.join(dirpath, "report.csv"), "w", encoding="utf-8") as file:
            reporter = cvpack.reporting.StateDataReporter(
                file,
                10,
                step=True,
                writers=[
                    xps.ExtensionWriter(
                        kinetic=True,
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
                    '"Extension Kinetic Energy (kJ/mole)"',
                    '"Extension Temperature (K)"\n',
                ]
            )


def test_extension_writer_dynamical_variables():
    """Test the extension writer with dynamical variables."""
    simulation = create_test_simulation()
    with tempfile.TemporaryDirectory() as dirpath:
        with open(os.path.join(dirpath, "report.csv"), "w", encoding="utf-8") as file:
            reporter = cvpack.reporting.StateDataReporter(
                file,
                10,
                step=True,
                writers=[
                    xps.ExtensionWriter(
                        dynamical_variables=True,
                    )
                ],
            )
            simulation.reporters.append(reporter)
            simulation.step(100)
        with open(os.path.join(dirpath, "report.csv"), encoding="utf-8") as file:
            line = file.readline()
            assert '#"Step"' in line
            assert '"phi0 (rad)"' in line


def test_extension_writer_forces():
    """Test the extension writer with forces on dynamical variables."""
    simulation = create_test_simulation()
    with tempfile.TemporaryDirectory() as dirpath:
        with open(os.path.join(dirpath, "report.csv"), "w", encoding="utf-8") as file:
            reporter = cvpack.reporting.StateDataReporter(
                file,
                10,
                step=True,
                writers=[
                    xps.ExtensionWriter(
                        forces=True,
                    )
                ],
            )
            simulation.reporters.append(reporter)
            simulation.step(100)
        with open(os.path.join(dirpath, "report.csv"), encoding="utf-8") as file:
            line = file.readline()
            assert '#"Step"' in line
            assert '"Force on phi0 (kJ/(mol*rad))"' in line


def test_extension_writer_collective_variables():
    """Test the extension writer with collective variables."""
    model = testsystems.AlanineDipeptideVacuum()
    mass = 3 * unit.dalton * (unit.nanometer / unit.radian) ** 2
    phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    # Use CollectiveVariableCoupling which creates a MetaCollectiveVariable
    coupling = xps.CollectiveVariableCoupling(
        "0.5*kappa*(phi-phi0)^2",
        [phi],
        [phi0],
        kappa=1000 * unit.kilojoules_per_mole / unit.radian**2,
    )
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond
    )
    integrator.setRandomNumberSeed(1234)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling),
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
                        collective_variables=True,
                    )
                ],
            )
            simulation.reporters.append(reporter)
            simulation.step(100)
        with open(os.path.join(dirpath, "report.csv"), encoding="utf-8") as file:
            line = file.readline()
            assert '#"Step"' in line
            assert '"phi (rad)"' in line


def test_extension_writer_effective_masses():
    """Test the extension writer with effective masses."""
    model = testsystems.AlanineDipeptideVacuum()
    mass = 3 * unit.dalton * (unit.nanometer / unit.radian) ** 2
    phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    # Use CollectiveVariableCoupling which creates a MetaCollectiveVariable
    coupling = xps.CollectiveVariableCoupling(
        "0.5*kappa*(phi-phi0)^2",
        [phi],
        [phi0],
        kappa=1000 * unit.kilojoules_per_mole / unit.radian**2,
    )
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond
    )
    integrator.setRandomNumberSeed(1234)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling),
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
                        effective_masses=True,
                    )
                ],
            )
            simulation.reporters.append(reporter)
            simulation.step(100)
        with open(os.path.join(dirpath, "report.csv"), encoding="utf-8") as file:
            line = file.readline()
            assert '#"Step"' in line
            assert '"emass[phi]' in line
            assert "Da" in line or "dalton" in line


def test_extension_writer_coupling_functions():
    """Test the extension writer with coupling functions."""
    model = testsystems.AlanineDipeptideVacuum()
    # Create a force with a parameter that will be a function of the DV
    force = openmm.CustomBondForce("scaling*energy/r")
    force.addGlobalParameter("scaling", 1.0)
    force.addEnergyParameterDerivative("scaling")
    force.addPerBondParameter("energy")
    # Add a dummy bond to make the force valid
    force.addBond(0, 1, [1.0])

    lambda_dv = xps.DynamicalVariable(
        name="lambda",
        unit=unit.dimensionless,
        mass=1.0 * unit.dalton * unit.nanometer**2,
        bounds=xps.ReflectiveBounds(0.0, 1.0, unit.dimensionless),
    )

    # Create InnerProductCoupling with a function that creates a protected parameter
    coupling = xps.InnerProductCoupling(
        [force],
        [lambda_dv],
        functions={"scaling": "(1-cos(alpha*lambda))/2"},
        alpha=pi * unit.radian,
    )

    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond
    )
    integrator.setRandomNumberSeed(1234)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling),
        xps.LockstepIntegrator(integrator),
        platform,
    )
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    simulation.context.setDynamicalVariableValues([0.5])
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
                        coupling_functions=True,
                    )
                ],
            )
            simulation.reporters.append(reporter)
            simulation.step(100)
        with open(os.path.join(dirpath, "report.csv"), encoding="utf-8") as file:
            line = file.readline()
            assert '#"Step"' in line
            # The scaling function should be reported
            assert '"scaling"' in line


def test_extension_writer_inner_product_coupling_multiple_functions():
    """Test the extension writer with InnerProductCoupling with multiple functions."""
    model = testsystems.AlanineDipeptideVacuum()
    # Create two forces with different parameters
    force1 = openmm.CustomBondForce("scaling1*energy/r")
    force1.addGlobalParameter("scaling1", 1.0)
    force1.addEnergyParameterDerivative("scaling1")
    force1.addPerBondParameter("energy")
    force1.addBond(0, 1, [1.0])

    force2 = openmm.CustomBondForce("scaling2*energy/r")
    force2.addGlobalParameter("scaling2", 1.0)
    force2.addEnergyParameterDerivative("scaling2")
    force2.addPerBondParameter("energy")
    force2.addBond(0, 1, [1.0])

    lambda_dv = xps.DynamicalVariable(
        name="lambda",
        unit=unit.dimensionless,
        mass=1.0 * unit.dalton * unit.nanometer**2,
        bounds=xps.ReflectiveBounds(0.0, 1.0, unit.dimensionless),
    )

    # Create InnerProductCoupling with multiple functions
    coupling = xps.InnerProductCoupling(
        [force1, force2],
        [lambda_dv],
        functions={
            "scaling1": "lambda",
            "scaling2": "(1-cos(alpha*lambda))/2",
        },
        alpha=pi * unit.radian,
    )

    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond
    )
    integrator.setRandomNumberSeed(1234)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling),
        xps.LockstepIntegrator(integrator),
        platform,
    )
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    simulation.context.setDynamicalVariableValues([0.5])
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
                        coupling_functions=True,
                    )
                ],
            )
            simulation.reporters.append(reporter)
            simulation.step(100)
        with open(os.path.join(dirpath, "report.csv"), encoding="utf-8") as file:
            line = file.readline()
            assert '#"Step"' in line
            # Both scaling functions should be reported
            assert '"scaling1"' in line
            assert '"scaling2"' in line


def test_extension_writer_inner_product_coupling_no_functions():
    """Test ExtensionWriter with InnerProductCoupling without functions."""
    model = testsystems.AlanineDipeptideVacuum()
    # Create a force where the DV name matches the parameter name directly
    force = openmm.CustomBondForce("lambda*energy/r")
    force.addGlobalParameter("lambda", 1.0)
    force.addEnergyParameterDerivative("lambda")
    force.addPerBondParameter("energy")
    force.addBond(0, 1, [1.0])

    lambda_dv = xps.DynamicalVariable(
        name="lambda",
        unit=unit.dimensionless,
        mass=1.0 * unit.dalton * unit.nanometer**2,
        bounds=xps.ReflectiveBounds(0.0, 1.0, unit.dimensionless),
    )

    # Create InnerProductCoupling without functions (identity mapping)
    coupling = xps.InnerProductCoupling(
        [force],
        [lambda_dv],
    )

    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond
    )
    integrator.setRandomNumberSeed(1234)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling),
        xps.LockstepIntegrator(integrator),
        platform,
    )
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    simulation.context.setDynamicalVariableValues([0.5])
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
                        coupling_functions=True,
                    )
                ],
            )
            simulation.reporters.append(reporter)
            simulation.step(100)
        with open(os.path.join(dirpath, "report.csv"), encoding="utf-8") as file:
            line = file.readline()
            assert '#"Step"' in line
            # lambda is a DV, not a function, so it should not appear as
            # a coupling function. The header should only have Step.
            assert line.strip() == '#"Step"'


def test_extension_writer_inner_product_coupling_with_other_parameters():
    """Test InnerProductCoupling with coupling_functions and other parameters."""
    model = testsystems.AlanineDipeptideVacuum()
    force = openmm.CustomBondForce("scaling*energy/r")
    force.addGlobalParameter("scaling", 1.0)
    force.addEnergyParameterDerivative("scaling")
    force.addPerBondParameter("energy")
    force.addBond(0, 1, [1.0])

    lambda_dv = xps.DynamicalVariable(
        name="lambda",
        unit=unit.dimensionless,
        mass=1.0 * unit.dalton * unit.nanometer**2,
        bounds=xps.ReflectiveBounds(0.0, 1.0, unit.dimensionless),
    )

    coupling = xps.InnerProductCoupling(
        [force],
        [lambda_dv],
        functions={"scaling": "lambda*lambda"},
    )

    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond
    )
    integrator.setRandomNumberSeed(1234)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling),
        xps.LockstepIntegrator(integrator),
        platform,
    )
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    simulation.context.setDynamicalVariableValues([0.5])
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
                        kinetic=True,
                        temperature=True,
                        dynamical_variables=True,
                        forces=True,
                        coupling_functions=True,
                    )
                ],
            )
            simulation.reporters.append(reporter)
            simulation.step(100)
        with open(os.path.join(dirpath, "report.csv"), encoding="utf-8") as file:
            line = file.readline()
            assert '#"Step"' in line
            assert '"Extension Kinetic Energy (kJ/mole)"' in line
            assert '"Extension Temperature (K)"' in line
            assert '"lambda (dimensionless)"' in line
            assert '"Force on lambda (kJ/(mol*dimensionless))"' in line
            assert '"scaling"' in line


def test_extension_writer_all_parameters():
    """Test the extension writer with all parameters enabled."""
    model = testsystems.AlanineDipeptideVacuum()
    mass = 3 * unit.dalton * (unit.nanometer / unit.radian) ** 2
    phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    coupling = xps.CollectiveVariableCoupling(
        "0.5*kappa*(phi-phi0)^2",
        [phi],
        [phi0],
        kappa=1000 * unit.kilojoules_per_mole / unit.radian**2,
    )
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond
    )
    integrator.setRandomNumberSeed(1234)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, coupling),
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
                        kinetic=True,
                        temperature=True,
                        dynamical_variables=True,
                        forces=True,
                        collective_variables=True,
                    )
                ],
            )
            simulation.reporters.append(reporter)
            simulation.step(100)
        with open(os.path.join(dirpath, "report.csv"), encoding="utf-8") as file:
            line = file.readline()
            assert '#"Step"' in line
            assert '"Extension Kinetic Energy (kJ/mole)"' in line
            assert '"Extension Temperature (K)"' in line
            assert '"phi0 (rad)"' in line
            assert '"Force on phi0 (kJ/(mol*rad))"' in line
            assert '"phi (rad)"' in line
