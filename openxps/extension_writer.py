"""
.. module:: openxps.state_data_writer
   :platform: Linux, MacOS, Windows
   :synopsis: A custom writer for reporting state data from an external context.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import openmm as mm
from cvpack.reporting.custom_writer import CustomWriter
from openmm import _openmm as mmswig
from openmm import app as mmapp
from openmm import unit as mmunit


class ExtensionWriter(CustomWriter):
    """
    A custom writer for reporting state data from an extension context.

    Parameters
    ----------
    potential
        If ``True``, the potential energy of the extension context will be reported.
    kinetic
        If ``True``, the kinetic energy of the extension context will be reported.
    total
        If ``True``, the total energy of the extension context will be reported.
    temperature
        If ``True``, the temperature of the extension context will be reported.

    Example
    -------
    >>> import openxps as xps
    >>> from math import pi
    >>> from sys import stdout
    >>> import openmm
    >>> import cvpack
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> mass = 3 * unit.dalton*(unit.nanometer/unit.radian)**2
    >>> phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.bounds.CIRCULAR)
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> kappa = 1000 * unit.kilojoules_per_mole / unit.radian**2
    >>> harmonic_force = xps.HarmonicCoupling(phi, phi0, kappa)
    >>> integrator = openmm.LangevinMiddleIntegrator(
    ...     300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond
    ... )
    >>> integrator.setRandomNumberSeed(1234)
    >>> platform = openmm.Platform.getPlatformByName("Reference")
    >>> simulation = xps.ExtendedSpaceSimulation(
    ...     model.topology,
    ...     xps.ExtendedSpaceSystem(model.system, harmonic_force),
    ...     xps.LockstepIntegrator(integrator),
    ...     platform
    ... )
    >>> simulation.context.setPositions(model.positions)
    >>> simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    >>> simulation.context.setDynamicalVariableValues([180 * unit.degree])
    >>> simulation.context.setDynamicalVariableVelocitiesToTemperature(
    ...     300 * unit.kelvin, 1234
    ... )
    >>> reporter = cvpack.reporting.StateDataReporter(
    ...     stdout,
    ...     10,
    ...     step=True,
    ...     kineticEnergy=True,
    ...     writers=[xps.ExtensionWriter(kinetic=True)],
    ... )
    >>> simulation.reporters.append(reporter)
    >>> simulation.step(100)  # doctest: +SKIP
    #"Step","Kinetic Energy (kJ/mole)","Extension Kinetic Energy (kJ/mole)"
    10,60.512...,1.7013...
    20,75.765...,2.5089...
    30,61.116...,1.3375...
    40,52.359...,0.4791...
    50,61.382...,0.7065...
    60,48.674...,0.6520...
    70,60.771...,1.3525...
    80,46.518...,2.0280...
    90,66.111...,0.9597...
    100,60.94...,0.9695...
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        potential: bool = False,
        kinetic: bool = False,
        total: bool = False,
        temperature: bool = False,
    ) -> None:
        self._potential = potential
        self._kinetic = kinetic
        self._total = total
        self._temperature = temperature
        self._needs_energy = potential or kinetic or total or temperature
        self._needs_velocities = kinetic or temperature
        self._temp_factor = 0.0

    def initialize(self, simulation: mmapp.Simulation) -> None:
        if self._temperature:
            number = len(simulation.context.getSystem().getDynamicalVariables())
            kb = mmunit.MOLAR_GAS_CONSTANT_R.value_in_unit(
                mmunit.kilojoules_per_mole / mmunit.kelvin
            )
            self._temp_factor = 2 / (number * kb)

    def getHeaders(self) -> list[str]:
        headers = []
        if self._potential:
            headers.append("Extension Potential Energy (kJ/mole)")
        if self._kinetic:
            headers.append("Extension Kinetic Energy (kJ/mole)")
        if self._total:
            headers.append("Extension Total Energy (kJ/mole)")
        if self._temperature:
            headers.append("Extension Temperature (K)")
        return headers

    def getValues(self, simulation: mmapp.Simulation) -> list[float]:
        context = simulation.extended_space_context
        state = context.getState(
            getEnergy=self._needs_energy, getVelocities=self._needs_velocities
        )
        if self._needs_energy:
            potential_energy = mmswig.State_getPotentialEnergy(state)
            kinetic_energy = mmswig.State_getKineticEnergy(state)
        if self._needs_velocities:
            velocities = mmswig.State__getVectorAsVec3(state, mm.State.Velocities)
            for dv, velocity in zip(
                context.getSystem().getDynamicalVariables(), velocities
            ):
                mass = dv.mass._value
                kinetic_energy -= 0.5 * mass * (velocity.y**2 + velocity.z**2)
        values = []
        if self._potential:
            values.append(potential_energy)
        if self._kinetic:
            values.append(kinetic_energy)
        if self._total:
            values.append(potential_energy + kinetic_energy)
        if self._temperature:
            values.append(self._temp_factor * kinetic_energy)
        return values
