"""
.. module:: openxps.state_data_writer
   :platform: Linux, MacOS, Windows
   :synopsis: A custom writer for reporting state data from an external context.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import cvpack
import openmm as mm
from cvpack.reporting.custom_writer import CustomWriter
from openmm import _openmm as mmswig
from openmm import unit as mmunit

from .simulation import ExtendedSpaceSimulation


class ExtensionWriter(CustomWriter):
    """
    A custom writer for reporting state data from an extension context.

    Parameters
    ----------
    kinetic
        If ``True``, the kinetic energy of the extension context will be reported.
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
        kinetic: bool = False,
        temperature: bool = False,
        dynamical_variables: bool = False,
        forces: bool = False,
        collective_variables: bool = False,
    ) -> None:
        self._kinetic = kinetic
        self._temperature = temperature
        self._dynamical_variables = dynamical_variables
        self._forces = forces
        self._collective_variables = collective_variables

        self._needs_energy = kinetic or temperature
        self._needs_velocities = kinetic or temperature
        self._needs_positions = dynamical_variables
        self._needs_forces = forces

        self._temp_factor = 0.0
        self._dv_objects = []
        self._cv_units = {}

    def initialize(self, simulation: ExtendedSpaceSimulation) -> None:
        context = simulation.extended_space_context
        coupling = context.getSystem().getCoupling()
        self._dv_objects = coupling.getDynamicalVariables()

        if self._temperature:
            number = len(coupling.getDynamicalVariables())
            kb = mmunit.MOLAR_GAS_CONSTANT_R.value_in_unit(
                mmunit.kilojoules_per_mole / mmunit.kelvin
            )
            self._temp_factor = 2 / (number * kb)

        if self._collective_variables:
            self._cv_units = {}
            for force in coupling.getForces():
                if isinstance(force, cvpack.MetaCollectiveVariable):
                    for cv in force.getInnerVariables():
                        if cv.getName() not in self._cv_units:
                            self._cv_units[cv.getName()] = cv.getUnit()

    def getHeaders(self) -> list[str]:
        headers = []
        if self._kinetic:
            headers.append("Extension Kinetic Energy (kJ/mole)")
        if self._temperature:
            headers.append("Extension Temperature (K)")
        if self._dynamical_variables:
            for dv in self._dv_objects:
                headers.append(f"{dv.name} ({dv.unit})")
        if self._forces:
            for dv in self._dv_objects:
                headers.append(f"Force on {dv.name} (kJ/(mol*{dv.unit}))")
        if self._collective_variables:
            for name, unit in self._cv_units.items():
                headers.append(f"{name} ({unit})")
        return headers

    def getValues(self, simulation: ExtendedSpaceSimulation) -> list[float]:  # noqa: PLR0912
        context = simulation.extended_space_context
        extension_context = context.getExtensionContext()
        state = extension_context.getState(
            getEnergy=self._needs_energy,
            getPositions=self._needs_positions,
            getVelocities=self._needs_velocities,
            getForces=self._needs_forces,
        )
        if self._needs_energy:
            kinetic_energy = mmswig.State_getKineticEnergy(state)
        if self._needs_velocities:
            velocities = mmswig.State__getVectorAsVec3(state, mm.State.Velocities)
            for dv, velocity in zip(
                context.getSystem().getDynamicalVariables(), velocities
            ):
                mass = dv.mass._value
                kinetic_energy -= 0.5 * mass * (velocity.y**2 + velocity.z**2)
        if self._needs_positions:
            positions = mmswig.State__getVectorAsVec3(state, mm.State.Positions)
        if self._forces:
            forces = mmswig.State__getVectorAsVec3(state, mm.State.Forces)
        values = []
        if self._kinetic:
            values.append(kinetic_energy)
        if self._temperature:
            values.append(self._temp_factor * kinetic_energy)
        if self._dynamical_variables:
            for index, dv in enumerate(self._dv_objects):
                value, _ = dv.bounds.wrap(positions[index].x, 0)
                values.append(value)
        if self._forces:
            for force in forces:
                values.append(force.x)
        if self._collective_variables:
            for name in self._cv_units:
                values.append(extension_context.getParameter(name))
        return values
