"""
.. module:: openxps.state_data_writer
   :platform: Linux, MacOS, Windows
   :synopsis: A custom writer for reporting state data from an external context.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

from cvpack.reporting.custom_writer import CustomWriter
from openmm import _openmm as mmswig
from openmm import app as mmapp
from openmm import unit as mmunit

from .context import ExtendedSpaceContext


class ExtensionWriter(CustomWriter):
    """
    A custom writer for reporting state data from an extension context.

    Parameters
    ----------
    context
        The extended space context whose extension-context state data will be reported.
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
    >>> import cvpack
    >>> import openmm
    >>> from openmm import app, unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> umbrella_potential = cvpack.MetaCollectiveVariable(
    ...     f"0.5*kappa*min(delta,{2*pi}-delta)^2; delta=abs(phi-phi0)",
    ...     [cvpack.Torsion(6, 8, 14, 16, name="phi")],
    ...     unit.kilojoule_per_mole,
    ...     kappa=1000 * unit.kilojoule_per_mole / unit.radian**2,
    ...     phi0=pi*unit.radian,
    ... )
    >>> integrator = openmm.LangevinMiddleIntegrator(
    ...     300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond
    ... )
    >>> integrator.setRandomNumberSeed(1234)
    >>> platform = openmm.Platform.getPlatformByName("Reference")
    >>> simulation = app.Simulation(
    ...     model.topology, model.system, integrator, platform
    ... )
    >>> mass = 3 * unit.dalton*(unit.nanometer/unit.radian)**2
    >>> phi0 = xps.ExtraDOF("phi0", unit.radian, mass, xps.bounds.CIRCULAR)
    >>> context = xps.ExtendedSpaceContext(
    ...     simulation.context, [phi0], umbrella_potential
    ... )
    >>> context.setPositions(model.positions)
    >>> context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    >>> context.setExtraValues([180 * unit.degree])
    >>> context.setExtraVelocitiesToTemperature(300 * unit.kelvin, 1234)
    >>> reporter = cvpack.reporting.StateDataReporter(
    ...     stdout,
    ...     10,
    ...     step=True,
    ...     kineticEnergy=True,
    ...     writers=[xps.ExtensionWriter(context, kinetic=True)],
    ... )
    >>> simulation.reporters.append(reporter)
    >>> simulation.step(100)  # doctest: +ELLIPSIS
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

    def __init__(  # pylint: disable=too-many-arguments
        self,
        context: ExtendedSpaceContext,
        potential: bool = False,
        kinetic: bool = False,
        total: bool = False,
        temperature: bool = False,
    ) -> None:
        if not isinstance(context, ExtendedSpaceContext):
            raise TypeError("The context must be an instance of ExtendedSpaceContext.")
        self._context = context
        self._potential = potential
        self._kinetic = kinetic
        self._total = total
        self._temperature = temperature
        self._needs_energy = potential or kinetic or total or temperature
        self._temp_factor = 0.0

    def initialize(self, simulation: mmapp.Simulation) -> None:
        if self._temperature:
            number = len(self._context.getExtraDOFs())
            kb = mmunit.MOLAR_GAS_CONSTANT_R.value_in_unit(
                mmunit.kilojoules_per_mole / mmunit.kelvin
            )
            self._temp_factor = 2 / (number * kb)

    def getHeaders(self) -> t.List[str]:
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

    def getValues(self, simulation: mmapp.Simulation) -> t.List[float]:
        state = self._context.getExtensionState(getEnergy=self._needs_energy)
        if self._needs_energy:
            potential_energy = mmswig.State_getPotentialEnergy(state)
            kinetic_energy = mmswig.State_getKineticEnergy(state)
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
