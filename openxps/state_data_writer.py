"""
.. module:: openxps.state_data_writer
   :platform: Linux, MacOS, Windows
   :synopsis: A custom writer for reporting state data from an external context.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import os
import typing as t

import openmm as mm
from cvpack.reporting.custom_writer import CustomWriter
from openmm import app as mmapp
from openmm import unit as mmunit


class StateDataWriter(mmapp.StateDataReporter, CustomWriter):
    """
    A custom writer for reporting state data from an external context.

    .. _openmm.app.StateDataReporter: http://docs.openmm.org/latest/api-python/
        generated/openmm.app.statedatareporter.StateDataReporter.html

    Parameters
    ----------
    stateGetter
        The ``getState`` callable from an external context or a ``functools.partial``
        object that wraps it.
    prefix
        A string to be prepended to the report headers.
    kwargs
        Additional keyword arguments to be passed to the `openmm.app.StateDataReporter`_
        constructor.

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
    >>> simulation = app.Simulation(model.topology, model.system, integrator, platform)
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
    ...     writers=[
    ...         xps.StateDataWriter(
    ...             context.getExtensionState,
    ...             "Extension",
    ...             kineticEnergy=True,
    ...         )
    ...     ],
    ... )
    >>> simulation.reporters.append(reporter)
    >>> simulation.step(100)
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

    _kB = mmunit.MOLAR_GAS_CONSTANT_R / mmunit.MOLAR_GAS_CONSTANT_R.unit

    def __init__(
        self,
        stateGetter: t.Callable[..., mm.State],
        prefix: str = "",
        **kwargs,
    ) -> None:
        self._state_getter = stateGetter
        self._prefix = prefix + " " * bool(prefix)
        super().__init__(os.devnull, 0, **kwargs)

    def initialize(self, simulation: mmapp.Simulation) -> None:
        return super()._initializeConstants(simulation)

    def getHeaders(self) -> t.List[str]:
        return [f"{self._prefix}{header}" for header in super()._constructHeaders()]

    def getValues(self, simulation: mmapp.Simulation) -> t.List[float]:
        return super()._constructReportValues(
            simulation,
            self._state_getter(
                getPositions=self._needsPositions,
                getVelocities=self._needsVelocities,
                getForces=self._needsForces,
                getEnergy=self._needEnergy,
            ),
        )
