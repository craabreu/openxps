"""
.. module:: openxps.simulation
   :platform: Linux, MacOS, Windows
   :synopsis: Simulation class for extended phase-space simulations with OpenMM.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm as mm
from openmm import app as mmapp

from .context import ExtendedSpaceContext
from .integrator import ExtendedSpaceIntegrator
from .system import ExtendedSpaceSystem


class ExtendedSpaceSimulation(mmapp.Simulation):
    """
    A :OpenMM:`Simulation` object that uses an :class:`ExtendedSpaceContext` to enable
    extended phase-space (XPS) simulations with dynamical variables.

    This class extends :OpenMM:`Simulation` to seamlessly integrate extended phase-space
    capabilities by creating an :class:`ExtendedSpaceContext` instead of a standard
    :OpenMM:`Context`. All other functionality is inherited from the base class.

    **Note**: The system and integrator provided as arguments are modified in place.

    Parameters
    ----------
    topology
        The :OpenMM:`Topology` describing the system to be simulated.
    system
        The :class:`ExtendedSpaceSystem` object to simulate.
    integrator
        An :class:`ExtendedSpaceIntegrator` object to be used for advancing the XPS
        simulation. Available implementations include :class:`LockstepIntegrator` for
        systems where both integrators use the same step size, and
        :class:`SplitIntegrator` for systems with different step sizes related by an
        even integer ratio.
    platform
        The :OpenMM:`Platform` to use for calculations. If None, the default Platform
        will be used.
    platformProperties
        A dictionary of platform-specific properties. If None, the default properties
        will be used.
    state
        If specified, the simulation state will be set to this :OpenMM:`State` object
        after initialization.

    Example
    -------
    >>> import openxps as xps
    >>> from math import pi
    >>> import cvpack
    >>> import openmm
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
    >>> simulation = xps.ExtendedSpaceSimulation(
    ...     model.topology,
    ...     xps.ExtendedSpaceSystem(model.system, harmonic_force),
    ...     xps.LockstepIntegrator(integrator),
    ...     openmm.Platform.getPlatformByName("Reference"),
    ... )
    >>> simulation.context.setPositions(model.positions)
    >>> simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    >>> simulation.context.setDynamicalVariableValues([180 * unit.degree])
    >>> simulation.context.setDynamicalVariableVelocitiesToTemperature(
    ...     300 * unit.kelvin, 1234
    ... )
    >>> simulation.step(100)
    >>> simulation.context.getDynamicalVariableValues()
    (... rad,)
    """

    def __init__(  # noqa: PLR0913 pylint: disable=super-init-not-called
        self,
        topology: mmapp.Topology,
        system: ExtendedSpaceSystem,
        integrator: ExtendedSpaceIntegrator,
        platform: t.Optional[mm.Platform] = None,
        platformProperties: t.Optional[dict] = None,
        state: t.Optional[mm.State] = None,
    ) -> None:
        self.topology = topology
        self.system = system

        args = [system, integrator]
        if platform is not None:
            args.append(platform)
            if platformProperties is not None:
                args.append(platformProperties)

        self.context = self.extended_space_context = ExtendedSpaceContext(*args)
        self.integrator = self.context.getIntegrator()

        self.currentStep = 0
        self.reporters = []

        if state is not None:
            self.context.setState(state)

        try:
            self._usesPBC = self.system.usesPeriodicBoundaryConditions()
        except Exception:
            self._usesPBC = topology.getUnitCellDimensions() is not None
