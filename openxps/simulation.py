"""
.. module:: openxps.simulation
   :platform: Linux, MacOS, Windows
   :synopsis: Simulation class for extended phase-space simulations with OpenMM.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import cvpack
import openmm as mm
from openmm import app as mmapp

from .context import ExtendedSpaceContext
from .dynamical_variable import DynamicalVariable
from .integrator import ExtendedSpaceIntegrator


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
    dynamical_variables
        A collection of dynamical variables (DVs) to be included in the XPS simulation.
    coupling_potential
        A :CVPack:`MetaCollectiveVariable` defining the potential energy term that
        couples the DVs to the physical coordinates. It must have units
        of ``kilojoules_per_mole``. All DVs must be included as parameters in the
        coupling potential.
    topology
        The :OpenMM:`Topology` describing the system to be simulated.
    system
        The :OpenMM:`System` object to simulate.
    integrator
        An :OpenMM:`Integrator` object or a tuple of two :OpenMM:`Integrator`
        objects to be used in the XPS simulation. If a tuple is provided, the
        first integrator is used for the physical system and the second one is
        used for the DVs.
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
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> umbrella_potential = cvpack.MetaCollectiveVariable(
    ...     f"0.5*kappa*min(delta,{2*pi}-delta)^2; delta=abs(phi-phi0)",
    ...     [phi],
    ...     unit.kilojoule_per_mole,
    ...     kappa=1000 * unit.kilojoule_per_mole / unit.radian**2,
    ...     phi0=pi*unit.radian,
    ... )
    >>> mass = 3 * unit.dalton*(unit.nanometer/unit.radian)**2
    >>> phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.bounds.CIRCULAR)
    >>> integrator = openmm.LangevinMiddleIntegrator(
    ...     300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond
    ... )
    >>> integrator.setRandomNumberSeed(1234)
    >>> simulation = xps.ExtendedSpaceSimulation(
    ...     [phi0],
    ...     umbrella_potential,
    ...     model.topology,
    ...     model.system,
    ...     xps.LockstepIntegrator(integrator),
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
        dynamical_variables: t.Sequence[DynamicalVariable],
        coupling_potential: cvpack.MetaCollectiveVariable,
        topology: mmapp.Topology,
        system: mm.System,
        integrator: ExtendedSpaceIntegrator,
        platform: t.Optional[mm.Platform] = None,
        platformProperties: t.Optional[dict] = None,
        state: t.Optional[mm.State] = None,
    ) -> None:
        # Store the topology and system
        self.topology = topology
        self.system = system

        # Determine if the system uses periodic boundary conditions
        self._usesPBC = False
        for force in system.getForces():
            if hasattr(force, "usesPeriodicBoundaryConditions"):
                self._usesPBC = force.usesPeriodicBoundaryConditions()
                if self._usesPBC:
                    break

        # Create the ExtendedSpaceContext
        args = [dynamical_variables, coupling_potential, system, integrator]
        if platform is not None:
            args.append(platform)
            if platformProperties is not None:
                args.append(platformProperties)

        self.context = ExtendedSpaceContext(*args)

        # The integrator is stored in the context and might have been modified
        # (for tuple of integrators, only the first is returned)
        self.integrator = self.context.getIntegrator()

        # Initialize other base class attributes
        self.currentStep = 0
        self.reporters = []

        # Load state if provided
        if state is not None:
            self.context.setState(state)
