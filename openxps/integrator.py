"""
.. module:: openxps.integrators
   :platform: Linux, MacOS, Windows
   :synopsis: Integrators for extended phase-space simulations with OpenMM.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from abc import abstractmethod
from copy import deepcopy

import cvpack
import openmm as mm
from openmm import _openmm as mmswig

from . import integrators
from .dynamical_variable import DynamicalVariable

#: Tuple of OpenMM integrator classes known to evaluate forces exclusively at the
#: beginning of each time step (kick-first or leapfrog schemes).
KNOWN_KICK_FIRST_INTEGRATORS = (
    mm.VerletIntegrator,
    mm.LangevinIntegrator,
    mm.LangevinMiddleIntegrator,
    mm.NoseHooverIntegrator,
)

KNOWN_TIME_REVERSAL_SYMMETRIC_INTEGRATORS = (
    integrators.VelocityVerletIntegrator,
    integrators.BAOABIntegrator,
)


class ExtendedSpaceIntegrator(mm.Integrator):
    """
    Base class for integrators that advance extended phase-space simulations.

    An extended phase-space integrator manages two separate OpenMM contexts: one for
    the physical system and one for the extension system containing dynamical variables.
    This class provides the interface for integrating both systems while maintaining
    coupling between them.

    Parameters
    ----------
    physical_integrator
        The integrator for the physical system.
    extension_integrator
        The integrator for the extension system.

    """

    def __init__(
        self,
        physical_integrator: mm.Integrator,
        extension_integrator: mm.Integrator,
    ) -> None:
        self._physical_integrator = physical_integrator
        self._extension_integrator = extension_integrator

    def __copy__(self):
        """
        Create a shallow copy of the integrator.

        Returns
        -------
        ExtendedSpaceIntegrator
            A copy of this integrator with copies of the physical and extension
            integrators.

        """
        return self.__class__(
            self._physical_integrator.__copy__(),
            self._extension_integrator.__copy__(),
        )

    @staticmethod
    def _update_physical_context(
        physical_context: mm.Context,
        extension_context: mm.Context,
        dynamical_variables: t.Sequence[DynamicalVariable],
    ) -> None:
        """
        Update the physical context with current values from the extension system.

        This function transfers the current positions of particles in the extension
        context to the corresponding parameters in the physical context. If a dynamical
        variable has bounds, the value is wrapped accordingly.

        Parameters
        ----------
        physical_context
            The OpenMM context containing the physical system.
        extension_context
            The OpenMM context containing the extension system.
        dynamical_variables
            The dynamical variables included in the extended phase-space system.

        """
        state = mmswig.Context_getState(extension_context, mm.State.Positions)
        positions = mmswig.State__getVectorAsVec3(state, mm.State.Positions)
        for i, dv in enumerate(dynamical_variables):
            value = positions[i].x
            if dv.bounds is not None:
                value, _ = dv.bounds.wrap(value, 0)
            mmswig.Context_setParameter(physical_context, dv.name, value)

    @staticmethod
    def _update_extension_context(
        extension_context: mm.Context,
        physical_context: mm.Context,
        coupling_potential: cvpack.MetaCollectiveVariable,
    ) -> None:
        """
        Update the extension context with current collective variable values.

        This function evaluates the collective variables that define the coupling
        potential in the physical context and transfers their values to the
        corresponding parameters in the extension context.

        Parameters
        ----------
        extension_context
            The OpenMM context containing the extension system.
        physical_context
            The OpenMM context containing the physical system.
        coupling_potential
            The potential that couples the physical and dynamical variables.

        """
        collective_variables = mmswig.CustomCVForce_getCollectiveVariableValues(
            coupling_potential,
            physical_context,
        )
        for i, value in enumerate(collective_variables):
            mmswig.Context_setParameter(
                extension_context,
                mmswig.CustomCVForce_getCollectiveVariableName(coupling_potential, i),
                value,
            )

    def getPhysicalIntegrator(self) -> mm.Integrator:
        """
        Get the integrator for the physical system.

        Returns
        -------
        mm.Integrator
            The physical system integrator.

        """
        return self._physical_integrator

    def getExtensionIntegrator(self) -> mm.Integrator:
        """
        Get the integrator for the extension system.

        Returns
        -------
        mm.Integrator
            The extension system integrator.

        """
        return self._extension_integrator

    @abstractmethod
    def integrate(
        self,
        physical_context: mm.Context,
        extension_context: mm.Context,
        dynamical_variables: t.Sequence[DynamicalVariable],
        coupling_potential: cvpack.MetaCollectiveVariable,
    ) -> None:
        """
        Advance the extended phase-space simulation by one step.

        This abstract method must be implemented by subclasses to define how the
        physical and extension systems are integrated together.

        Parameters
        ----------
        physical_context
            The OpenMM context containing the physical system.
        extension_context
            The OpenMM context containing the extension system.
        dynamical_variables
            The dynamical variables included in the extended phase-space system.
        coupling_potential
            The potential that couples the physical and dynamical variables.

        """
        raise NotImplementedError("This method must be implemented by the subclass.")


class LockstepIntegrator(ExtendedSpaceIntegrator):
    """
    An integrator that advances physical and extension systems in lockstep.

    This class integrates both the physical and extension systems simultaneously, then
    synchronizes the contexts after each step. It assumes that both integrators apply
    all forces at the beginning of each time step (a kick-first scheme). If either
    integrator does not follow this scheme, the overall integration will be incorrect.

    .. note::
        The physical and extension integrators have their step sizes changed to the
        specified value if necessary.

    Parameters
    ----------
    physical_integrator
        The integrator for the physical system.
    extension_integrator
        The integrator for the extension system. If None, a copy of the physical
        integrator is used.
    assume_kick_first
        If True, skip the validation that checks whether the integrators are known
        to follow a kick-first scheme. Use this at your own risk if you know your
        integrators are compatible but are not in the known list. Default is False.

    """

    def __init__(
        self,
        physical_integrator: mm.Integrator,
        extension_integrator: t.Optional[mm.Integrator] = None,
        assume_kick_first: bool = False,
    ) -> None:
        if extension_integrator is None:
            extension_integrator = deepcopy(physical_integrator)
        elif not assume_kick_first and not all(
            isinstance(integrator, KNOWN_KICK_FIRST_INTEGRATORS)
            for integrator in (physical_integrator, extension_integrator)
        ):
            raise ValueError(
                "The physical and extension integrators must follow a "
                "kick-first scheme. If you are certain your integrators are "
                "compatible, set assume_kick_first=True."
            )
        super().__init__(physical_integrator, extension_integrator)

    @staticmethod
    def integrate(
        physical_context: mm.Context,
        steps: int,
        extension_context: mm.Context,
        dynamical_variables: t.Sequence[DynamicalVariable],
        coupling_potential: cvpack.MetaCollectiveVariable,
    ) -> None:
        """
        Advances the extended phase-space simulation by integrating the physical and
        extension systems in lockstep over a specified number of time steps.

        Parameters
        ----------
        physical_context
            The OpenMM context containing the physical system.
        steps
            The number of time steps to advance the simulation.
        dynamical_variables
            The dynamical variables included in the extended phase-space system.
        extension_context
            The OpenMM context containing the extension system.
        coupling_potential
            The potential that couples the physical and dynamical variables.

        """
        for _ in range(steps):
            mmswig.Integrator_step(physical_context._integrator, 1)
            mmswig.Integrator_step(extension_context._integrator, 1)
            ExtendedSpaceIntegrator._update_physical_context(
                physical_context, extension_context, dynamical_variables
            )
            ExtendedSpaceIntegrator._update_extension_context(
                extension_context, physical_context, coupling_potential
            )
