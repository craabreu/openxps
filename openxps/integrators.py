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

from .dynamical_variable import DynamicalVariable


def _update_physical_context(
    physical_context: mm.Context,
    extension_context: mm.Context,
    dynamical_variables: t.Sequence[DynamicalVariable],
) -> None:
    state = mmswig.Context_getState(extension_context, mm.State.Positions)
    positions = mmswig.State__getVectorAsVec3(state, mm.State.Positions)
    for i, dv in enumerate(dynamical_variables):
        value = positions[i].x
        if dv.bounds is not None:
            value, _ = dv.bounds.wrap(value, 0)
        mmswig.Context_setParameter(physical_context, dv.name, value)


def _update_extension_context(
    extension_context: mm.Context,
    physical_context: mm.Context,
    coupling_potential: cvpack.MetaCollectiveVariable,
) -> None:
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


class ExtendedSpaceIntegrator(mm.Integrator):
    def __init__(
        self,
        physical_integrator: mm.Integrator,
        extension_integrator: mm.Integrator,
    ) -> None:
        self._physical_integrator = physical_integrator
        self._extension_integrator = extension_integrator

    def __copy__(self):
        return self.__class__(
            self._physical_integrator.__copy__(),
            self._extension_integrator.__copy__(),
        )

    def getPhysicalIntegrator(self) -> mm.Integrator:
        return self._physical_integrator

    def getExtensionIntegrator(self) -> mm.Integrator:
        return self._extension_integrator

    @abstractmethod
    def integrate(
        self,
        physical_context: mm.Context,
        extension_context: mm.Context,
        dynamical_variables: t.Sequence[DynamicalVariable],
        coupling_potential: cvpack.MetaCollectiveVariable,
    ) -> None:
        raise NotImplementedError("This method must be implemented by the subclass.")


class InTandemIntegrator(ExtendedSpaceIntegrator):
    def __init__(
        self,
        physical_integrator: mm.Integrator,
        extension_integrator: t.Optional[mm.Integrator] = None,
    ) -> None:
        super().__init__(
            physical_integrator,
            deepcopy(physical_integrator)
            if extension_integrator is None
            else extension_integrator,
        )

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
        extension systems in tandem over a specified number of time steps.

        This function assumes that both integrators have the same time step size and
        both evaluate forces exclusively at the onset of each time step. If either of
        these conditions is not met, the behavior of the step method is undefined.

        Parameters
        ----------
        physical_context
            The OpenMM context containing the physical system.
        steps
            The number of time steps to advance the simulation.
        dynamical_variables
            The dynamical variables included in the extended phase-space system.
        extension_context : mm.Context
            The OpenMM context containing the extension system.
        coupling_potential
            The potential that couples the physical and dynamical variables.

        """
        for _ in range(steps):
            mmswig.Integrator_step(physical_context._integrator, 1)
            mmswig.Integrator_step(extension_context._integrator, 1)
            _update_physical_context(
                physical_context, extension_context, dynamical_variables
            )
            _update_extension_context(
                extension_context, physical_context, coupling_potential
            )
