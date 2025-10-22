"""
.. module:: openxps.integrator
   :platform: Linux, MacOS, Windows
   :synopsis: Integrators for extended phase-space simulations with OpenMM.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from abc import abstractmethod
from copy import deepcopy

import cvpack
import numpy as np
import openmm as mm
from openmm import _openmm as mmswig
from openmm import unit as mmunit

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

#: Tuple of OpenMM integrator classes known to be reversible in the sense of operator
#: splitting, i.e., they can be represented as a palindromic sequence of operations.
KNOWN_REVERSIBLE_INTEGRATORS = (
    integrators.VelocityVerletIntegrator,
    integrators.BAOABIntegrator,
)


class ExtendedSpaceIntegrator(mm.Integrator):
    """Base class for integrators that advance extended phase-space simulations.

    An extended phase-space integrator manages two separate OpenMM integrators: one for
    the physical system and one for the extension system containing dynamical variables.

    The step size of this integrator is the maximum of the step sizes of the physical
    and extension integrators.

    All :OpenMM:`Integrator` methods are applied to the physical system integrator,
    except for ``step``, ``getStepSize``, and ``setStepSize``, which are applied to
    both the physical and extension system integrators.

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
        self._initialize()

    def __copy__(self) -> "ExtendedSpaceIntegrator":
        """Return an unconfigured copy of the integrator."""
        return self.__class__(
            self._physical_integrator.__copy__(),
            self._extension_integrator.__copy__(),
        )

    def __getstate__(self) -> str:
        """Get the state of the integrator as a string."""
        return (
            self._physical_integrator.__getstate__()
            + "\f"
            + self._extension_integrator.__getstate__()
        )

    def __setstate__(self, state: str) -> None:
        """Set the state of the integrator from a string."""
        physical_state, extension_state = state.split("\f")
        # Deserialize integrators from XML strings
        self._physical_integrator = mm.XmlSerializer.deserialize(physical_state)
        self._extension_integrator = mm.XmlSerializer.deserialize(extension_state)
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the integrator."""
        self.this = self._physical_integrator.this
        self._step_size = max(
            mmswig.Integrator_getStepSize(self._physical_integrator),
            mmswig.Integrator_getStepSize(self._extension_integrator),
        )
        self._extension_context = None
        self._physical_context = None
        self._dynamical_variables = None
        self._coupling_potential = None

    def _update_physical_context(self) -> None:
        """Update the physical context with the current state of the extension system.

        This function transfers the current positions of particles in the extension
        context to the corresponding parameters in the physical context. If a dynamical
        variable has bounds, the value is wrapped accordingly.

        """
        state = mmswig.Context_getState(self._extension_context, mm.State.Positions)
        positions = mmswig.State__getVectorAsVec3(state, mm.State.Positions)
        for i, dv in enumerate(self._dynamical_variables):
            value = positions[i].x
            if dv.bounds is not None:
                value, _ = dv.bounds.wrap(value, 0)
            mmswig.Context_setParameter(self._physical_context, dv.name, value)

    def _update_extension_context(self) -> None:
        """Update the extension context with the current collective variable values.

        This function evaluates the collective variables that define the coupling
        potential in the physical context and transfers their values to the
        corresponding parameters in the extension context.

        """
        collective_variables = mmswig.CustomCVForce_getCollectiveVariableValues(
            self._coupling_potential,
            self._physical_context,
        )
        for i, value in enumerate(collective_variables):
            mmswig.Context_setParameter(
                self._extension_context,
                mmswig.CustomCVForce_getCollectiveVariableName(
                    self._coupling_potential, i
                ),
                value,
            )

    def configure(
        self,
        physical_context: mm.Context,
        extension_context: mm.Context,
        dynamical_variables: t.Sequence[DynamicalVariable],
        coupling_potential: cvpack.MetaCollectiveVariable,
    ) -> None:
        """Configure the integrator.

        Store pointers to the given contexts, dynamical variables, and coupling
        potential in the integrator. This allows the integrator to update the contexts
        during the simulation.

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
        self._physical_context = physical_context
        self._extension_context = extension_context
        self._dynamical_variables = dynamical_variables
        self._coupling_potential = coupling_potential

    def getPhysicalIntegrator(self) -> mm.Integrator:
        """Get the integrator for the physical system.

        Returns
        -------
        mm.Integrator
            The integrator for the physical system.
        """
        return self._physical_integrator

    def getExtensionIntegrator(self) -> mm.Integrator:
        """Get the integrator for the extension system.

        Returns
        -------
        mm.Integrator
            The integrator for the extension system.
        """
        return self._extension_integrator

    def getStepSize(self) -> mmunit.Quantity:
        """Get the step size for the integrator.

        This is the maximum of the step sizes of the physical and extension integrators.

        Returns
        -------
        mmunit.Quantity
            The step size for the integrator.
        """
        return self._step_size * mmunit.picosecond

    def setStepSize(self, size: mmunit.Quantity | float) -> None:
        """Set the step size for the extended phase-space integrator.

        This scales the step size of the physical and extension integrators by a factor
        defined by the ratio of the given step size to the largest current step size.

        Parameters
        ----------
        size
            The step size for the extended phase-space integrator.
        """
        if mmunit.is_quantity(size):
            size = size.value_in_unit(mmunit.picosecond)
        factor = size / self._step_size
        mmswig.Integrator_setStepSize(
            self._physical_integrator,
            factor * mmswig.Integrator_getStepSize(self._physical_integrator),
        )
        mmswig.Integrator_setStepSize(
            self._extension_integrator,
            factor * mmswig.Integrator_getStepSize(self._extension_integrator),
        )
        self._step_size = size

    @abstractmethod
    def step(self, steps: int) -> None:
        """Advance the extended phase-space simulation by a specified number of steps.

        Parameters
        ----------
        steps
            The number of time steps to advance the simulation.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")


class LockstepIntegrator(ExtendedSpaceIntegrator):
    """
    An integrator that advances the physical and extension systems in lockstep.

    This class integrates both the physical and extension systems simultaneously, then
    synchronizes the contexts after each step. It assumes that both integrators apply
    all forces at the beginning of each time step (a kick-first scheme). If either
    integrator does not follow this scheme, the overall integration will be incorrect.

    The step sizes of the physical and extension integrators must be equal.

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
        integrators are compatible. Default is False.

    Raises
    ------
    ValueError
        If the physical and extension integrators do not follow a kick-first scheme or
        do not have the same step size.
    """

    def __init__(
        self,
        physical_integrator: mm.Integrator,
        extension_integrator: t.Optional[mm.Integrator] = None,
        assume_kick_first: bool = False,
    ) -> None:
        if extension_integrator is None:
            extension_integrator = deepcopy(physical_integrator)
        elif not np.isclose(
            mmswig.Integrator_getStepSize(physical_integrator),
            mmswig.Integrator_getStepSize(extension_integrator),
        ):
            raise ValueError(
                "The physical and extension integrators must have the same step size."
            )
        elif not assume_kick_first and not all(
            isinstance(integrator, KNOWN_KICK_FIRST_INTEGRATORS)
            for integrator in (physical_integrator, extension_integrator)
        ):
            raise ValueError(
                "The physical and extension integrators must follow a kick-first "
                "scheme. If you are certain your integrators do follow such a scheme, "
                "set assume_kick_first=True."
            )
        super().__init__(physical_integrator, extension_integrator)

    def step(self, steps: int) -> None:
        """Advance the extended phase-space simulation by a specified number of steps.

        Parameters
        ----------
        steps
            The number of time steps to advance the simulation.
        """
        for _ in range(steps):
            mmswig.Integrator_step(self._physical_integrator, 1)
            mmswig.Integrator_step(self._extension_integrator, 1)
            self._update_physical_context()
            self._update_extension_context()


class SplitIntegrator(ExtendedSpaceIntegrator):
    r"""
    An integrator that advances the physical and extension systems using an operator
    splitting scheme.

    This integrator assumes that the physical and extension integrators are reversible
    in terms of operator splitting, i.e., they can be represented as a palindromic
    sequence of operations. If either integrator does not follow this scheme, the
    overall integration scheme will be incorrect.

    The step sizes of the physical and extension integrators must be related by an even
    integer ratio, with either the physical or extension integrator having the larger
    step size.

    This class integrates the physical and extension systems in a :math:`B^n A B^n`
    pattern, where :math:`n` is the number of substeps, :math:`A` is the integrator with
    the larger step size :math:`\Delta t` (considered as the overall step size), and
    :math:`B` is the integrator with the smaller step size :math:`\Delta t/(2n)`.

    Parameters
    ----------
    physical_integrator
        The integrator for the physical system.
    extension_integrator
        The integrator for the extension system. If None, a copy of the physical
        integrator is used with half the step size of the physical integrator.
    assume_reversible
        If True, skip the validation that checks whether the integrators are known
        to be reversible in terms of operator splitting. Use this at your own risk if
        you know your integrators are compatible but are not in the known list.
        Default is False.

    Raises
    ------
    ValueError
        If the physical and extension integrators are not reversible in terms of
        operator splitting, or do not have a step size ratio equal to an even integer
        number.
    """

    def __init__(
        self,
        physical_integrator: mm.Integrator,
        extension_integrator: t.Optional[mm.Integrator] = None,
        assume_reversible: bool = False,
    ) -> None:
        physical_step_size = mmswig.Integrator_getStepSize(physical_integrator)
        if extension_integrator is None:
            extension_integrator = deepcopy(physical_integrator)
            extension_step_size = physical_step_size / 2
            mmswig.Integrator_setStepSize(extension_integrator, extension_step_size)
        elif not assume_reversible and not all(
            isinstance(integrator, KNOWN_REVERSIBLE_INTEGRATORS)
            for integrator in (physical_integrator, extension_integrator)
        ):
            raise ValueError(
                "The physical and extension integrators must be reversible in terms of "
                "operator splitting. If you are certain your integrators are "
                "reversible, set assume_reversible=True."
            )
        else:
            extension_step_size = mmswig.Integrator_getStepSize(extension_integrator)
        step_size = max(physical_step_size, extension_step_size)
        substep_size = min(physical_step_size, extension_step_size)
        if not np.isclose(np.remainder(step_size, 2 * substep_size), 0):
            raise ValueError(
                "The physical and extension integrator step sizes must be related by "
                "an even integer ratio."
            )
        super().__init__(physical_integrator, extension_integrator)

    def _initialize(self) -> None:
        super()._initialize()
        physical_step_size = mmswig.Integrator_getStepSize(self._physical_integrator)
        extension_step_size = mmswig.Integrator_getStepSize(self._extension_integrator)
        if physical_step_size > extension_step_size:
            self._middle_integrator = self._physical_integrator
            self._update_middle_context = self._update_physical_context
            self._end_integrator = self._extension_integrator
            self._update_end_context = self._update_extension_context
            ratio = physical_step_size / (2 * extension_step_size)
        else:
            self._middle_integrator = self._extension_integrator
            self._update_middle_context = self._update_extension_context
            self._end_integrator = self._physical_integrator
            self._update_end_context = self._update_physical_context
            ratio = extension_step_size / (2 * physical_step_size)
        self._num_substeps = int(np.rint(ratio))

    def step(self, steps: int) -> None:
        """Advance the extended phase-space simulation by integrating the physical and
        extension systems in lockstep over a specified number of time steps.

        Parameters
        ----------
        steps : int
            The number of time steps to advance the simulation.
        """
        for _ in range(steps):
            mmswig.Integrator_step(self._end_integrator, self._num_substeps)
            self._update_middle_context()
            mmswig.Integrator_step(self._middle_integrator, 1)
            self._update_end_context()
            mmswig.Integrator_step(self._end_integrator, self._num_substeps)
            self._update_middle_context()
