"""
.. module:: openxps.integrator
   :platform: Linux, MacOS, Windows
   :synopsis: Integrators for extended phase-space simulations with OpenMM.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import openmm as mm
from openmm import _openmm as mmswig
from openmm import unit as mmunit

from . import integrators
from .couplings import Coupling
from .integrators.mixin import IntegratorMixin
from .utils import STRING_SEPARATOR

#: Tuple of OpenMM integrator classes known to evaluate forces exclusively at the
#: beginning of each time step (force-first or leapfrog schemes).
KNOWN_FORCE_FIRST_INTEGRATORS = (
    mm.VerletIntegrator,
    mm.LangevinIntegrator,
    mm.LangevinMiddleIntegrator,
    mm.NoseHooverIntegrator,
)

#: Tuple of OpenMM integrator classes known to be symmetric in the sense of operator
#: splitting, i.e., they can be represented as a palindromic sequence of operations.
KNOWN_SYMMETRIC_INTEGRATORS = (
    integrators.VelocityVerletIntegrator,
    integrators.BAOABIntegrator,
)


class ExtendedSpaceIntegrator(mm.Integrator, ABC):
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
            + STRING_SEPARATOR
            + self._extension_integrator.__getstate__()
        )

    def __setstate__(self, state: str) -> None:
        """Set the state of the integrator from a string."""
        physical_state, extension_state = state.split(STRING_SEPARATOR)
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
        self._coupling = None

    def configure(
        self,
        physical_context: mm.Context,
        extension_context: mm.Context,
        coupling: Coupling,
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
        coupling
            The potential that couples the physical and dynamical variables.
        """
        for integrator, context in (
            (self._physical_integrator, physical_context),
            (self._extension_integrator, extension_context),
        ):
            if isinstance(integrator, IntegratorMixin):
                integrator.register_with_system(context.getSystem())
        self._physical_context = physical_context
        self._extension_context = extension_context
        self._coupling = coupling
        self._dynamical_variables = coupling.getDynamicalVariables()

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
    all forces at the beginning of each time step (a force-first scheme). If either
    integrator does not follow this scheme, the overall integration will be incorrect.

    The step sizes of the physical and extension integrators must be equal.

    Parameters
    ----------
    physical_integrator
        The integrator for the physical system.
    extension_integrator
        The integrator for the extension system. If None, a copy of the physical
        integrator is used.
    assume_force_first
        If True, skip the validation that checks whether the integrators are known
        to follow a force-first scheme. Use this at your own risk if you know your
        integrators are compatible. Default is False.

    Raises
    ------
    ValueError
        If the physical and extension integrators do not follow a force-first scheme or
        do not have the same step size.
    """

    def __init__(
        self,
        physical_integrator: mm.Integrator,
        extension_integrator: t.Optional[mm.Integrator] = None,
        assume_force_first: bool = False,
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
        if not (
            assume_force_first
            or (
                self._is_force_first(physical_integrator)
                and self._is_force_first(extension_integrator)
            )
        ):
            raise ValueError(
                "The physical and extension integrators must follow a force-first "
                "scheme. If you are certain your integrators do follow such a scheme, "
                "set assume_force_first=True."
            )
        super().__init__(physical_integrator, extension_integrator)

    @staticmethod
    def _is_force_first(integrator: mm.Integrator) -> bool:
        """Check if an integrator follows a force-first scheme."""
        if isinstance(integrator, KNOWN_FORCE_FIRST_INTEGRATORS):
            return True
        if isinstance(integrator, IntegratorMixin):
            return integrator.isForceFirst()
        return False

    def step(self, steps: int) -> None:
        """Advance the extended phase-space simulation by a specified number of steps.

        Parameters
        ----------
        steps
            The number of time steps to advance the simulation.
        """
        for _ in range(steps):
            self._physical_integrator.step(1)
            self._extension_integrator.step(1)
            self._coupling.updatePhysicalContext(
                self._physical_context,
                self._extension_context,
            )
            self._coupling.updateExtensionContext(
                self._physical_context,
                self._extension_context,
            )


class SplitIntegrator(ExtendedSpaceIntegrator):
    r"""
    An integrator that advances the physical and extension systems using an operator
    splitting scheme.

    This integrator assumes that the physical and extension integrators are symmetric
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
    assume_symmetric
        If True, skip the validation that checks whether the integrators are known
        to be symmetric in terms of operator splitting. Use this at your own risk if
        you know your integrators are compatible but are not in the known list.
        Default is False.

    Raises
    ------
    ValueError
        If the physical and extension integrators are not symmetric in terms of
        operator splitting, or do not have a step size ratio equal to an even integer
        number.
    """

    def __init__(
        self,
        physical_integrator: mm.Integrator,
        extension_integrator: t.Optional[mm.Integrator] = None,
        assume_symmetric: bool = False,
    ) -> None:
        physical_step_size = mmswig.Integrator_getStepSize(physical_integrator)
        if extension_integrator is None:
            extension_integrator = deepcopy(physical_integrator)
            extension_step_size = physical_step_size / 2
            mmswig.Integrator_setStepSize(extension_integrator, extension_step_size)
        else:
            extension_step_size = mmswig.Integrator_getStepSize(extension_integrator)
        if not (
            assume_symmetric
            or (
                self._is_symmetric(physical_integrator)
                and self._is_symmetric(extension_integrator)
            )
        ):
            raise ValueError(
                "The physical and extension integrators must be symmetric in terms of "
                "operator splitting. If you are certain your integrators are "
                "symmetric, set assume_symmetric=True."
            )
        step_size = max(physical_step_size, extension_step_size)
        substep_size = min(physical_step_size, extension_step_size)
        if not self._is_even_division(step_size, substep_size):
            raise ValueError(
                "The physical and extension integrator step sizes must be related by "
                "an even integer ratio."
            )
        super().__init__(physical_integrator, extension_integrator)

    @staticmethod
    def _is_symmetric(integrator: mm.Integrator) -> bool:
        """Check if an integrator is symmetric in terms of operator splitting."""
        if isinstance(integrator, KNOWN_SYMMETRIC_INTEGRATORS):
            return True
        if isinstance(integrator, IntegratorMixin):
            return not integrator.isForceFirst()
        return False

    @staticmethod
    def _is_even_division(a: float, b: float) -> bool:
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        return np.isclose(a / b - round(a / b), 0) and int(round(a / b)) % 2 == 0

    def _initialize(self) -> None:
        super()._initialize()
        physical_step_size = mmswig.Integrator_getStepSize(self._physical_integrator)
        extension_step_size = mmswig.Integrator_getStepSize(self._extension_integrator)
        if physical_step_size > extension_step_size:
            self._middle_integrator = self._physical_integrator
            self._end_integrator = self._extension_integrator
            ratio = physical_step_size / (2 * extension_step_size)
        else:
            self._middle_integrator = self._extension_integrator
            self._end_integrator = self._physical_integrator
            ratio = extension_step_size / (2 * physical_step_size)
        self._num_substeps = int(np.rint(ratio))

    def configure(
        self,
        physical_context: mm.Context,
        extension_context: mm.Context,
        coupling: Coupling,
    ) -> None:
        super().configure(physical_context, extension_context, coupling)
        if self._middle_integrator is self._physical_integrator:
            self._update_middle_context = self._coupling.updatePhysicalContext
            self._update_end_context = self._coupling.updateExtensionContext
        else:
            self._update_middle_context = self._coupling.updateExtensionContext
            self._update_end_context = self._coupling.updatePhysicalContext

    def step(self, steps: int) -> None:
        """Advance the extended phase-space simulation by integrating the physical and
        extension systems in lockstep over a specified number of time steps.

        Parameters
        ----------
        steps : int
            The number of time steps to advance the simulation.
        """
        step_count = self._physical_context.getStepCount()
        for _ in range(steps):
            self._end_integrator.step(self._num_substeps)
            self._update_middle_context(self._physical_context, self._extension_context)
            self._middle_integrator.step(1)
            self._update_end_context(self._physical_context, self._extension_context)
            self._end_integrator.step(self._num_substeps)
            self._update_middle_context(self._physical_context, self._extension_context)
        self._physical_context.setStepCount(step_count + steps)
