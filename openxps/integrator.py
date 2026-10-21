"""
.. module:: openxps.integrator
   :platform: Linux, MacOS, Windows
   :synopsis: Integrators for extended phase-space simulations with OpenMM.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from abc import ABC, abstractmethod
from copy import copy, deepcopy

import numpy as np
import openmm as mm
from cvpack.units import Quantity
from openmm import _openmm as mmswig
from openmm import unit as mmunit

from . import integrators
from .couplings import Coupling
from .integrators.utils import IntegratorMixin

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
    integrators.SymmetricVerletIntegrator,
    integrators.SymmetricLangevinIntegrator,
)


def _integrator_state(integrator: mm.Integrator) -> tuple[type, str, dict]:
    attributes = {
        key: value.in_unit_system(mmunit.md_unit_system)
        if mmunit.is_quantity(value)
        else value
        for key, value in vars(integrator).items()
        if key != "this"
    }
    return type(integrator), mm.XmlSerializer.serialize(integrator), attributes


def _restore_integrator(cls: type, xml: str, attributes: dict) -> mm.Integrator:
    integrator = cls.__new__(cls)
    integrator.this = mm.XmlSerializer.deserialize(xml).this
    integrator.__dict__.update(
        {
            key: Quantity(value) if mmunit.is_quantity(value) else value
            for key, value in attributes.items()
        }
    )
    return integrator


def _copy_without_capture(integrator: mm.Integrator) -> mm.Integrator:
    """Copy an integrator for use as an extension integrator."""
    if isinstance(integrator, IntegratorMixin):
        return integrator._withoutEnergyDerivatives()
    return deepcopy(integrator)


def _decorrelate_random_seed(physical: mm.Integrator, extension: mm.Integrator) -> None:
    if hasattr(physical, "getRandomNumberSeed"):
        seed = physical.getRandomNumberSeed()
        extension.setRandomNumberSeed(0 if seed == 0 else seed % (2**31 - 1) + 1)


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
        self._assume_valid = getattr(self, "_assume_valid", False)
        self._initialize()

    def __copy__(self) -> "ExtendedSpaceIntegrator":
        """Return an unconfigured copy of the integrator."""
        clone = self.__class__.__new__(self.__class__)
        clone._physical_integrator = copy(self._physical_integrator)
        clone._extension_integrator = copy(self._extension_integrator)
        clone._assume_valid = self._assume_valid
        clone._initialize()
        return clone

    def __deepcopy__(self, memo: dict) -> "ExtendedSpaceIntegrator":
        return self.__copy__()

    def __getstate__(self) -> dict:
        """Get the state of the integrator as a picklable dictionary."""
        return {
            "physical": _integrator_state(self._physical_integrator),
            "extension": _integrator_state(self._extension_integrator),
            "assume_valid": self._assume_valid,
        }

    def __setstate__(self, state: dict) -> None:
        """Set the state of the integrator from a dictionary."""
        self._physical_integrator = _restore_integrator(*state["physical"])
        self._extension_integrator = _restore_integrator(*state["extension"])
        self._assume_valid = state["assume_valid"]
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
        self._capture_point = None

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
        for integrator, context, isExtension in (
            (self._physical_integrator, physical_context, False),
            (self._extension_integrator, extension_context, True),
        ):
            if isinstance(integrator, IntegratorMixin):
                integrator.registerWithSystem(context.getSystem(), isExtension)
        self._physical_context = physical_context
        self._extension_context = extension_context
        self._coupling = coupling
        self._dynamical_variables = coupling.getDynamicalVariables()
        self._capture_point = None
        extension = self._extension_integrator
        if (
            isinstance(extension, IntegratorMixin)
            and extension.getEnergyDerivativeNames()
        ):
            raise ValueError(
                "The extension integrator must not compute energy derivatives; pass "
                "energyDerivatives only to the physical integrator."
            )
        physical = self._physical_integrator
        if (
            isinstance(physical, IntegratorMixin)
            and physical.getEnergyDerivativeNames()
        ):
            captured = set(physical.getEnergyDerivativeNames())
            required = set(coupling.getEnergyDerivativeNames())
            if captured != required:
                raise ValueError(
                    "The energyDerivatives of the physical integrator, "
                    f"{sorted(captured)}, must be the parameters whose derivatives "
                    f"the coupling reads, {sorted(required)}."
                )
            self._checkCapturePoint(physical.getCapturePoint())
            self._capture_point = physical.getCapturePoint()

    def _checkCapturePoint(self, point: str) -> None:
        """Raise if the integration scheme cannot use derivatives captured there."""

    def _updateExtensionFromCapture(
        self, physical_context: mm.Context, extension_context: mm.Context
    ) -> None:
        """Do what ``updateExtensionContext`` does, with the captured derivatives."""
        self._coupling.updateExtensionCollectiveVariables(
            physical_context, extension_context
        )
        self._coupling.updateExtensionDerivatives(
            extension_context, self._physical_integrator.getCapturedDerivatives()
        )

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
        self._physical_integrator.setStepSize(
            factor * mmswig.Integrator_getStepSize(self._physical_integrator)
        )
        self._extension_integrator.setStepSize(
            factor * mmswig.Integrator_getStepSize(self._extension_integrator)
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

    If the physical integrator is one of the integrators in :mod:`openxps.integrators`
    built with ``energyDerivatives``, the derivatives read by the coupling are taken
    from the force evaluation of each physical step instead of a separate evaluation.
    The names must be those returned by the coupling's ``getEnergyDerivativeNames``.

    Raises
    ------
    ValueError
        If the physical and extension integrators do not follow a force-first scheme or
        do not have the same step size.

    Example
    -------
    >>> import openxps as xps
    >>> import openmm as mm
    >>> from openmm import unit

    Using OpenMM's LangevinMiddleIntegrator integrator (force-first)

    >>> integrator = mm.LangevinMiddleIntegrator(
    ...     300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtosecond
    ... )
    >>> xps_integrator = xps.LockstepIntegrator(integrator)
    >>> print(xps_integrator.getStepSize())
    0.001 ps

    Using CSVR integrator with force-first scheme for the extension system

    >>> csvr = xps.integrators.CSVRIntegrator(
    ...     300 * unit.kelvin, 10 / unit.picosecond, 1 * unit.femtosecond,
    ...     forceFirst=True
    ... )
    >>> xps_integrator_ff = xps.LockstepIntegrator(integrator, csvr)
    """

    def __init__(
        self,
        physical_integrator: mm.Integrator,
        extension_integrator: mm.Integrator | None = None,
        assume_force_first: bool = False,
    ) -> None:
        self._assume_valid = assume_force_first
        if extension_integrator is None:
            extension_integrator = _copy_without_capture(physical_integrator)
            _decorrelate_random_seed(physical_integrator, extension_integrator)
        elif not np.isclose(
            mmswig.Integrator_getStepSize(physical_integrator),
            mmswig.Integrator_getStepSize(extension_integrator),
        ):
            raise ValueError("The step sizes must be equal.")
        if not (
            assume_force_first
            or (
                self._is_force_first(physical_integrator)
                and self._is_force_first(extension_integrator)
            )
        ):
            raise ValueError(
                "The integrators must follow a force-first scheme.\n"
                "If you are certain they do, set assume_force_first=True."
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
        capture = self._capture_point
        for _ in range(steps):
            self._physical_integrator.step(1)
            if capture == "start":
                # Derivatives at the positions this extension step must see
                self._coupling.updateExtensionDerivatives(
                    self._extension_context,
                    self._physical_integrator.getCapturedDerivatives(),
                )
            self._extension_integrator.step(1)
            self._coupling.updatePhysicalContext(
                self._physical_context,
                self._extension_context,
            )
            if capture is None:
                self._coupling.updateExtensionContext(
                    self._physical_context,
                    self._extension_context,
                )
            elif capture == "end":
                self._updateExtensionFromCapture(
                    self._physical_context, self._extension_context
                )
            else:
                self._coupling.updateExtensionCollectiveVariables(
                    self._physical_context, self._extension_context
                )
        if capture == "start" and steps > 0:
            # Leave the extension context at the final configuration for observers
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

    If the physical integrator is one of the integrators in :mod:`openxps.integrators`
    built with ``energyDerivatives``, the derivatives read by the coupling are taken
    from the force evaluation at the end of its steps instead of a separate
    evaluation. When the physical integrator takes several substeps, every substep
    computes the energy and only the last value is used.

    Raises
    ------
    ValueError
        If the physical and extension integrators are not symmetric in terms of
        operator splitting, or do not have a step size ratio equal to an even integer
        number.

    Example
    -------
    >>> import openxps as xps
    >>> import openmm as mm
    >>> from openmm import unit

    Using symmetric Verlet integrator with half step size

    >>> integrator = xps.integrators.SymmetricVerletIntegrator(4 * unit.femtosecond)
    >>> xps_integrator = xps.SplitIntegrator(integrator)
    >>> print(xps_integrator.getStepSize())
    0.004 ps
    >>> xps_integrator.getNumSubsteps()
    1

    Using symmetric Langevin integrator with 4:1 step size ratio

    >>> physical = xps.integrators.SymmetricLangevinIntegrator(
    ...     300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond
    ... )
    >>> extension = xps.integrators.SymmetricLangevinIntegrator(
    ...     300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtosecond
    ... )
    >>> xps_integrator = xps.SplitIntegrator(physical, extension)
    >>> print(xps_integrator.getStepSize())
    0.004 ps
    >>> xps_integrator.getNumSubsteps()
    2

    Using symmetric Langevin integrator with 1:4 step size ratio

    >>> physical = xps.integrators.SymmetricLangevinIntegrator(
    ...     300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond
    ... )
    >>> extension = xps.integrators.SymmetricLangevinIntegrator(
    ...     300 * unit.kelvin, 1 / unit.picosecond, 8 * unit.femtosecond
    ... )
    >>> xps_integrator = xps.SplitIntegrator(physical, extension)
    >>> print(xps_integrator.getStepSize())
    0.008 ps
    >>> xps_integrator.getNumSubsteps()
    2
    """

    def __init__(
        self,
        physical_integrator: mm.Integrator,
        extension_integrator: mm.Integrator | None = None,
        assume_symmetric: bool = False,
    ) -> None:
        self._assume_valid = assume_symmetric
        physical_step_size = mmswig.Integrator_getStepSize(physical_integrator)
        if extension_integrator is None:
            extension_integrator = _copy_without_capture(physical_integrator)
            _decorrelate_random_seed(physical_integrator, extension_integrator)
            extension_step_size = physical_step_size / 2
            extension_integrator.setStepSize(extension_step_size)
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
                "The integrators must be symmetric in terms of operator splitting.\n"
                "If you are certain they are, set assume_symmetric=True."
            )
        step_size = max(physical_step_size, extension_step_size)
        substep_size = min(physical_step_size, extension_step_size)
        if not self._is_even_division(step_size, substep_size):
            raise ValueError("The step sizes must be related by an even integer ratio.")
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

    def _checkCapturePoint(self, point: str) -> None:
        if point != "end":
            raise ValueError(
                "SplitIntegrator can only use energy derivatives captured at the end "
                "of a physical step; use a symmetric (non-force-first) physical "
                "integrator."
            )

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
        if self._capture_point is not None:
            if self._middle_integrator is self._physical_integrator:
                self._update_end_context = self._updateExtensionFromCapture
            else:
                self._update_middle_context = self._updateExtensionFromCapture

    def getNumSubsteps(self) -> int:
        """Get the number of substeps for the integrator.

        Returns
        -------
        int
            The number of substeps for the integrator.
        """
        return self._num_substeps

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
