"""
.. module:: openxps.coupling
   :platform: Linux, MacOS, Windows
   :synopsis: Coupling between physical and extended phase-space systems.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from copy import copy

import cvpack
import openmm as mm
from cvpack.serialization import Serializable
from openmm import _openmm as mmswig
from openmm import unit as mmunit

from .dynamical_variable import DynamicalVariable


class Coupling(Serializable):
    """Abstract base class for couplings between physical and extended phase-space
    systems.

    A coupling connects the physical system's collective variables to the extended
    phase-space dynamical variables, enabling enhanced sampling simulations.

    Subclasses must implement the :meth:`addToSystem` and
    :meth:`addToExtensionSystem` methods.

    """

    def __init__(self, forces: t.Iterable[mm.Force]) -> None:
        self._forces = list(forces)

    def __add__(self, other: "Coupling") -> "CouplingSum":
        return CouplingSum([self, other])

    def getForces(self) -> list[mm.Force]:
        """Get the list of OpenMM Force objects associated with this coupling.

        Returns
        -------
        list[openmm.Force]
            A list of Force objects contained within this coupling.
        """
        return self._forces

    def getForce(self, index: int) -> mm.Force:
        """Retrieve a single OpenMM Force object by its index from this coupling.

        Parameters
        ----------
        index : int
            The index of the Force object to retrieve.

        Returns
        -------
        openmm.Force
            The Force object at the specified index.

        Raises
        ------
        IndexError
            If the index is out of range.
        """
        return self._forces[index]

    def addToSystem(self, system: mm.System) -> None:
        """Add this coupling to an OpenMM system.

        Parameters
        ----------
        system : openmm.System
            The system to which the coupling should be added.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def addToExtensionSystem(
        self,
        extension_system: mm.System,
        dynamical_variables: t.Sequence[DynamicalVariable],
    ) -> None:
        """Add the flipped version of this coupling to the extension system.

        The flipped force swaps the roles of physical collective variables and
        extended dynamical variables, creating a force suitable for the extension
        system where dynamical variables are treated as particles.

        Parameters
        ----------
        extension_system : openmm.System
            The extension system to which the flipped coupling should be added.
        dynamical_variables : Sequence[DynamicalVariable]
            The extended dynamical variables to be promoted to collective variables.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def getPhysicalParameters(
        self, physical_context: mm.Context
    ) -> dict[str, mmunit.Quantity]:
        """Get parameter names and values to update the physical context with.

        Parameters
        ----------
        physical_context
            The physical context to get the physical parameters from.

        Returns
        -------
        dict[str, mmunit.Quantity]
            A dictionary with the names of the parameters as keys and their values in
            the physical context as values.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def updateExtensionContext(
        self, extension_context: mm.Context, physical_context: mm.Context
    ) -> None:
        """Update the extension context with the current physical parameters.

        Parameters
        ----------
        extension_context
            The extension context to update with the physical parameters.
        physical_context
            The physical context to get the physical parameters from.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def getParameterDefaultValues(self) -> dict[str, mmunit.Quantity]:
        parameters = {}
        for force in self._forces:
            for key, value in force.getParameterDefaultValues().items():
                if key in parameters and parameters[key] != value:
                    raise ValueError(
                        f"Parameter {key} has conflicting default values in "
                        f"coupling: {parameters[key]} != {value}"
                    )
                parameters[key] = value
        return parameters


class CouplingSum(Coupling):
    """A sum of couplings.

    Parameters
    ----------
    couplings
        The couplings to be added.

    Examples
    --------
    >>> from copy import copy
    >>> import cvpack
    >>> import openxps as xps
    >>> from openmm import unit
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> psi = cvpack.Torsion(4, 6, 8, 14, name="psi")
    >>> dvmass = 3 * unit.dalton * (unit.nanometer / unit.radian)**2
    >>> phi_s = xps.DynamicalVariable("phi_s", unit.radian, dvmass, xps.bounds.CIRCULAR)
    >>> psi_s = xps.DynamicalVariable("psi_s", unit.radian, dvmass, xps.bounds.CIRCULAR)
    >>> coupling = xps.HarmonicCoupling(
    ...     phi, phi_s, 1000 * unit.kilojoule_per_mole / unit.radian**2
    ... ) + xps.HarmonicCoupling(
    ...     psi, psi_s, 500 * unit.kilojoule_per_mole / unit.radian**2
    ... )
    """

    def __init__(self, couplings: t.Iterable[Coupling]) -> None:
        self._couplings = []
        forces = []
        for coupling in couplings:
            if isinstance(coupling, CouplingSum):
                self._couplings.extend(coupling.getCouplings())
            else:
                self._couplings.append(coupling)
            forces.extend(coupling.getForces())
        super().__init__(forces)

    def __repr__(self) -> str:
        return "+".join(f"({repr(coupling)})" for coupling in self._couplings)

    def __copy__(self) -> "CouplingSum":
        return CouplingSum(map(copy, self._couplings))

    def __getstate__(self) -> dict[str, Coupling]:
        return {f"coupling{i}": coupling for i, coupling in enumerate(self._couplings)}

    def __setstate__(self, keywords: dict[str, Coupling]) -> None:
        self._couplings = list(keywords.values())

    def getCouplings(self) -> t.Sequence[Coupling]:
        """Get the couplings included in the summed coupling."""
        return self._couplings

    def addToSystem(self, system: mm.System) -> None:
        for coupling in self._couplings:
            coupling.addToSystem(system)

    def addToExtensionSystem(
        self,
        extension_system: mm.System,
        dynamical_variables: t.Sequence[DynamicalVariable],
    ) -> None:
        for coupling in self._couplings:
            coupling.addToExtensionSystem(extension_system, dynamical_variables)

    def updateExtensionContext(
        self, extension_context: mm.Context, physical_context: mm.Context
    ):
        for coupling in self._couplings:
            coupling.updateExtensionContext(extension_context, physical_context)


CouplingSum.registerTag("!openxps.CouplingSum")


class CustomCoupling(Coupling):
    """A custom coupling derived from an algebraic expression.

    This class uses a :CVPack:`MetaCollectiveVariable` object to create a flexible
    coupling defined by a mathematical expression. It automatically handles the
    transformation needed for extended phase-space simulations.

    Parameters
    ----------
    function
        An algebraic expression that defines the coupling energy as a function of
        collective variables and parameters.
    collective_variables
        The collective variables used in the coupling function.
    **parameters
        Named parameters that appear in the function expression. These can include
        extended dynamical variable names that will be promoted to context parameters.

    Examples
    --------
    >>> from copy import copy
    >>> import cvpack
    >>> import openxps as xps
    >>> from openmm import unit
    >>> from math import pi
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> xps.CustomCoupling(
    ...     f"0.5*kappa*min(delta,{2*pi}-delta)^2; delta=abs(phi-phi0)",
    ...     [phi],
    ...     kappa=1000 * unit.kilojoules_per_mole / unit.radian**2,
    ...     phi0=pi * unit.radian,
    ... )
    CustomCoupling("0.5*kappa*min(delta,6.28318...-delta)^2; delta=abs(phi-phi0)")
    """

    def __init__(
        self,
        function: str,
        collective_variables: t.Iterable[cvpack.CollectiveVariable],
        **parameters: t.Any,
    ) -> None:
        force = cvpack.MetaCollectiveVariable(
            function,
            collective_variables,
            unit=mmunit.kilojoule_per_mole,
            name="coupling",
            **parameters,
        )
        super().__init__([force])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self._forces[0].getEnergyFunction()}")'

    def __copy__(self) -> "CustomCoupling":
        new = CustomCoupling.__new__(CustomCoupling)
        new.__setstate__(self.__getstate__())
        return new

    def __getstate__(self) -> dict[str, t.Any]:
        return self._forces[0].__getstate__()

    def __setstate__(self, keywords: dict[str, t.Any]) -> None:
        self._forces = [cvpack.MetaCollectiveVariable(**keywords)]

    def addToSystem(self, system: mm.System) -> None:
        self._forces[0].addToSystem(system)

    def addToExtensionSystem(
        self,
        extension_system: mm.System,
        dynamical_variables: t.Sequence[DynamicalVariable],
    ) -> None:
        """
        Add the flipped version of this coupling to the extension system.

        The flipped force replaces dynamical variable parameters with collective
        variables that represent the dynamical variables as particles. Physical
        collective variables become parameters set to zero.

        Parameters
        ----------
        extension_system
            The extension system to which the flipped coupling should be added.
        dynamical_variables
            The extended dynamical variables to be promoted to collective variables.

        Examples
        --------
        >>> import cvpack
        >>> import openxps as xps
        >>> from openmm import unit
        >>> from math import pi
        >>> import openmm as mm
        >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
        >>> coupling = xps.CustomCoupling(
        ...     "0.5*kappa*min(delta,{2*pi}-delta)^2; delta=abs(phi-phi0)",
        ...     [phi],
        ...     kappa=1000 * unit.kilojoules_per_mole / unit.radian**2,
        ...     phi0=pi * unit.radian,
        ... )
        >>> phi0 = xps.DynamicalVariable(
        ...     "phi0",
        ...     unit.radian,
        ...     3 * unit.dalton * (unit.nanometer / unit.radian)**2,
        ...     xps.bounds.CIRCULAR
        ... )
        >>> extension_system = mm.System()
        >>> extension_system.addParticle(phi0.mass / phi0.mass.unit)
        0
        >>> coupling.addToExtensionSystem(extension_system, [phi0])
        >>> extension_system.getNumForces()
        1
        """
        force = self._forces[0]
        parameters = force.getParameterDefaultValues()

        dvs_to_flip = [dv for dv in dynamical_variables if dv.name in parameters]
        for dv in dvs_to_flip:
            parameters.pop(dv.name)
        parameters.update(
            {cv.getName(): 0.0 * cv.getUnit() for cv in force.getInnerVariables()}
        )

        flipped_force = cvpack.MetaCollectiveVariable(
            force.getEnergyFunction(),
            [
                dv.createCollectiveVariable(i)
                for i, dv in enumerate(dynamical_variables)
                if dv in dvs_to_flip
            ],
            unit=mmunit.kilojoule_per_mole,
            name="coupling",
            **parameters,
        )
        extension_system.addForce(flipped_force)

    def updateExtensionContext(
        self, extension_context: mm.Context, physical_context: mm.Context
    ) -> None:
        force = self._forces[0]
        collective_variables = mmswig.CustomCVForce_getCollectiveVariableValues(
            force, physical_context
        )
        for index, value in enumerate(collective_variables):
            name = mmswig.CustomCVForce_getCollectiveVariableName(force, index)
            mmswig.Context_setParameter(extension_context, name, value)


CustomCoupling.registerTag("!openxps.CustomCoupling")


class HarmonicCoupling(CustomCoupling):
    r"""A harmonic coupling between a dynamical variable and a collective variable.

    The coupling energy is given by:

    .. math::

        U = \frac{1}{2} \kappa \left(s - q({\bf r})\right)^2

    where :math:`s` is an extended dynamical variable, :math:`q({\bf r})` is a
    physical collective variable, and :math:`\kappa` is a coupling constant.

    Parameters
    ----------
    collective_variable
        The collective variable used in the coupling.
    dynamical_variable
        The dynamical variable used in the coupling.
    force_constant
        The force constant for the coupling.

    Examples
    --------
    >>> from copy import copy
    >>> import cvpack
    >>> import openxps as xps
    >>> from openmm import unit
    >>> dvmass = 3 * unit.dalton * (unit.nanometer / unit.radian)**2
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> phi_s = xps.DynamicalVariable("phi_s", unit.radian, dvmass, xps.bounds.CIRCULAR)
    >>> kappa = 1000 * unit.kilojoule_per_mole / unit.radian**2
    >>> xps.HarmonicCoupling(phi, phi_s, kappa)
    HarmonicCoupling("0.5*kappa_phi_phi_s*((phi-phi_s-6.28...*floor(...))^2)")
    """

    def __init__(
        self,
        collective_variable: cvpack.CollectiveVariable,
        dynamical_variable: DynamicalVariable,
        force_constant: mmunit.Quantity,
    ) -> None:
        self._validateArguments(collective_variable, dynamical_variable, force_constant)
        kappa = f"kappa_{collective_variable.getName()}_{dynamical_variable.name}"
        function = (
            f"0.5*{kappa}*({dynamical_variable.distanceTo(collective_variable)}^2)"
        )
        super().__init__(
            function=function,
            collective_variables=[collective_variable],
            **{kappa: force_constant},
            **{dynamical_variable.name: 0.0 * dynamical_variable.unit},
        )

    def _validateArguments(self, cv, dv, kappa):
        pair = f"{cv.getName()} and {dv.name}"
        if not cv.getUnit().is_compatible(dv.unit):
            raise ValueError(f"Incompatible units for {pair}.")
        if (dv.isPeriodic() == cv.getPeriodicBounds() is None) or (
            cv.getPeriodicBounds() != dv.bounds.asQuantity()
        ):
            raise ValueError(f"Incompatible periodicity for {pair}.")
        if mmunit.is_quantity(kappa) and not kappa.unit.is_compatible(
            mmunit.kilojoule_per_mole / dv.unit**2
        ):
            raise ValueError(f"Incompatible force constant units for {pair}.")


HarmonicCoupling.registerTag("!openxps.HarmonicCoupling")
