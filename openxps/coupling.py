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
    """
    Abstract base class for couplings between physical and extended phase-space
    systems.

    A coupling connects the physical system's collective variables to the
    extended phase-space dynamical variables, enabling enhanced sampling simulations.
    Subclasses must implement the :meth:`addToSystem` and :meth:`flip` methods.

    """

    def __init__(self, forces: t.Iterable[mm.Force]) -> None:
        self._forces = list(forces)

    def __add__(self, other: "Coupling") -> "CouplingSum":
        return CouplingSum([self, other])

    def getForces(self) -> list[mm.Force]:
        return self._forces

    def addToSystem(self, system: mm.System) -> None:
        """
        Add this coupling to an OpenMM system.

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

    def flip(
        self,
        dynamical_variables: t.Sequence[DynamicalVariable],
    ) -> "Coupling":
        """
        Create a flipped version of this coupling.

        The flipped force swaps the roles of physical collective variables and
        extended dynamical variables, creating a force suitable for the extension
        system where dynamical variables are treated as particles.

        Parameters
        ----------
        dynamical_variables : Sequence[DynamicalVariable]
            The extended dynamical variables to be promoted to collective variables.

        Returns
        -------
        Coupling
            A new coupling with swapped variable roles.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def getExtensionParameters(
        self, physical_context: mm.Context
    ) -> dict[str, mmunit.Quantity]:
        """Get parameter names and values to update the extension context with.

        Parameters
        ----------
        physical_context
            The physical context to get the extension parameters from.

        Returns
        -------
        dict[str, mmunit.Quantity]
            A dictionary with the names of the parameters as keys and their values in
            the extension context as values.
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
        return "+".join(f"({repr(force)})" for force in self._couplings)

    def __copy__(self) -> "CouplingSum":
        return CouplingSum(map(copy, self._couplings))

    def __getstate__(self) -> dict[str, Coupling]:
        return {f"force{i}": force for i, force in enumerate(self._couplings)}

    def __setstate__(self, keywords: dict[str, Coupling]) -> None:
        self._couplings = list(keywords.values())

    def getCouplings(self) -> t.Sequence[Coupling]:
        """Get the couplings included in the summed coupling."""
        return self._couplings

    def getParameterDefaultValues(self) -> dict[str, mmunit.Quantity]:
        """Get parameter names and default values from all couplings.

        Returns
        -------
        dict[str, mmunit.Quantity]
            A dictionary with parameter names as keys and their default values.
        """
        parameters = {}
        for force in self._couplings:
            force_params = force.getParameterDefaultValues()
            for key, value in force_params.items():
                if key in parameters and parameters[key] != value:
                    raise ValueError(
                        f"Parameter {key} has conflicting default values in "
                        f"couplings: {parameters[key]} != {value}"
                    )
                parameters[key] = value
        return parameters

    def addToSystem(self, system: mm.System) -> None:
        for force in self._couplings:
            force.addToSystem(system)

    def flip(
        self,
        dynamical_variables: t.Sequence[DynamicalVariable],
    ) -> "CouplingSum":
        return CouplingSum(
            [force.flip(dynamical_variables) for force in self._couplings]
        )

    def getExtensionParameters(
        self, physical_context: mm.Context
    ) -> dict[str, mmunit.Quantity]:
        parameter_dicts = [
            force.getExtensionParameters(physical_context) for force in self._couplings
        ]
        parameters = {}
        for parameter_dict in parameter_dicts:
            for key, value in parameter_dict.items():
                if key in parameters and parameters[key] != value:
                    raise ValueError(
                        f"Parameter {key} has conflicting values in couplings: "
                    )
                parameters[key] = value
        return parameters


CouplingSum.registerTag("!openxps.CouplingSum")


class CustomCoupling(Coupling):
    """
    A custom coupling that uses an algebraic expression to couple physical
    collective variables with extended dynamical variables.

    This class extends :CVPack:`MetaCollectiveVariable` to provide a flexible
    coupling defined by a mathematical expression. It automatically
    handles the transformation needed for extended phase-space simulations.

    Parameters
    ----------
    function : str
        An algebraic expression that defines the coupling energy as a function of
        collective variables and parameters.
    collective_variables : Iterable[cvpack.CollectiveVariable]
        The collective variables used in the coupling function.
    **parameters : Any
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
        """
        Initialize a custom coupling.

        Parameters
        ----------
        function : str
            An algebraic expression defining the coupling energy.
        collective_variables : Iterable[cvpack.CollectiveVariable]
            The collective variables used in the function.
        **parameters : Any
            Named parameters appearing in the function expression.

        """
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

    def getInnerValues(self, context: mm.Context) -> dict[str, mmunit.Quantity]:
        return self._forces[0].getInnerValues(context)

    def getEnergyFunction(self) -> str:
        return self._forces[0].getEnergyFunction()

    def getValue(self, context: mm.Context) -> mmunit.Quantity:
        return self._forces[0].getValue(context)

    def getNumCollectiveVariables(self) -> int:
        return self._forces[0].getNumCollectiveVariables()

    def flip(
        self, dynamical_variables: t.Sequence[DynamicalVariable]
    ) -> "CustomCoupling":
        """
        Create a flipped version of this coupling for the extension system.

        The flipped force replaces dynamical variable parameters with collective
        variables that represent the dynamical variables as particles. Physical
        collective variables become parameters set to zero.

        Parameters
        ----------
        dynamical_variables : Sequence[DynamicalVariable]
            The extended dynamical variables to be promoted to collective variables.

        Returns
        -------
        CustomCoupling
            A new coupling with swapped variable roles, suitable for adding
            to the extension system.

        Examples
        --------
        >>> import cvpack
        >>> import openxps as xps
        >>> from openmm import unit
        >>> from math import pi
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
        >>> flipped = coupling.flip([phi0])
        >>> flipped.getForces()[0].getParameterDefaultValues()
        {'kappa': 1000 kJ/(mol rad**2), 'phi': 0.0 rad}
        """
        force = self._forces[0]
        parameters = force.getParameterDefaultValues()
        # Only pop DVs that are actually parameters in this force
        dvs_to_flip = [dv for dv in dynamical_variables if dv.name in parameters]
        for dv in dvs_to_flip:
            parameters.pop(dv.name)
        parameters.update(
            {cv.getName(): 0.0 * cv.getUnit() for cv in force.getInnerVariables()}
        )
        # Create CVs only for the DVs that were parameters
        return CustomCoupling(
            function=force.getEnergyFunction(),
            collective_variables=[
                dv.createCollectiveVariable(i)
                for i, dv in enumerate(dynamical_variables)
                if dv in dvs_to_flip
            ],
            **parameters,
        )

    def getExtensionParameters(
        self, physical_context: mm.Context
    ) -> dict[str, mmunit.Quantity]:
        """Get parameters to update the extension context.

        This method evaluates the collective variables in the physical context and
        returns their names and values as parameters.

        Parameters
        ----------
        physical_context
            The physical context to get the extension parameters from.

        Returns
        -------
        dict[str, mmunit.Quantity]
            A dictionary with the names of the parameters as keys and their values in
            the extension context as values.
        """
        force = self._forces[0]
        collective_variables = mmswig.CustomCVForce_getCollectiveVariableValues(
            force, physical_context
        )
        return {
            mmswig.CustomCVForce_getCollectiveVariableName(force, index): value
            for index, value in enumerate(collective_variables)
        }


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
