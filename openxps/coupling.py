"""
.. module:: openxps.coupling
   :platform: Linux, MacOS, Windows
   :synopsis: Coupling between physical and extended phase-space systems.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import cvpack
import openmm as mm
from openmm import _openmm as mmswig
from openmm import unit as mmunit

from .dynamical_variable import DynamicalVariable


class CouplingForce(mm.Force):
    """
    Abstract base class for coupling forces between physical and extended phase-space
    systems.

    A coupling force connects the physical system's collective variables to the
    extended phase-space dynamical variables, enabling enhanced sampling simulations.
    Subclasses must implement the :meth:`addToSystem` and :meth:`flip` methods.

    """

    def addToSystem(self, system: mm.System) -> None:
        """
        Add this coupling force to an OpenMM system.

        Parameters
        ----------
        system : openmm.System
            The system to which the coupling force should be added.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def flip(self) -> "CouplingForce":
        """
        Create a flipped version of this coupling force.

        The flipped force swaps the roles of physical collective variables and
        extended dynamical variables, creating a force suitable for the extension
        system where dynamical variables are treated as particles.

        Returns
        -------
        CouplingForce
            A new coupling force with swapped variable roles.

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


class CustomCouplingForce(cvpack.MetaCollectiveVariable, CouplingForce):
    """
    A custom coupling force that uses an algebraic expression to couple physical
    collective variables with extended dynamical variables.

    This class extends :CVPack:`MetaCollectiveVariable` to provide a flexible
    coupling force defined by a mathematical expression. It automatically
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
    >>> import cvpack
    >>> import openxps as xps
    >>> from openmm import unit
    >>> from math import pi
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> coupling = xps.CustomCouplingForce(
    ...     "0.5*kappa*min(delta,{2*pi}-delta)^2; delta=abs(phi-phi0)",
    ...     [phi],
    ...     kappa=1000 * unit.kilojoules_per_mole / unit.radian**2,
    ...     phi0=pi * unit.radian,
    ... )
    """

    def __init__(
        self,
        function: str,
        collective_variables: t.Iterable[cvpack.CollectiveVariable],
        **parameters: t.Any,
    ) -> None:
        """
        Initialize a custom coupling force.

        Parameters
        ----------
        function : str
            An algebraic expression defining the coupling energy.
        collective_variables : Iterable[cvpack.CollectiveVariable]
            The collective variables used in the function.
        **parameters : Any
            Named parameters appearing in the function expression.

        """
        super().__init__(
            function,
            collective_variables,
            unit=mmunit.kilojoule_per_mole,
            name="coupling_force",
            **parameters,
        )

    def __setstate__(self, keywords: dict[str, t.Any]) -> None:
        """Restore the object state during deserialization."""
        super().__init__(**keywords)

    def flip(
        self, dynamical_variables: t.Sequence[DynamicalVariable]
    ) -> "CustomCouplingForce":
        """
        Create a flipped version of this coupling force for the extension system.

        The flipped force replaces dynamical variable parameters with collective
        variables that represent the dynamical variables as particles. Physical
        collective variables become parameters set to zero.

        Parameters
        ----------
        dynamical_variables : Sequence[DynamicalVariable]
            The extended dynamical variables to be promoted to collective variables.

        Returns
        -------
        CustomCouplingForce
            A new coupling force with swapped variable roles, suitable for adding
            to the extension system.

        Examples
        --------
        >>> import cvpack
        >>> import openxps as xps
        >>> from openmm import unit
        >>> from math import pi
        >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
        >>> coupling = xps.CustomCouplingForce(
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
        >>> flipped.getParameterDefaultValues()
        {'kappa': 1000 kJ/(mol rad**2), 'phi': 0.0 rad}
        """
        parameters = self.getParameterDefaultValues()
        for dv in dynamical_variables:
            parameters.pop(dv.name)
        parameters.update(
            {cv.getName(): 0.0 * cv.getUnit() for cv in self.getInnerVariables()}
        )
        return CustomCouplingForce(
            function=self.getEnergyFunction(),
            collective_variables=[
                dv.createCollectiveVariable(index)
                for index, dv in enumerate(dynamical_variables)
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
        collective_variables = mmswig.CustomCVForce_getCollectiveVariableValues(
            self, physical_context
        )
        return {
            mmswig.CustomCVForce_getCollectiveVariableName(self, index): value
            for index, value in enumerate(collective_variables)
        }


CustomCouplingForce.registerTag("!openxps.CustomCouplingForce")
