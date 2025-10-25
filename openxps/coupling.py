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


class CouplingForce(mm.Force):
    """
    Abstract base class for coupling forces between physical and extended phase-space
    systems.

    A coupling force connects the physical system's collective variables to the
    extended phase-space dynamical variables, enabling enhanced sampling simulations.
    Subclasses must implement the :meth:`addToSystem` and :meth:`flip` methods.

    """

    def __add__(self, other: "CouplingForce") -> "CouplingForceSum":
        return CouplingForceSum([self, other])

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

    def flip(
        self,
        dynamical_variables: t.Sequence[DynamicalVariable],
    ) -> "CouplingForce":
        """
        Create a flipped version of this coupling force.

        The flipped force swaps the roles of physical collective variables and
        extended dynamical variables, creating a force suitable for the extension
        system where dynamical variables are treated as particles.

        Parameters
        ----------
        dynamical_variables : Sequence[DynamicalVariable]
            The extended dynamical variables to be promoted to collective variables.

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
    >>> from copy import copy
    >>> import cvpack
    >>> import openxps as xps
    >>> from openmm import unit
    >>> from math import pi
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> xps.CustomCouplingForce(
    ...     f"0.5*kappa*min(delta,{2*pi}-delta)^2; delta=abs(phi-phi0)",
    ...     [phi],
    ...     kappa=1000 * unit.kilojoules_per_mole / unit.radian**2,
    ...     phi0=pi * unit.radian,
    ... )
    CustomCouplingForce("0.5*kappa*min(delta,6.28318...-delta)^2; delta=abs(phi-phi0)")
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

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self.getEnergyFunction()}")'

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
        # Only pop DVs that are actually parameters in this force
        dvs_to_flip = [dv for dv in dynamical_variables if dv.name in parameters]
        for dv in dvs_to_flip:
            parameters.pop(dv.name)
        parameters.update(
            {cv.getName(): 0.0 * cv.getUnit() for cv in self.getInnerVariables()}
        )
        # Create CVs only for the DVs that were parameters
        return CustomCouplingForce(
            function=self.getEnergyFunction(),
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
        collective_variables = mmswig.CustomCVForce_getCollectiveVariableValues(
            self, physical_context
        )
        return {
            mmswig.CustomCVForce_getCollectiveVariableName(self, index): value
            for index, value in enumerate(collective_variables)
        }


CustomCouplingForce.registerTag("!openxps.CustomCouplingForce")


class HarmonicCouplingForce(CustomCouplingForce):
    r"""
    A harmonic coupling force that uses a harmonic potential to couple a physical
    collective variable with an extended dynamical variable.

    The coupling energy is given by:
    .. math::
        E = \frac{1}{2} \kappa \left(s - q({\bf r})\right)^2
    where :math:`s` is the extended dynamical variable, :math:`q({\bf r})` is the
    physical collective variable, and :math:`kappa` is the coupling force constant.

    Parameters
    ----------
    collective_variable
        The collective variable used in the coupling force.
    dynamical_variable
        The dynamical variable used in the coupling force.
    force_constant
        The force constant for the coupling force.

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
    >>> xps.HarmonicCouplingForce(phi, phi_s, kappa)
    HarmonicCouplingForce("0.5*kappa_phi_phi_s*((phi-phi_s-6.28...*floor(...))^2)")
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


HarmonicCouplingForce.registerTag("!openxps.HarmonicCouplingForce")


class CouplingForceSum(CouplingForce, Serializable):
    """A sum of coupling forces.

    Parameters
    ----------
    coupling_forces
        The coupling forces to be added.

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
    >>> coupling_force = xps.HarmonicCouplingForce(
    ...     phi, phi_s, 1000 * unit.kilojoule_per_mole / unit.radian**2
    ... ) + xps.HarmonicCouplingForce(
    ...     psi, psi_s, 500 * unit.kilojoule_per_mole / unit.radian**2
    ... )
    """

    def __init__(self, coupling_forces: t.Iterable[CouplingForce]) -> None:
        self._coupling_forces = []
        for force in coupling_forces:
            if isinstance(force, CouplingForceSum):
                self._coupling_forces.extend(force.getCouplingForces())
            else:
                self._coupling_forces.append(force)

    def __repr__(self) -> str:
        return "+".join(f"({repr(force)})" for force in self._coupling_forces)

    def __copy__(self) -> "CouplingForceSum":
        return CouplingForceSum(map(copy, self._coupling_forces))

    def __getstate__(self) -> dict[str, CouplingForce]:
        return {f"force{i}": force for i, force in enumerate(self._coupling_forces)}

    def __setstate__(self, keywords: dict[str, CouplingForce]) -> None:
        self._coupling_forces = list(keywords.values())

    def getCouplingForces(self) -> t.Sequence[CouplingForce]:
        """Get the coupling forces included in the summed coupling force."""
        return self._coupling_forces

    def getParameterDefaultValues(self) -> dict[str, mmunit.Quantity]:
        """Get parameter names and default values from all coupling forces.

        Returns
        -------
        dict[str, mmunit.Quantity]
            A dictionary with parameter names as keys and their default values.
        """
        parameters = {}
        for force in self._coupling_forces:
            force_params = force.getParameterDefaultValues()
            for key, value in force_params.items():
                if key in parameters and parameters[key] != value:
                    raise ValueError(
                        f"Parameter {key} has conflicting default values in "
                        f"coupling forces: {parameters[key]} != {value}"
                    )
                parameters[key] = value
        return parameters

    def addToSystem(self, system: mm.System) -> None:
        for force in self._coupling_forces:
            force.addToSystem(system)

    def flip(
        self,
        dynamical_variables: t.Sequence[DynamicalVariable],
    ) -> "CouplingForceSum":
        return CouplingForceSum(
            [force.flip(dynamical_variables) for force in self._coupling_forces]
        )

    def getExtensionParameters(
        self, physical_context: mm.Context
    ) -> dict[str, mmunit.Quantity]:
        parameter_dicts = [
            force.getExtensionParameters(physical_context)
            for force in self._coupling_forces
        ]
        parameters = {}
        for parameter_dict in parameter_dicts:
            for key, value in parameter_dict.items():
                if key in parameters and parameters[key] != value:
                    raise ValueError(
                        f"Parameter {key} has conflicting values in coupling forces: "
                    )
                parameters[key] = value
        return parameters


CouplingForceSum.registerTag("!openxps.CouplingForceSum")
