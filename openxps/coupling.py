"""
.. module:: openxps.coupling
   :platform: Linux, MacOS, Windows
   :synopsis: Coupling between physical and extended phase-space systems.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import cvpack
import openmm as mm
from cvpack.serialization import Serializable
from openmm import XmlSerializer
from openmm import _openmm as mmswig
from openmm import unit as mmunit

from openxps.utils import Function, preprocess_args

from .dynamical_variable import DynamicalVariable


class Coupling(Serializable):
    """Abstract base class for couplings between physical and extension systems.

    A coupling connects the physical system's coordinates to the extension system's
    dynamical variables, enabling enhanced sampling simulations.

    Subclasses must implement the :meth:`addToPhysicalSystem` and
    :meth:`addToExtensionSystem` methods.

    Parameters
    ----------
    forces
        A sequence of :OpenMM:`Force` objects.
    dynamical_variables
        A sequence of :class:`DynamicalVariable` objects.
    """

    def __init__(
        self,
        forces: t.Iterable[mm.Force],
        dynamical_variables: t.Sequence[DynamicalVariable],
    ) -> None:
        self._forces = list(forces)
        self._dynamical_variables = [dv.in_md_units() for dv in dynamical_variables]
        unique_names = {dv.name for dv in self._dynamical_variables}
        if len(unique_names) != len(self._dynamical_variables):
            raise ValueError("The dynamical variables must have unique names.")
        self._dv_indices = {
            dv.name: index for index, dv in enumerate(self._dynamical_variables)
        }
        self._flipped_force = None
        self._checkGlobalParameters()

    def __add__(self, other: "Coupling") -> "CouplingSum":
        return CouplingSum([self, other])

    def __copy__(self) -> "Coupling":
        new = self.__class__.__new__(self.__class__)
        new.__setstate__(self.__getstate__())
        return new

    def __getstate__(self) -> dict[str, t.Any]:
        return {
            "forces": self._forces,
            "dynamical_variables": self._dynamical_variables,
            "dv_indices": self._dv_indices,
        }

    def __setstate__(self, state: dict[str, t.Any]) -> None:
        self._forces = state["forces"]
        self._dynamical_variables = state["dynamical_variables"]
        self._dv_indices = state["dv_indices"]
        self._flipped_force = None

    def _checkGlobalParameters(self) -> dict[str, mmunit.Quantity]:
        parameters = {}
        for force in self._forces:
            force_parameters = {}
            for index in range(force.getNumGlobalParameters()):
                name = force.getGlobalParameterName(index)
                force_parameters[name] = force.getGlobalParameterDefaultValue(index)
            for key, value in force_parameters.items():
                if key in parameters and parameters[key] != value:
                    raise ValueError(
                        f"Parameter {key} has conflicting default values in "
                        f"coupling: {parameters[key]} != {value}"
                    )
                parameters[key] = value

    def _createFlippedForce(self) -> mm.Force | None:
        return None

    @staticmethod
    def _addForceToSystem(force: mm.Force, system: mm.System) -> None:
        if isinstance(force, cvpack.CollectiveVariable):
            force.addToSystem(system)
        else:
            system.addForce(force)

    def _updateDynamicalVariableIndices(
        self, dynamical_variables: t.Sequence[DynamicalVariable]
    ) -> None:
        """Update the indices of the dynamical variables associated with this coupling.

        Parameters
        ----------
        dynamical_variables
            All the dynamical variables in the system, regardless of whether they are
            associated with this coupling or not.
        """
        for index, dv in enumerate(dynamical_variables):
            if dv.name in self._dv_indices:
                self._dv_indices[dv.name] = index

    def getForces(self) -> list[mm.Force]:
        """Get the list of OpenMM Force objects associated with this coupling.

        Returns
        -------
        list[openmm.Force]
            A list of Force objects contained within this coupling.
        """
        return self._forces

    def getDynamicalVariables(self) -> t.Sequence[DynamicalVariable]:
        """Get the dynamical variables associated with this coupling.

        Returns
        -------
        list[DynamicalVariable]
            A list of DynamicalVariable objects contained within this coupling.
        """
        return self._dynamical_variables

    def getForce(self, index: int) -> mm.Force:
        """Retrieve a single OpenMM Force object from this coupling.

        Parameters
        ----------
        index
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

    def getDynamicalVariable(self, index: int) -> DynamicalVariable:
        """Retrieve a single dynamical variable from this coupling.

        Parameters
        ----------
        index
            The index of the DynamicalVariable object to retrieve.

        Returns
        -------
        DynamicalVariable
            The DynamicalVariable object at the specified index.

        Raises
        ------
        IndexError
            If the index is out of range.
        """
        return self._dynamical_variables[index]

    def getProtectedParameters(self) -> set[str]:
        """Get parameters of the physical context that should not be manually modified.

        Returns
        -------
        set[str]
            The protected parameters.
        """
        if self._flipped_force is None:
            raise ValueError("This coupling has not been added to an extension system.")
        return {
            self._flipped_force.getCollectiveVariableName(index)
            for index in range(self._flipped_force.getNumCollectiveVariables())
        }

    def addToPhysicalSystem(self, system: mm.System) -> None:
        """Add this coupling to an OpenMM system.

        Parameters
        ----------
        system
            The system to which the coupling should be added.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.

        """
        for force in self._forces:
            self._addForceToSystem(force, system)

    def addToExtensionSystem(self, system: mm.System) -> None:
        """
        Add the flipped version of this coupling to the extension system.

        The flipped force replaces dynamical variable parameters with collective
        variables that represent the dynamical variables as particles. Physical
        collective variables become parameters set to zero.

        Parameters
        ----------
        extension_system
            The extension system to which the flipped coupling should be added.

        Examples
        --------
        >>> import cvpack
        >>> import openxps as xps
        >>> from openmm import unit
        >>> from math import pi
        >>> import openmm as mm
        >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
        >>> phi0 = xps.DynamicalVariable(
        ...     "phi0",
        ...     unit.radian,
        ...     3 * unit.dalton * (unit.nanometer / unit.radian)**2,
        ...     xps.bounds.CIRCULAR
        ... )
        >>> coupling = xps.CollectiveVariableCoupling(
        ...     "0.5*kappa*min(delta,{2*pi}-delta)^2; delta=abs(phi-phi0)",
        ...     [phi],
        ...     [phi0],
        ...     kappa=1000 * unit.kilojoules_per_mole / unit.radian**2,
        ... )
        >>> extension_system = mm.System()
        >>> extension_system.addParticle(phi0.mass / phi0.mass.unit)
        0
        >>> coupling.addToExtensionSystem(extension_system)
        >>> extension_system.getNumForces()
        1
        """
        self._flipped_force = self._createFlippedForce()
        self._addForceToSystem(self._flipped_force, system)

    def updatePhysicalContext(
        self,
        physical_context: mm.Context,
        extension_context: mm.Context,
    ) -> None:
        """Update the physical context with the current extension parameters.

        Parameters
        ----------
        physical_context
            The physical context to update with the extension parameters.
        extension_context
            The extension context to get the extension parameters from.

        """
        collective_variables = mmswig.CustomCVForce_getCollectiveVariableValues(
            self._flipped_force, extension_context
        )
        for index, value in enumerate(collective_variables):
            mmswig.Context_setParameter(
                physical_context,
                self._flipped_force.getCollectiveVariableName(index),
                value,
            )

    def updateExtensionContext(
        self,
        physical_context: mm.Context,
        extension_context: mm.Context,
    ) -> None:
        """Update the extension context with the current physical parameters.

        Parameters
        ----------
        physical_context
            The physical context to get the physical parameters from.
        extension_context
            The extension context to update with the physical parameters.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def getCollectiveVariableValues(
        self, physical_context: mm.Context
    ) -> dict[str, float]:
        """Get the values of the collective variables.

        Parameters
        ----------
        physical_context
            The physical context to get the collective variable values from.
        """
        collective_variables = {}
        for force in self._forces:
            if isinstance(force, mm.CustomCVForce):
                cv_values = force.getCollectiveVariableValues(physical_context)
                for index, value in enumerate(cv_values):
                    collective_variables[
                        mmswig.CustomCVForce_getCollectiveVariableName(force, index)
                    ] = value
        return collective_variables


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
        dv_dict = {}
        for coupling in couplings:
            if isinstance(coupling, CouplingSum):
                self._couplings.extend(coupling.getCouplings())
            else:
                self._couplings.append(coupling)
            forces.extend(coupling.getForces())
            for dv in coupling.getDynamicalVariables():
                if dv.name not in dv_dict:
                    dv_dict[dv.name] = dv
                elif dv_dict[dv.name] != dv:
                    raise ValueError(
                        f'The dynamical variable "{dv.name}" has '
                        "conflicting definitions in the couplings."
                    )
        super().__init__(forces, sorted(dv_dict.values(), key=lambda dv: dv.name))
        self._broadcastDynamicalVariableIndices()
        self._checkCollectiveVariables()

    def __repr__(self) -> str:
        return "+".join(f"({repr(coupling)})" for coupling in self._couplings)

    def __copy__(self) -> "CouplingSum":
        new = CouplingSum.__new__(CouplingSum)
        new.__setstate__(self.__getstate__())
        return new

    def __getstate__(self) -> dict[str, t.Any]:
        return {"couplings": self._couplings}

    def __setstate__(self, state: dict[str, t.Any]) -> None:
        self.__init__(state["couplings"])

    def _broadcastDynamicalVariableIndices(self) -> None:
        for coupling in self._couplings:
            coupling._updateDynamicalVariableIndices(self._dynamical_variables)

    def _checkCollectiveVariables(self) -> None:
        cvs = {}
        for coupling in self._couplings:
            for force in coupling.getForces():
                if isinstance(force, mm.CustomCVForce):
                    for index in range(force.getNumCollectiveVariables()):
                        name = force.getCollectiveVariableName(index)
                        xml_string = XmlSerializer.serialize(
                            force.getCollectiveVariable(index)
                        )
                        if name in cvs and cvs[name] != xml_string:
                            raise ValueError(
                                f'The collective variable "{name}" has conflicting '
                                "definitions in the couplings."
                            )
                        cvs[name] = xml_string

    def getCouplings(self) -> t.Sequence[Coupling]:
        """Get the couplings included in the summed coupling."""
        return self._couplings

    def getProtectedParameters(self) -> set[str]:
        return set.union(
            *[coupling.getProtectedParameters() for coupling in self._couplings]
        )

    def addToExtensionSystem(self, system: mm.System) -> None:
        for coupling in self._couplings:
            coupling.addToExtensionSystem(system)

    def updatePhysicalContext(
        self,
        physical_context: mm.Context,
        extension_context: mm.Context,
    ):
        for coupling in self._couplings:
            coupling.updatePhysicalContext(physical_context, extension_context)

    def updateExtensionContext(
        self,
        physical_context: mm.Context,
        extension_context: mm.Context,
    ):
        for coupling in self._couplings:
            coupling.updateExtensionContext(physical_context, extension_context)


CouplingSum.registerTag("!openxps.CouplingSum")


class CollectiveVariableCoupling(Coupling):
    """Coupling between dynamical variables and physical collective variables.

    This class uses a :CVPack:`MetaCollectiveVariable` to create a coupling defined by
    a mathematical expression involving physical collective variables and parameters.

    Parameters
    ----------
    function
        A mathematical expression that defines the coupling energy as a function of
        collective variables and parameters.
    collective_variables
        The physical collective variables used in the coupling function.
    dynamical_variables
        The extended dynamical variables used in the coupling function.
    **parameters
        Named parameters that appear in the mathematical expression.

    Examples
    --------
    >>> from copy import copy
    >>> import cvpack
    >>> import openxps as xps
    >>> from openmm import unit
    >>> from math import pi
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> mass = 3 * unit.dalton * (unit.nanometer / unit.radian)**2
    >>> phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.bounds.CIRCULAR)
    >>> xps.CollectiveVariableCoupling(
    ...     f"0.5*kappa*min(delta,{2*pi}-delta)^2; delta=abs(phi-phi0)",
    ...     [phi],
    ...     [phi0],
    ...     kappa=1000 * unit.kilojoules_per_mole / unit.radian**2,
    ... )
    CollectiveVariableCoupling("0.5*kappa*min(delta,6.28...-delta)^2; ...")
    """

    @preprocess_args
    def __init__(
        self,
        function: str,
        collective_variables: t.Iterable[cvpack.CollectiveVariable],
        dynamical_variables: t.Iterable[DynamicalVariable],
        **parameters: mmunit.Quantity,
    ) -> None:
        dv_names = {dv.name for dv in dynamical_variables}
        filtered_params = {k: v for k, v in parameters.items() if k not in dv_names}

        force = cvpack.MetaCollectiveVariable(
            function,
            collective_variables,
            unit=mmunit.kilojoule_per_mole,
            name="coupling",
            **filtered_params,
            **{dv.name: 0.0 * dv.unit for dv in dynamical_variables},
        )
        super().__init__([force], dynamical_variables)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self._forces[0].getEnergyFunction()}")'

    def _createFlippedForce(self) -> mm.CustomCVForce:
        force = self._forces[0]
        parameters = force.getParameterDefaultValues()
        dvs_to_flip = [dv for dv in self._dynamical_variables if dv.name in parameters]
        for dv in dvs_to_flip:
            parameters.pop(dv.name)
        parameters.update(
            {cv.getName(): 0.0 * cv.getUnit() for cv in force.getInnerVariables()}
        )
        return cvpack.MetaCollectiveVariable(
            force.getEnergyFunction(),
            [
                dv.createCollectiveVariable(self._dv_indices[dv.name])
                for dv in dvs_to_flip
            ],
            unit=mmunit.kilojoule_per_mole,
            name="coupling",
            **parameters,
        )

    def updateExtensionContext(
        self,
        physical_context: mm.Context,
        extension_context: mm.Context,
    ) -> None:
        force = self._forces[0]
        collective_variables = mmswig.CustomCVForce_getCollectiveVariableValues(
            force, physical_context
        )
        for index, value in enumerate(collective_variables):
            name = mmswig.CustomCVForce_getCollectiveVariableName(force, index)
            mmswig.Context_setParameter(extension_context, name, value)


CollectiveVariableCoupling.registerTag("!openxps.CollectiveVariableCoupling")


class HarmonicCoupling(CollectiveVariableCoupling):
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
            dynamical_variables=[dynamical_variable],
            **{kappa: force_constant},
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


class InnerProductCoupling(Coupling):
    r"""Coupling defined by the inner product of two separable vector-valued functions.

    Use this class if the coupling energy is the inner product of a vector-valued
    function :math:`{\boldsymbol \lambda}` of the extended dynamical variables
    :math:`{\bf s}` and a vector-valued function :math:`{\bf u}` of the physical
    coordinates :math:`{\bf r}`, i.e.,

    .. math::

        U = {\boldsymbol \lambda}({\bf s}) \cdot {\bf u}({\bf r})
          = \sum_{i=1}^n \lambda_i({\bf s}) u_i({\bf r})

    where :math:`n` is the dimensionality of the vector-valued functions.

    Parameters
    ----------
    forces
        A sequence of :OpenMM:`Force` objects whose sum depends linearly on a vector of
        :math:`n` global context parameters.
    dynamical_variables
        The dynamical variables of the extended phase-space system.
    functions
        A dictionary defining global context parameters (keys) as mathematical functions
        (values) of the dynamical variables. It is not necessary to include identity
        functions (i.e., ``key=value``).
    **parameters
        Named parameters that appear in the mathematical expressions.

    Examples
    --------
    >>> from copy import copy
    >>> from collections import namedtuple
    >>> import cvpack
    >>> import openxps as xps
    >>> from openmm import unit
    >>> from math import pi
    >>> import openmm as mm
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()

    A function to remove Coulomb interactions:

    >>> Parameters = namedtuple(
    ...     "Parameters",
    ...     ["index", "charge", "sigma", "epsilon"],
    ... )
    >>> def remove_coulomb_interaction(
    ...     force: mm.NonbondedForce, p1: Parameters, p2: Parameters
    ... ) -> None:
    ...     force.addException(
    ...         p1.index,
    ...         p2.index,
    ...         0.0,
    ...         (p1.sigma + p2.sigma)/2,
    ...         (p1.epsilon * p2.epsilon).sqrt(),
    ...     )

    Remove carbonyl oxygen <=> amide hydrogen interactions:

    >>> nbforce = next(
    ...     f for f in model.system.getForces()
    ...     if isinstance(f, mm.NonbondedForce)
    ... )
    >>> O1 = Parameters(5, *nbforce.getParticleParameters(5))
    >>> H1 = Parameters(7, *nbforce.getParticleParameters(7))
    >>> O2 = Parameters(15, *nbforce.getParticleParameters(15))
    >>> H2 = Parameters(17, *nbforce.getParticleParameters(17))
    >>> remove_coulomb_interaction(nbforce, O1, H2)
    >>> remove_coulomb_interaction(nbforce, O2, H1)

    Add scaled Coulomb interactions:

    >>> force = mm.CustomBondForce(f"scaling*chargeProd/r")
    >>> _ = force.addGlobalParameter("scaling", 1.0)
    >>> _ = force.addEnergyParameterDerivative("scaling")
    >>> _ = force.addPerBondParameter("chargeProd")
    >>> _ = force.addBond(O1.index, H2.index, [O1.charge*H2.charge])
    >>> _ = force.addBond(O2.index, H1.index, [O2.charge*H1.charge])

    Create a coupling between the dynamical variable and the nonbonded force:

    >>> lambda_dv = xps.DynamicalVariable(
    ...     name="lambda",
    ...     unit=unit.dimensionless,
    ...     mass=1.0 * unit.dalton * unit.nanometer**2,
    ...     bounds=xps.bounds.Reflective(0.0, 1.0, unit.dimensionless),
    ... )
    >>> coupling = xps.InnerProductCoupling(
    ...     [force],
    ...     [lambda_dv],
    ...     functions={"scaling": "(1-cos(alpha*lambda))/2"},
    ...     alpha=pi*unit.radian,
    ... )
    >>> context = xps.ExtendedSpaceContext(
    ...     xps.ExtendedSpaceSystem(model.system, coupling),
    ...     xps.LockstepIntegrator(mm.VerletIntegrator(1.0 * mmunit.femtosecond)),
    ...     mm.Platform.getPlatformByName("Reference"),
    ... )
    >>> context.setPositions(model.positions)
    >>> context.setDynamicalVariableValues([1.0])
    """

    @preprocess_args
    def __init__(
        self,
        forces: t.Iterable[mm.Force],
        dynamical_variables: t.Iterable[DynamicalVariable],
        functions: t.Optional[dict[str, str]] = None,
        **parameters: mmunit.Quantity,
    ) -> None:
        forces = [
            cvpack.OpenMMForceWrapper(
                force,
                mmunit.kilojoule_per_mole,
                name=force.getName(),
            )
            if not isinstance(force, cvpack.CollectiveVariable)
            else force
            for force in forces
        ]
        self._functions = [
            Function(name, expression, **parameters)
            for name, expression in (functions or {}).items()
        ]
        self._parameters = parameters
        super().__init__(forces, dynamical_variables)
        self._dynamic_parameters = self._getDynamicParameters()

    def __getstate__(self) -> dict[str, t.Any]:
        state = super().__getstate__()
        state["functions"] = self._functions
        state["parameters"] = self._parameters
        return state

    def __setstate__(self, state: dict[str, t.Any]) -> None:
        self._functions = state["functions"]
        self._parameters = state["parameters"]
        super().__setstate__(state)
        self._dynamic_parameters = self._getDynamicParameters()

    def _getDynamicParameters(self) -> set[str]:
        all_force_parameters = {
            force.getGlobalParameterName(index)
            for force in self._forces
            for index in range(force.getNumGlobalParameters())
        }
        function_names = {fn.getName() for fn in self._functions}
        if self._functions:
            function_variables = set.union(
                *(fn.getVariables() for fn in self._functions),
            )
        else:
            function_variables = set()

        if missing := function_names - all_force_parameters:
            raise ValueError(
                "These functions are not global parameters in the provided forces: "
                + ", ".join(missing)
            )

        all_dvs = {dv.name for dv in self._dynamical_variables}

        if functions_missing_dvs := [
            fn.getName() for fn in self._functions if not (fn.getVariables() & all_dvs)
        ]:
            raise ValueError(
                "These functions do not depend on any dynamical variables: "
                + ", ".join(functions_missing_dvs)
            )

        dvs_in_force_parameters = all_dvs & all_force_parameters
        dvs_in_function_variables = all_dvs & function_variables

        if dvs_in_both := dvs_in_function_variables & dvs_in_force_parameters:
            raise ValueError(
                "These dynamical variables are both function variables and global "
                f"context parameters: {', '.join(dvs_in_both)}"
            )
        if dvs_in_neither := all_dvs - (
            dvs_in_function_variables | dvs_in_force_parameters
        ):
            raise ValueError(
                "These dynamical variables are neither function variables nor global "
                "context parameters: " + ", ".join(dvs_in_neither)
            )

        dynamic_parameters = dvs_in_force_parameters | function_names
        for force in self._forces:
            force_parameters = {
                force.getGlobalParameterName(index)
                for index in range(force.getNumGlobalParameters())
            }
            has_derivatives = {
                force.getEnergyParameterDerivativeName(index)
                for index in range(force.getNumEnergyParameterDerivatives())
            }
            if missing := force_parameters & dynamic_parameters - has_derivatives:
                raise ValueError(
                    "The following parameters require a derivative and are present in "
                    f"force {force.getName()}, but no derivative was requested: "
                    + ", ".join(missing)
                )
        return dynamic_parameters

    def _createFlippedForce(self) -> mm.CustomCVForce:
        inner_product = "+".join(
            f"{parameter}*{self._derivativeName(parameter)}"
            for parameter in self._dynamic_parameters
        )
        energy_function = ";".join(
            [inner_product]
            + [f"{fn.getName()}={fn.getExpression()}" for fn in self._functions]
        )
        flipped_force = mm.CustomCVForce(energy_function)
        all_dvs = [dv.name for dv in self._dynamical_variables]
        for parameter in self._dynamic_parameters:
            flipped_force.addGlobalParameter(self._derivativeName(parameter), 0.0)
        for fn in self._functions:
            flipped_force.addCollectiveVariable(
                fn.getName(), fn.createCollectiveVariable(all_dvs)
            )
        for dv in self._dynamical_variables:
            if dv.name in self._dynamic_parameters:
                flipped_force.addCollectiveVariable(
                    dv.name, dv.createCollectiveVariable(self._dv_indices[dv.name])
                )
        return flipped_force

    @staticmethod
    def _derivativeName(parameter: str) -> str:
        return "derivative_with_respect_to_" + parameter

    def updateExtensionContext(
        self,
        physical_context: mm.Context,
        extension_context: mm.Context,
    ) -> None:
        state = mmswig.Context_getState(physical_context, mm.State.ParameterDerivatives)
        for name, value in mmswig.State_getEnergyParameterDerivatives(state).items():
            if name in self._dynamic_parameters:
                mmswig.Context_setParameter(
                    extension_context, self._derivativeName(name), value
                )


InnerProductCoupling.registerTag("!openxps.InnerProductCoupling")
