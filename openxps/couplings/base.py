"""
Base class for couplings.

.. module:: openxps.couplings.base
   :platform: Linux, MacOS, Windows
   :synopsis: Base class for couplings between physical and extended phase-space systems

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import cvpack
import openmm as mm
from cvpack.serialization import Serializable
from openmm import XmlSerializer
from openmm import _openmm as mmswig
from openmm import unit as mmunit

from ..dynamical_variable import DynamicalVariable


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
