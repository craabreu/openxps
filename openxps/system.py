"""
.. module:: openxps.system
   :platform: Linux, MacOS, Windows
   :synopsis: System for extended phase-space simulations with OpenMM.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import cvpack
import openmm as mm

from .coupling import CouplingForce
from .dynamical_variable import DynamicalVariable


class ExtendedSpaceSystem(mm.System):
    """An :OpenMM:`System` object that includes extra dynamical variables (DVs) and
    allows for extended phase-space (XPS) simulations.

    A given :CVPack:`MetaCollectiveVariable` is added to the system to couple the
    physical coordinates and the DVs.

    Parameters
    ----------
    dynamical_variables
        A collection of dynamical variables (DVs) to be included in the XPS simulation.
    coupling_potential
        A :CVPack:`MetaCollectiveVariable` defining the potential energy term that
        couples the DVs to the physical coordinates. It must have units
        of ``kilojoules_per_mole``. All DVs must be included as parameters in the
        coupling potential.
    system
        The :OpenMM:`System` to be used in the XPS simulation.

    Example
    -------
    >>> import openxps as xps
    >>> from math import pi
    >>> import openmm
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> umbrella_potential = xps.CustomCouplingForce(
    ...     f"0.5*kappa*min(delta,{2*pi}-delta)^2; delta=abs(phi-phi0)",
    ...     [cvpack.Torsion(6, 8, 14, 16, name="phi")],
    ...     kappa=1000 * unit.kilojoules_per_mole / unit.radian**2,
    ...     phi0=pi*unit.radian,
    ... )
    >>> mass = 3 * unit.dalton*(unit.nanometer/unit.radian)**2
    >>> phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.bounds.CIRCULAR)
    >>> system = xps.ExtendedSpaceSystem(
    ...     [phi0],
    ...     umbrella_potential,
    ...     model.system,
    ... )
    >>> system.getDynamicalVariables()
    (DynamicalVariable(name='phi0', unit=rad, mass=3 nm**2 Da/(rad**2), bounds=...),)
    >>> system.getExtensionSystem().getNumParticles()
    1
    """

    def __init__(
        self,
        dynamical_variables: t.Iterable[DynamicalVariable],
        coupling_potential: CouplingForce,
        system: mm.System,
    ) -> None:
        try:
            dynamical_variables = tuple(dv.in_md_units() for dv in dynamical_variables)
        except AttributeError as e:
            raise TypeError(
                "All dynamical variables must be instances of DynamicalVariable."
            ) from e
        self._validateCouplingForce(coupling_potential, dynamical_variables)
        coupling_potential.addToSystem(system)
        self.this = system
        self._extension_system = self._createExtensionSystem(
            dynamical_variables, coupling_potential
        )
        self._dvs = dynamical_variables
        self._coupling_potential = coupling_potential

    def _validateCouplingForce(
        self,
        coupling_potential: CouplingForce,
        dynamical_variables: t.Sequence[DynamicalVariable],
    ) -> None:
        if not isinstance(coupling_potential, CouplingForce):
            raise TypeError(
                "The coupling potential must be an instance of CouplingForce."
            )
        missing_parameters = [
            dv.name
            for dv in dynamical_variables
            if dv.name not in coupling_potential.getParameterDefaultValues()
        ]
        if missing_parameters:
            raise ValueError(
                "These dynamical variables are not coupling potential parameters: "
                + ", ".join(missing_parameters)
            )

    def _createExtensionSystem(
        self,
        dynamical_variables: t.Sequence[DynamicalVariable],
        coupling_potential: CouplingForce,
    ) -> mm.System:
        extension_system = mm.System()
        for dv in dynamical_variables:
            extension_system.addParticle(dv.mass / dv.mass.unit)

        parameters = coupling_potential.getParameterDefaultValues()
        for dv in dynamical_variables:
            parameters.pop(dv.name)
        parameters.update(
            {cv.getName(): 0.0 for cv in coupling_potential.getInnerVariables()}
        )

        flipped_potential = cvpack.MetaCollectiveVariable(
            function=coupling_potential.getEnergyFunction(),
            variables=[
                dv.createCollectiveVariable(index)
                for index, dv in enumerate(dynamical_variables)
            ],
            unit=coupling_potential.getUnit(),
            periodicBounds=coupling_potential.getPeriodicBounds(),
            name=coupling_potential.getName(),
            **parameters,
        )
        flipped_potential.addToSystem(extension_system)

        return extension_system

    def getDynamicalVariables(self) -> tuple[DynamicalVariable]:
        """
        Get the dynamical variables included in the extended phase-space system.

        Returns
        -------
        t.Tuple[DynamicalVariable]
            A tuple containing the dynamical variables.
        """
        return self._dvs

    def getCouplingForce(self) -> CouplingForce:
        """
        Get the coupling potential included in the extended phase-space system.

        Returns
        -------
        CouplingForce
            The coupling potential.
        """
        return self._coupling_potential

    def getExtensionSystem(self) -> mm.System:
        """
        Get the extension system included in the extended phase-space system.

        Returns
        -------
        mm.System
            The extension system.
        """
        return self._extension_system
