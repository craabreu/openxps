"""
.. module:: openxps.system
   :platform: Linux, MacOS, Windows
   :synopsis: System for extended phase-space simulations with OpenMM.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm as mm

from .coupling import Coupling
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
    coupling
        A :CVPack:`MetaCollectiveVariable` defining the potential energy term that
        couples the DVs to the physical coordinates. It must have units
        of ``kilojoules_per_mole``. All DVs must be included as parameters in the
        coupling.
    system
        The :OpenMM:`System` to be used in the XPS simulation.

    Example
    -------
    >>> import openxps as xps
    >>> from math import pi
    >>> import openmm
    >>> import cvpack
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> mass = 3 * unit.dalton*(unit.nanometer/unit.radian)**2
    >>> phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.bounds.CIRCULAR)
    >>> harmonic_force = xps.HarmonicCoupling(
    ...     cvpack.Torsion(6, 8, 14, 16, name="phi"),
    ...     phi0,
    ...     1000 * unit.kilojoules_per_mole / unit.radian**2,
    ... )
    >>> system = xps.ExtendedSpaceSystem([phi0], harmonic_force, model.system)
    >>> system.getDynamicalVariables()
    (DynamicalVariable(name='phi0', unit=rad, mass=3 nm**2 Da/(rad**2), bounds=...),)
    >>> system.getExtensionSystem().getNumParticles()
    1
    """

    def __init__(
        self,
        dynamical_variables: t.Iterable[DynamicalVariable],
        coupling: Coupling,
        system: mm.System,
    ) -> None:
        try:
            dynamical_variables = tuple(dv.in_md_units() for dv in dynamical_variables)
        except AttributeError as e:
            raise TypeError(
                "All dynamical variables must be instances of DynamicalVariable."
            ) from e
        self._validateCoupling(coupling, dynamical_variables)
        coupling.addToSystem(system)
        self.this = system
        self._extension_system = self._createExtensionSystem(
            dynamical_variables, coupling
        )
        self._dvs = dynamical_variables
        self._coupling = coupling

    def _validateCoupling(
        self,
        coupling: Coupling,
        dynamical_variables: t.Sequence[DynamicalVariable],
    ) -> None:
        if not isinstance(coupling, Coupling):
            raise TypeError("The coupling must be an instance of Coupling.")
        missing_parameters = [
            dv.name
            for dv in dynamical_variables
            if dv.name not in coupling.getParameterDefaultValues()
        ]
        if missing_parameters:
            raise ValueError(
                "These dynamical variables are not coupling parameters: "
                + ", ".join(missing_parameters)
            )

    def _createExtensionSystem(
        self,
        dynamical_variables: t.Sequence[DynamicalVariable],
        coupling: Coupling,
    ) -> mm.System:
        extension_system = mm.System()
        for dv in dynamical_variables:
            extension_system.addParticle(dv.mass / dv.mass.unit)
        flipped_potential = coupling.flip(dynamical_variables)
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

    def getCoupling(self) -> Coupling:
        """
        Get the coupling included in the extended phase-space system.

        Returns
        -------
        Coupling
            The coupling.
        """
        return self._coupling

    def getExtensionSystem(self) -> mm.System:
        """
        Get the extension system included in the extended phase-space system.

        Returns
        -------
        mm.System
            The extension system.
        """
        return self._extension_system
