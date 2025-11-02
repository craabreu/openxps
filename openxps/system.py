"""
.. module:: openxps.system
   :platform: Linux, MacOS, Windows
   :synopsis: System for extended phase-space simulations with OpenMM.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import numpy as np
import openmm as mm
from openmm import unit as mmunit

from .couplings import Coupling
from .dynamical_variable import DynamicalVariable


class ExtendedSpaceSystem(mm.System):
    """An :OpenMM:`System` object that includes extra dynamical variables (DVs) and
    allows for extended phase-space (XPS) simulations.

    Parameters
    ----------
    system
        The :OpenMM:`System` to be used in the XPS simulation.
    coupling
        A :class:`Coupling` object, required to couple the physical and extended
        phase-space systems. The dynamical variables are obtained from this coupling.

    Keyword Arguments
    ------------------
    tether_period
        The period of oscillation of a harmonic potential that tethers the y and z
        coordinates of the extension system particles to the origin in the yz-plane.
        This is not necessary in typical XPS simulations.

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
    >>> phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    >>> harmonic_force = xps.HarmonicCoupling(
    ...     cvpack.Torsion(6, 8, 14, 16, name="phi"),
    ...     phi0,
    ...     1000 * unit.kilojoules_per_mole / unit.radian**2,
    ... )
    >>> system = xps.ExtendedSpaceSystem(model.system, harmonic_force)
    >>> system.getDynamicalVariables()
    (DynamicalVariable(name='phi0', unit=rad, mass=3 nm**2 Da/(rad**2), bounds=...),)
    >>> system.getExtensionSystem().getNumParticles()
    1
    """

    def __init__(
        self,
        system: mm.System,
        coupling: Coupling,
        *,
        tether_period: t.Optional[mmunit.Quantity] = None,
    ) -> None:
        self._coupling = coupling
        coupling.addToPhysicalSystem(system)
        self.this = system.this
        self._extension_system = mm.System()
        for dv in coupling.getDynamicalVariables():
            self._extension_system.addParticle(dv.mass / dv.mass.unit)
        coupling.addToExtensionSystem(self._extension_system)
        if tether_period is not None:
            self._tethering_force = self._create_tethering_force(tether_period)
            self._extension_system.addForce(self._tethering_force)
        else:
            self._tethering_force = None

    def _create_tethering_force(
        self, tether_period: mmunit.Quantity
    ) -> mm.CustomExternalForce:
        """Create a force tethering all particles to the origin in the yz-plane."""
        tethering_force = mm.CustomExternalForce(
            "0.5*kappa*(y^2 + z^2); kappa=mass_4_pi_sq/tether_period^2"
        )
        tethering_force.setName("Tethering")
        tethering_force.addGlobalParameter("tether_period", tether_period)
        tethering_force.addPerParticleParameter("mass_4_pi_sq")
        for index, dv in enumerate(self._coupling.getDynamicalVariables()):
            tethering_force.addParticle(index, [4 * np.pi**2 * dv.mass / dv.mass.unit])
        return tethering_force

    def getDynamicalVariables(self) -> tuple[DynamicalVariable]:
        """
        Get the dynamical variables included in the extended phase-space system.

        Returns
        -------
        t.Tuple[DynamicalVariable]
            A tuple containing the dynamical variables.
        """
        return tuple(self._coupling.getDynamicalVariables())

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

    def getTetheringForce(self) -> t.Optional[mm.CustomExternalForce]:
        """
        Get the tethering force included in the extended phase-space system.

        If no tethering force has been added, returns None.

        Returns
        -------
        t.Optional[mm.CustomExternalForce]
            The tethering force.
        """
        return self._tethering_force
