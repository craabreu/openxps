"""
.. module:: openxps.system
   :platform: Linux, MacOS, Windows
   :synopsis: System for extended phase-space simulations with OpenMM.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import openmm as mm

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
    >>> system = xps.ExtendedSpaceSystem(model.system, harmonic_force)
    >>> system.getDynamicalVariables()
    (DynamicalVariable(name='phi0', unit=rad, mass=3 nm**2 Da/(rad**2), bounds=...),)
    >>> system.getExtensionSystem().getNumParticles()
    1
    """

    def __init__(self, system: mm.System, coupling: Coupling) -> None:
        self._coupling = coupling
        coupling.addToPhysicalSystem(system)
        self.this = system.this
        self._extension_system = mm.System()
        for dv in coupling.getDynamicalVariables():
            self._extension_system.addParticle(dv.mass / dv.mass.unit)
        coupling.addToExtensionSystem(self._extension_system)

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
