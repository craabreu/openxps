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
    ) -> None:
        if not isinstance(coupling, Coupling):
            raise TypeError("The coupling must be an instance of Coupling.")
        self._coupling = coupling
        self._extension_system = mm.System()
        for dv in coupling.getDynamicalVariables():
            self._extension_system.addParticle(dv.mass / dv.mass.unit)
        coupling.addToExtensionSystem(self._extension_system)
        self._checkOtherForces(system)
        coupling.addToPhysicalSystem(system)
        self.this = system.this

    def _checkOtherForces(self, system: mm.System) -> None:
        """Reject forces outside the coupling that depend on its parameters.

        The coupling sets these parameters from the dynamical variables, but the
        extension system only feels the forces of the coupling itself, so a force
        outside it would act on the physical system without a counterpart.
        """
        protected = self._coupling.getProtectedParameters()
        for force in system.getForces():
            if not hasattr(force, "getNumGlobalParameters"):
                continue
            names = {
                force.getGlobalParameterName(i)
                for i in range(force.getNumGlobalParameters())
            }
            if shared := names & protected:
                raise ValueError(
                    f"Force {force.getName()} is not part of the coupling but depends "
                    "on parameters controlled by it: "
                    + ", ".join(sorted(shared))
                    + ". Include this force in the coupling."
                )

    def getDynamicalVariables(self) -> tuple[DynamicalVariable]:
        """
        Get the dynamical variables included in the extended phase-space system.

        Returns
        -------
        t.Tuple[DynamicalVariable]
            The dynamical variables, converted to MD units. Their order is the one
            expected by the methods of :class:`ExtendedSpaceContext` that get or set
            the values and velocities of dynamical variables.
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
