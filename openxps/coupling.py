"""
.. module:: openxps.coupling
   :platform: Linux, MacOS, Windows
   :synopsis: Coupling between physical and extended phase-space systems.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import cvpack
import openmm as mm


class CouplingPotential:
    """
    Abstract base class for couplings between physical and extended phase-space systems.

    """

    def addToSystem(self, system: mm.System) -> None:
        raise NotImplementedError("Subclasses must implement this method.")


class CustomCoupling(CouplingPotential, cvpack.MetaCollectiveVariable):
    __doc__ = cvpack.MetaCollectiveVariable.__doc__

    def addToSystem(self, system: mm.System) -> None:
        """Add this coupling potential to a system.

        Parameters
        ----------
        system
            The system to which the coupling potential should be added.
        """
        system.addForce(self)
