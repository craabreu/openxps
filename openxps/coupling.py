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


class CustomCouplingPotential(cvpack.MetaCollectiveVariable, CouplingPotential):
    __doc__ = cvpack.MetaCollectiveVariable.__doc__


CustomCouplingPotential.registerTag("!openxps.CustomCouplingPotential")
