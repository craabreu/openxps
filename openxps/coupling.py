"""
.. module:: openxps.coupling
   :platform: Linux, MacOS, Windows
   :synopsis: Coupling between physical and extended phase-space systems.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import cvpack
import openmm as mm
from openmm import unit as mmunit


class CouplingPotential:
    """
    Abstract base class for couplings between physical and extended phase-space systems.

    """

    def addToSystem(self, system: mm.System) -> None:
        raise NotImplementedError("Subclasses must implement this method.")


class CustomCouplingPotential(cvpack.MetaCollectiveVariable, CouplingPotential):
    __doc__ = cvpack.MetaCollectiveVariable.__doc__

    def __init__(
        self,
        function: str,
        collective_variables: t.Iterable[cvpack.CollectiveVariable],
        **parameters: t.Any,
    ) -> None:
        super().__init__(
            function,
            collective_variables,
            unit=mmunit.kilojoule_per_mole,
            name="coupling_potential",
            **parameters,
        )

    def __setstate__(self, keywords: dict[str, t.Any]) -> None:
        super().__init__(**keywords)


CustomCouplingPotential.registerTag("!openxps.CustomCouplingPotential")
