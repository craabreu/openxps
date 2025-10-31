"""
Coupling sum.

.. module:: openxps.couplings.coupling_sum
   :platform: Linux, MacOS, Windows
   :synopsis: A sum of couplings

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm as mm
from openmm import XmlSerializer

from .base import Coupling


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
