"""
.. module:: openxps.context
   :platform: Linux, MacOS, Windows
   :synopsis: Context for extended phase-space simulations with OpenMM.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
import openmm as mm
from openmm import app as mmapp
from openmm import unit as mmunit
from cvpack.reporting.custom_writer import CustomWriter


from .extra_dof import ExtraDOF


class BypassWriter(CustomWriter):
    """
    A custom writer for phase-space extension data.

    Parameters
    ----------
        filename
            The name of the file to be written.
    """

    _kB = mmunit.MOLAR_GAS_CONSTANT_R / mmunit.MOLAR_GAS_CONSTANT_R.unit

    def __init__(
        self,
        stateGetter: t.Callable[..., mm.State],
        kineticEnergy: bool = False,
        temperature: bool = False,
    ) -> None:
        self._kineticEnergy = kineticEnergy
        self._temperatures = temperatures

    def getHeaders(self) -> t.List[str]:
        headers = []
        if self._kineticEnergy:
            headers.append("Extra Kinetic Energy (kJ/mol)")
        for xdof in self._temperatures:
            headers.append(f"Temperature[{xdof.name}] (K)")
        return headers

    def getReportValues(
        self, simulation: mmapp.Simulation, state: State
    ) -> t.List[float]:
        state = simulation.context.getExtensionState(
            getEnergy=self._kineticEnergy,
            getVelocities=bool(self._temperatures),
        )
        values = []
        if self._kineticEnergy:
            values.append(state.getKineticEnergy() / mmunit.kilojoules_per_mole)
        for i, xdof in enumerate(self._temperatures):
            values.append(state.getExtraTemperature(xdof.index))
        if self._temperatures:
            velocities = state.getVelocities(asNumpy=True)
            for i, xdof in enumerate(self._temperatures):
                values.append(xdof.mass * velocities[i, 0] ** 2 / self._kB)
        return values
