"""
.. module:: openxps.integrators.massive_ggmt
   :platform: Linux, MacOS, Windows
   :synopsis: Massive Generalized Gaussian Moment Thermostat (GGMT) integrator.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import openmm as mm
from openmm import unit as mmunit

from .mixin import IntegratorMixin


class MassiveGGMTIntegrator(IntegratorMixin, mm.CustomIntegrator):
    """
    Implements a massive variant of the Generalized Gaussian Moment Thermostat (GGMT)
    integrator.

    Parameters
    ----------
    temperature
        The temperature.
    timeConstant
        The time constant of the thermostat.
    stepSize
        The integration step size.
    splitting
        The splitting scheme. A sequence of "V", "R", and "O" characters representing
        the velocity boost, position update, and thermostat steps, respectively.
    """

    def __init__(
        self,
        temperature: mmunit.Quantity,
        timeConstant: mmunit.Quantity,
        stepSize: mmunit.Quantity,
        splitting: str,
    ) -> None:
        if set(splitting) != {"V", "R", "O"}:
            raise ValueError(f"Invalid splitting scheme: {splitting}")
        super().__init__(stepSize)
        self._add_variables(temperature, timeConstant)
        self.addUpdateContextState()
        for letter in splitting:
            n = splitting.count(letter)
            timestep = "dt" if n == 1 else f"{1 / n}*dt"
            if letter == "V":
                self._add_boost(timestep)
            elif letter == "R":
                self._add_translation(timestep)
            elif letter == "O":
                self._add_thermostat_boost(f"{1 / (2 * n)}*dt")
                self._add_scaling(timestep)
                self._add_thermostat_boost(f"{1 / (2 * n)}*dt")

    def _add_variables(
        self, temperature: mmunit.Quantity, timeConstant: mmunit.Quantity
    ) -> None:
        kT = mmunit.MOLAR_GAS_CONSTANT_R * temperature
        self.addGlobalVariable("kT", kT)
        self.addGlobalVariable("Q1", kT * timeConstant**2)
        self.addGlobalVariable("Q2", 8 / 3 * kT**3 * timeConstant**2)
        self.addPerDofVariable("v1", 0)
        self.addPerDofVariable("v2", 0)

    def _add_translation(self, timestep: str) -> None:
        self.addComputePerDof("x", f"x + {timestep}*v")

    def _add_boost(self, timestep: str) -> None:
        self.addComputePerDof("v", f"v + {timestep}*f/m")
        self.addConstrainVelocities()

    def _add_thermostat_boost(self, timestep: str) -> None:
        self.addComputePerDof("v1", f"v1 + 0.5*{timestep}*(m*v^2 - kT)/Q1")
        self.addComputePerDof("v2", f"v2 + 0.5*{timestep}*((m*v^2)^2/3 - kT^2)/Q2")

    def _add_scaling(self, timestep: str) -> None:
        self.addComputePerDof(
            "v",
            f"v*exp(-{timestep}*(v1 + kT*v2))/sqrt(1 + 2*{timestep}*m*v^2*v2/3)",
        )

    def register_with_system(self, system: mm.System) -> None:
        if system.getNumConstraints() > 0:
            raise ValueError("Massive GGMT integrators do not support constraints.")


class SymmetricMassiveGGMTIntegrator(MassiveGGMTIntegrator):
    """
    Implements a symmetric, massive variant of the Generalized Gaussian Moment
    Thermostat (GGMT) integrator.

    Parameters
    ----------
    temperature
        The temperature.
    timeConstant
        The time constant of the thermostat.
    stepSize
        The integration step size.
    """

    def __init__(
        self,
        temperature: mmunit.Quantity,
        timeConstant: mmunit.Quantity,
        stepSize: mmunit.Quantity,
    ) -> None:
        super().__init__(temperature, timeConstant, stepSize, "VRORV")


class ForceFirstMassiveGGMTIntegrator(MassiveGGMTIntegrator):
    """
    Implements a force-first, massive variant of the Generalized Gaussian Moment
    Thermostat (GGMT) integrator.

    Parameters
    ----------
    temperature
        The temperature.
    timeConstant
        The time constant of the thermostat.
    stepSize
        The integration step size.
    """

    def __init__(
        self,
        temperature: mmunit.Quantity,
        timeConstant: mmunit.Quantity,
        stepSize: mmunit.Quantity,
    ) -> None:
        super().__init__(temperature, timeConstant, stepSize, "VROR")
