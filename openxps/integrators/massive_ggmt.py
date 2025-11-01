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
    bathLoops
        The number of internal loops in the thermostat. A larger number will increase
        the accuracy and stability of the integrator but will also increase the
        computational cost.
    forceFirst
        If True, the integrator will apply a force-first scheme rather than a
        symmetric operator splitting scheme.

    Example
    -------
    >>> import openxps as xps
    >>> from openmm import unit
    >>> # Symmetric scheme (default)
    >>> integrator = xps.integrators.MassiveGGMTIntegrator(
    ...     300 * unit.kelvin, 20 * unit.femtoseconds, 2 * unit.femtoseconds, 4
    ... )
    >>> integrator
    Per-dof variables:
      v1, v2
    Global variables:
      kT = 2.494338785445972
      Q1 = 0.000997735514178389
      Q2 = 0.016553698576853793
    Computation steps:
       0: allow forces to update the context state
       1: v <- v + 0.5*dt*f/m
       2: constrain velocities
       3: x <- x + 0.5*dt*v
       4: v1 <- v1 + 0.125*dt*(m*v^2 - kT)/Q1
       5: v2 <- v2 + 0.125*dt*((m*v^2)^2/3 - kT^2)/Q2
       6: v <- v*exp(-0.25*dt*(v1+kT*v2))/sqrt(1+0.5*dt*m*v^2*v2/3)
       7: v1 <- v1 + 0.25*dt*(m*v^2 - kT)/Q1
       8: v2 <- v2 + 0.25*dt*((m*v^2)^2/3 - kT^2)/Q2
       9: v <- v*exp(-0.25*dt*(v1+kT*v2))/sqrt(1+0.5*dt*m*v^2*v2/3)
      10: v1 <- v1 + 0.25*dt*(m*v^2 - kT)/Q1
      11: v2 <- v2 + 0.25*dt*((m*v^2)^2/3 - kT^2)/Q2
      12: v <- v*exp(-0.25*dt*(v1+kT*v2))/sqrt(1+0.5*dt*m*v^2*v2/3)
      13: v1 <- v1 + 0.25*dt*(m*v^2 - kT)/Q1
      14: v2 <- v2 + 0.25*dt*((m*v^2)^2/3 - kT^2)/Q2
      15: v <- v*exp(-0.25*dt*(v1+kT*v2))/sqrt(1+0.5*dt*m*v^2*v2/3)
      16: v1 <- v1 + 0.125*dt*(m*v^2 - kT)/Q1
      17: v2 <- v2 + 0.125*dt*((m*v^2)^2/3 - kT^2)/Q2
      18: x <- x + 0.5*dt*v
      19: v <- v + 0.5*dt*f/m
      20: constrain velocities
    >>> # Force-first scheme
    >>> integrator_ff = xps.integrators.MassiveGGMTIntegrator(
    ...     300 * unit.kelvin, 20 * unit.femtoseconds, 2 * unit.femtoseconds, 4,
    ...     forceFirst=True
    ... )
    """

    def __init__(
        self,
        temperature: mmunit.Quantity,
        timeConstant: mmunit.Quantity,
        stepSize: mmunit.Quantity,
        bathLoops: int,
        forceFirst: bool = False,
    ) -> None:
        if bathLoops < 1:
            raise ValueError("The number of bath loops must be at least 1.")
        super().__init__(stepSize)
        self._forceFirst = forceFirst
        self._add_variables(temperature, timeConstant)
        self.addUpdateContextState()
        splitting = "VROR" if forceFirst else "VRORV"
        for letter in splitting:
            fraction = 1 / splitting.count(letter)
            if letter == "V":
                self._add_boost(fraction)
            elif letter == "R":
                self._add_translation(fraction)
            elif letter == "O":
                self._add_thermostat(fraction, bathLoops)

    def _add_variables(
        self, temperature: mmunit.Quantity, timeConstant: mmunit.Quantity
    ) -> None:
        kT = mmunit.MOLAR_GAS_CONSTANT_R * temperature
        self.addGlobalVariable("kT", kT)
        self.addGlobalVariable("Q1", kT * timeConstant**2)
        self.addGlobalVariable("Q2", 8 / 3 * kT**3 * timeConstant**2)
        self.addPerDofVariable("v1", 0)
        self.addPerDofVariable("v2", 0)

    def _add_translation(self, fraction: float) -> None:
        self.addComputePerDof("x", f"x + {fraction}*dt*v")

    def _add_boost(self, fraction: float) -> None:
        self.addComputePerDof("v", f"v + {fraction}*dt*f/m")
        self.addConstrainVelocities()

    def _update_v1_v2(self, fraction: float) -> None:
        self.addComputePerDof("v1", f"v1 + {fraction}*dt*(m*v^2 - kT)/Q1")
        self.addComputePerDof("v2", f"v2 + {fraction}*dt*((m*v^2)^2/3 - kT^2)/Q2")

    def _update_v(self, fraction: float) -> None:
        self.addComputePerDof(
            "v",
            f"v*exp(-{fraction}*dt*(v1+kT*v2))/sqrt(1+{2 * fraction}*dt*m*v^2*v2/3)",
        )

    def _add_thermostat(self, fraction: float, bathLoops: int) -> None:
        subfraction = fraction / bathLoops
        self._update_v1_v2(0.5 * subfraction)
        for _ in range(bathLoops - 1):
            self._update_v(subfraction)
            self._update_v1_v2(subfraction)
        self._update_v(subfraction)
        self._update_v1_v2(0.5 * subfraction)

    def register_with_system(self, system: mm.System) -> None:
        if system.getNumConstraints() > 0:
            raise ValueError("Massive GGMT integrators do not support constraints.")
