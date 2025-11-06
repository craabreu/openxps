"""
.. module:: openxps.integrators.regulated_nhl
   :platform: Linux, MacOS, Windows
   :synopsis: Regulated Nosé-Hoover-Langevin (NHL) integrator.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import openmm as mm
from openmm import unit as mmunit

from .mixins import (
    FrictionCoefficientMixin,
    IntegratorMixin,
    TemperatureMixin,
    TimeConstantMixin,
)


class RegulatedNHLIntegrator(
    IntegratorMixin,
    TemperatureMixin,
    TimeConstantMixin,
    FrictionCoefficientMixin,
    mm.CustomIntegrator,
):
    """A Regulated Nosé-Hoover-Langevin (NHL) integrator :cite:`Abreu2021`.

    Parameters
    ----------
    temperature
        The temperature.
    timeConstant
        The time constant of the thermostat.
    frictionCoeff
        The friction coefficient which couples the system to the heat bath.
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
    >>> from openmmtools import testsystems

    Symmetric scheme (default)

    >>> integrator = xps.integrators.RegulatedNHLIntegrator(
    ...     temperature=300 * unit.kelvin,
    ...     timeConstant=20 * unit.femtoseconds,
    ...     frictionCoeff=1 / unit.picosecond,
    ...     stepSize=2 * unit.femtoseconds,
    ... )
    >>> integrator
    Per-dof variables:
      v1
    Global variables:
      kT = 2.494338785445972
      invQ = 300680.88760681514
      a = 0.9980019986673331
      b = 0.06318236032318289
    Computation steps:
       0: allow forces to update the context state
       1: v <- v + 0.5*dt*f/m
       2: x <- x + c*tanh(v/c)*0.5*dt; c=sqrt(1*kT/m)
       3: v1 <- v1 + 0.5*dt*(m*v*c*tanh(v/c) - kT)*invQ; c = sqrt(1*kT/m)
       4: v <- v*exp(-v1*0.5*dt)
       5: v1 <- a*v1 + b*gaussian
       6: v <- v*exp(-v1*0.5*dt)
       7: v1 <- v1 + 0.5*dt*(m*v*c*tanh(v/c) - kT)*invQ; c = sqrt(1*kT/m)
       8: x <- x + c*tanh(v/c)*0.5*dt; c=sqrt(1*kT/m)
       9: v <- v + 0.5*dt*f/m

    Force-first scheme

    >>> integrator_ff = xps.integrators.RegulatedNHLIntegrator(
    ...     temperature=300 * unit.kelvin,
    ...     timeConstant=20 * unit.femtoseconds,
    ...     frictionCoeff=1 / unit.picosecond,
    ...     stepSize=2 * unit.femtoseconds,
    ...     forceFirst=True
    ... )
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> try:
    ...     integrator_ff.registerWithSystem(model.system)
    ... except ValueError as e:
    ...     print(e)
    Regulated NHL integrators do not support constraints.
    """

    def __init__(  # noqa: PLR0913
        self,
        temperature: mmunit.Quantity,
        timeConstant: mmunit.Quantity,
        frictionCoeff: mmunit.Quantity,
        stepSize: mmunit.Quantity,
        regulation_parameter: float = 1,
        bathLoops: int = 1,
        forceFirst: bool = False,
    ) -> None:
        if bathLoops < 1:
            raise ValueError("The number of bath loops must be at least 1.")
        super().__init__(stepSize)
        self._forceFirst = forceFirst
        self._regulation_parameter = regulation_parameter
        self._add_temperature(temperature)
        self._add_timeConstant(timeConstant)
        self._add_frictionCoeff(frictionCoeff)
        self.addPerDofVariable("v1", 0)
        self.addUpdateContextState()
        self.setKineticEnergyExpression(
            f"0.5*m*v*c*tanh(v/c); c=sqrt({self._regulation_parameter}*kT/m)"
        )
        self._add_boost(1 if forceFirst else 0.5)
        self._add_translation(0.5 / bathLoops)
        for _ in range(bathLoops - 1):
            self._add_thermostat(1 / bathLoops)
            self._add_translation(1 / bathLoops)
        self._add_thermostat(1 / bathLoops)
        self._add_translation(0.5 / bathLoops)
        if not forceFirst:
            self._add_boost(0.5)

    def _add_translation(self, fraction: float) -> None:
        self.addComputePerDof(
            "x",
            f"x + c*tanh(v/c)*{fraction}*dt; c=sqrt({self._regulation_parameter}*kT/m)",
        )

    def _add_boost(self, fraction: float) -> None:
        self.addComputePerDof("v", f"v + {fraction}*dt*f/m")

    def _add_v1_boost(self, fraction: float) -> str:
        self.addComputePerDof(
            "v1",
            f"v1 + {fraction}*dt*(m*v*c*tanh(v/c) - kT)*invQ"
            f"; c = sqrt({self._regulation_parameter}*kT/m)",
        )

    def _add_v_scaling(self, fraction: float) -> str:
        return self.addComputePerDof("v", f"v*exp(-v1*{fraction}*dt)")

    def _add_thermostat(self, fraction: float) -> None:
        self._add_v1_boost(0.5 * fraction)
        self._add_v_scaling(0.5 * fraction)
        self.addComputePerDof("v1", "a*v1 + b*gaussian")
        self._add_v_scaling(0.5 * fraction)
        self._add_v1_boost(0.5 * fraction)

    def registerWithSystem(self, system: mm.System) -> None:
        if system.getNumConstraints() > 0:
            raise ValueError("Regulated NHL integrators do not support constraints.")
