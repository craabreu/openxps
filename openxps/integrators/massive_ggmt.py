"""
.. module:: openxps.integrators.massive_ggmt
   :platform: Linux, MacOS, Windows
   :synopsis: Massive Generalized Gaussian Moment Thermostat (GGMT) integrator.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import openmm as mm
from openmm import unit as mmunit

from openxps.integrators.utils import IntegratorMixin, add_property
from openxps.utils import preprocess_args


@add_property("temperature", mmunit.kelvin)
@add_property("time constant", 1 / mmunit.picosecond)
class MassiveGGMTIntegrator(IntegratorMixin, mm.CustomIntegrator):
    """A massive Generalized Gaussian Moment Thermostat integrator :cite:`Liu2000`.

    The Generalized Gaussian Moment Thermostat (GGMT) is effective at maintaining
    temperature control under steady-state conditions, such as in adiabatic free energy
    dynamics (AFED) simulations :cite:`Abrams2008`.

    .. warning::
        This integrator does not support constraints due to its massive nature, i.e.,
        independent thermostatting of each degree of freedom.

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
    >>> from openmmtools import testsystems

    Symmetric scheme (default)

    >>> integrator = xps.integrators.MassiveGGMTIntegrator(
    ...     temperature=300 * unit.kelvin,
    ...     timeConstant=40 * unit.femtoseconds,
    ...     stepSize=2 * unit.femtoseconds,
    ...     bathLoops=2
    ... )
    >>> integrator
    Per-dof variables:
      v1, v2
    Global variables:
      kT = 2.4943387...
      invQ = 250.567...
      invQ2 = 15.1023...
    Computation steps:
       0: allow forces to update the context state
       1: v <- v + 0.5*dt*f/m
       2: x <- x + 0.25*dt*v
       3: v1 <- v1 + 0.25*dt*(m*v^2 - kT)*invQ
       4: v2 <- v2 + 0.25*dt*((m*v^2)^2/3 - kT^2)*invQ2
       5: v <- v*exp(-0.5*dt*(v1 + kT*v2))/sqrt(1 + 1.0*dt*m*v^2*v2/3)
       6: v2 <- v2 + 0.25*dt*((m*v^2)^2/3 - kT^2)*invQ2
       7: v1 <- v1 + 0.25*dt*(m*v^2 - kT)*invQ
       8: x <- x + 0.5*dt*v
       9: v1 <- v1 + 0.25*dt*(m*v^2 - kT)*invQ
      10: v2 <- v2 + 0.25*dt*((m*v^2)^2/3 - kT^2)*invQ2
      11: v <- v*exp(-0.5*dt*(v1 + kT*v2))/sqrt(1 + 1.0*dt*m*v^2*v2/3)
      12: v2 <- v2 + 0.25*dt*((m*v^2)^2/3 - kT^2)*invQ2
      13: v1 <- v1 + 0.25*dt*(m*v^2 - kT)*invQ
      14: x <- x + 0.25*dt*v
      15: v <- v + 0.5*dt*f/m

    Force-first scheme

    >>> integrator_ff = xps.integrators.MassiveGGMTIntegrator(
    ...     temperature=300 * unit.kelvin,
    ...     timeConstant=40 * unit.femtoseconds,
    ...     stepSize=2 * unit.femtoseconds,
    ...     bathLoops=2,
    ...     forceFirst=True
    ... )
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> try:
    ...     integrator_ff.registerWithSystem(model.system)
    ... except ValueError as e:
    ...     print(e)
    Massive GGMT integrators do not support constraints.
    """

    @preprocess_args
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
        self._init_temperature(temperature)
        self._init_time_constant(timeConstant)
        self._add_variables()
        self.addUpdateContextState()
        self._add_boost(1 if forceFirst else 0.5)
        self._add_translation(0.5 / bathLoops)
        for _ in range(bathLoops - 1):
            self._add_thermostat(1 / bathLoops)
            self._add_translation(1 / bathLoops)
        self._add_thermostat(1 / bathLoops)
        self._add_translation(0.5 / bathLoops)
        if not forceFirst:
            self._add_boost(0.5)

    def _add_variables(self) -> None:
        self.addGlobalVariable("kT", 0)
        self.addGlobalVariable("invQ", 0)
        self.addGlobalVariable("invQ2", 0)
        self.addPerDofVariable("v1", 0)
        self.addPerDofVariable("v2", 0)
        self._update_global_variables()

    def _update_global_variables(self) -> None:
        tau = self.getTimeConstant()
        kt = mmunit.MOLAR_GAS_CONSTANT_R * self.getTemperature()
        self.setGlobalVariableByName("kT", kt)
        self.setGlobalVariableByName("invQ", 1 / (kt * tau**2))
        self.setGlobalVariableByName("invQ2", 3 / (8 * kt**3 * tau**2))

    def _add_translation(self, fraction: float) -> None:
        self.addComputePerDof("x", f"x + {fraction}*dt*v")

    def _add_boost(self, fraction: float) -> None:
        self.addComputePerDof("v", f"v + {fraction}*dt*f/m")

    def _add_v1_boost(self, fraction: float) -> None:
        self.addComputePerDof("v1", f"v1 + {fraction}*dt*(m*v^2 - kT)*invQ")

    def _add_v2_boost(self, fraction: float) -> None:
        self.addComputePerDof("v2", f"v2 + {fraction}*dt*((m*v^2)^2/3 - kT^2)*invQ2")

    def _add_thermostat(self, fraction: float) -> None:
        self._add_v1_boost(0.5 * fraction)
        self._add_v2_boost(0.5 * fraction)
        self.addComputePerDof(
            "v",
            f"v*exp(-{fraction}*dt*(v1 + kT*v2))"
            "/"
            f"sqrt(1 + {2 * fraction}*dt*m*v^2*v2/3)",
        )
        self._add_v2_boost(0.5 * fraction)
        self._add_v1_boost(0.5 * fraction)

    def registerWithSystem(self, system: mm.System) -> None:
        if system.getNumConstraints() > 0:
            raise ValueError("Massive GGMT integrators do not support constraints.")
