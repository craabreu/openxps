"""
.. module:: openxps.integrators.massive_ggmt
   :platform: Linux, MacOS, Windows
   :synopsis: Massive Generalized Gaussian Moment Thermostat (GGMT) integrator.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import numpy as np
import openmm as mm
from openmm import unit as mmunit

from openxps.integrators.utils import IntegratorMixin, add_property
from openxps.utils import preprocess_args


@add_property("temperature", mmunit.kelvin)
@add_property("time constant", 1 / mmunit.picosecond)
@add_property("friction coefficient", 1 / mmunit.picosecond)
class MassiveGGMTIntegrator(IntegratorMixin, mm.CustomIntegrator):
    r"""A massive, second- and fourth-moment GGMT integrator :cite:`Liu2000`.

    The massive GGMT integrator is effective at maintaining temperature control under
    steady-state conditions, such as in d-AFED/TAMD :cite:`Maragliano2006,Abrams2008`
    or UFED :cite:`Chen2012` simulations.

    This implementation optionally introduces stochasticity by applying Langevin-type
    noise to the variables controlling the second- and fourth-moment of the velocity
    distribution.

    .. note::
        If stochasticity is enabled, the friction coefficient is initially set to the
        inverse of the time constant, but this value can be customized via
        :meth:`setFrictionCoefficient`.

    .. warning::
        This integrator is incompatible with systems subject to constraints.

    Parameters
    ----------
    temperature
        The temperature of the heat bath.
    timeConstant
        The time constant (a.k.a. coupling/damping/relaxation time) of the thermostat.
    stepSize
        The integration step size.
    bathLoops
        The number of internal loops in the thermostat. A larger number increases
        accuracy and stability at the expense of computational cost.

    Keyword Arguments
    -----------------
    forceFirst
        If True, the integrator will apply a force-first scheme. Otherwise, it will
        apply a symmetric operator splitting scheme.
    stochastic
        If True, the integrator will apply stochasticity to the variables controlling
        the second- and fourth-moment of the velocity distribution.

    Example
    -------
    >>> import openxps as xps
    >>> from openmm import unit
    >>> from openmmtools import testsystems

    Symmetric scheme (default)

    >>> integrator = xps.integrators.MassiveGGMTIntegrator(
    ...     temperature=300 * unit.kelvin,
    ...     timeConstant=100 * unit.femtoseconds,
    ...     stepSize=2 * unit.femtoseconds,
    ...     bathLoops=2
    ... )
    >>> integrator
    Per-dof variables:
      v1, v2
    Global variables:
      kT = 2.494338...
      invQ = 40.090...
      invQ2 = 2.416...
    Computation steps:
       0: allow forces to update the context state
       1: v <- v + 0.5*dt*f/m
       2: x <- x + 0.25*dt*v
       3: v1 <- v1 + 0.25*dt*(m*v^2 - kT)*invQ
       4: v2 <- v2 + 0.25*dt*((m*v^2)^2/3 - kT^2)*invQ2
       5: v <- v*exp(-0.5*dt*(v1 + kT*v2))/sqrt(1 + 0.333...*dt*m*v^2*v2)
       6: v2 <- v2 + 0.25*dt*((m*v^2)^2/3 - kT^2)*invQ2
       7: v1 <- v1 + 0.25*dt*(m*v^2 - kT)*invQ
       8: x <- x + 0.5*dt*v
       9: v1 <- v1 + 0.25*dt*(m*v^2 - kT)*invQ
      10: v2 <- v2 + 0.25*dt*((m*v^2)^2/3 - kT^2)*invQ2
      11: v <- v*exp(-0.5*dt*(v1 + kT*v2))/sqrt(1 + 0.333...*dt*m*v^2*v2)
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
    ...     integrator_ff.registerWithSystem(model.system, False)
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
        bathLoops: int = 1,
        *,
        forceFirst: bool = False,
        stochastic: bool = False,
    ) -> None:
        if bathLoops < 1:
            raise ValueError("The number of bath loops must be at least 1.")
        super().__init__(stepSize)
        self._forceFirst = forceFirst
        self._init_temperature(temperature)
        self._init_time_constant(timeConstant)
        self._init_friction_coefficient(1 / timeConstant if stochastic else 0.0)
        self._bath_loops = bathLoops
        self._stochastic = stochastic
        self._add_variables()
        self.addUpdateContextState()
        self._add_boost(1 if forceFirst else 0.5)
        self._add_translation(0.5 / self._bath_loops)
        for _ in range(self._bath_loops - 1):
            self._add_thermostat(1 / self._bath_loops)
            self._add_translation(1 / self._bath_loops)
        self._add_thermostat(1 / self._bath_loops)
        self._add_translation(0.5 / self._bath_loops)
        if not forceFirst:
            self._add_boost(0.5)

    def _add_variables(self) -> None:
        self.addGlobalVariable("kT", 0)
        self.addGlobalVariable("invQ", 0)
        self.addGlobalVariable("invQ2", 0)
        self.addPerDofVariable("v1", 0)
        self.addPerDofVariable("v2", 0)
        if self._stochastic:
            self.addGlobalVariable("a", 0)
            self.addGlobalVariable("b1", 0)
            self.addGlobalVariable("b2", 0)
        self._update_global_variables()

    def _update_global_variables(self) -> None:
        tau = self.getTimeConstant()
        kt = mmunit.MOLAR_GAS_CONSTANT_R * self.getTemperature()
        self.setGlobalVariableByName("kT", kt)
        self.setGlobalVariableByName("invQ", 1 / (kt * tau**2))
        self.setGlobalVariableByName("invQ2", 3 / (8 * kt**3 * tau**2))
        if self._stochastic:
            friction = self.getFrictionCoefficient()
            dt = self.getStepSize() / self._bath_loops
            a = np.exp(-friction * dt)
            b1 = np.sqrt(1 - a**2) / tau
            b2 = b1 * np.sqrt(3 / 8) / kt
            self.setGlobalVariableByName("a", a)
            self.setGlobalVariableByName("b1", b1)
            self.setGlobalVariableByName("b2", b2)

    def _add_translation(self, fraction: float) -> None:
        self.addComputePerDof("x", f"x + {fraction}*dt*v")

    def _add_boost(self, fraction: float) -> None:
        self.addComputePerDof("v", f"v + {fraction}*dt*f/m")

    def _add_v1_boost(self, fraction: float) -> None:
        self.addComputePerDof("v1", f"v1 + {fraction}*dt*(m*v^2 - kT)*invQ")

    def _add_v2_boost(self, fraction: float) -> None:
        self.addComputePerDof("v2", f"v2 + {fraction}*dt*((m*v^2)^2/3 - kT^2)*invQ2")

    def _add_v_scaling(self, fraction: float) -> None:
        self.addComputePerDof(
            "v",
            f"v*exp(-{fraction}*dt*(v1 + kT*v2))"
            "/"
            f"sqrt(1 + {(2 * fraction) / 3}*dt*m*v^2*v2)",
        )

    def _add_thermostat(self, fraction: float) -> None:
        self._add_v1_boost(0.5 * fraction)
        self._add_v2_boost(0.5 * fraction)
        if self._stochastic:
            self._add_v_scaling(0.5 * fraction)
            self.addComputePerDof("v1", "a*v1 + b1*gaussian")
            self.addComputePerDof("v2", "a*v2 + b2*gaussian")
            self._add_v_scaling(0.5 * fraction)
        else:
            self._add_v_scaling(fraction)
        self._add_v2_boost(0.5 * fraction)
        self._add_v1_boost(0.5 * fraction)

    def registerWithSystem(self, system: mm.System, isExtension: bool) -> None:
        if system.getNumConstraints() > 0:
            raise ValueError("Massive GGMT integrators do not support constraints.")
