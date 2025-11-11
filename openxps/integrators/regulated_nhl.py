"""
.. module:: openxps.integrators.regulated_nhl
   :platform: Linux, MacOS, Windows
   :synopsis: Regulated Nosé-Hoover-Langevin (NHL) integrator.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import numpy as np
import openmm as mm
from openmm import unit as mmunit

from openxps.utils import preprocess_args

from .utils import IntegratorMixin, add_property


@add_property("temperature", mmunit.kelvin)
@add_property("time constant", mmunit.picosecond)
@add_property("friction coefficient", 1 / mmunit.picosecond)
class RegulatedNHLIntegrator(IntegratorMixin, mm.CustomIntegrator):
    r"""A Regulated Nosé-Hoover-Langevin (NHL) integrator :cite:`Abreu2021`.

    The Regulated Nosé-Hoover-Langevin (NHL) integrator is effective at maintaining
    temperature control under steady-state conditions, such as in d-AFED/TAMD
    :cite:`Maragliano2006,Abrams2008` or UFED :cite:`Chen2012` simulations.

    .. note::
        The friction coefficient is initially set to the inverse of the time constant,
        but this value can be customized via :meth:`setFrictionCoefficient`.

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

    Keyword Arguments
    -----------------
    regulationParameter
        The regulation parameter (see :cite:`Abreu2021` for details).
    forceFirst
        If True, the integrator will apply a force-first scheme. Otherwise, it will
        apply a symmetric operator splitting scheme.

    Example
    -------
    >>> import openxps as xps
    >>> from openmm import unit
    >>> from openmmtools import testsystems

    Symmetric scheme (default)

    >>> integrator = xps.integrators.RegulatedNHLIntegrator(
    ...     temperature=300 * unit.kelvin,
    ...     timeConstant=100 * unit.femtoseconds,
    ...     stepSize=2 * unit.femtoseconds,
    ... )
    >>> integrator
    Per-dof variables:
      v1
    Global variables:
      kT = 2.494...
      invQ = 40.090...
      a = 0.9801...
      b = 1.9801...
    Computation steps:
       0: allow forces to update the context state
       1: v <- v + 0.5*dt*f/m
       2: x <- x + 0.5*dt*c*tanh(v/c); c=sqrt(1*kT/m)
       3: v1 <- v1 + 0.5*dt*(m*v*c*tanh(v/c) - kT)*invQ; c=sqrt(1*kT/m)
       4: v <- v*exp(-0.5*dt*v1)
       5: v1 <- a*v1 + b*gaussian
       6: v <- v*exp(-0.5*dt*v1)
       7: v1 <- v1 + 0.5*dt*(m*v*c*tanh(v/c) - kT)*invQ; c=sqrt(1*kT/m)
       8: x <- x + 0.5*dt*c*tanh(v/c); c=sqrt(1*kT/m)
       9: v <- v + 0.5*dt*f/m

    Force-first scheme

    >>> integrator_ff = xps.integrators.RegulatedNHLIntegrator(
    ...     temperature=300 * unit.kelvin,
    ...     timeConstant=100 * unit.femtoseconds,
    ...     stepSize=2 * unit.femtoseconds,
    ...     forceFirst=True
    ... )
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> try:
    ...     integrator_ff.registerWithSystem(model.system, False)
    ... except ValueError as e:
    ...     print(e)
    Regulated NHL integrators do not support constraints.
    """

    @preprocess_args
    def __init__(  # noqa: PLR0913
        self,
        temperature: mmunit.Quantity,
        timeConstant: mmunit.Quantity,
        stepSize: mmunit.Quantity,
        *,
        regulationParameter: float = 1,
        forceFirst: bool = False,
    ) -> None:
        super().__init__(stepSize)
        self._forceFirst = forceFirst
        self._regulation_parameter = regulationParameter
        self._init_temperature(temperature)
        self._init_time_constant(timeConstant)
        self._init_friction_coefficient(1 / timeConstant)
        self._add_variables()
        self.addUpdateContextState()
        self.setKineticEnergyExpression(
            f"0.5*m*v*c*tanh(v/c); c=sqrt({self._regulation_parameter}*kT/m)"
        )
        self._add_boost(1 if forceFirst else 0.5)
        self._add_translation(0.5)
        self._add_thermostat(1)
        self._add_translation(0.5)
        if not forceFirst:
            self._add_boost(0.5)

    def _add_variables(self) -> None:
        self.addGlobalVariable("kT", 0)
        self.addGlobalVariable("invQ", 0)
        self.addGlobalVariable("a", 0)
        self.addGlobalVariable("b", 0)
        self.addPerDofVariable("v1", 0)
        self._update_global_variables()

    def _update_global_variables(self) -> None:
        tau = self.getTimeConstant()
        dt = self.getStepSize()
        friction = self.getFrictionCoefficient()
        kt = mmunit.MOLAR_GAS_CONSTANT_R * self.getTemperature()
        self.setGlobalVariableByName("kT", kt)
        self.setGlobalVariableByName("invQ", 1 / (kt * tau**2))
        self.setGlobalVariableByName("a", np.exp(-friction * dt))
        self.setGlobalVariableByName("b", np.sqrt(1 - np.exp(-2 * friction * dt)) / tau)

    def _add_translation(self, fraction: float) -> None:
        self.addComputePerDof(
            "x",
            f"x + {fraction}*dt*c*tanh(v/c); c=sqrt({self._regulation_parameter}*kT/m)",
        )

    def _add_boost(self, fraction: float) -> None:
        self.addComputePerDof("v", f"v + {fraction}*dt*f/m")

    def _add_v1_boost(self, fraction: float) -> str:
        self.addComputePerDof(
            "v1",
            f"v1 + {fraction}*dt*(m*v*c*tanh(v/c) - kT)*invQ"
            f"; c=sqrt({self._regulation_parameter}*kT/m)",
        )

    def _add_v_scaling(self, fraction: float) -> str:
        return self.addComputePerDof("v", f"v*exp(-{fraction}*dt*v1)")

    def _add_thermostat(self, fraction: float) -> None:
        self._add_v1_boost(0.5 * fraction)
        self._add_v_scaling(0.5 * fraction)
        self.addComputePerDof("v1", "a*v1 + b*gaussian")
        self._add_v_scaling(0.5 * fraction)
        self._add_v1_boost(0.5 * fraction)

    def registerWithSystem(self, system: mm.System, isExtension: bool) -> None:
        if system.getNumConstraints() > 0:
            raise ValueError("Regulated NHL integrators do not support constraints.")
