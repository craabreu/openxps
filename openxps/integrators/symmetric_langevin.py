"""
.. module:: openxps.integrators.symmetric_langevin
   :platform: Linux, MacOS, Windows
   :synopsis: BAOAB integrator.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import numpy as np
import openmm as mm
from openmm import unit as mmunit

from openxps.utils import preprocess_args

from .utils import IntegratorMixin, add_property


@add_property("temperature")
@add_property("friction coefficient")
class SymmetricLangevinIntegrator(IntegratorMixin, mm.CustomIntegrator):
    """A symmetric Langevin integrator using the BAOAB algorithm :cite:`Leimkuhler2013`.

    Parameters
    ----------
    temperature
        The temperature of the heat bath.
    frictionCoeff
        The friction coefficient which couples the system to the heat bath.
    stepSize
        The step size with which to integrate the system.

    Example
    -------
    >>> import openxps as xps
    >>> from openmm import unit
    >>> integrator = xps.integrators.SymmetricLangevinIntegrator(
    ...     300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtosecond
    ... )
    >>> integrator
    Per-dof variables:
      x1
    Global variables:
      kT = 2.49433...
      a = 0.999000...
      b = 0.044699...
    Computation steps:
       0: allow forces to update the context state
       1: v <- v + 0.5*dt*f/m
       2: constrain velocities
       3: x <- x + 0.5*dt*v
       4: v <- a*v + b*sqrt(kT/m)*gaussian
       5: x <- x + 0.5*dt*v
       6: x1 <- x
       7: constrain positions
       8: v <- v + (x-x1)/dt + 0.5*dt*f/m
       9: constrain velocities
    """

    @preprocess_args
    def __init__(
        self,
        temperature: mmunit.Quantity,
        frictionCoeff: mmunit.Quantity,
        stepSize: mmunit.Quantity,
    ) -> None:
        super().__init__(stepSize)
        self._temperature = temperature
        self._friction_coefficient = frictionCoeff
        self._add_variables()
        self.addUpdateContextState()
        self.addComputePerDof("v", "v + 0.5*dt*f/m")
        self.addConstrainVelocities()
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian")
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + (x-x1)/dt + 0.5*dt*f/m")
        self.addConstrainVelocities()

    def _add_variables(self) -> None:
        self.addGlobalVariable("kT", 0)
        self.addGlobalVariable("a", 0)
        self.addGlobalVariable("b", 0)
        self.addPerDofVariable("x1", 0)
        self._update_global_variables()

    def _update_global_variables(self) -> None:
        kt = mmunit.MOLAR_GAS_CONSTANT_R * self.getTemperature()
        friction = self.getFrictionCoefficient()
        dt = self.getStepSize()
        a = np.exp(-friction * dt)
        b = np.sqrt(1 - a**2)
        self.setGlobalVariableByName("kT", kt)
        self.setGlobalVariableByName("a", a)
        self.setGlobalVariableByName("b", b)

    def setStepSize(self, size: mmunit.Quantity) -> None:
        super().setStepSize(size)
        self._update_global_variables()
