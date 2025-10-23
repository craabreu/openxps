"""
.. module:: openxps.integrators.baoab
   :platform: Linux, MacOS, Windows
   :synopsis: BAOAB integrator.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import numpy as np
import openmm as mm
from openmm import unit as mmunit

from .mixin import IntegratorMixin


class BAOABIntegrator(IntegratorMixin, mm.CustomIntegrator):
    """
    A BAOAB integrator.

    This is a reversible integrator.

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
    >>> integrator = xps.integrators.BAOABIntegrator(
    ...     300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtosecond
    ... )
    >>> integrator
    Per-dof variables:
      x1
    Global variables:
      a = 0.999000499833375
      b = 0.044699008184376096
      kT = 2.494338785445972
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

    def __init__(
        self,
        temperature: mmunit.Quantity,
        frictionCoeff: mmunit.Quantity,
        stepSize: mmunit.Quantity,
    ) -> None:
        super().__init__(stepSize)
        self.addGlobalVariable("a", np.exp(-frictionCoeff * stepSize))
        self.addGlobalVariable("b", np.sqrt(1 - np.exp(-2 * frictionCoeff * stepSize)))
        self.addGlobalVariable("kT", mmunit.MOLAR_GAS_CONSTANT_R * temperature)
        self.addPerDofVariable("x1", 0)
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
