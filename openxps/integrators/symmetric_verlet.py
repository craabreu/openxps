"""
.. module:: openxps.integrators.symmetric_verlet
   :platform: Linux, MacOS, Windows
   :synopsis: Velocity Verlet integrator.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import openmm as mm
from openmm import unit as mmunit

from .mixins import IntegratorMixin


class SymmetricVerletIntegrator(IntegratorMixin, mm.CustomIntegrator):
    """A symmetric velocity Verlet integrator.

    Parameters
    ----------
    stepSize
        The step size with which to integrate the system.

    Example
    -------
    >>> import openxps as xps
    >>> from openmm import unit
    >>> integrator = xps.integrators.SymmetricVerletIntegrator(1 * unit.femtosecond)
    >>> integrator
    Per-dof variables:
      x1
    Computation steps:
       0: allow forces to update the context state
       1: v <- v + 0.5*dt*f/m
       2: x <- x + dt*v
       3: x1 <- x
       4: constrain positions
       5: v <- v + (x-x1)/dt + 0.5*dt*f/m
       6: constrain velocities
    """

    def __init__(self, stepSize: mmunit.Quantity):
        super().__init__(stepSize)
        self.addPerDofVariable("x1", 0)
        self.addUpdateContextState()
        self.addComputePerDof("v", "v + 0.5*dt*f/m")
        self.addComputePerDof("x", "x + dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + (x-x1)/dt + 0.5*dt*f/m")
        self.addConstrainVelocities()
