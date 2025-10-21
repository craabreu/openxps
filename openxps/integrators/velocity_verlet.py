"""
.. module:: openxps.integrators.velocity_verlet
   :platform: Linux, MacOS, Windows
   :synopsis: Velocity Verlet integrator.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import openmm as mm
from openmm import unit as mmunit


class VelocityVerletIntegrator(mm.CustomIntegrator):
    """
    A velocity Verlet integrator.

    Parameters
    ----------
    stepSize
        The step size with which to integrate the system.

    """

    def __init__(self, stepSize: mmunit.Quantity):
        super().__init__(stepSize)
        self.addGlobalVariable("x1", 0)
        self.addUpdateContextState()
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + (x-x1)/dt + 0.5*dt*f/m")
        self.addConstrainVelocities()
