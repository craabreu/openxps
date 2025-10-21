"""
.. module:: openxps.integrators.baoab
   :platform: Linux, MacOS, Windows
   :synopsis: BAOAB integrator.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import numpy as np
import openmm as mm
from openmm import unit as mmunit


class BAOABIntegrator(mm.CustomIntegrator):
    """
    A BAOAB integrator.

    Parameters
    ----------
    temperature
        The temperature of the heat bath.
    frictionCoeff
        The friction coefficient which couples the system to the heat bath.
    stepSize
        The step size with which to integrate the system.
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
