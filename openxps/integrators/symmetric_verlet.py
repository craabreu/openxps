"""
.. module:: openxps.integrators.symmetric_verlet
   :platform: Linux, MacOS, Windows
   :synopsis: Velocity Verlet integrator.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm as mm
from openmm import unit as mmunit

from openxps.utils import preprocess_args

from .utils import IntegratorMixin


class SymmetricVerletIntegrator(IntegratorMixin, mm.CustomIntegrator):
    """A symmetric velocity Verlet integrator.

    Parameters
    ----------
    stepSize
        The step size with which to integrate the system.
    energyDerivatives
        Names of global parameters whose energy derivatives are computed in the force
        evaluation of each step, for :class:`~openxps.InnerProductCoupling`. See
        :meth:`~openxps.integrators.utils.IntegratorMixin.getCapturedDerivatives`.

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

    @preprocess_args
    def __init__(
        self, stepSize: mmunit.Quantity, energyDerivatives: t.Iterable[str] = ()
    ):
        super().__init__(stepSize)
        self._init_energy_derivatives(energyDerivatives)
        self.addPerDofVariable("x1", 0)
        self.addUpdateContextState()
        self._add_derivative_capture("start")
        self.addComputePerDof("v", "v + 0.5*dt*f/m")
        self.addComputePerDof("x", "x + dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self._add_derivative_capture("end")
        self.addComputePerDof("v", "v + (x-x1)/dt + 0.5*dt*f/m")
        self.addConstrainVelocities()
