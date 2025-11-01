"""
.. module:: openxps.integrators.csvr
   :platform: Linux, MacOS, Windows
   :synopsis: Canonical Sampling through Velocity Rescaling (CSVR) integrator.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import numpy as np
import openmm as mm
from openmm import unit as mmunit

from .mixin import IntegratorMixin


class CSVRIntegrator(IntegratorMixin, mm.CustomIntegrator):
    """
    Implements the Canonical Sampling through Velocity Rescaling (CSVR) integrator,
    also known as the Bussi-Donadio-Parrinello thermostat.

    Parameters
    ----------
    temperature
        The temperature.
    frictionCoeff
        The friction coefficient.
    stepSize
        The integration step size.
    forceFirst
        If True, the integrator will apply a force-first scheme rather than a
        symmetric operator splitting scheme.

    Example
    -------
    >>> import openxps as xps
    >>> from openmm import unit
    >>> # Symmetric scheme (default)
    >>> integrator = xps.integrators.CSVRIntegrator(
    ...     300 * unit.kelvin, 10 / unit.picosecond, 2 * unit.femtoseconds
    ... )
    >>> integrator
    Per-dof variables:
      x1
    Global variables:
      sumRsq = 0.0
      mvv = 0.0
      kT = 2.494338785445972
      friction = 10.0
    Computation steps:
       0: allow forces to update the context state
       1: v <- v + 0.5*dt*f/m
       2: constrain velocities
       3: x <- x + 0.5*dt*v
       4: x1 <- x
       5: constrain positions
       6: v <- v + (x - x1)/(0.5*dt)
       7: constrain velocities
       8: mvv <- sum(m*v*v)
       9: v <- v*sqrt(A + BC*(R1 ^ 2 + sumRsq) + 2*sqrt(A*BC)*R1); ...
      10: x <- x + 0.5*dt*v
      11: x1 <- x
      12: constrain positions
      13: v <- v + (x - x1)/(0.5*dt)
      14: constrain velocities
      15: v <- v + 0.5*dt*f/m
      16: constrain velocities
    >>> # Force-first scheme
    >>> integrator_ff = xps.integrators.CSVRIntegrator(
    ...     300 * unit.kelvin, 10 / unit.picosecond, 2 * unit.femtoseconds,
    ...     forceFirst=True
    ... )
    """

    def __init__(
        self,
        temperature: mmunit.Quantity,
        frictionCoeff: mmunit.Quantity,
        stepSize: mmunit.Quantity,
        forceFirst: bool = False,
    ) -> None:
        super().__init__(stepSize)
        self._forceFirst = forceFirst
        self._num_dof = None
        self._rng = np.random.default_rng(None)
        self._add_global_variables(temperature, frictionCoeff)
        self.addUpdateContextState()
        splitting = "VROR" if forceFirst else "VRORV"
        for letter in splitting:
            n = splitting.count(letter)
            timestep = "dt" if n == 1 else f"{1 / n}*dt"
            if letter == "V":
                self._add_boost(timestep)
            elif letter == "R":
                self._add_translation(timestep)
            elif letter == "O":
                self._add_rescaling(timestep)

    def _add_global_variables(
        self, temperature: mmunit.Quantity, frictionCoeff: mmunit.Quantity
    ) -> None:
        self.addPerDofVariable("x1", 0)
        self.addGlobalVariable("sumRsq", 0)
        self.addGlobalVariable("mvv", 0)
        self.addGlobalVariable("kT", mmunit.MOLAR_GAS_CONSTANT_R * temperature)
        self.addGlobalVariable("friction", frictionCoeff)

    def _add_translation(self, timestep: str) -> None:
        self.addComputePerDof("x", f"x + {timestep}*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", f"v + (x - x1)/({timestep})")
        self.addConstrainVelocities()

    def _add_boost(self, timestep: str) -> None:
        self.addComputePerDof("v", f"v + {timestep}*f/m")
        self.addConstrainVelocities()

    def _add_rescaling(self, timestep: str) -> None:
        self.addComputeSum("mvv", "m*v*v")
        self.addComputePerDof(
            "v",
            "v*sqrt(A + BC*(R1 ^ 2 + sumRsq) + 2*sqrt(A*BC)*R1)"
            "; R1 = gaussian"
            "; BC = (1 - A)*kT/mvv"
            f"; A = exp(-friction*{timestep})",
        )

    def _sums_of_squared_gaussians(self, num_steps: int) -> np.ndarray:
        sumRsq = 2.0 * self._rng.standard_gamma((self._num_dof - 1) // 2, num_steps)
        if self._num_dof % 2 == 0:
            sumRsq += self._rng.standard_normal(num_steps) ** 2
        return sumRsq

    def register_with_system(self, system: mm.System) -> None:
        self._num_dof = IntegratorMixin._countDegreesOfFreedom(system)

    def setRandomNumberSeed(self, seed: int) -> None:
        """
        This method overrides the :class:`openmm.CustomIntegrator` method to also set
        the seed of the random number generator used to pick numbers from the gamma
        distribution.
        Parameters
        ----------
        seed
            The seed to use for the random number generator.
        """
        self._rng = np.random.default_rng(seed + 2**31)
        super().setRandomNumberSeed(self._rng.integers(-(2**31), 2**31))

    def step(self, steps: int) -> None:
        """
        This method overrides the :class:`openmm.CustomIntegrator` method to include the
        efficient computation of the sum of squares of normally distributed random
        numbers.
        Parameters
        ----------
        steps
            The number of steps to take.
        """
        for sumRsq in self._sums_of_squared_gaussians(steps):
            self.setGlobalVariableByName("sumRsq", sumRsq)
            super().step(1)
