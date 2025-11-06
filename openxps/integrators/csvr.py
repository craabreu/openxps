"""
.. module:: openxps.integrators.csvr
   :platform: Linux, MacOS, Windows
   :synopsis: Canonical Sampling through Velocity Rescaling (CSVR) integrator.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import numpy as np
import openmm as mm
from openmm import unit as mmunit

from .mixins import IntegratorMixin


class CSVRIntegrator(IntegratorMixin, mm.CustomIntegrator):
    """The Canonical Sampling through Velocity Rescaling integrator :cite:`Bussi2007`.

    The Canonical Sampling through Velocity Rescaling (CSVR) thermostat scales all
    velocities by a single factor stochastically determined in order to preserve the
    canonical distribution at the specified temperature.

    .. note::
        In an extended phase-space simulation, the integrator must be used in
        one-dimensional mode for the extension system.

    .. warning::
        When used outside the context of OpenXPS, the :meth:`registerWithSystem` method
        must be called before starting a simulation.

    Parameters
    ----------
    temperature
        The temperature.
    frictionCoeff
        The friction coefficient.
    stepSize
        The integration step size.
    oneDimensional
        If True, the integrator will be used in one-dimensional mode, with only the
        x-coordinate of the velocity vectors being rescaled.
    forceFirst
        If True, the integrator will apply a force-first scheme rather than a
        symmetric operator splitting scheme.

    Example
    -------
    >>> import openxps as xps
    >>> import openmm as mm
    >>> from openmm import unit
    >>> from openmmtools import testsystems

    Symmetric scheme (default)

    >>> integrator = xps.integrators.CSVRIntegrator(
    ...     temperature=300 * unit.kelvin,
    ...     frictionCoeff=10 / unit.picosecond,
    ...     stepSize=2 * unit.femtoseconds,
    ... )
    >>> integrator
    Per-dof variables:
      x1
    Global variables:
      sumRsq = 0.0
      mvv = 0.0
      kT = 2.49433...
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
       9: v <- v*sqrt(A + BC*(R1^2 + sumRsq) + 2*sqrt(A*BC)*R1); ...
      10: x <- x + 0.5*dt*v
      11: x1 <- x
      12: constrain positions
      13: v <- v + (x - x1)/(0.5*dt)
      14: constrain velocities
      15: v <- v + 0.5*dt*f/m
      16: constrain velocities

    Force-first scheme

    >>> integrator_ff = xps.integrators.CSVRIntegrator(
    ...     temperature=300 * unit.kelvin,
    ...     frictionCoeff=10 / unit.picosecond,
    ...     stepSize=2 * unit.femtoseconds,
    ...     forceFirst=True
    ... )
    >>> integrator_ff.getNumDegreesOfFreedom() is None
    True
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> integrator_ff.registerWithSystem(model.system)
    >>> integrator_ff.getNumDegreesOfFreedom()
    51
    >>> integrator_ff.setRandomNumberSeed(1234)
    >>> platform = mm.Platform.getPlatformByName("Reference")
    >>> context = mm.Context(model.system, integrator_ff, platform)
    >>> context.setPositions(model.positions)
    >>> context.setVelocitiesToTemperature(300 * unit.kelvin, 4321)
    >>> integrator_ff.step(10)
    """

    def __init__(
        self,
        temperature: mmunit.Quantity,
        frictionCoeff: mmunit.Quantity,
        stepSize: mmunit.Quantity,
        oneDimensional: bool = False,
        forceFirst: bool = False,
    ) -> None:
        super().__init__(stepSize)
        self._forceFirst = forceFirst
        self._oneDimensional = oneDimensional
        self._num_dof = None
        self._rng = np.random.default_rng(None)
        self._add_global_variables(temperature, frictionCoeff)
        self.addUpdateContextState()
        self._add_boost(1 if forceFirst else 0.5)
        self._add_translation(0.5)
        self._add_rescaling(1)
        self._add_translation(0.5)
        if not forceFirst:
            self._add_boost(0.5)

    def _add_global_variables(
        self, temperature: mmunit.Quantity, frictionCoeff: mmunit.Quantity
    ) -> None:
        self.addPerDofVariable("x1", 0)
        self.addGlobalVariable("sumRsq", 0)
        self.addGlobalVariable("mvv", 0)
        self.addGlobalVariable("kT", mmunit.MOLAR_GAS_CONSTANT_R * temperature)
        self.addGlobalVariable("friction", frictionCoeff)

    def _add_translation(self, fraction: float) -> None:
        self.addComputePerDof("x", f"x + {fraction}*dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", f"v + (x - x1)/({fraction}*dt)")
        self.addConstrainVelocities()

    def _add_boost(self, fraction: float) -> None:
        self.addComputePerDof("v", f"v + {fraction}*dt*f/m")
        self.addConstrainVelocities()

    def _add_rescaling(self, fraction: float) -> None:
        self.addComputeSum("mvv", "m*_x(v)^2/3" if self._oneDimensional else "m*v*v")
        self.addComputePerDof(
            "v",
            "v*sqrt(A + BC*(R1^2 + sumRsq) + 2*sqrt(A*BC)*R1)"
            "; R1 = gaussian"
            "; BC = (1 - A)*kT/mvv"
            f"; A = exp(-{fraction}*dt*friction)",
        )

    def _sums_of_squared_gaussians(self, num_steps: int) -> np.ndarray:
        sumRsq = 2.0 * self._rng.standard_gamma((self._num_dof - 1) // 2, num_steps)
        if self._num_dof % 2 == 0:
            sumRsq += self._rng.standard_normal(num_steps) ** 2
        return sumRsq

    def isOneDimensional(self) -> bool:
        """Check if the integrator is one-dimensional."""
        return self._oneDimensional

    def getNumDegreesOfFreedom(self) -> int:
        """Get the number of degrees of freedom in the system."""
        return self._num_dof

    def registerWithSystem(self, system: mm.System) -> None:
        if self._oneDimensional:
            if system.getNumConstraints() > 0:
                raise ValueError(
                    "One-dimensional CSVR integrator cannot be used with constraints."
                )
            self._num_dof = system.getNumParticles()
        else:
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
