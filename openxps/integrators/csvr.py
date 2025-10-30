"""
.. module:: openxps.integrators.csvr
   :platform: Linux, MacOS, Windows
   :synopsis: Canonical Sampling through Velocity Rescaling (CSVR) integrator.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import numpy as np
import openmm as mm
from openmm import unit as mmunit

from ..dynamical_variable import DynamicalVariable
from .mixin import IntegratorMixin

SplittingScheme: t.TypeAlias = t.Literal["VROR", "VRORV"]
CSVR_FORCE_FIRST: SplittingScheme = "VROR"
CSVR_SYMMETRIC: SplittingScheme = "VRORV"


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
    numDOFs: int
        The number of degrees of freedom in the system.
    splitting
        The splitting scheme. Valid options are ``CSVR_FORCE_FIRST`` and
        ``CSVR_SYMMETRIC``.
    """

    def __init__(
        self,
        temperature: mmunit.Quantity,
        frictionCoeff: mmunit.Quantity,
        stepSize: mmunit.Quantity,
        numDOFs: int,
        splitting: SplittingScheme,
    ) -> None:
        super().__init__(stepSize)
        self._num_dof = numDOFs
        self._rng = np.random.default_rng(None)
        self._add_global_variables(temperature, frictionCoeff)
        self.addUpdateContextState()
        for letter in splitting:
            n = splitting.count(letter)
            timestep = "dt" if n == 1 else f"{1 / n}*dt"
            if letter == "V":
                self._add_boost(timestep)
            elif letter == "R":
                self._add_translation(timestep)
            elif letter == "O":
                self._add_rescaling(timestep)
            else:
                raise ValueError("Valid splitting scheme letters are R, V, and O")

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


class SymmetricCSVRIntegrator(CSVRIntegrator):
    """
    Implements the Canonical Sampling through Velocity Rescaling (CSVR) integrator,
    also known as the Bussi-Donadio-Parrinello thermostat.

    .. note::
        Either a system or a sequence of dynamical variables must be provided.

    Parameters
    ----------
    temperature
        The temperature.
    frictionCoeff
        The friction coefficient.
    stepSize
        The integration step size.

    Keyword Arguments
    -----------------
    physical_system
        The physical :OpenMM:`System` to be used in the XPS simulation, or `None` if
        this integrator is intended to be used for an extension system.
    dynamical_variables
        A sequence of :class:`DynamicalVariable` objects, or `None` if this integrator
        is intended to be used for a physical system.
    """

    def __init__(
        self,
        temperature: mmunit.Quantity,
        frictionCoeff: mmunit.Quantity,
        stepSize: mmunit.Quantity,
        *,
        physical_system: t.Optional[mm.System] = None,
        dynamical_variables: t.Optional[t.Sequence[DynamicalVariable]] = None,
    ) -> None:
        super().__init__(
            temperature,
            frictionCoeff,
            stepSize,
            IntegratorMixin.countDegreesOfFreedom(physical_system, dynamical_variables),
            CSVR_SYMMETRIC,
        )


class ForceFirstCSVRIntegrator(CSVRIntegrator):
    """
    Implements a force-first variant of the Canonical Sampling through Velocity
    Rescaling (CSVR) integrator, also known as the Bussi-Donadio-Parrinello thermostat.

    .. note::
        Either a system or a sequence of dynamical variables must be provided.

    Parameters
    ----------
    temperature
        The temperature.
    frictionCoeff
        The friction coefficient.
    stepSize
        The integration step size.

    Keyword Arguments
    -----------------
    physical_system
        The physical :OpenMM:`System` to be used in the XPS simulation, or `None` if
        this integrator is intended to be used for an extension system.
    dynamical_variables
        A sequence of :class:`DynamicalVariable` objects, or `None` if this integrator
        is intended to be used for a physical system.
    """

    def __init__(
        self,
        temperature: mmunit.Quantity,
        frictionCoeff: mmunit.Quantity,
        stepSize: mmunit.Quantity,
        *,
        physical_system: t.Optional[mm.System] = None,
        dynamical_variables: t.Optional[t.Sequence[DynamicalVariable]] = None,
    ) -> None:
        super().__init__(
            temperature,
            frictionCoeff,
            stepSize,
            IntegratorMixin.countDegreesOfFreedom(physical_system, dynamical_variables),
            CSVR_FORCE_FIRST,
        )
