"""
.. module:: openxps.integrators.mixins
   :platform: Linux, MacOS, Windows
   :synopsis: A mixin for integrators that provides extra functionality.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import numpy as np
import openmm as mm
from openmm import unit as mmunit

BLOCK_START = (6, 7)
BLOCK_END = 8


class IntegratorMixin:
    """A mixin for integrators that provides extra functionality."""

    _forceFirst: bool = False

    def __repr__(self) -> str:
        """Return a human-readable version of each integrator step."""
        readable_lines = []

        self.getNumPerDofVariables() > 0 and readable_lines.append("Per-dof variables:")
        per_dof = []
        for index in range(self.getNumPerDofVariables()):
            per_dof.append(self.getPerDofVariableName(index))
        readable_lines.append("  " + ", ".join(per_dof))

        self.getNumGlobalVariables() > 0 and readable_lines.append("Global variables:")
        for index in range(self.getNumGlobalVariables()):
            name = self.getGlobalVariableName(index)
            value = self.getGlobalVariable(index)
            readable_lines.append(f"  {name} = {value}")

        readable_lines.append("Computation steps:")

        step_type_str = [
            "{target} <- {expr}",
            "{target} <- {expr}",
            "{target} <- sum({expr})",
            "constrain positions",
            "constrain velocities",
            "allow forces to update the context state",
            "if ({expr}):",
            "while ({expr}):",
            "end",
        ]
        indent_level = 0
        for step in range(self.getNumComputations()):
            line = ""
            step_type, target, expr = self.getComputationStep(step)
            command = step_type_str[step_type].format(target=target, expr=expr)
            if step_type == BLOCK_END:
                indent_level -= 1
            line += f"{step:4d}: " + "   " * indent_level + command
            if step_type in BLOCK_START:
                indent_level += 1
            readable_lines.append(line)
        return "\n".join(readable_lines)

    def registerWithSystem(self, system: mm.System) -> None:
        """Register the integrator with the system."""
        pass

    @staticmethod
    def _countDegreesOfFreedom(system: mm.System) -> int:
        """Count the degrees of freedom in a system.

        Parameters
        ----------
        system
            The :OpenMM:`System` to count the degrees of freedom of.

        Returns
        -------
        int
            The number of degrees of freedom in the system.
        """
        dof = 0
        for i in range(system.getNumParticles()):
            if system.getParticleMass(i) > 0 * mmunit.dalton:
                dof += 3
        for i in range(system.getNumConstraints()):
            p1, p2, _ = system.getConstraintParameters(i)
            if (system.getParticleMass(p1) > 0 * mmunit.dalton) or (
                system.getParticleMass(p2) > 0 * mmunit.dalton
            ):
                dof -= 1
        if any(
            isinstance(system.getForce(i), mm.CMMotionRemover)
            for i in range(system.getNumForces())
        ):
            dof -= 3
        return dof

    def isForceFirst(self) -> bool:
        """Check if the integrator follows a force-first scheme.

        Returns
        -------
        bool
            True if the integrator is force-first, False otherwise.
        """
        return self._forceFirst


class TemperatureMixin:
    """A mixin for integrators that provide temperature functionality."""

    _temperature: mmunit.Quantity
    _kT: mmunit.Quantity

    def _handle_temperature(
        self, temperature: mmunit.Quantity, addOrSet: t.Callable
    ) -> None:
        """Handle the temperature-related variables."""
        self._temperature = temperature
        self._kT = mmunit.MOLAR_GAS_CONSTANT_R * temperature
        addOrSet("kT", self._kT)

    def _add_temperature(self, temperature: mmunit.Quantity) -> None:
        """Add the temperature variable to the integrator."""
        self._handle_temperature(temperature, self.addGlobalVariable)

    def getTemperature(self) -> mmunit.Quantity:
        """Get the temperature.

        Returns
        -------
        mmunit.Quantity
            The temperature.
        """
        return self._temperature

    def setTemperature(self, temperature: mmunit.Quantity) -> None:
        """Set the temperature.

        Parameters
        ----------
        temperature
            The temperature.
        """
        self._handle_temperature(temperature, self.setGlobalVariable)


class FrictionCoefficientMixin:
    """A mixin for integrators that provide friction coefficient functionality."""

    _frictionCoeff: mmunit.Quantity

    def _handle_frictionCoeff(
        self, frictionCoeff: mmunit.Quantity, addOrSet: t.Callable
    ) -> None:
        """Handle the friction coefficient-related variables."""
        self._frictionCoeff = frictionCoeff
        addOrSet("a", np.exp(-frictionCoeff * self.getStepSize()))
        addOrSet("b", np.sqrt(1 - np.exp(-2 * frictionCoeff * self.getStepSize())))

    def _add_frictionCoeff(self, frictionCoeff: mmunit.Quantity) -> None:
        """Add the friction coefficient variable to the integrator.

        Parameters
        ----------
        frictionCoeff
            The friction coefficient.
        """
        self._handle_frictionCoeff(frictionCoeff, self.addGlobalVariable)

    def setFrictionCoeff(self, frictionCoeff: mmunit.Quantity) -> None:
        """Set the friction coefficient.

        Parameters
        ----------
        frictionCoeff
            The friction coefficient.
        """
        self._handle_frictionCoeff(frictionCoeff, self.setGlobalVariable)

    def getFrictionCoeff(self) -> mmunit.Quantity:
        """Get the friction coefficient.

        Returns
        -------
        mmunit.Quantity
            The friction coefficient.
        """
        return self._frictionCoeff


class TimeConstantMixin:
    """A mixin for integrators that provide time constant functionality."""

    _timeConstant: mmunit.Quantity
    _invQ: mmunit.Quantity

    def _handle_timeConstant(
        self, timeConstant: mmunit.Quantity, addOrSet: t.Callable
    ) -> None:
        """Handle the time constant-related variables."""
        self._timeConstant = timeConstant
        self._invQ = 1 / (mmunit.MOLAR_GAS_CONSTANT_R * timeConstant**2)
        addOrSet("invQ", self._invQ)

    def _add_timeConstant(self, timeConstant: mmunit.Quantity) -> None:
        """Add the time constant variable to the integrator.

        Parameters
        ----------
        timeConstant
            The time constant.

        """
        self._handle_timeConstant(timeConstant, self.addGlobalVariable)

    def getTimeConstant(self) -> mmunit.Quantity:
        """Get the time constant.

        Returns
        -------
        mmunit.Quantity
            The time constant.
        """
        return self._timeConstant

    def setTimeConstant(self, timeConstant: mmunit.Quantity) -> None:
        """Set the time constant.

        Parameters
        ----------
        timeConstant
            The time constant.
        """
        self._timeConstant = timeConstant
        self._handle_timeConstant(timeConstant, self.setGlobalVariable)
