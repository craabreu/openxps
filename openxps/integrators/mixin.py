"""
.. module:: openxps.integrators.mixin
   :platform: Linux, MacOS, Windows
   :synopsis: A mixin for integrators that provides extra functionality.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

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

    def register_with_system(self, system: mm.System) -> None:
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
