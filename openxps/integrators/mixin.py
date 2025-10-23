"""
.. module:: openxps.integrators.mixin
   :platform: Linux, MacOS, Windows
   :synopsis: A mixin for integrators that provides extra functionality.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

BLOCK_START = (6, 7)
BLOCK_END = 8


class IntegratorMixin:
    """A mixin for integrators that provides extra functionality."""

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
