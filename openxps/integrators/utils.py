"""
.. module:: openxps.integrators.utils
   :platform: Linux, MacOS, Windows
   :synopsis: A mixin for integrators that provides extra functionality.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import textwrap
import typing as t

import openmm as mm
from cvpack.units import Quantity
from openmm import unit as mmunit

from openxps.utils import preprocess_args

BLOCK_START = (6, 7)
BLOCK_END = 8

T = t.TypeVar("T")


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


def add_property(property: str) -> t.Callable[[type[T]], type[T]]:
    camel_case_name = "".join(s.capitalize() for s in property.split())
    snake_case_name = "_".join(s.lower() for s in property.split())

    @preprocess_args
    def setter(self, value: mmunit.Quantity) -> None:
        setattr(self, f"_{snake_case_name}", Quantity(value))
        if hasattr(self, "_update_global_variables"):
            self._update_global_variables()

    setter.__doc__ = textwrap.dedent(
        f"""\
        Set the {property}.

        Parameters
        ----------
        value
            The {property}.
        """
    )

    def getter(self) -> mmunit.Quantity:
        return getattr(self, f"_{snake_case_name}")

    getter.__doc__ = textwrap.dedent(
        f"""\
        Get the {property}.

        Returns
        -------
        openmm.unit.Quantity
            The {property}.
        """
    )

    def decorator(cls: type[T]) -> type[T]:
        setattr(cls, f"_{snake_case_name}", None)
        setattr(cls, f"set{camel_case_name}", setter)
        setattr(cls, f"get{camel_case_name}", getter)
        return cls

    return decorator
