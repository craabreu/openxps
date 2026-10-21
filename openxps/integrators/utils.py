"""
.. module:: openxps.integrators.utils
   :platform: Linux, MacOS, Windows
   :synopsis: A mixin for integrators that provides extra functionality.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import textwrap
import typing as t
import xml.etree.ElementTree as ET
from copy import deepcopy

import openmm as mm
from cvpack.units import Quantity
from openmm import unit as mmunit

from openxps.utils import preprocess_args

BLOCK_START = (6, 7)
BLOCK_END = 8

#: Prefix of the global variables that hold captured energy derivatives.
CAPTURE_PREFIX = "xpsEnergyDerivative"
CAPTURE_FLAG = "xpsCaptureEnergyDerivatives"

T = t.TypeVar("T")


class IntegratorMixin:
    """A mixin for integrators that provides extra functionality."""

    _forceFirst: bool = False
    _energy_derivatives: tuple[str, ...] | None = None

    def __repr__(self) -> str:
        """Return a human-readable version of each integrator step."""
        readable_lines = []

        if self.getNumPerDofVariables() > 0:
            readable_lines.append("Per-dof variables:")
            per_dof = [
                self.getPerDofVariableName(index)
                for index in range(self.getNumPerDofVariables())
            ]
            readable_lines.append("  " + ", ".join(per_dof))

        if self.getNumGlobalVariables() > 0:
            readable_lines.append("Global variables:")
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

    def setStepSize(self, size: mmunit.Quantity | float) -> None:
        """Set the step size and refresh any step-size-dependent global variables."""
        super().setStepSize(size)
        if hasattr(self, "_update_global_variables"):
            self._update_global_variables()

    def registerWithSystem(self, system: mm.System, isExtension: bool) -> None:
        """Register the integrator with the system."""
        pass

    def _add_dof_mask(self) -> None:
        """Add the per-DOF variable that gates thermostat updates.

        It is 1 for every degree of freedom unless :meth:`_set_dof_mask` restricts it.
        """
        self.addPerDofVariable("active", 1)

    @staticmethod
    def _masked(expression: str) -> str:
        """Multiply the main part of a Lepton expression by the DOF mask."""
        head, separator, definitions = expression.partition(";")
        return f"active*({head}){separator}{definitions}"

    def _set_dof_mask(self, system: mm.System, isExtension: bool) -> None:
        """Restrict thermostat updates to the x components of an extension system.

        Each extension particle represents one dynamical variable along x, while its
        y and z components are inert and have zero velocity. Thermostatting them
        would make their thermostat variables drift without bound.
        """
        mask = mm.Vec3(1, 0, 0) if isExtension else mm.Vec3(1, 1, 1)
        self.setPerDofVariableByName("active", [mask] * system.getNumParticles())

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

    def getEnergyDerivativeNames(self) -> tuple[str, ...]:
        """Get the parameters whose energy derivatives each step computes.

        Returns
        -------
        tuple[str, ...]
            The names given as ``energyDerivatives`` at construction.
        """
        if self._energy_derivatives is None:
            # Lost if OpenMM unpickled the integrator, which keeps only its XML
            self._energy_derivatives = self._readEnergyDerivatives()
        return self._energy_derivatives

    def _readEnergyDerivatives(self) -> tuple[str, ...]:
        """Recover the captured parameters from the computation steps."""
        prefix = "deriv(energy, "
        names = {}
        for index in range(self.getNumComputations()):
            kind, variable, expression = self.getComputationStep(index)
            if kind == mm.CustomIntegrator.ComputeGlobal and variable.startswith(
                CAPTURE_PREFIX
            ):
                names[int(variable[len(CAPTURE_PREFIX) :])] = expression[
                    len(prefix) : -1
                ]
        return tuple(names[k] for k in sorted(names))

    def _withoutEnergyDerivatives(self) -> "IntegratorMixin":
        """Return a copy of the integrator that computes no energy derivatives.

        Computation steps cannot be removed from a :OpenMM:`CustomIntegrator`, so the
        copy is rebuilt from its XML serialization without the capture steps.
        """
        root = ET.fromstring(mm.XmlSerializer.serialize(self))
        variables = root.find("GlobalVariables")
        for element in list(variables):
            name = element.get("name", "")
            if name.startswith(CAPTURE_PREFIX) or name == CAPTURE_FLAG:
                variables.remove(element)
        computations = root.find("Computations")
        in_block = False
        for element in list(computations):
            if element.get("computationExpression", "").startswith(CAPTURE_FLAG):
                in_block = True  # the if block that holds the capture steps
            elif element.get("computationVariable", "").startswith(CAPTURE_PREFIX):
                pass  # a capture step
            elif in_block and element.get("computationType") == str(
                mm.CustomIntegrator.BlockEnd
            ):
                in_block = False
            else:
                continue
            computations.remove(element)
        copy = self.__class__.__new__(self.__class__)
        copy.this = mm.XmlSerializer.deserialize(
            ET.tostring(root, encoding="unicode")
        ).this
        attributes = {key: value for key, value in vars(self).items() if key != "this"}
        copy.__dict__.update(deepcopy(attributes))
        copy._energy_derivatives = ()
        return copy

    def getCapturePoint(self) -> str:
        """Get where in a step the energy derivatives are computed.

        They are computed in the force evaluation the integrator already does: at
        the start of the step for force-first integrators, at the end otherwise.

        Returns
        -------
        str
            ``"start"`` or ``"end"``.
        """
        return "start" if self._forceFirst else "end"

    def getCapturedDerivatives(self) -> dict[str, float]:
        """Get the energy derivatives computed in the last step.

        Returns
        -------
        dict[str, float]
            The derivatives, in MD units, keyed by parameter name.
        """
        return {
            name: self.getGlobalVariableByName(f"{CAPTURE_PREFIX}{index}")
            for index, name in enumerate(self.getEnergyDerivativeNames())
        }

    def isCaptureEnabled(self) -> bool:
        """Whether the steps compute the energy derivatives.

        Returns
        -------
        bool
            False if the integrator was built without ``energyDerivatives`` or
            :meth:`setCaptureEnabled` turned the computation off.
        """
        return bool(self.getEnergyDerivativeNames()) and (
            self.getGlobalVariableByName(CAPTURE_FLAG) != 0
        )

    def setCaptureEnabled(self, enabled: bool) -> None:
        """Turn the computation of the energy derivatives on or off.

        While it is off, the steps skip the derivative computations, so that OpenMM
        evaluates the forces alone, and :meth:`getCapturedDerivatives` keeps returning
        the last values computed. :class:`~openxps.SplitIntegrator` uses this to
        compute the derivatives in the last of the substeps of a half step only.

        Parameters
        ----------
        enabled
            Whether the steps compute the energy derivatives.

        Raises
        ------
        ValueError
            If the integrator was built without ``energyDerivatives``.
        """
        if not self.getEnergyDerivativeNames():
            raise ValueError("The integrator was built without energyDerivatives.")
        self.setGlobalVariableByName(CAPTURE_FLAG, 1.0 if enabled else 0.0)

    def _init_energy_derivatives(self, names: t.Iterable[str]) -> None:
        self._energy_derivatives = tuple(names)

    def _add_derivative_capture(self, point: str) -> None:
        """Add the derivative computations if ``point`` is the capture point.

        Each integrator calls this right after ``addUpdateContextState`` with
        ``"start"`` and, with ``"end"``, after its last position update and position
        constraint and before its last use of ``f``, so that OpenMM computes the
        energy and the forces in a single evaluation.

        The computations sit in an if block on a global variable that
        :meth:`setCaptureEnabled` controls. When the block is skipped, OpenMM does not
        compute the energy.
        """
        if point != self.getCapturePoint() or not self._energy_derivatives:
            return
        self.addGlobalVariable(CAPTURE_FLAG, 1)
        self.beginIfBlock(f"{CAPTURE_FLAG} > 0.5")
        for index, name in enumerate(self._energy_derivatives):
            variable = f"{CAPTURE_PREFIX}{index}"
            self.addGlobalVariable(variable, 0)
            self.addComputeGlobal(variable, f"deriv(energy, {name})")
        self.endBlock()


def add_property(property: str, unit: mmunit.Unit) -> t.Callable[[type[T]], type[T]]:
    """A decorator to add a property to the integrator.

    The property is added to the integrator as a class attribute with the name
    `_{snake_case_name}`. Three methods are added to the class:

    * `_init_{snake_case_name}(value)`: Initializes the property.
    * `set{PascalCaseName}(value)`: Sets the property and updates the global variables
        if the integrator has a `_update_global_variables` method. A setter defined
        in the class body takes precedence and is left untouched.
    * `get{PascalCaseName}()`: Gets the property.

    Parameters
    ----------
    property
        The name of the property.
    unit
        The unit of the property.
    """
    if unit.in_unit_system(mmunit.md_unit_system) != unit:
        raise ValueError(f"The unit of {property} must be in the MD unit system.")
    PascalCaseName = "".join(s.capitalize() for s in property.split())
    snake_case_name = "_".join(s.lower() for s in property.split())

    def set_value(self, value: mmunit.Quantity | float) -> None:
        args = [value] if mmunit.is_quantity(value) else [value, unit]
        setattr(self, f"_{snake_case_name}", Quantity(*args))

    @preprocess_args
    def set_and_update(self, value: mmunit.Quantity | float) -> None:
        set_value(self, value)
        if hasattr(self, "_update_global_variables"):
            self._update_global_variables()

    set_and_update.__doc__ = textwrap.dedent(
        f"""\
        Set the {property}.

        Parameters
        ----------
        value
            The {property}.
        """
    )

    def get_value(self) -> mmunit.Quantity:
        return getattr(self, f"_{snake_case_name}")

    get_value.__doc__ = textwrap.dedent(
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
        setattr(cls, f"_init_{snake_case_name}", set_value)
        if f"set{PascalCaseName}" not in vars(cls):
            setattr(cls, f"set{PascalCaseName}", set_and_update)
        setattr(cls, f"get{PascalCaseName}", get_value)
        return cls

    return decorator
