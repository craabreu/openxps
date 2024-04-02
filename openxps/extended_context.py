"""
.. module:: openxps.extended_context
   :platform: Linux, MacOS, Windows
   :synopsis: Context for extended phase-space simulations with OpenMM.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from copy import copy

import cvpack
import openmm as mm
from cvpack.serialization import Serializable
from openmm import XmlSerializer as mmxml
from openmm import _openmm as mmswig
from openmm import unit as mmunit

from .extra_dof import ExtraDOF
from .systems import PhysicalSystem


class ExtendedSpaceIntegrator(mm.Integrator, Serializable):
    """
    An integrator for extended phase-space simulations with OpenMM.

    Parameters
    ----------
    step_size
        The size of each time step. If a float is provided, it is interpreted as
        a time in picoseconds.
    extra_dofs
        The extra degrees of freedom to extend the phase space with.
    system_integrator
        A blueprint for how to integrate the physical degrees of freedom. The extended
        space integrator will inherit all the methods and attributes of this integrator.
    extension_integrator
        A blueprint for how to integrate the extra degrees of freedom. It defaults to
        the same integrator used for the physical degrees of freedom.

    Example
    -------
    >>> import openxps as xps
    >>> from openmm import unit
    >>> phi_dv = xps.ExtraDOF(
    ...     "phi_dv",
    ...     unit.radian,
    ...     3 * unit.dalton*(unit.nanometer/unit.radian)**2,
    ...     xps.bounds.Periodic(-180, 180, unit.degree)
    ... )
    >>> integrator = xps.ExtendedSpaceIntegrator(
    ...     4 * unit.femtosecond,
    ...     [phi_dv],
    ...     mm.LangevinMiddleIntegrator(
    ...         300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond
    ...     ),
    ... )
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        stepSize: t.Union[mmunit.Quantity, float],
        extraDOFs: t.Iterable[ExtraDOF],
        systemIntegrator: mm.Integrator,
        extensionIntegrator: t.Optional[mm.Integrator] = None,
    ) -> None:
        self._extra_dofs = tuple(extraDOFs)
        self._this_integrator = copy(systemIntegrator)
        self.this = self._this_integrator.this
        self._extension_integrator = copy(extensionIntegrator or systemIntegrator)
        self.setStepSize(stepSize)
        self._context = None
        self._initialized = False

    def __getatrr__(self, name: str) -> t.Any:
        return getattr(self._this_integrator, name)

    def __getstate__(self) -> t.Dict[str, t.Any]:
        return {
            "step_size": cvpack.units.Quantity(self.getStepSize()),
            "extra_dofs": self._extra_dofs,
            "system_integrator": mmxml.serialize(self),
            "extension_integrator": mmxml.serialize(self._extension_integrator),
        }

    def __setstate__(self, state: t.Dict[str, t.Any]) -> None:
        self.__init__(
            state["step_size"],
            state["extra_dofs"],
            mmxml.deserialize(state["system_integrator"]),
            mmxml.deserialize(state["extension_integrator"]),
        )

    def getExtensionIntegrator(self) -> mm.Integrator:
        """
        Get the integrator for the extra degrees of freedom.

        Returns
        -------
        mm.Integrator
            The integrator for the extra degrees of freedom.
        """
        return self._extension_integrator

    def initialize(self, context: "ExtendedSpaceContext") -> None:
        """
        Set the extended phase-space context attached to the integrator.

        Parameters
        ----------
        context
            The extended phase-space context attached to the integrator.
        """
        self._context = context
        self._initialized = True

    def setStepSize(self, size: t.Union[mmunit.Quantity, float]) -> None:
        """
        Set the size of each time step.

        Parameters
        ----------
        size
            The size of each time step. If a float is provided, it is interpreted as
            a time in picoseconds.
        """
        self._extension_integrator.setStepSize(size / 2)
        return super().setStepSize(size)

    def step(self, steps):
        r"""
        step(self, steps)
        Advance a simulation through time by taking a series of time steps.

        Parameters
        ----------
        steps : int
            the number of time steps to take
        """
        if not self._initialized:
            raise RuntimeError("The integrator has not been initialized.")
        # pylint: disable=protected-access
        for _ in range(steps):
            self._extension_integrator.step(1)
            self._context._updateParameters()
            super().step(1)
            self._context._updateExtensionParameters()
            self._extension_integrator.step(1)
        self._context._updateParameters()
        # pylint: enable=protected-access


ExtendedSpaceIntegrator.registerTag("!openxps.ExtendedSpaceIntegrator")


class ExtendedSpaceContext(mm.Context):
    """
    A context for extended phase-space simulations with OpenMM.

    Parameters
    ----------
    context
        An OpenMM context containing the physical system to be extended.
    extra_dofs
        The extra degrees of freedom to extend the phase space with.
    extension_integrator
        A blueprint for how to integrate the extra degrees of freedom. It defaults to
        the same integrator present in the context.

    Example
    -------
    >>> import openxps as xps
    >>> from math import pi
    >>> import cvpack
    >>> import openmm
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> umbrella_potential = cvpack.MetaCollectiveVariable(
    ...     f"0.5*kappa*min(delta,{2*pi}-delta)^2; delta=abs(phi_cv-phi_dv)",
    ...     [cvpack.Torsion(6, 8, 14, 16, name="phi_cv")],
    ...     unit.kilojoule_per_mole,
    ...     kappa=1000 * unit.kilojoule_per_mole / unit.radian**2,
    ...     phi_dv=0*unit.radian,
    ... )
    >>> umbrella_potential.addToSystem(model.system)
    >>> integrator = openmm.LangevinMiddleIntegrator(
    ...     300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond
    ... )
    >>> platform = openmm.Platform.getPlatformByName("Reference")
    >>> mass = 3 * unit.dalton*(unit.nanometer/unit.radian)**2
    >>> phi_dv = xps.ExtraDOF("phi_dv", unit.radian, mass, xps.bounds.CIRCULAR)
    >>> context = xps.ExtendedSpaceContext(
    ...     openmm.Context(model.system, integrator, platform), [phi_dv]
    ... )
    >>> context.setPositions(model.positions)
    >>> context.setVelocitiesToTemperature(300 * unit.kelvin)
    >>> context.setExtraValues([180 * unit.degree])
    >>> context.setExtraVelocitiesToTemperature(300 * unit.kelvin)
    >>> context.getIntegrator().step(100)
    """

    def __init__(
        self,
        context: mm.Context,
        extraDOFs: t.Iterable[ExtraDOF],
        extensionIntegrator: t.Optional[mm.Integrator] = None,
    ) -> None:
        self._extra_dofs = tuple(extraDOFs)
        self._system = PhysicalSystem(context.getSystem(), self._extra_dofs)
        self._integrator = ExtendedSpaceIntegrator(
            context.getIntegrator().getStepSize(),
            self._extra_dofs,
            context.getIntegrator(),
            extensionIntegrator,
        )

        platform = context.getPlatform()
        properties = {
            name: platform.getPropertyValue(context, name)
            for name in platform.getPropertyNames()
        }
        super().__init__(self._system, self._integrator, platform, properties)

        extension_system = mm.System()
        for xdof in self._extra_dofs:
            index = extension_system.addParticle(
                xdof.mass.value_in_unit_system(mm.unit.md_unit_system)
            )
            force = mm.CustomExternalForce(f"{xdof.name}_force*x")
            force.addGlobalParameter(f"{xdof.name}_force", 0)
            force.addParticle(index, [])
            extension_system.addForce(force)

        self._extension = mm.Context(
            extension_system,
            self._integrator.getExtensionIntegrator(),
            mm.Platform.getPlatformByName("Reference"),
        )
        self._integrator.initialize(self)
        self._has_set_positions = False
        self._has_set_extra_dof_values = False

    def _updateParameters(self) -> None:
        state = mmswig.Context_getState(self._extension, mm.State.Positions)
        positions = mmswig.State__getVectorAsVec3(state, mm.State.Positions)
        for i, xdof in enumerate(self._extra_dofs):
            value = positions[i].x
            if xdof.bounds is not None:
                value, _ = xdof.bounds.wrap(value, 0)
            self.setParameter(xdof.name, value)

    def _updateExtensionParameters(self) -> None:
        state = mmswig.Context_getState(self, mm.State.ParameterDerivatives)
        derivatives = mmswig.State_getEnergyParameterDerivatives(state)
        for xdof in self._extra_dofs:
            self._extension.setParameter(f"{xdof.name}_force", -derivatives[xdof.name])

    def setPositions(self, positions: cvpack.units.MatrixQuantity) -> None:
        super().setPositions(positions)
        self._updateExtensionParameters()
        self._has_set_positions = True

    def setExtraValues(self, values: t.Iterable[mmunit.Quantity]) -> None:
        """
        Set the values of the extra degrees of freedom.

        Parameters
        ----------
        values
            A dictionary containing the values of the extra degrees of freedom.
        """
        if not self._has_set_positions:
            raise RuntimeError("The positions have not been set.")
        positions = []
        for xdof, value in zip(self._extra_dofs, values):
            if mmunit.is_quantity(value):
                value = value.value_in_unit(xdof.unit)
            positions.append(mm.Vec3(value, 0, 0))
            if xdof.bounds is not None:
                value, _ = xdof.bounds.wrap(value, 0)
            self.setParameter(xdof.name, value)
        self._extension.setPositions(positions)
        self._updateExtensionParameters()
        self._has_set_extra_dof_values = True

    def getExtraValues(self) -> t.Tuple[mmunit.Quantity]:
        """
        Get the values of the extra degrees of freedom.

        Returns
        -------
        t.Tuple[mmunit.Quantity]
            A tuple containing the values of the extra degrees of freedom.
        """
        if not self._has_set_extra_dof_values:
            raise RuntimeError("The extra degrees of freedom have never been set.")
        values = []
        for xdof in self._extra_dofs:
            value = self.getParameter(xdof.name)
            if xdof.bounds is not None:
                value, _ = xdof.bounds.wrap(value, 0)
            values.append(value * xdof.unit)
        return tuple(values)

    def setExtraVelocities(self, velocities: t.Iterable[mmunit.Quantity]) -> None:
        """
        Set the velocities of the extra degrees of freedom.

        Parameters
        ----------
        velocities
            A dictionary containing the velocities of the extra degrees of freedom.
        """
        if not self._has_set_extra_dof_values:
            raise RuntimeError("The extra degrees of freedom have never been set.")
        velocities = list(velocities)
        for i, xdof in enumerate(self._extra_dofs):
            value = velocities[i]
            if mmunit.is_quantity(value):
                value = value.value_in_unit(xdof.unit / mmunit.picosecond)
            velocities[i] = mm.Vec3(value, 0, 0)
        self._extension.setVelocities(velocities)

    def setExtraVelocitiesToTemperature(
        self, temperature: mmunit.Quantity, seed: t.Optional[int] = None
    ) -> None:
        """
        Set the velocities of the extra degrees of freedom to a temperature.

        Parameters
        ----------
        temperature
            The temperature to set the velocities to.
        """
        if not self._has_set_extra_dof_values:
            raise RuntimeError("The extra degrees of freedom have never been set.")
        if seed is None:
            self._extension.setVelocitiesToTemperature(temperature)
        else:
            self._extension.setVelocitiesToTemperature(temperature, seed)
        state = mmswig.Context_getState(self._extension, mm.State.Velocities)
        velocities = mmswig.State__getVectorAsVec3(state, mm.State.Velocities)
        self._extension.setVelocities([mm.Vec3(v.x, 0, 0) for v in velocities])

    def getExtraVelocities(self) -> t.Tuple[mmunit.Quantity]:
        """
        Get the velocities of the extra degrees of freedom.

        Returns
        -------
        t.Tuple[mmunit.Quantity]
            A tuple containing the velocities of the extra degrees of freedom.
        """
        if not self._has_set_extra_dof_values:
            raise RuntimeError("The extra degrees of freedom have never been set.")
        state = self._extension.getState(mm.State.Velocities)
        velocities = state.getVelocities().value_in_unit(
            mmunit.nanometer / mmunit.picosecond
        )
        for i, xdof in enumerate(self._extra_dofs):
            velocities[i] = velocities[i].x * xdof.unit / mmunit.picosecond
        return tuple(velocities)
