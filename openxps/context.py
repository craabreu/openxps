"""
.. module:: openxps.context
   :platform: Linux, MacOS, Windows
   :synopsis: Context for extended phase-space simulations with OpenMM.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from collections import defaultdict
from copy import copy
from functools import partial
from types import MethodType

import cvpack
import openmm as mm
from openmm import _openmm as mmswig
from openmm import unit as mmunit

from .extra_dof import ExtraDOF


def get_physical_system(
    context: mm.Context, extraDOFs: t.Tuple[ExtraDOF], addMissingDerivatives: bool
) -> mm.System:
    """
    Get a reference to the physical system from an extended phase-space context.

    If the system forces do not depend on all extra degrees of freedom, an error is
    raised. If missing derivative requests are found in the system forces and
    addMissingDerivatives is True, they are added automatically. Otherwise, an error is
    raised. The context is reinitialized if missing derivatives are added.

    Parameters
    ----------
    context
        The OpenMM context containing the physical system.
    extraDOFs
        The extra degrees of freedom to extend the phase space with.
    addMissingDerivatives
        Whether to add missing derivative requests in the system forces.

    Returns
    -------
    System
        The physical system.

    Raises
    ------
    ValueError
        If the system forces do not depend on all extra degrees of freedom.
    ValueError
        If missing derivative requests are found in the system forces and
        addMissingDerivatives is False.
    """
    system = context.getSystem()
    dependent_forces = defaultdict(list)
    for index, force in enumerate(system.getForces()):
        if hasattr(force, "getNumGlobalParameters"):
            for i in range(force.getNumGlobalParameters()):
                dependent_forces[force.getGlobalParameterName(i)].append(index)

    missing_parameters = [
        xdof.name for xdof in extraDOFs if xdof.name not in dependent_forces
    ]
    if missing_parameters:
        raise ValueError(
            f"No forces depend on these global parameters: {missing_parameters}."
        )

    missing_derivatives = defaultdict(list)
    for xdof in extraDOFs:
        for index in dependent_forces[xdof.name]:
            force = system.getForce(index)
            if not any(
                force.getEnergyParameterDerivativeName(i) == xdof.name
                for i in range(force.getNumEnergyParameterDerivatives())
            ):
                missing_derivatives[index].append(xdof.name)

    if missing_derivatives:
        if not addMissingDerivatives:
            raise ValueError(
                "Missing derivative requests in system forces. "
                "Set addMissingDerivatives=True to add them automatically."
            )
        for index, names in missing_derivatives.items():
            force = system.getForce(index)
            for name in names:
                force.addEnergyParameterDerivative(name)
        context.reinitialize(preserveState=True)

    return system


def update_physical_parameters(
    physicalContext: mm.Context,
    extensionContext: mm.Context,
    extraDOFs: t.Tuple[ExtraDOF],
) -> None:
    """
    Update the parameters of the context that contains the physical degrees of freedom,
    making them consistent with the values of the extra degrees of freedom.

    Parameters
    ----------
    physicalContext
        The context containing the physical degrees of freedom.
    extensionContext
        The context containing the extra degrees of freedom.
    extraDOFs
        The extra degrees of freedom to extend the phase space with.
    """
    state = mmswig.Context_getState(extensionContext, mm.State.Positions)
    positions = mmswig.State__getVectorAsVec3(state, mm.State.Positions)
    for i, xdof in enumerate(extraDOFs):
        value = positions[i].x
        if xdof.bounds is not None:
            value, _ = xdof.bounds.wrap(value, 0)
        mmswig.Context_setParameter(physicalContext, xdof.name, value)


def update_extension_parameters(
    physicalContext: mm.Context,
    extensionContext: mm.Context,
    extraDOFs: t.Tuple[ExtraDOF],
) -> None:
    """
    Update the parameters of the context containing the extra degrees of freedom,
    making them consistent with the values of the physical degrees of freedom.

    Parameters
    ----------
    physicalContext
        The context containing the physical degrees of freedom.
    extensionContext
        The context containing the extra degrees of freedom.
    extraDOFs
        The extra degrees of freedom to extend the phase space with.
    """
    state = mmswig.Context_getState(physicalContext, mm.State.ParameterDerivatives)
    derivatives = mmswig.State_getEnergyParameterDerivatives(state)
    for xdof in extraDOFs:
        mmswig.Context_setParameter(
            extensionContext,
            f"{xdof.name}_force",
            -derivatives[xdof.name],
        )


def integrate_extended_space(
    physicalContext: mm.Context,
    steps: int,
    extraDOFs: t.Tuple[ExtraDOF],
    extensionContext: mm.Context,
) -> None:
    """
    Perform a series of time steps in an extended phase-space simulation.

    Parameters
    ----------
    physicalContext
        The context containing the physical degrees of freedom.
    steps
        The number of time steps to take.
    extraDOFs
        The extra degrees of freedom to extend the phase space with.
    extensionContext
        The context containing the extra degrees of freedom.
    """
    extension_integrator = extensionContext.getIntegrator()
    physical_integrator = physicalContext.getIntegrator()

    try:
        mmswig.Integrator_step(extension_integrator, 1)
        update_physical_parameters(physicalContext, extensionContext, extraDOFs)
    except mm.OpenMMException as error:
        raise RuntimeError("Extra degrees of freedom have not been set.") from error
    for _ in range(steps - 1):
        mmswig.Integrator_step(physical_integrator, 1)
        update_extension_parameters(physicalContext, extensionContext, extraDOFs)
        mmswig.Integrator_step(extension_integrator, 2)
        update_physical_parameters(physicalContext, extensionContext, extraDOFs)
    mmswig.Integrator_step(physical_integrator, 1)
    update_extension_parameters(physicalContext, extensionContext, extraDOFs)
    mmswig.Integrator_step(extension_integrator, 1)
    update_physical_parameters(physicalContext, extensionContext, extraDOFs)


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
    ...     phi_dv=pi*unit.radian,
    ... )
    >>> umbrella_potential.addToSystem(model.system)
    >>> integrator = openmm.LangevinMiddleIntegrator(
    ...     300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond
    ... )
    >>> platform = openmm.Platform.getPlatformByName("Reference")
    >>> mass = 3 * unit.dalton*(unit.nanometer/unit.radian)**2
    >>> phi_dv = xps.ExtraDOF("phi_dv", unit.radian, mass, xps.bounds.CIRCULAR)
    >>> context = xps.ExtendedSpaceContext(
    ...     openmm.Context(model.system, integrator, platform),
    ...     [phi_dv],
    ...     addMissingDerivatives=True,
    ... )
    >>> context.setPositions(model.positions)
    >>> context.setVelocitiesToTemperature(300 * unit.kelvin)
    >>> context.setExtraValues([180 * unit.degree])
    >>> context.setExtraVelocitiesToTemperature(300 * unit.kelvin)
    >>> context.getIntegrator().step(100)
    >>> context.getExtraValues()
    (Quantity(value=..., unit=radian),)
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        context: mm.Context,
        extraDOFs: t.Iterable[ExtraDOF],
        extensionIntegrator: t.Optional[mm.Integrator] = None,
        *,
        addMissingDerivatives: bool = False,
    ) -> None:
        self._extra_dofs = tuple(extraDOFs)
        self._system = get_physical_system(
            context, self._extra_dofs, addMissingDerivatives
        )
        self._integrator = context.getIntegrator()
        self.this = context.this
        extension_integrator = extensionIntegrator or copy(self._integrator)
        extension_system = mm.System()
        for xdof in self._extra_dofs:
            index = extension_system.addParticle(
                xdof.mass.value_in_unit_system(mm.unit.md_unit_system)
            )
            force = mm.CustomExternalForce(f"{xdof.name}_force*x")
            force.addGlobalParameter(f"{xdof.name}_force", 0)
            force.addParticle(index, [])
            extension_system.addForce(force)

        self._extension_context = mm.Context(
            extension_system,
            extension_integrator,
            mm.Platform.getPlatformByName("Reference"),
        )

        self._integrator.step = MethodType(
            partial(
                integrate_extended_space,
                extraDOFs=self._extra_dofs,
                extensionContext=self._extension_context,
            ),
            self,
        )

    def setPositions(self, positions: cvpack.units.MatrixQuantity) -> None:
        super().setPositions(positions)
        update_extension_parameters(self, self._extension_context, self._extra_dofs)

    def setExtraValues(self, values: t.Iterable[mmunit.Quantity]) -> None:
        """
        Set the values of the extra degrees of freedom.

        Parameters
        ----------
        values
            A dictionary containing the values of the extra degrees of freedom.
        """
        positions = []
        for xdof, value in zip(self._extra_dofs, values):
            if mmunit.is_quantity(value):
                value = value.value_in_unit(xdof.unit)
            positions.append(mm.Vec3(value, 0, 0))
            if xdof.bounds is not None:
                value, _ = xdof.bounds.wrap(value, 0)
            self.setParameter(xdof.name, value)
        mmswig.Context_setPositions(self._extension_context, positions)
        update_extension_parameters(self, self._extension_context, self._extra_dofs)

    def getExtraValues(self) -> t.Tuple[mmunit.Quantity]:
        """
        Get the values of the extra degrees of freedom.

        Returns
        -------
        t.Tuple[mmunit.Quantity]
            A tuple containing the values of the extra degrees of freedom.
        """
        state = mmswig.Context_getState(self._extension_context, mm.State.Positions)
        positions = mmswig.State__getVectorAsVec3(state, mm.State.Positions)
        for i, xdof in enumerate(self._extra_dofs):
            value = positions[i].x
            if xdof.bounds is not None:
                value, _ = xdof.bounds.wrap(value, 0)
            positions[i] = value * xdof.unit
        return tuple(positions)

    def setExtraVelocities(self, velocities: t.Iterable[mmunit.Quantity]) -> None:
        """
        Set the velocities of the extra degrees of freedom.

        Parameters
        ----------
        velocities
            A dictionary containing the velocities of the extra degrees of freedom.
        """
        velocities = list(velocities)
        for i, xdof in enumerate(self._extra_dofs):
            value = velocities[i]
            if mmunit.is_quantity(value):
                value = value.value_in_unit(xdof.unit / mmunit.picosecond)
            velocities[i] = mm.Vec3(value, 0, 0)
        mmswig.Context_setVelocities(self._extension_context, velocities)

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
        args = (temperature,) if seed is None else (temperature, seed)
        try:
            self._extension_context.setVelocitiesToTemperature(*args)
        except mm.OpenMMException as error:
            raise RuntimeError("Extra degrees of freedom have not been set.") from error
        state = mmswig.Context_getState(self._extension_context, mm.State.Velocities)
        velocities = mmswig.State__getVectorAsVec3(state, mm.State.Velocities)
        mmswig.Context_setVelocities(
            self._extension_context, [mm.Vec3(v.x, 0, 0) for v in velocities]
        )

    def getExtraVelocities(self) -> t.Tuple[mmunit.Quantity]:
        """
        Get the velocities of the extra degrees of freedom.

        Returns
        -------
        t.Tuple[mmunit.Quantity]
            A tuple containing the velocities of the extra degrees of freedom.
        """
        state = mmswig.Context_getState(
            self._extension_context, mm.State.Positions | mm.State.Velocities
        )
        positions = mmswig.State__getVectorAsVec3(state, mm.State.Positions)
        velocities = mmswig.State__getVectorAsVec3(state, mm.State.Velocities)
        for i, xdof in enumerate(self._extra_dofs):
            value = positions[i].x
            rate = velocities[i].x
            if xdof.bounds is not None:
                value, rate = xdof.bounds.wrap(value, rate)
            velocities[i] = rate * xdof.unit / mmunit.picosecond
        return tuple(velocities)
