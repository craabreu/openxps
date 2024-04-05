"""
.. module:: openxps.context
   :platform: Linux, MacOS, Windows
   :synopsis: Context for extended phase-space simulations with OpenMM.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from copy import copy
from functools import partial
from types import MethodType

import cvpack
import openmm as mm
from openmm import _openmm as mmswig
from openmm import unit as mmunit

from .bounds import Periodic
from .extra_dof import ExtraDOF


class ExtendedSpaceContext(mm.Context):
    """
    Wraps an :OpenMM:`Context` object to include extra degrees of freedom (DOFs) and
    allow for extended phase-space (XPS) simulations.

    The system and integrator attached to the context are modified in-place.

    A provided :CVPack:`MetaCollectiveVariable` is added to the system to couple the
    physical and extra DOFs. The integrator's ``step`` method is replaced with a custom
    function that advances the physical and extra DOFs alternately using the Strang
    splitting algorithm :cite:`Strang_1968`. This is identical to the disparate-mass
    version of the Reversible Reference System Propagator Algorithm (rRESPA) method
    :cite:`Tuckerman_1992`.

    Parameters
    ----------
    context
        The original OpenMM context containing the physical system.
    extra_dofs
        A group of extra degrees of freedom to be included in the XPS simulation.
    coupling_potential
        A meta-collective variable, with units of ``kilojoule_per_mole`` or equivalent,
        which defines the potential energy term that couples the physical system to the
        extra DOFs.
    extension_integrator
        An integrator for the extra degrees of freedom. If not provided, the original
        context's integrator is used as a template.

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
    >>> integrator = openmm.LangevinMiddleIntegrator(
    ...     300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond
    ... )
    >>> platform = openmm.Platform.getPlatformByName("Reference")
    >>> mass = 3 * unit.dalton*(unit.nanometer/unit.radian)**2
    >>> phi_dv = xps.ExtraDOF("phi_dv", unit.radian, mass, xps.bounds.CIRCULAR)
    >>> context = xps.ExtendedSpaceContext(
    ...     openmm.Context(model.system, integrator, platform),
    ...     [phi_dv],
    ...     umbrella_potential,
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
        extra_dofs: t.Iterable[ExtraDOF],
        coupling_potential: cvpack.MetaCollectiveVariable,
        extension_integrator: t.Optional[mm.Integrator] = None,
    ) -> None:
        self._extra_dofs = tuple(extra_dofs)
        self.this = context.this
        self._system = context.getSystem()
        self._integrator = context.getIntegrator()
        self._coupling_potential = coupling_potential
        extension_integrator = extension_integrator or copy(self._integrator)

        self._validate()
        self._coupling_potential.addToSystem(self._system)
        self.reinitialize(preserveState=True)

        extension_system = mm.System()
        for xdof in self._extra_dofs:
            extension_system.addParticle(
                xdof.mass.value_in_unit_system(mm.unit.md_unit_system)
            )
        flipped_potential = self._flipMetaCV(coupling_potential, self._extra_dofs)
        flipped_potential.addToSystem(extension_system)

        self._extension_context = mm.Context(
            extension_system,
            extension_integrator,
            mm.Platform.getPlatformByName("Reference"),
        )

        self._integrator.step = MethodType(
            partial(
                integrate_extended_space,
                extra_dofs=self._extra_dofs,
                extension_context=self._extension_context,
                coupling_potential=coupling_potential,
            ),
            self,
        )

    def _validate(self) -> None:
        if not all(isinstance(xdof, ExtraDOF) for xdof in self._extra_dofs):
            raise TypeError(
                "All extra degrees of freedom must be instances of ExtraDOF."
            )
        if not isinstance(self._coupling_potential, cvpack.MetaCollectiveVariable):
            raise TypeError(
                "The coupling potential must be an instance of MetaCollectiveVariable."
            )
        if not self._coupling_potential.getUnit().is_compatible(
            mmunit.kilojoule_per_mole
        ):
            raise ValueError("The coupling potential must have units of energy/mole.")
        context_parameters = set(self.getParameters())
        force_parameters = self._coupling_potential.getParameterDefaultValues()
        parameter_units = {
            name: quantity.unit for name, quantity in force_parameters.items()
        }
        if parameters := set(parameter_units) & context_parameters:
            raise ValueError(
                f"The context already contains {parameters} among its parameters."
            )
        xdof_units = {xdof.name: xdof.unit for xdof in self._extra_dofs}
        if parameters := set(xdof_units) - set(parameter_units):
            raise ValueError(
                f"The coupling potential parameters do not include {parameters}."
            )
        for name, unit in xdof_units.items():
            if not unit.is_compatible(parameter_units[name]):
                raise ValueError(f"Unit mismatch for parameter '{name}'.")

    def _flipMetaCV(
        self,
        meta_cv: cvpack.MetaCollectiveVariable,
        extra_dofs: t.Tuple[ExtraDOF],
    ) -> cvpack.MetaCollectiveVariable:
        """
        Flip a meta-collective variable, turning all inner collective variables into
        named parameters and all extra-dof-related parameters into inner collective
        variables.

        Parameters
        ----------
        meta_cv
            The meta-collective variable to be flipped.
        extra_dofs
            The extra degrees of freedom to be included in the extended phase space.

        Returns
        -------
        cvpack.MetaCollectiveVariable
            The flipped meta-collective variable.
        """
        extra_dof_cvs = []
        for index, xdof in enumerate(extra_dofs):
            if isinstance(xdof.bounds, Periodic):
                bounds = [xdof.bounds.lower, xdof.bounds.upper] * xdof.unit
            else:
                bounds = None
            force = mm.CustomExternalForce("x")
            force.addParticle(index, [])
            cv = cvpack.OpenMMForceWrapper(force, xdof.unit, bounds, xdof.name)
            extra_dof_cvs.append(cv)

        parameters = meta_cv.getParameterDefaultValues()
        for xdof in extra_dofs:
            parameters.pop(xdof.name)
        parameters.update(meta_cv.getInnerValues(self))

        return cvpack.MetaCollectiveVariable(
            function=meta_cv.getEnergyFunction(),
            variables=extra_dof_cvs,
            unit=meta_cv.getUnit(),
            periodicBounds=meta_cv.getPeriodicBounds(),
            name=meta_cv.getName(),
            **parameters,
        )

    def setPositions(self, positions: cvpack.units.MatrixQuantity) -> None:
        """
        Sets the positions of all particles in the physical system.

        This method extends the base ``setPositions`` method of OpenMM's Context class
        to ensure that the extended degrees of freedom are also updated accordingly.

        Parameters
        ----------
        positions
            The positions for each particle in the system.
        """
        super().setPositions(positions)
        update_extension_context(
            self, self._extension_context, self._coupling_potential
        )

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
        update_extension_context(
            self, self._extension_context, self._coupling_potential
        )

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


def update_physical_context(
    physical_context: mm.Context,
    extension_context: mm.Context,
    extra_dofs: t.Tuple[ExtraDOF],
) -> None:
    """
    Update the parameters of the context that contains the physical degrees of freedom,
    making them consistent with the values of the extra degrees of freedom.

    Parameters
    ----------
    physical_context
        The context containing the physical degrees of freedom.
    extension_context
        The context containing the extra degrees of freedom.
    extra_dofs
        The extra degrees of freedom to extend the phase space with.
    """
    state = mmswig.Context_getState(extension_context, mm.State.Positions)
    positions = mmswig.State__getVectorAsVec3(state, mm.State.Positions)
    for i, xdof in enumerate(extra_dofs):
        value = positions[i].x
        if xdof.bounds is not None:
            value, _ = xdof.bounds.wrap(value, 0)
        mmswig.Context_setParameter(physical_context, xdof.name, value)


def update_extension_context(
    physical_context: mm.Context,
    extension_context: mm.Context,
    coupling_potential: cvpack.MetaCollectiveVariable,
) -> None:
    """
    Update the parameters of the context containing the extra degrees of freedom,
    making them consistent with the current state of the physical system.

    Parameters
    ----------
    physical_context
        The context containing the physical degrees of freedom.
    extension_context
        The context containing the extra degrees of freedom.
    coupling_potential
        The potential that couples the physical and extra degrees of freedom.
    """
    for name, value in coupling_potential.getInnerValues(physical_context).items():
        mmswig.Context_setParameter(extension_context, name, value / value.unit)


def integrate_extended_space(
    physical_context: mm.Context,
    steps: int,
    extra_dofs: t.Tuple[ExtraDOF],
    extension_context: mm.Context,
    coupling_potential: cvpack.MetaCollectiveVariable,
) -> None:
    """
    Advances the extended phase-space simulation by integrating both the physical
    system and the extra degrees of freedom (DOFs) over a specified number of time
    steps.

    This function orchestrates the simulation process by alternating between advancing
    the physical system and updating the extra DOFs in accordance with the Reversible
    Reference System Propagator Algorithm (rRESPA) method :cite:`Tuckerman_1992`.
    This ensures that changes in the extra DOFs are reflected in the physical system
    and vice versa, maintaining consistency across the extended phase-space.

    Parameters
    ----------
    physical_context
        The OpenMM context representing the physical system. This context is advanced
        according to its associated integrator.
    steps
        The number of time steps to advance the simulation. Each step involves updating
        both the physical system and the extra DOFs.
    extra_dofs
        A tuple containing the extra degrees of freedom to be included in the
        simulation. These DOFs are integrated separately from the physical system
        but are synchronized at each step to ensure consistent simulation conditions.
    extension_context : mm.Context
        The OpenMM context for the extra DOFs. This context is advanced separately from
        the physical context but is synchronized with it to reflect the mutual influence
        between the physical system and the extra DOFs.
    coupling_potential
        The potential that couples the physical and extra degrees of freedom.
    flipped_potential
        The flipped version of the coupling potential.

    Raises
    ------
    RuntimeError
        If the particle positions or extra degrees of freedom have not been properly
        initialized in the context.
    """

    extension_integrator = extension_context.getIntegrator()
    physical_integrator = physical_context.getIntegrator()

    try:
        mmswig.Integrator_step(extension_integrator, 1)
        update_physical_context(physical_context, extension_context, extra_dofs)
    except mm.OpenMMException as error:
        raise RuntimeError("Extra degrees of freedom have not been set.") from error
    for _ in range(steps - 1):
        mmswig.Integrator_step(physical_integrator, 1)
        update_extension_context(
            physical_context, extension_context, coupling_potential
        )
        mmswig.Integrator_step(extension_integrator, 2)
        update_physical_context(physical_context, extension_context, extra_dofs)
    mmswig.Integrator_step(physical_integrator, 1)
    update_extension_context(physical_context, extension_context, coupling_potential)
    mmswig.Integrator_step(extension_integrator, 1)
    update_physical_context(physical_context, extension_context, extra_dofs)
