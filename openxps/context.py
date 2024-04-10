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

    **Note**: The system and integrator attached to the context are modified in-place.

    A provided :CVPack:`MetaCollectiveVariable` is added to the system to couple the
    physical DOFs and the extra ones. The integrator's ``step`` method is replaced with
    a custom function that advances the physical and extension systems in tandem.

    Parameters
    ----------
    context
        The original OpenMM context containing the physical system.
    extra_dofs
        A group of extra degrees of freedom to be included in the XPS simulation.
    coupling_potential
        A meta-collective variable defining the potential energy term that couples the
        physical system to the extra DOFs. It must have units of ``kilojoules_per_mole``
        or equivalent.
    integrator_template
        An :OpenMM:`Integrator` object to be used as a template for the algorithm that
        advances the extra DOFs. If not provided, the physical system's integrator is
        used as a template.

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
    ...     f"0.5*kappa*min(delta,{2*pi}-delta)^2; delta=abs(phi-phi0)",
    ...     [cvpack.Torsion(6, 8, 14, 16, name="phi")],
    ...     unit.kilojoule_per_mole,
    ...     kappa=1000 * unit.kilojoule_per_mole / unit.radian**2,
    ...     phi0=pi*unit.radian,
    ... )
    >>> integrator = openmm.LangevinMiddleIntegrator(
    ...     300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtosecond
    ... )
    >>> platform = openmm.Platform.getPlatformByName("Reference")
    >>> mass = 3 * unit.dalton*(unit.nanometer/unit.radian)**2
    >>> phi0 = xps.ExtraDOF("phi0", unit.radian, mass, xps.bounds.CIRCULAR)
    >>> context = xps.ExtendedSpaceContext(
    ...     openmm.Context(model.system, integrator, platform),
    ...     [phi0],
    ...     umbrella_potential,
    ... )
    >>> context.setPositions(model.positions)
    >>> context.setVelocitiesToTemperature(300 * unit.kelvin)
    >>> context.setExtraValues([180 * unit.degree])
    >>> context.setExtraVelocitiesToTemperature(300 * unit.kelvin)
    >>> context.getIntegrator().step(100)
    >>> context.getExtraValues()
    (Quantity(value=..., unit=radian),)
    >>> state = context.getExtensionState(getEnergy=True)
    >>> state.getPotentialEnergy()
    Quantity(value=..., unit=kilojoule/mole)
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        context: mm.Context,
        extra_dofs: t.Iterable[ExtraDOF],
        coupling_potential: cvpack.MetaCollectiveVariable,
        integrator_template: t.Optional[mm.Integrator] = None,
    ) -> None:
        self.this = context.this
        self._system = context.getSystem()
        self._integrator = context.getIntegrator()
        self._extra_dofs = tuple(extra_dofs)
        self._coupling_potential = coupling_potential
        self._validate()
        self._coupling_potential.addToSystem(self._system)
        self._extra_kinetic_energy = self._createKineticEnergyProxy()
        self._extra_kinetic_energy.addToSystem(self._system)
        self.reinitialize(preserveState=True)
        self._extension_context = self._createExtensionContext(integrator_template)
        self._integrator.step = MethodType(
            partial(
                integrate_extended_space,
                extra_dofs=self._extra_dofs,
                extension_context=self._extension_context,
                coupling_potential=coupling_potential,
                extra_kinetic_energy=self._extra_kinetic_energy,
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
            raise ValueError("The coupling potential must have units of molar energy.")
        context_parameters = set(self.getParameters())
        force_parameters = self._coupling_potential.getParameterDefaultValues()
        parameter_units = {
            name: quantity.unit for name, quantity in force_parameters.items()
        }
        if parameters := sorted(set(parameter_units) & context_parameters):
            raise ValueError(
                f"The context already contains {parameters} among its parameters."
            )
        xdof_units = {xdof.name: xdof.unit for xdof in self._extra_dofs}
        if parameters := sorted(set(xdof_units) - set(parameter_units)):
            raise ValueError(
                f"The coupling potential parameters do not include {parameters}."
            )
        for name, unit in xdof_units.items():
            if not unit.is_compatible(parameter_units[name]):
                raise ValueError(f"Unit mismatch for parameter '{name}'.")

    def _createExtensionContext(
        self, integrator_template: t.Union[mm.Integrator, None]
    ) -> mm.Context:
        extension_integrator = copy(integrator_template or self._integrator)
        extension_integrator.setStepSize(self._integrator.getStepSize())

        extension_system = mm.System()
        for xdof in self._extra_dofs:
            extension_system.addParticle(
                xdof.mass.value_in_unit_system(mm.unit.md_unit_system)
            )

        extra_dof_cvs = []
        for index, xdof in enumerate(self._extra_dofs):
            bounds = xdof.bounds
            if bounds is None:
                force = mm.CustomExternalForce("x")
            else:
                force = mm.CustomExternalForce(bounds.leptonExpression("x"))
                bounds = bounds.asQuantity() if isinstance(bounds, Periodic) else None
            force.addParticle(index, [])
            extra_dof_cvs.append(
                cvpack.OpenMMForceWrapper(force, xdof.unit, bounds, xdof.name)
            )

        meta_cv = self._coupling_potential
        parameters = meta_cv.getParameterDefaultValues()
        for xdof in self._extra_dofs:
            parameters.pop(xdof.name)
        parameters.update(meta_cv.getInnerValues(self))

        flipped_potential = cvpack.MetaCollectiveVariable(
            function=meta_cv.getEnergyFunction(),
            variables=extra_dof_cvs,
            unit=meta_cv.getUnit(),
            periodicBounds=meta_cv.getPeriodicBounds(),
            name=meta_cv.getName(),
            **parameters,
        )
        flipped_potential.addToSystem(extension_system)

        return mm.Context(
            extension_system,
            extension_integrator,
            mm.Platform.getPlatformByName("Reference"),
        )

    def _createKineticEnergyProxy(self) -> cvpack.OpenMMForceWrapper:
        force = mm.CustomExternalForce("m*v^2/2")
        force.addPerParticleParameter("m")
        force.addPerParticleParameter("v")
        for xdof in self._extra_dofs:
            force.addParticle(0, [xdof.mass, 0])
        return cvpack.OpenMMForceWrapper(
            force, mmunit.kilojoule_per_mole, None, "extra_kinetic_energy"
        )

    def getExtraDOFs(self) -> t.Tuple[ExtraDOF]:
        """
        Get the extra degrees of freedom included in the extended phase-space system.

        Returns
        -------
        t.Tuple[ExtraDOF]
            A tuple containing the extra degrees of freedom.
        """
        return self._extra_dofs

    def setPositions(self, positions: cvpack.units.MatrixQuantity) -> None:
        """
        Sets the positions of all particles in the physical system.

        Parameters
        ----------
        positions
            The positions for each particle in the system.
        """
        super().setPositions(positions)
        for name, value in self._coupling_potential.getInnerValues(self).items():
            self._extension_context.setParameter(name, value / value.unit)

    def setExtraValues(self, values: t.Iterable[mmunit.Quantity]) -> None:
        """
        Set the values of the extra degrees of freedom.

        Parameters
        ----------
        values
            A sequence of quantities containing the values and units of all extra
            degrees of freedom.
        """
        positions = []
        for xdof, value in zip(self._extra_dofs, values):
            if mmunit.is_quantity(value):
                value = value.value_in_unit(xdof.unit)
            positions.append(mm.Vec3(value, 0, 0))
            if xdof.bounds is not None:
                value, _ = xdof.bounds.wrap(value, 0)
            self.setParameter(xdof.name, value)
        self._extension_context.setPositions(positions)
        for name, value in self._coupling_potential.getInnerValues(self).items():
            self._extension_context.setParameter(name, value / value.unit)

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
            self._extra_kinetic_energy.setParticleParameters(i, 0, [xdof.mass, value])
        self._extension_context.setVelocities(velocities)
        self._extra_kinetic_energy.updateParametersInContext(self)

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
        self._extension_context.setVelocitiesToTemperature(*args)
        state = mmswig.Context_getState(self._extension_context, mm.State.Velocities)
        velocities = mmswig.State__getVectorAsVec3(state, mm.State.Velocities)
        self.setExtraVelocities([v.x for v in velocities])

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

    def getExtensionState(self, **kwargs: t.Any) -> mm.State:
        """
        Get an :OpenMM:`State` object containing the state of the extension system.

        Parameters
        ----------
        kwargs
            The same arguments accepted by the :OpenMM:`Context.getState` method.

        Returns
        -------
        mm.State
            The state of the extension system.
        """
        return self._extension_context.getState(**kwargs)


def integrate_extended_space(
    physical_context: mm.Context,
    steps: int,
    extra_dofs: t.Tuple[ExtraDOF],
    extension_context: mm.Context,
    coupling_potential: cvpack.MetaCollectiveVariable,
    extra_kinetic_energy: cvpack.OpenMMForceWrapper,
) -> None:
    """
    Advances the extended phase-space simulation by integrating the physical and
    extension systems, in tandem, over a specified number of time steps.

    Parameters
    ----------
    physical_context
        The OpenMM context containing the physical system.
    steps
        The number of time steps to advance the simulation.
    extra_dofs
        The extra degrees of freedom included in the extended phase-space system.
    extension_context : mm.Context
        The OpenMM context containing the extension system.
    coupling_potential
        The potential that couples the physical and extra degrees of freedom.
    extra_kinetic_energy
        The force that calculates the kinetic energy of the extra degrees of freedom.

    Raises
    ------
    mm.OpenMMException
        If the particle positions or extra degrees of freedom have not been properly
        initialized in the context.
    """

    extension_integrator = extension_context.getIntegrator()
    physical_integrator = physical_context.getIntegrator()

    for _ in range(steps):
        mmswig.Integrator_step(physical_integrator, 1)
        mmswig.Integrator_step(extension_integrator, 1)

        state = mmswig.Context_getState(
            extension_context, mm.State.Positions | mm.State.Velocities
        )
        positions = mmswig.State__getVectorAsVec3(state, mm.State.Positions)
        velocities = mmswig.State__getVectorAsVec3(state, mm.State.Velocities)
        collective_variables = coupling_potential.getInnerValues(physical_context)

        for i, xdof in enumerate(extra_dofs):
            value = positions[i].x
            if xdof.bounds is not None:
                value, _ = xdof.bounds.wrap(value, 0)
            mmswig.Context_setParameter(physical_context, xdof.name, value)
            extra_kinetic_energy.setParticleParameters(
                i, 0, [xdof.mass, velocities[i].x]
            )
        extra_kinetic_energy.updateParametersInContext(physical_context)

        for name, value in collective_variables.items():
            mmswig.Context_setParameter(extension_context, name, value / value.unit)
