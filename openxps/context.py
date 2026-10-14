"""
.. module:: openxps.context
   :platform: Linux, MacOS, Windows
   :synopsis: Context for extended phase-space simulations with OpenMM.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm as mm
from openmm import _openmm as mmswig
from openmm import unit as mmunit

from .integrator import ExtendedSpaceIntegrator
from .system import ExtendedSpaceSystem
from .utils import BINARY_SEPARATOR


class ExtendedSpaceContext(mm.Context):
    """An :OpenMM:`Context` object that includes extra dynamical variables (DVs) and
    allows for extended phase-space (XPS) simulations.

    Parameters
    ----------
    system
        The :class:`ExtendedSpaceSystem` to be used in the XPS simulation.
    integrator
        An :class:`ExtendedSpaceIntegrator` object to be used for advancing the XPS
        simulation. Available implementations include :class:`LockstepIntegrator` for
        systems where both integrators use the same step size, and
        :class:`SplitIntegrator` for systems with different step sizes related by an
        even integer ratio.
    platform
        The :OpenMM:`Platform` to use for calculations.
    properties
        A dictionary of values for platform-specific properties.

    Example
    -------
    >>> import openxps as xps
    >>> from math import pi
    >>> import openmm
    >>> import cvpack
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> mass = 3 * unit.dalton*(unit.nanometer/unit.radian)**2
    >>> phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> kappa = 1000 * unit.kilojoule_per_mole / unit.radian**2
    >>> harmonic_force = xps.HarmonicCoupling(phi, phi0, kappa)
    >>> temp = 300 * unit.kelvin
    >>> integrator = openmm.LangevinMiddleIntegrator(
    ...     temp, 1 / unit.picosecond, 4 * unit.femtosecond
    ... )
    >>> integrator.setRandomNumberSeed(1234)
    >>> platform = openmm.Platform.getPlatformByName("Reference")
    >>> context = xps.ExtendedSpaceContext(
    ...     xps.ExtendedSpaceSystem(model.system, harmonic_force),
    ...     xps.LockstepIntegrator(integrator),
    ...     platform,
    ... )
    >>> context.setPositions(model.positions)
    >>> context.setVelocitiesToTemperature(temp, 1234)
    >>> context.setDynamicalVariableValues([180 * unit.degree])
    >>> context.setDynamicalVariableVelocitiesToTemperature(temp, 1234)
    >>> context.getIntegrator().step(100)
    >>> print("phi0 =", *context.getDynamicalVariableValues())
    phi0 = ... rad
    >>> state = context.getExtensionContext().getState(getEnergy=True)
    >>> print("U =", state.getPotentialEnergy(), "K =", state.getKineticEnergy())
    U = ... kJ/mol K = ... kJ/mol
    """

    def __init__(
        self,
        system: ExtendedSpaceSystem,
        integrator: ExtendedSpaceIntegrator,
        platform: mm.Platform | None = None,
        properties: dict | None = None,
    ) -> None:
        self._validate(system, integrator)
        args = [system, integrator.getPhysicalIntegrator()]
        if platform is not None:
            args.append(platform)
            if properties is not None:
                args.append(properties)
        super().__init__(*args)
        extension_context = mm.Context(
            system.getExtensionSystem(),
            integrator.getExtensionIntegrator(),
            mm.Platform.getPlatformByName("Reference"),
        )
        integrator.configure(
            physical_context=self,
            extension_context=extension_context,
            coupling=system.getCoupling(),
        )
        self._system = system
        self._dvs = system.getDynamicalVariables()
        self._coupling = system.getCoupling()
        self._integrator = integrator
        self._extension_context = extension_context

    def _validate(
        self,
        system: ExtendedSpaceSystem,
        integrator: ExtendedSpaceIntegrator,
    ) -> None:
        if not isinstance(system, ExtendedSpaceSystem):
            raise TypeError("The system must be an instance of ExtendedSpaceSystem.")
        if not isinstance(integrator, ExtendedSpaceIntegrator):
            raise TypeError(
                "The integrator must be an instance of ExtendedSpaceIntegrator."
            )

    def getSystem(self) -> ExtendedSpaceSystem:
        """
        Get the system included in the extended phase-space context.

        Returns
        -------
        ExtendedSpaceSystem
            The system.
        """
        return self._system

    def getIntegrator(self) -> ExtendedSpaceIntegrator:
        """
        Get the integrator included in the extended phase-space context.

        Returns
        -------
        ExtendedSpaceIntegrator
            The integrator.
        """
        return self._integrator

    def synchronize(self) -> None:
        """
        Push the dynamical variable values to the physical context and the
        collective variable values to the extension context.

        Call this after changing the physical state through a channel that bypasses
        this class, such as an :OpenMM:`LocalEnergyMinimizer`.
        """
        self._coupling.updatePhysicalContext(self, self._extension_context)
        self._coupling.updateExtensionContext(self, self._extension_context)

    def _pushSharedParameters(self) -> None:
        """Copy parameters present in both contexts from the physical one."""
        protected = self._coupling.getProtectedParameters()
        extension_parameters = self._extension_context.getParameters()
        for name, value in self.getParameters().items():
            if name in extension_parameters and name not in protected:
                self._extension_context.setParameter(name, value)

    def getParameter(self, name: str) -> float:
        """
        Get the value of a global parameter of the physical or extension system.

        Parameters
        ----------
        name
            The name of the parameter. A parameter that exists only in the extension
            system (e.g. a parameter of a coupling function) is read from the
            extension context.

        Returns
        -------
        float
            The value of the parameter.
        """
        if name not in self.getParameters() and (
            name in self._extension_context.getParameters()
        ):
            return self._extension_context.getParameter(name)
        return super().getParameter(name)

    def setParameter(self, name: str, value: mmunit.Quantity) -> None:
        """
        Set the value of a global parameter defined by a Force object in the System.

        Notes
        -----
        Parameters that represent dynamical variables are protected and cannot be set
        through this method; use :meth:`setDynamicalVariableValues` instead. A
        parameter that exists in both systems (e.g. a coupling constant) is updated
        in both contexts, and one that exists only in the extension system (e.g. a
        parameter of a coupling function) is updated in the extension context. Both
        contexts are then synchronized.

        Parameters
        ----------
        name
            The name of the parameter to set.
        value
            The value of the parameter.
        """
        if name in self._coupling.getProtectedParameters():
            raise ValueError(
                f'Cannot manually set the parameter "{name}". This parameter is '
                "set automatically via setDynamicalVariableValues."
            )
        in_extension = name in self._extension_context.getParameters()
        if name in self.getParameters() or not in_extension:
            super().setParameter(name, value)
        if in_extension:
            self._extension_context.setParameter(name, value)
        self.synchronize()

    def reinitialize(self, preserveState: bool = False) -> None:
        """
        Reinitialize the physical context and restore the dynamical variable
        parameters from the extension context, whose state is always preserved.

        Parameters shared by both contexts take their values from the physical
        context, so they are reset to their defaults when the physical state is not
        preserved.

        Parameters
        ----------
        preserveState
            Whether to preserve the state of the physical context.
        """
        super().reinitialize(preserveState)
        self._extension_context.reinitialize(True)
        self._pushSharedParameters()
        self._coupling.updatePhysicalContext(self, self._extension_context)
        if preserveState:
            self._coupling.updateExtensionContext(self, self._extension_context)

    def setState(self, state: mm.State) -> None:
        """
        Set the state of the physical context and synchronize both contexts.

        Parameters restored from the state that also exist in the extension system
        (e.g. a coupling constant) are copied to the extension context.

        Parameters
        ----------
        state
            The :OpenMM:`State` to apply to the physical context.
        """
        super().setState(state)
        self._pushSharedParameters()
        self.synchronize()

    def setPositions(self, positions: mmunit.Quantity) -> None:
        """
        Sets the positions of all particles in the physical system.

        Parameters
        ----------
        positions
            The positions for each particle in the system.
        """
        super().setPositions(positions)
        self._coupling.updateExtensionContext(self, self._extension_context)

    def setDynamicalVariableValues(self, values: t.Iterable[mmunit.Quantity]) -> None:
        """
        Set the values of the dynamical variables.

        Parameters
        ----------
        values
            The values of all dynamical variables, in the order of
            :meth:`ExtendedSpaceSystem.getDynamicalVariables`. As in OpenMM, a value
            given without units is taken in MD units (e.g. radians for angles, even
            if the dynamical variable was declared in degrees).
        """
        values = list(values)
        if len(values) != len(self._dvs):
            raise ValueError(
                f"Expected values for {len(self._dvs)} dynamical variables, "
                f"got {len(values)}."
            )
        positions = []
        for dv, quantity in zip(self._dvs, values):
            if mmunit.is_quantity(quantity):
                value = quantity.value_in_unit(dv.unit)
            else:
                value = quantity
            positions.append(mm.Vec3(value, 0, 0))
        self._extension_context.setPositions(positions)
        self._coupling.updatePhysicalContext(self, self._extension_context)

    def getDynamicalVariableValues(self) -> tuple[mmunit.Quantity]:
        """
        Get the values of the dynamical variables, wrapped into their bounds.

        Returns
        -------
        t.Tuple[mmunit.Quantity]
            The values of the dynamical variables, in the order of
            :meth:`ExtendedSpaceSystem.getDynamicalVariables`. They are given in MD
            units, whatever unit each dynamical variable was declared in.
        """
        state = mmswig.Context_getState(self._extension_context, mm.State.Positions)
        positions = mmswig.State__getVectorAsVec3(state, mm.State.Positions)
        values = []
        for i, dv in enumerate(self._dvs):
            value, _ = dv.bounds.wrap(positions[i].x, 0)
            values.append(value * dv.unit)
        return tuple(values)

    def setDynamicalVariableVelocities(
        self, velocities: t.Iterable[mmunit.Quantity]
    ) -> None:
        """
        Set the velocities of the dynamical variables.

        Parameters
        ----------
        velocities
            The velocities of all dynamical variables, in the order of
            :meth:`ExtendedSpaceSystem.getDynamicalVariables`. As in OpenMM, a
            velocity given without units is taken in MD units (e.g. radians per
            picosecond for angles).
        """
        velocities = list(velocities)
        if len(velocities) != len(self._dvs):
            raise ValueError(
                f"Expected velocities for {len(self._dvs)} dynamical variables, "
                f"got {len(velocities)}."
            )
        for i, dv in enumerate(self._dvs):
            value = velocities[i]
            if mmunit.is_quantity(value):
                value = value.value_in_unit(dv.unit / mmunit.picosecond)
            velocities[i] = mm.Vec3(value, 0, 0)
        self._extension_context.setVelocities(velocities)

    def setDynamicalVariableVelocitiesToTemperature(
        self, temperature: mmunit.Quantity, seed: int | None = None
    ) -> None:
        """
        Set the velocities of the dynamical variables to a temperature.

        Parameters
        ----------
        temperature
            The temperature to set the velocities to.
        """
        args = (temperature,) if seed is None else (temperature, seed)
        self._extension_context.setVelocitiesToTemperature(*args)
        state = mmswig.Context_getState(self._extension_context, mm.State.Velocities)
        velocities = mmswig.State__getVectorAsVec3(state, mm.State.Velocities)
        self._extension_context.setVelocities([mm.Vec3(v.x, 0, 0) for v in velocities])

    def getDynamicalVariableVelocities(self) -> tuple[mmunit.Quantity]:
        """
        Get the velocities of the dynamical variables.

        Returns
        -------
        t.Tuple[mmunit.Quantity]
            The velocities of the dynamical variables, in the order of
            :meth:`ExtendedSpaceSystem.getDynamicalVariables`. They are given in MD
            units, whatever unit each dynamical variable was declared in.
        """
        state = mmswig.Context_getState(
            self._extension_context, mm.State.Positions | mm.State.Velocities
        )
        positions = mmswig.State__getVectorAsVec3(state, mm.State.Positions)
        velocities = mmswig.State__getVectorAsVec3(state, mm.State.Velocities)
        dv_velocities = []
        for i, dv in enumerate(self._dvs):
            _, rate = dv.bounds.wrap(positions[i].x, velocities[i].x)
            dv_velocities.append(rate * dv.unit / mmunit.picosecond)
        return tuple(dv_velocities)

    def getExtensionContext(self) -> mm.Context:
        """
        Get a reference to the OpenMM context containing the extension system.

        Returns
        -------
        mm.Context
            The context containing the extension system.
        """
        return self._extension_context

    def createCheckpoint(self) -> bytes:
        r"""Create a checkpoint recording the current state of the Context.

        This should be treated as an opaque block of binary data. See
        :meth:`loadCheckpoint` for more details.

        Returns
        -------
        bytes
            The checkpoint data

        """
        return (
            mmswig.Context_createCheckpoint(self)
            + BINARY_SEPARATOR
            + mmswig.Context_createCheckpoint(self._extension_context)
        )

    def loadCheckpoint(self, checkpoint: bytes) -> None:
        r"""Load a checkpoint that was written by :meth:`createCheckpoint`.

        See :OpenMM:`Context` for more details.

        Parameters
        ----------
        checkpoint
            The checkpoint data to load.
        """
        physical_checkpoint, extension_checkpoint = checkpoint.split(BINARY_SEPARATOR)
        mmswig.Context_loadCheckpoint(self, physical_checkpoint)
        mmswig.Context_loadCheckpoint(self._extension_context, extension_checkpoint)
