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

    **Note**: The system and integrator provided as arguments are modified in place.

    A given :CVPack:`MetaCollectiveVariable` is added to the system to couple the
    physical coordinates and the DVs. The integrator's ``step`` method is replaced with
    a custom function that advances both the physical and extension systems in  tandem.

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
    >>> temp = 300 * unit.kelvin
    >>> integrator = openmm.LangevinMiddleIntegrator(
    ...     temp, 1 / unit.picosecond, 4 * unit.femtosecond
    ... )
    >>> integrator.setRandomNumberSeed(1234)
    >>> platform = openmm.Platform.getPlatformByName("Reference")
    >>> mass = 3 * unit.dalton*(unit.nanometer/unit.radian)**2
    >>> phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.bounds.CIRCULAR)
    >>> height = 2 * unit.kilojoule_per_mole
    >>> sigma = 18 * unit.degree
    >>> context = xps.ExtendedSpaceContext(
    ...     xps.ExtendedSpaceSystem(
    ...         [phi0],
    ...         umbrella_potential,
    ...         model.system,
    ...     ),
    ...     xps.LockstepIntegrator(integrator),
    ...     platform,
    ... )
    >>> context.setPositions(model.positions)
    >>> context.setVelocitiesToTemperature(temp, 1234)
    >>> context.setDynamicalVariableValues([180 * unit.degree])
    >>> context.setDynamicalVariableVelocitiesToTemperature(temp, 1234)
    >>> context.getIntegrator().step(100)
    >>> context.getDynamicalVariableValues()
    (... rad,)
    >>> state = context.getExtensionContext().getState(getEnergy=True)
    >>> state.getPotentialEnergy(), state.getKineticEnergy()
    (... kJ/mol, ... kJ/mol)
    """

    def __init__(
        self,
        system: ExtendedSpaceSystem,
        integrator: ExtendedSpaceIntegrator,
        platform: t.Optional[mm.Platform] = None,
        properties: t.Optional[dict] = None,
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
            dynamical_variables=system.getDynamicalVariables(),
            coupling_potential=system.getCouplingPotential(),
        )
        self._system = system
        self._dvs = system.getDynamicalVariables()
        self._coupling_potential = system.getCouplingPotential()
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

    def setParameter(self, name: str, value: mmunit.Quantity) -> None:
        """
        Set the value of a global parameter defined by a Force object in the System.

        Notes
        -----
        If the parameter is a dynamical variable, the value will be wrapped to the
        appropriate boundary condition if necessary.

        Parameters
        ----------
        name
            The name of the parameter to set.
        value
            The value of the parameter.
        """
        dv_names = [dv.name for dv in self._dvs]
        if name in dv_names:
            i = dv_names.index(name)
            dv = self._dvs[i]
            wrapped_value, _ = dv.bounds.wrap(value.value_in_unit(dv.unit), 0)
            state = mmswig.Context_getState(self._extension_context, mm.State.Positions)
            positions = list(mmswig.State__getVectorAsVec3(state, mm.State.Positions))
            positions[i] = mm.Vec3(wrapped_value, 0, 0)
            self._extension_context.setPositions(positions)
            super().setParameter(name, wrapped_value)
        else:
            super().setParameter(name, value)

    def setPositions(self, positions: mmunit.Quantity) -> None:
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

    def setDynamicalVariableValues(self, values: t.Iterable[mmunit.Quantity]) -> None:
        """
        Set the values of the dynamical variables.

        Parameters
        ----------
        values
            A sequence of quantities containing the values and units of all extra
            degrees of freedom.
        """
        positions = []
        for dv, quantity in zip(self._dvs, values):
            if mmunit.is_quantity(quantity):
                value = quantity.value_in_unit(dv.unit)
            else:
                value = quantity
            positions.append(mm.Vec3(value, 0, 0))
            wrapped_value, _ = dv.bounds.wrap(value, 0)
            super().setParameter(dv.name, wrapped_value)
        self._extension_context.setPositions(positions)

    def getDynamicalVariableValues(self) -> tuple[mmunit.Quantity]:
        """
        Get the values of the dynamical variables.

        Returns
        -------
        t.Tuple[mmunit.Quantity]
            A tuple containing the values of the dynamical variables.
        """
        return tuple(self.getParameter(dv.name) * dv.unit for dv in self._dvs)

    def setDynamicalVariableVelocities(
        self, velocities: t.Iterable[mmunit.Quantity]
    ) -> None:
        """
        Set the velocities of the dynamical variables.

        Parameters
        ----------
        velocities
            A dictionary containing the velocities of the dynamical variables.
        """
        velocities = list(velocities)
        for i, dv in enumerate(self._dvs):
            value = velocities[i]
            if mmunit.is_quantity(value):
                value = value.value_in_unit(dv.unit / mmunit.picosecond)
            velocities[i] = mm.Vec3(value, 0, 0)
        self._extension_context.setVelocities(velocities)

    def setDynamicalVariableVelocitiesToTemperature(
        self, temperature: mmunit.Quantity, seed: t.Optional[int] = None
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
            A tuple containing the velocities of the dynamical variables.
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

    def createCheckpoint(self) -> str:
        r"""Create a checkpoint recording the current state of the Context.

        This should be treated as an opaque block of binary data. See
        :meth:`loadCheckpoint` for more details.

        Returns
        -------
        str
            A string containing the checkpoint data

        """
        return (
            mmswig.Context_createCheckpoint(self)
            + BINARY_SEPARATOR
            + mmswig.Context_createCheckpoint(self._extension_context)
        )

    def loadCheckpoint(self, checkpoint):
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
