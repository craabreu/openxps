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

from .dynamical_variable import DynamicalVariable
from .integrators import ExtendedSpaceIntegrator


class ExtendedSpaceContext(mm.Context):
    """An :OpenMM:`Context` object that includes extra dynamical variables (DVs) and
    allows for extended phase-space (XPS) simulations.

    **Note**: The system and integrator provided as arguments are modified in place.

    A given :CVPack:`MetaCollectiveVariable` is added to the system to couple the
    physical coordinates and the DVs. The integrator's ``step`` method is replaced with
    a custom function that advances both the physical and extension systems in tandem.

    Parameters
    ----------
    dynamical_variables
        A collection of dynamical variables (DVs) to be included in the XPS simulation.
    coupling_potential
        A :CVPack:`MetaCollectiveVariable` defining the potential energy term that
        couples the DVs to the physical coordinates. It must have units
        of ``kilojoules_per_mole``. All DVs must be included as parameters in the
        coupling potential.
    system
        The :OpenMM:`System` to be used in the XPS simulation.
    integrator
        An :OpenMM:`Integrator` object or a tuple of two :OpenMM:`Integrator` objects
        to be used in the XPS simulation. If a tuple is provided, the first integrator
        is used for the physical system and the second one is used for the DVs.
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
    ...     [phi0],
    ...     umbrella_potential,
    ...     model.system,
    ...     xps.InTandemIntegrator(integrator),
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
        dynamical_variables: t.Sequence[DynamicalVariable],
        coupling_potential: cvpack.MetaCollectiveVariable,
        system: mm.System,
        integrator: ExtendedSpaceIntegrator,
        platform: t.Optional[mm.Platform] = None,
        properties: t.Optional[dict] = None,
    ) -> None:
        self._dvs = tuple(dynamical_variables)
        self._coupling_potential = coupling_potential
        self._xps_integrator = integrator
        self._validate()
        coupling_potential.addToSystem(system)
        args = [system, integrator.getPhysicalIntegrator()]
        if platform is not None:
            args.append(platform)
            if properties is not None:
                args.append(properties)
        super().__init__(*args)
        self._extension_context = self._createExtensionContext(
            integrator.getExtensionIntegrator()
        )
        self._integrator.step = MethodType(
            partial(
                integrator.integrate,
                dynamical_variables=self._dvs,
                extension_context=self._extension_context,
                coupling_potential=coupling_potential,
            ),
            self,
        )

    def _validate(self) -> None:
        if not all(isinstance(dv, DynamicalVariable) for dv in self._dvs):
            raise TypeError(
                "All dynamical variables must be instances of DynamicalVariable."
            )
        if not isinstance(self._coupling_potential, cvpack.MetaCollectiveVariable):
            raise TypeError(
                "The coupling potential must be an instance of MetaCollectiveVariable."
            )
        if not self._coupling_potential.getUnit().is_compatible(
            mmunit.kilojoule_per_mole
        ):
            raise ValueError("The coupling potential must have units of molar energy.")
        missing_parameters = [
            dv.name
            for dv in self._dvs
            if dv.name not in self._coupling_potential.getParameterDefaultValues()
        ]
        if missing_parameters:
            raise ValueError(
                "These dynamical variables are not coupling potential parameters: "
                + ", ".join(missing_parameters)
            )

        if not isinstance(self._xps_integrator, ExtendedSpaceIntegrator):
            raise TypeError(
                "The integrator must be an instance of ExtendedSpaceIntegrator."
            )

    def _createExtensionContext(
        self, integrator_template: t.Union[mm.Integrator, None]
    ) -> mm.Context:
        extension_integrator = copy(integrator_template or self._integrator)
        extension_integrator.setStepSize(self._integrator.getStepSize())

        extension_system = mm.System()
        for dv in self._dvs:
            extension_system.addParticle(dv.mass / dv.mass.unit)

        meta_cv = self._coupling_potential
        parameters = meta_cv.getParameterDefaultValues()
        for dv in self._dvs:
            parameters.pop(dv.name)
        parameters.update(meta_cv.getInnerValues(self))

        flipped_potential = cvpack.MetaCollectiveVariable(
            function=meta_cv.getEnergyFunction(),
            variables=[
                dv.createCollectiveVariable(index) for index, dv in enumerate(self._dvs)
            ],
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

    def getDynamicalVariables(self) -> tuple[DynamicalVariable]:
        """
        Get the dynamical variables included in the extended phase-space system.

        Returns
        -------
        t.Tuple[DynamicalVariable]
            A tuple containing the dynamical variables.
        """
        return self._dvs

    def getCouplingPotential(self) -> cvpack.MetaCollectiveVariable:
        """
        Get the coupling potential included in the extended phase-space system.

        Returns
        -------
        cvpack.MetaCollectiveVariable
            The coupling potential.
        """
        return self._coupling_potential

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
            value = value.value_in_unit(dv.unit)
            if dv.bounds is not None:
                value, _ = dv.bounds.wrap(value, 0)
            state = mmswig.Context_getState(self, mm.State.Positions)
            positions = mmswig.State__getVectorAsVec3(state, mm.State.Positions)
            positions[i].x = value
            self._extension_context.setPositions(positions)
        else:
            super().setParameter(name, value)

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
            if dv.bounds is not None:
                value, _ = dv.bounds.wrap(value, 0)
            super().setParameter(dv.name, value)
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
            value = positions[i].x
            rate = velocities[i].x
            if dv.bounds is not None:
                value, rate = dv.bounds.wrap(value, rate)
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
