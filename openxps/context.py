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

from .bias_potential import BiasPotential
from .dynamical_variable import DynamicalVariable


class ExtendedSpaceContext(mm.Context):  # pylint: disable=too-many-instance-attributes
    """
    Wraps an :OpenMM:`Context` object to include dynamical variables (DVs) and
    allow for extended phase-space (XPS) simulations.

    **Note**: The system and integrator attached to the context are modified in-place.

    A provided :CVPack:`MetaCollectiveVariable` is added to the system to couple the
    physical DVs and the extra ones. The integrator's ``step`` method is replaced with
    a custom function that advances the physical and extension systems in tandem.

    Parameters
    ----------
    context
        The original OpenMM context containing the physical system.
    dynamical_variables
        A group of dynamical variables to be included in the XPS simulation.
    coupling_potential
        A meta-collective variable defining the potential energy term that couples the
        physical system to the DVs. It must have units of ``kilojoules_per_mole``
        or equivalent.
    integrator_template
        An :OpenMM:`Integrator` object to be used as a template for the algorithm that
        advances the DVs. If not provided, the physical system's integrator is
        used as a template.
    bias_potential
        A bias potential applied to the DVs. If not provided, no bias is applied.

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
    ...     openmm.Context(model.system, integrator, platform),
    ...     [phi0],
    ...     umbrella_potential,
    ...     bias_potential=xps.MetadynamicsBias(
    ...         [phi0], [sigma], height, temp, 10, [100]
    ...     ),
    ... )
    >>> context.setPositions(model.positions)
    >>> context.setVelocitiesToTemperature(temp, 1234)
    >>> context.setExtraValues([180 * unit.degree])
    >>> context.setExtraVelocitiesToTemperature(temp, 1234)
    >>> context.getIntegrator().step(100)
    >>> context.getExtraValues()
    (... rad,)
    >>> context.addBiasKernel()
    >>> state = context.getExtensionContext().getState(getEnergy=True)
    >>> state.getPotentialEnergy(), state.getKineticEnergy()
    (... kJ/mol, ... kJ/mol)
    """

    def __init__(  # pylint: disable=super-init-not-called,too-many-arguments
        self,
        context: mm.Context,
        dynamical_variables: t.Sequence[DynamicalVariable],
        coupling_potential: cvpack.MetaCollectiveVariable,
        integrator_template: t.Optional[mm.Integrator] = None,
        bias_potential: t.Optional[BiasPotential] = None,
    ) -> None:
        self.this = context.this
        self._system = context.getSystem()
        self._integrator = context.getIntegrator()
        self._dvs = tuple(dynamical_variables)
        self._coupling_potential = coupling_potential
        self._validate()
        self._coupling_potential.addToSystem(self._system)
        self.reinitialize(preserveState=True)
        self._bias_potential = bias_potential
        self._extension_context = self._createExtensionContext(integrator_template)

        self._integrator.step = MethodType(
            partial(
                integrate_extended_space,
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
        context_parameters = set(self.getParameters())
        force_parameters = self._coupling_potential.getParameterDefaultValues()
        parameter_units = {
            name: quantity.unit for name, quantity in force_parameters.items()
        }
        if parameters := sorted(set(parameter_units) & context_parameters):
            raise ValueError(
                f"The context already contains {parameters} among its parameters."
            )
        dv_units = {dv.name: dv.unit for dv in self._dvs}
        if parameters := sorted(set(dv_units) - set(parameter_units)):
            raise ValueError(
                f"The coupling potential parameters do not include {parameters}."
            )
        for name, unit in dv_units.items():
            if not unit.is_compatible(parameter_units[name]):
                raise ValueError(f"Unit mismatch for parameter '{name}'.")

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

        if self._bias_potential is not None:
            self._bias_potential.initialize(self._dvs)
            self._bias_potential.addToSystem(extension_system)

        return mm.Context(
            extension_system,
            extension_integrator,
            mm.Platform.getPlatformByName("Reference"),
        )

    def addBiasKernel(self) -> None:
        """
        Add a Gaussian kernel to the bias potential.
        """
        try:
            self._bias_potential.addKernel(self._extension_context)
        except AttributeError as error:
            raise AttributeError(
                "No bias potential was provided when creating the context."
            ) from error

    def getExtraDOFs(self) -> tuple[DynamicalVariable]:
        """
        Get the dynamical variables included in the extended phase-space system.

        Returns
        -------
        t.Tuple[DynamicalVariable]
            A tuple containing the dynamical variables.
        """
        return self._dvs

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
        Set the values of the dynamical variables.

        Parameters
        ----------
        values
            A sequence of quantities containing the values and units of all extra
            degrees of freedom.
        """
        positions = []
        for dv, value in zip(self._dvs, values):
            if mmunit.is_quantity(value):
                value = value.value_in_unit(dv.unit)
            positions.append(mm.Vec3(value, 0, 0))
            if dv.bounds is not None:
                value, _ = dv.bounds.wrap(value, 0)
            self.setParameter(dv.name, value)
        self._extension_context.setPositions(positions)
        for name, value in self._coupling_potential.getInnerValues(self).items():
            self._extension_context.setParameter(name, value / value.unit)

    def getExtraValues(self) -> tuple[mmunit.Quantity]:
        """
        Get the values of the dynamical variables.

        Returns
        -------
        t.Tuple[mmunit.Quantity]
            A tuple containing the values of the dynamical variables.
        """
        return tuple(self.getParameter(dv.name) * dv.unit for dv in self._dvs)

    def setExtraVelocities(self, velocities: t.Iterable[mmunit.Quantity]) -> None:
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

    def setExtraVelocitiesToTemperature(
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

    def getExtraVelocities(self) -> tuple[mmunit.Quantity]:
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
        for i, dv in enumerate(self._dvs):
            value = positions[i].x
            rate = velocities[i].x
            if dv.bounds is not None:
                value, rate = dv.bounds.wrap(value, rate)
            velocities[i] = rate * dv.unit / mmunit.picosecond
        return tuple(velocities)

    def getExtensionContext(self) -> mm.Context:
        """
        Get a reference to the OpenMM context containing the extension system.

        Returns
        -------
        mm.Context
            The context containing the extension system.
        """
        return self._extension_context


def integrate_extended_space(
    physical_context: mm.Context,
    steps: int,
    dynamical_variables: tuple[DynamicalVariable],
    extension_context: mm.Context,
    coupling_potential: cvpack.MetaCollectiveVariable,
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
    dynamical_variables
        The dynamical variables included in the extended phase-space system.
    extension_context : mm.Context
        The OpenMM context containing the extension system.
    coupling_potential
        The potential that couples the physical and dynamical variables.

    Raises
    ------
    mm.OpenMMException
        If the particle positions or dynamical variables have not been properly
        initialized in the context.
    """

    for _ in range(steps):
        # pylint: disable=protected-access
        mmswig.Integrator_step(physical_context._integrator, 1)
        mmswig.Integrator_step(extension_context._integrator, 1)
        # pylint: enable=protected-access

        state = mmswig.Context_getState(extension_context, mm.State.Positions)
        positions = mmswig.State__getVectorAsVec3(state, mm.State.Positions)
        for i, dv in enumerate(dynamical_variables):
            value = positions[i].x
            if dv.bounds is not None:
                value, _ = dv.bounds.wrap(value, 0)
            mmswig.Context_setParameter(physical_context, dv.name, value)

        collective_variables = mmswig.CustomCVForce_getCollectiveVariableValues(
            coupling_potential,
            physical_context,
        )
        for i, value in enumerate(collective_variables):
            mmswig.Context_setParameter(
                extension_context,
                mmswig.CustomCVForce_getCollectiveVariableName(coupling_potential, i),
                value,
            )
