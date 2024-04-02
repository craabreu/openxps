"""
.. module:: openxps.extended_context
   :platform: Linux, MacOS, Windows
   :synopsis: Context for extended phase-space simulations with OpenMM.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from copy import copy

import openmm as mm
from openmm import XmlSerializer as mmxml
from openmm import _openmm as mmswig
from openmm import unit as mmunit

from .extra_dof import ExtraDOF
from .serializable import Serializable
from .systems import PhysicalSystem
from .utils import preprocess_args


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

    @preprocess_args
    def __init__(  # pylint: disable=super-init-not-called
        self,
        step_size: t.Union[mmunit.Quantity, float],
        extra_dofs: t.Iterable[ExtraDOF],
        system_integrator: mm.Integrator,
        extension_integrator: t.Optional[mm.Integrator] = None,
    ) -> None:
        self._extra_dofs = tuple(extra_dofs)
        self._extension = copy(extension_integrator or system_integrator)
        self._this_integrator = copy(system_integrator)
        self.this = self._this_integrator.this
        self.setStepSize(step_size)
        self._system_context = self._extension_context = None
        self._initialized = False

    def __getatrr__(self, name: str) -> t.Any:
        return getattr(self._this_integrator, name)

    def __getstate__(self) -> t.Dict[str, t.Any]:
        return {
            "step_size": self.getStepSize(),
            "extra_dofs": self._extra_dofs,
            "system_integrator": mmxml.serialize(self),
            "extension_integrator": mmxml.serialize(self._extension),
        }

    def __setstate__(self, state: t.Dict[str, t.Any]) -> None:
        self.__init__(
            state["step_size"],
            state["extra_dofs"],
            mmxml.deserialize(state["system_integrator"]),
            mmxml.deserialize(state["extension_integrator"]),
        )

    def get_extension(self) -> mm.Integrator:
        """
        Get the integrator for the extra degrees of freedom.

        Returns
        -------
        mm.Integrator
            The integrator for the extra degrees of freedom.
        """
        return self._extension

    def initialize(
        self, system_context: mm.Context, extension_context: mm.Context
    ) -> None:
        """
        Set the contexts for the physical and extended systems.

        Parameters
        ----------
        system_context
            The context for the physical system.
        extension_context
            The context for the extra degrees of freedom.
        """
        self._system_context = system_context
        self._extension_context = extension_context
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
        self._extension.setStepSize(size / 2)
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
        return mmswig.Integrator_step(self, steps)


ExtendedSpaceIntegrator.register_tag("!openxps.ExtendedSpaceIntegrator")


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
    >>> context = openmm.Context(model.system, integrator, platform)
    >>> mass = 3 * unit.dalton*(unit.nanometer/unit.radian)**2
    >>> phi_dv = xps.ExtraDOF("phi_dv", unit.radian, mass, xps.bounds.CIRCULAR)
    >>> xps_context = xps.ExtendedSpaceContext(context, [phi_dv])
    """

    def __init__(
        self,
        context: mm.Context,
        extra_dofs: t.Iterable[ExtraDOF],
        extension_integrator: t.Optional[mm.Integrator] = None,
    ) -> None:
        self._extra_dofs = tuple(extra_dofs)
        self._system = PhysicalSystem(context.getSystem(), self._extra_dofs)
        self._integrator = ExtendedSpaceIntegrator(
            context.getIntegrator().getStepSize(),
            self._extra_dofs,
            context.getIntegrator(),
            extension_integrator,
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
            force = mm.CustomExternalForce(f"-{xdof.name}_derivative*x")
            force.addGlobalParameter(f"{xdof.name}_derivative", 0)
            force.addParticle(index, [])
            extension_system.addForce(force)

        self._extension = mm.Context(
            extension_system,
            self._integrator.get_extension(),
            mm.Platform.getPlatformByName("Reference"),
        )
        self._integrator.initialize(self, self._extension)

    def set_extra_dof_values(self, values: t.Iterable[mmunit.Quantity]) -> None:
        """
        Set the values of the extra degrees of freedom.

        Parameters
        ----------
        values
            A dictionary containing the values of the extra degrees of freedom.
        wrap
            If ``True``, the values are wrapped around the bounds of the extra degrees
            of freedom.
        """
        values = list(values)
        for i, xdof in enumerate(self._extra_dofs):
            value = values[i]
            if mmunit.is_quantity(value):
                value = value.value_in_unit(xdof.unit)
            values[i] = mm.Vec3(value, 0, 0)
            self.setParameter(xdof.name, value)
        self._extension.setPositions(values)

    def get_extra_dof_values(self) -> t.Tuple[mmunit.Quantity]:
        """
        Get the values of the extra degrees of freedom.

        Returns
        -------
        t.Tuple[mmunit.Quantity]
            A tuple containing the values of the extra degrees of freedom.
        """
        values = []
        for xdof in self._extra_dofs:
            value = self.getParameter(xdof.name)
            if xdof.bounds is not None:
                value = xdof.bounds.wrap(value)
            values.append(value * xdof.unit)
        return tuple(values)
