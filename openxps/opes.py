"""
.. module:: openxps.opes
   :platform: Linux, MacOS, Windows
   :synopsis: On-the-fly probability enhanced sampling of dynamical variables.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm as mm
import openmm_opes
from openmm import app as mmapp

from .bounds import NoBounds, ReflectiveBounds
from .context import ExtendedSpaceContext
from .metadynamics import ExtendedSpaceBiasVariable
from .system import ExtendedSpaceSystem
from .utils import ExtensionSimulation


def _biasVariables(
    system: ExtendedSpaceSystem,
    variables: t.Sequence[ExtendedSpaceBiasVariable],
    kwargs: t.Mapping[str, t.Any] | None = None,
) -> tuple[list[mmapp.BiasVariable], bool]:
    """Bias variables on the extension particles, and whether any is reflective.

    Mirrors :class:`ExtendedSpaceMetadynamics`: each dynamical variable, in MD
    units, must belong to the system and have bounds, since the bias lives on a
    finite grid. ``bounded`` is True if any variable has reflective bounds; with
    unbounded variables rejected, the non-periodic variables are exactly the
    reflective ones. It must not be given in ``kwargs``, the keyword arguments a
    sampler forwards to openmm-opes.
    """
    if kwargs and "bounded" in kwargs:
        raise TypeError(
            "bounded is set from the bounds of the dynamical variables; do not pass it"
        )
    system_dvs = system.getDynamicalVariables()
    bias_variables = []
    bounded = False
    for variable in variables:
        dv = variable.dynamical_variable.in_md_units()
        if dv not in system_dvs:
            raise ValueError(f"Dynamical variable {dv.name} not found in system")
        if isinstance(dv.bounds, NoBounds):
            raise ValueError(
                f"Dynamical variable {dv.name} has no bounds, but OPES needs a "
                "finite grid. Use ReflectiveBounds over the range the variable may "
                "explore."
            )
        bounded = bounded or isinstance(dv.bounds, ReflectiveBounds)
        index = system_dvs.index(dv)
        bias_variables.append(
            mmapp.BiasVariable(
                dv.createCollectiveVariable(index),
                dv.bounds.lower * dv.bounds.unit,
                dv.bounds.upper * dv.bounds.unit,
                variable.sigma,
                dv.isPeriodic(),
                variable.grid_width,
            )
        )
    return bias_variables, bounded


def _extensionSimulation(simulation):
    """A view of ``simulation`` whose context is the extension context.

    openmm-opes calls back into the sampler with the object it was given, so an
    already converted simulation is returned unchanged.
    """
    if isinstance(simulation, ExtensionSimulation):
        return simulation
    if not isinstance(getattr(simulation, "context", None), ExtendedSpaceContext):
        raise TypeError(
            "Expected an ExtendedSpaceSimulation, whose context holds the extension "
            f"system, not {type(simulation).__name__}"
        )
    return ExtensionSimulation(simulation)


def _extensionContext(context: mm.Context) -> mm.Context:
    """The extension context of an :class:`ExtendedSpaceContext`, else ``context``."""
    if isinstance(context, ExtendedSpaceContext):
        return context.getExtensionContext()
    return context


class ExtendedSpaceOPES(openmm_opes.OPES):
    """On-the-fly probability enhanced sampling (OPES) of dynamical variables.

    A bias potential is applied to dynamical variables of an
    :class:`~openxps.ExtendedSpaceSystem` to enhance the sampling of the physical
    coordinates coupled to them. This is :OpenMMOPES:`OPES <opes>` acting on the
    extension system: OPES and OPES-explore (``exploreMode=True``), multiple walkers
    and all other options are those of openmm-opes, documented there.

    Parameters
    ----------
    system
        The :class:`~openxps.ExtendedSpaceSystem` to be simulated. The bias force
        is added to its extension system when this object is created, so it must
        be created before the :class:`~openxps.ExtendedSpaceSimulation` or
        :class:`~openxps.ExtendedSpaceContext` that uses the system.
    variables
        The dynamical variables to bias and their kernel widths, as
        :class:`~openxps.ExtendedSpaceBiasVariable` objects. Variables with
        :class:`~openxps.bounds.NoBounds` are rejected, since the bias is stored on
        a finite grid. Periodic and reflective variables cannot be mixed.
    **kwargs
        The remaining arguments of :OpenMMOPES:`OPES <opes>` (``temperature``,
        ``barrier``, ``frequency``, ``varianceFrequency``, ...), except ``bounded``,
        which follows from the variables' bounds.

    .. note::
        Kernels are those of openmm-opes: Gaussian by default and minimum-image on
        periodic variables, whereas :class:`~openxps.ExtendedSpaceMetadynamics`
        uses von Mises kernels. ``sigma`` keeps its openmm-opes meaning: with
        ``varianceFrequency=None``, the standard deviation of the unbiased
        distribution.

    .. note::
        ``extraBias`` entries are evaluated in the extension context, so they must
        be forces of the extension system, such as other samplers on it.

    .. note::
        To restart, save the sampler with ``getState()`` alongside the simulation's
        state, and after ``setState()`` on a new sampler call ``updateContext()`` on
        the new context (or take a step): until then, the context holds no bias and
        ``getBiasEnergy()`` returns minus the barrier.

    Example
    -------
    >>> import openxps as xps
    >>> from openmm import unit
    >>> import cvpack
    >>> import openmm
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> mass = 3 * unit.dalton*(unit.nanometer/unit.radian)**2
    >>> phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    >>> kappa = 1000 * unit.kilojoules_per_mole / unit.radian**2
    >>> system = xps.ExtendedSpaceSystem(
    ...     model.system, xps.HarmonicCoupling(phi, phi0, kappa)
    ... )
    >>> opes = xps.ExtendedSpaceOPES(
    ...     system,
    ...     [xps.ExtendedSpaceBiasVariable(phi0, 18 * unit.degrees)],
    ...     temperature=300 * unit.kelvin,
    ...     barrier=20 * unit.kilojoules_per_mole,
    ...     frequency=100,
    ...     varianceFrequency=None,
    ... )
    >>> integrator = openmm.LangevinMiddleIntegrator(
    ...     300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond
    ... )
    >>> integrator.setRandomNumberSeed(1234)
    >>> simulation = xps.ExtendedSpaceSimulation(
    ...     model.topology,
    ...     system,
    ...     xps.LockstepIntegrator(integrator),
    ...     openmm.Platform.getPlatformByName("Reference"),
    ... )
    >>> context = simulation.context
    >>> context.setPositions(model.positions)
    >>> context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    >>> context.setDynamicalVariableValues([180 * unit.degree])
    >>> context.setDynamicalVariableVelocitiesToTemperature(300 * unit.kelvin, 1234)
    >>> opes.step(simulation, 200)
    >>> opes.getNumKernels()  # the second kernel merged into the first
    1
    """

    def __init__(
        self,
        system: ExtendedSpaceSystem,
        variables: t.Sequence[ExtendedSpaceBiasVariable],
        **kwargs,
    ) -> None:
        bias_variables, bounded = _biasVariables(system, variables, kwargs)
        super().__init__(
            system.getExtensionSystem(), bias_variables, bounded=bounded, **kwargs
        )

    def step(self, simulation, steps: int) -> None:
        """Advance an :class:`~openxps.ExtendedSpaceSimulation` by ``steps`` steps."""
        super().step(_extensionSimulation(simulation), steps)

    @staticmethod
    def stepTogether(simulation, samplers, steps: int) -> None:
        """Advance a simulation biased by several extended-space samplers at once.

        See :OpenMMOPES:`OPES <opes>`. The samplers may be :class:`ExtendedSpaceOPES` or
        :class:`ExtendedSpaceCombinedOPES` objects acting on the same extension
        system.
        """
        openmm_opes.OPES.stepTogether(_extensionSimulation(simulation), samplers, steps)

    def getCollectiveVariables(self, simulation):
        """Current values of the biased dynamical variables, in MD units."""
        return super().getCollectiveVariables(_extensionSimulation(simulation))

    def getBiasEnergy(self, context: mm.Context):
        """The bias energy applied in an :class:`~openxps.ExtendedSpaceContext`."""
        return super().getBiasEnergy(_extensionContext(context))

    def updateContext(self, context: mm.Context) -> None:
        """Push the current bias into an :class:`~openxps.ExtendedSpaceContext`."""
        super().updateContext(_extensionContext(context))


class ExtendedSpaceCombinedOPES(openmm_opes.CombinedOPES):
    """OPES and OPES-explore combined (OPES_COM) on dynamical variables.

    This is :OpenMMOPES:`CombinedOPES <combined-opes>` acting on the extension
    system of an :class:`~openxps.ExtendedSpaceSystem`: both components deposit
    kernels on the same dynamical variables, and the total bias is their sum. See
    :class:`ExtendedSpaceOPES` for how the variables are handled; the same notes
    on kernels and ``extraBias`` apply.

    Parameters
    ----------
    system
        The :class:`~openxps.ExtendedSpaceSystem` to be simulated. The bias force
        is added to its extension system when this object is created, so it must
        be created before the simulation or context that uses the system.
    variables
        The dynamical variables to bias and their kernel widths, as
        :class:`~openxps.ExtendedSpaceBiasVariable` objects.
    **kwargs
        The remaining arguments of :OpenMMOPES:`CombinedOPES <combined-opes>`
        (``temperature``, ``barrier``, ``exploreBarrier``, ``frequency``,
        ``varianceFrequency``, ...), except ``bounded``, which follows from the
        variables' bounds.

    Example
    -------
    >>> import openxps as xps
    >>> from openmm import unit
    >>> import cvpack
    >>> import openmm
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> mass = 3 * unit.dalton*(unit.nanometer/unit.radian)**2
    >>> phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    >>> kappa = 1000 * unit.kilojoules_per_mole / unit.radian**2
    >>> system = xps.ExtendedSpaceSystem(
    ...     model.system, xps.HarmonicCoupling(phi, phi0, kappa)
    ... )
    >>> opes = xps.ExtendedSpaceCombinedOPES(
    ...     system,
    ...     [xps.ExtendedSpaceBiasVariable(phi0, 18 * unit.degrees)],
    ...     temperature=300 * unit.kelvin,
    ...     barrier=20 * unit.kilojoules_per_mole,
    ...     exploreBarrier=10 * unit.kilojoules_per_mole,
    ...     frequency=100,
    ...     varianceFrequency=None,
    ... )
    >>> integrator = openmm.LangevinMiddleIntegrator(
    ...     300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond
    ... )
    >>> integrator.setRandomNumberSeed(1234)
    >>> simulation = xps.ExtendedSpaceSimulation(
    ...     model.topology,
    ...     system,
    ...     xps.LockstepIntegrator(integrator),
    ...     openmm.Platform.getPlatformByName("Reference"),
    ... )
    >>> context = simulation.context
    >>> context.setPositions(model.positions)
    >>> context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    >>> context.setDynamicalVariableValues([180 * unit.degree])
    >>> context.setDynamicalVariableVelocitiesToTemperature(300 * unit.kelvin, 1234)
    >>> opes.step(simulation, 200)
    >>> opes.getNumKernels()  # each component merged its second kernel
    (1, 1)
    """

    def __init__(
        self,
        system: ExtendedSpaceSystem,
        variables: t.Sequence[ExtendedSpaceBiasVariable],
        **kwargs,
    ) -> None:
        bias_variables, bounded = _biasVariables(system, variables, kwargs)
        super().__init__(
            system.getExtensionSystem(), bias_variables, bounded=bounded, **kwargs
        )

    def step(self, simulation, steps: int) -> None:
        """Advance an :class:`~openxps.ExtendedSpaceSimulation` by ``steps`` steps."""
        super().step(_extensionSimulation(simulation), steps)

    stepTogether = staticmethod(ExtendedSpaceOPES.stepTogether)

    def getCollectiveVariables(self, simulation):
        """Current values of the biased dynamical variables, in MD units."""
        return super().getCollectiveVariables(_extensionSimulation(simulation))

    def getBiasEnergy(self, context: mm.Context):
        """The total bias energy in an :class:`~openxps.ExtendedSpaceContext`."""
        return super().getBiasEnergy(_extensionContext(context))

    def getComponentBiasEnergies(self, context: mm.Context):
        """The OPES and OPES-explore bias energies in an
        :class:`~openxps.ExtendedSpaceContext`."""
        return super().getComponentBiasEnergies(_extensionContext(context))

    def updateContext(self, context: mm.Context) -> None:
        """Push the current biases into an :class:`~openxps.ExtendedSpaceContext`."""
        super().updateContext(_extensionContext(context))
