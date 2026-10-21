"""
.. module:: openxps.metadynamics
   :platform: Linux, Windows, macOS
   :synopsis: Bias potentials applied to dynamical variables.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from dataclasses import dataclass, field
from functools import reduce

import numpy as np
from cvpack.serialization import Serializable
from openmm import app as mmapp
from openmm import unit as mmunit

from .bounds import NoBounds, PeriodicBounds
from .dynamical_variable import DynamicalVariable
from .system import ExtendedSpaceSystem
from .utils import preprocess_args


def _von_mises(displacement: np.ndarray, sigma: float, period: float) -> np.ndarray:
    """A Gaussian in the chord distance ``(period/pi)*sin(pi*displacement/period)``.

    It is smooth and periodic, and matches a Gaussian of width ``sigma`` when the
    displacement is much smaller than the period.
    """
    factor = 2 * np.pi / period
    return np.exp((np.cos(factor * displacement) - 1) / (factor * sigma) ** 2)


def _axis_kernel(
    grid: np.ndarray,
    center: float,
    sigma: float,
    bounds: str,
    lower: float,
    upper: float,
) -> np.ndarray:
    """The bias kernel along one axis, on the given grid points.

    Periodic variables get a von Mises kernel. Reflective variables get the sum of
    all mirror images of a Gaussian across both walls, which are images of period
    ``2*(upper - lower)`` centered at ``center`` and ``2*lower - center``. Two von
    Mises kernels of that period therefore account for every image, and the kernel
    has zero slope at both walls.
    """
    if bounds == PeriodicBounds.kind:
        return _von_mises(grid - center, sigma, upper - lower)
    period = 2 * (upper - lower)
    return _von_mises(grid - center, sigma, period) + _von_mises(
        grid + center - 2 * lower, sigma, period
    )


class _SimulationWrapper:
    def __init__(self, simulation: mmapp.Simulation):
        self._simulation = simulation
        self.context = simulation.context.getExtensionContext()
        self.step = simulation.step

    @property
    def currentStep(self):
        return self._simulation.currentStep


@dataclass(frozen=True)
class ExtendedSpaceBiasVariable(Serializable):
    """
    A dynamical variable augmented with a bias kernel.

    In :class:`ExtendedSpaceMetadynamics`, the kernel shape depends on the bounds of
    the dynamical variable: a von Mises kernel for :class:`PeriodicBounds` and a
    Gaussian mirrored across both walls for :class:`ReflectiveBounds`. In
    :class:`~openxps.ExtendedSpaceOPES` and :class:`~openxps.ExtendedSpaceCombinedOPES`,
    the kernels are those of openmm-opes, and ``sigma`` is the standard deviation of
    the unbiased distribution of the variable.

    Parameters
    ----------
    dynamical_variable
        The dynamical variable to be augmented.
    sigma
        The bandwidth of the bias kernel, which approximates the standard deviation
        of a Gaussian when it is small relative to the bounds.
    grid_width
        The width of the grid for the bias kernel. If ``None``, the grid width is
        automatically determined based on the dynamical variable's bounds and the
        bandwidth.

    Example
    -------
    >>> import openxps as xps
    >>> from openmm import unit
    >>> dv = xps.DynamicalVariable(
    ...     "phi",
    ...     unit.radian,
    ...     3 * unit.dalton*(unit.nanometer/unit.radian)**2,
    ...     xps.PeriodicBounds(-180, 180, unit.degree)
    ... )
    >>> bias_variable = xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)
    >>> bias_variable.dynamical_variable
    DynamicalVariable(name='phi', unit=rad, mass=3 nm**2 Da/(rad**2), bounds=...)
    """

    dynamical_variable: DynamicalVariable
    sigma: mmunit.Quantity
    grid_width: int | None = field(default_factory=lambda: None)

    def __post_init__(self) -> None:
        if not mmunit.is_quantity(self.sigma):
            raise TypeError("sigma must be a quantity")
        if not self.sigma.unit.is_compatible(self.dynamical_variable.unit):
            raise ValueError(
                "sigma must be compatible with the dynamical variable's unit"
            )
        if self.sigma / self.sigma.unit <= 0:
            raise ValueError("sigma must be positive")
        if not (self.grid_width is None or isinstance(self.grid_width, int)):
            raise TypeError("grid_width must be an integer or None")
        if self.grid_width is not None and self.grid_width <= 1:
            raise ValueError("grid_width must be greater than 1")


ExtendedSpaceBiasVariable.__init__ = preprocess_args(ExtendedSpaceBiasVariable.__init__)

ExtendedSpaceBiasVariable.registerTag("!openxps.metadynamics.ExtendedSpaceBiasVariable")


class ExtendedSpaceMetadynamics(mmapp.Metadynamics):
    """
    Performs Extended Phase-Space (XPS) Metadynamics simulations, in which a bias
    potential is applied to dynamical variables (DVs) to enhance sampling of the
    physical coordinates.

    Parameters
    ----------
    system
        The :class:`ExtendedSpaceSystem` to be used in the XPS simulation.
    variables
        A sequence of :class:`ExtendedSpaceBiasVariable` objects to specify the biased
        dynamical variables and their corresponding Gaussian kernels.
    temperature
        The temperature at which the simulation is being run. This is used in computing
        the free energy.
    biasFactor
        Used in scaling the height of the Gaussians added to the bias. The dynamical
        variables are sampled as if the effective temperature of the simulation were
        temperature*biasFactor.
    height
        The initial height of the Gaussians to add (in units of energy).
    frequency
        The interval in time steps at which Gaussians should be added to the bias
        potential.
    saveFrequency
        The interval in time steps at which to write out the current biases to disk. At
        the same time it writes biases, it also checks for updated biases written by
        other processes and loads them in. This must be a multiple of frequency.
    biasDir
        The directory to which biases should be written, and from which biases written
        by other processes should be loaded.

    .. note::
        The bias is stored on a grid spanning the bounds of each dynamical variable,
        so variables with :class:`NoBounds` are rejected. Use
        :class:`ReflectiveBounds` over the range the variable may explore instead.
        If the variable never reaches the walls, the reflections never act, and if
        it does, the coupling turns them into soft walls on the physical system.

    .. note::
        Periodic and reflective variables can be mixed. A reflective variable is
        tabulated as a periodic one of twice its range, holding the bias and its
        mirror image, which makes the interpolated bias exactly flat at the walls.
        This doubles the table along each reflective axis.

    .. note::
        The bias force is added to the extension system of ``system`` when this
        object is created, so it must be created before the
        :class:`~openxps.ExtendedSpaceSimulation` or
        :class:`~openxps.ExtendedSpaceContext` that uses the system.

    Example
    -------
    >>> import openxps as xps
    >>> from openmm import unit
    >>> import cvpack
    >>> from math import pi
    >>> import openmm
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> mass = 3 * unit.dalton*(unit.nanometer/unit.radian)**2
    >>> phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> kappa = 1000 * unit.kilojoules_per_mole / unit.radian**2
    >>> harmonic_force = xps.HarmonicCoupling(phi, phi0, kappa)
    >>> system = xps.ExtendedSpaceSystem(model.system, harmonic_force)
    >>> bias_variable = xps.ExtendedSpaceBiasVariable(phi0, 18 * unit.degrees)
    >>> temperature = 300 * unit.kelvin
    >>> metadynamics = xps.ExtendedSpaceMetadynamics(
    ...     system=system,
    ...     variables=[bias_variable],
    ...     temperature=temperature,
    ...     biasFactor=5,
    ...     height=2 * unit.kilojoule_per_mole,
    ...     frequency=100,
    ... )
    >>> simulation = xps.ExtendedSpaceSimulation(
    ...     model.topology,
    ...     system,
    ...     xps.LockstepIntegrator(
    ...         openmm.LangevinMiddleIntegrator(
    ...             temperature, 1 / unit.picosecond, 4 * unit.femtosecond
    ...         )
    ...     ),
    ...     openmm.Platform.getPlatformByName("Reference"),
    ... )
    >>> context = simulation.context
    >>> context.setPositions(model.positions)
    >>> context.setVelocitiesToTemperature(temperature, 1234)
    >>> context.setDynamicalVariableValues([180 * unit.degree])
    >>> context.setDynamicalVariableVelocitiesToTemperature(temperature, 1234)
    >>> metadynamics.step(simulation, 100)
    >>> print(metadynamics.getFreeEnergy())
    [...] kJ/mol
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        system: ExtendedSpaceSystem,
        variables: t.Sequence[ExtendedSpaceBiasVariable],
        temperature: mmunit.Quantity,
        biasFactor: float,
        height: mmunit.Quantity,
        frequency: int,
        saveFrequency: int | None = None,
        biasDir: str | None = None,
    ) -> None:
        system_dvs = system.getDynamicalVariables()
        bias_variables = []
        for variable in variables:
            dv = variable.dynamical_variable.in_md_units()
            if dv not in system_dvs:
                raise ValueError(f"Dynamical variable {dv.name} not found in system")
            if isinstance(dv.bounds, NoBounds):
                raise ValueError(
                    f"Dynamical variable {dv.name} has no bounds, but metadynamics "
                    "needs a finite grid. Use ReflectiveBounds over the range the "
                    "variable may explore."
                )
            index = system_dvs.index(dv)
            bias_variables.append(
                mmapp.BiasVariable(
                    dv.createCollectiveVariable(index),
                    dv.bounds.lower * dv.bounds.unit,
                    dv.bounds.upper * dv.bounds.unit,
                    variable.sigma,
                    True,
                    variable.grid_width,
                )
            )
        # Every table axis is periodic, so OpenMM must not reject the mix of periodic
        # and reflective variables. Their true periodicity is restored afterwards.
        super().__init__(
            system.getExtensionSystem(),
            bias_variables,
            temperature,
            biasFactor,
            height,
            frequency,
            saveFrequency,
            biasDir,
        )
        self._bounds_kinds = [
            variable.dynamical_variable.bounds.kind for variable in variables
        ]
        for v, kind in zip(self.variables, self._bounds_kinds):
            v.periodic = kind == PeriodicBounds.kind
        self._widths = [
            v.gridWidth if v.periodic else 2 * v.gridWidth - 1 for v in self.variables
        ]
        self._limits = []
        for v in self.variables:
            upper = v.maxValue if v.periodic else 2 * v.maxValue - v.minValue
            self._limits += [v.minValue, upper]
        self._updateTable()

    def step(self, simulation, steps):
        super().step(_SimulationWrapper(simulation), steps)

    def getCollectiveVariables(self, simulation):
        return self._force.getCollectiveVariableValues(
            simulation.context.getExtensionContext()
        )

    def _addGaussian(self, position, height, context):
        """Add a kernel to the bias, overriding the truncated Gaussian of OpenMM."""
        axis_kernels = []
        for value, variable, kind in zip(position, self.variables, self._bounds_kinds):
            lower, upper = variable.minValue, variable.maxValue
            grid = np.linspace(lower, upper, variable.gridWidth)
            kernel = _axis_kernel(grid, value, variable.biasWidth, kind, lower, upper)
            if kind == PeriodicBounds.kind:
                kernel[-1] = kernel[0]
            axis_kernels.append(kernel)
        kernel = reduce(np.multiply.outer, reversed(axis_kernels))
        height = height.value_in_unit(mmunit.kilojoules_per_mole)
        self._selfBias += height * kernel
        self._totalBias += height * kernel
        self._updateTable()
        self._force.updateParametersInContext(context)

    def _updateTable(self):
        """Copy the total bias into the periodic table.

        A reflective variable on ``[lower, upper]`` behaves as a periodic one of period
        ``2*(upper - lower)`` whose bias is mirrored about ``upper``. Its table axis
        holds the bias followed by its mirror image, so the periodic spline that
        OpenMM fits is symmetric about both walls and has zero slope there.
        """
        bias = self._totalBias
        # The bias array stores the last variable along its first axis.
        for axis, v in enumerate(reversed(self.variables)):
            if not v.periodic:
                mirror = np.flip(np.take(bias, range(v.gridWidth - 1), axis=axis), axis)
                bias = np.concatenate((bias, mirror), axis=axis)
        if len(self.variables) == 1:
            self._table.setFunctionParameters(bias.flatten(), *self._limits)
        else:
            self._table.setFunctionParameters(
                *self._widths, bias.flatten(), *self._limits
            )
