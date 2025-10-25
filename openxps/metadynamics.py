"""
.. module:: openxps.metadynamics
   :platform: Linux, Windows, macOS
   :synopsis: Bias potentials applied to dynamical variables.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from dataclasses import dataclass

from cvpack.serialization import Serializable
from openmm import app as mmapp
from openmm import unit as mmunit

from .bounds import NoBounds
from .dynamical_variable import DynamicalVariable
from .system import ExtendedSpaceSystem
from .utils import preprocess_args


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
    A dynamical variable augmented with a Gaussian bias kernel.

    Parameters
    ----------
    dynamical_variable
        The dynamical variable to be augmented.
    sigma
        The bandwidth (standard deviation) of the Gaussian bias kernel.
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
    ...     xps.bounds.Periodic(-180, 180, unit.degree)
    ... )
    >>> bias_variable = xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)
    >>> bias_variable.dynamical_variable
    DynamicalVariable(name='phi', unit=rad, mass=3 nm**2 Da/(rad**2), bounds=...)
    """

    dynamical_variable: DynamicalVariable
    sigma: mmunit.Quantity
    grid_width: t.Optional[int] = None

    def __post_init__(self) -> None:
        if not mmunit.is_quantity(self.sigma):
            raise TypeError("sigma must be a quantity")
        if not self.sigma.unit.is_compatible(self.dynamical_variable.unit):
            raise ValueError(
                "sigma must be compatible with the dynamical variable's unit"
            )
        if not (self.grid_width is None or isinstance(self.grid_width, int)):
            raise TypeError("grid_width must be an integer or None")


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
    >>> phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.bounds.CIRCULAR)
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> kappa = 1000 * unit.kilojoules_per_mole / unit.radian**2
    >>> harmonic_force = xps.HarmonicCouplingForce(phi, phi0, kappa)
    >>> system = xps.ExtendedSpaceSystem([phi0], harmonic_force, model.system)
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
    >>> metadynamics.getFreeEnergy()
    [... ... ...] kJ/mol
    """

    def __init__(  # noqa: PLR0913
        self,
        system: ExtendedSpaceSystem,
        variables: t.Sequence[ExtendedSpaceBiasVariable],
        temperature: mmunit.Quantity,
        biasFactor: float,
        height: mmunit.Quantity,
        frequency: int,
        saveFrequency: t.Optional[int] = None,
        biasDir: t.Optional[str] = None,
    ) -> None:
        system_dvs = system.getDynamicalVariables()
        bias_variables = []
        for variable in variables:
            dv = variable.dynamical_variable.in_md_units()
            if dv not in system_dvs:
                raise ValueError(f"Dynamical variable {dv.name} not found in system")
            if isinstance(dv.bounds, NoBounds):
                raise ValueError(f"Dynamical variable {dv.name} has no bounds")
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

    def step(self, simulation, steps):
        super().step(_SimulationWrapper(simulation), steps)

    def getCollectiveVariables(self, simulation):
        return self._force.getCollectiveVariableValues(
            simulation.context.getExtensionContext()
        )
