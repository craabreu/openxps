"""
.. module:: openxps.metadynamics
   :platform: Linux, Windows, macOS
   :synopsis: Bias potentials applied to dynamical variables.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import functools
import typing as t

import numpy as np
import openmm as mm
from openmm import unit as mmunit

from .bias_potential import BiasPotential
from .dynamical_variable import DynamicalVariable

MAX_DIMENSIONS = 3


class MetadynamicsBias(BiasPotential):
    r"""
    A Metadynamics potential applied to dynamical variables.

    .. math::

        V({\bf x}) = \sum_{i=1}^{N} h_i \exp\left\{
            -\frac{1}{2} \left(
                {\bf x} - {\bf c}_i
            \right)^T [{\bf D}({\boldsymbol \sigma}_i)]^{-2} \left(
                {\bf x} - {\bf c}_i
            \right)
        \right\}

    where :math:`{\bf x}` is the vector of dynamical variables, :math:`h_i` is
    the height of the Gaussian kernel, :math:`{\bf c}_i` is the center of the kernel,
    :math:`{\boldsymbol \sigma}_i` is the bandwidth vector of the kernel, and
    :math:`{\bf D}(\cdot)` builds a diagonal matrix from a vector.

    The bias potential can be interpolated on a grid using natural splines for 1D, 2D,
    or 3D functions. This is done by providing the number of points in each dimension
    of the grid. If the grid is not provided, the bias potential is evaluated directly
    from the mathematical expression, which gets more expensive as the number of added
    kernels increases.

    Parameters
    ----------
    dynamical_variables
        The dynamical variables subject to the bias potential.
    bandwidth
        The bandwidth vector of the Gaussian kernel. Each element must have units of
        the corresponding dynamical variable.
    initial_height
        The initial height of the Gaussian kernels. It must have units of molar energy.
    temperature
        The temperature of the system (with units).
    bias_factor
        The bias factor that controls the equilibrium bias potential and the rate of
        convergence. It must be larger than 1.
    grid_sizes
        The number of points in each dimension of the grid used to interpolate the bias
        potential.

    Raises
    ------
    ValueError
        If the bias factor is not larger than 1 or if temperature does not have valid
        units.
    """

    def __init__(  # noqa: PLR0913
        self,
        dynamical_variables: t.Sequence[DynamicalVariable],
        bandwidth: t.Sequence[mmunit.Quantity],
        initial_height: mmunit.Quantity,
        temperature: mmunit.Quantity,
        bias_factor: float,
        grid_sizes: t.Union[t.Sequence[int], None],
    ) -> None:
        if bias_factor <= 1:
            raise ValueError("bias_factor must be a float larger than 1")
        self._bandwidth = [
            quantity.value_in_unit(dv.unit)
            for quantity, dv in zip(bandwidth, dynamical_variables)
        ]
        self._initial_height, self._kb_delta_t = [
            quantity.value_in_unit(mmunit.kilojoules_per_mole)
            for quantity in (
                initial_height,
                mmunit.MOLAR_GAS_CONSTANT_R * temperature * (bias_factor - 1),
            )
        ]
        if grid_sizes is None:
            summands = []
            distances = []
            for dv, sigma in zip(dynamical_variables, self._bandwidth):
                distance = f"{dv.name}-{dv.name}_center"
                if dv.isPeriodic():
                    length = dv.bounds.period
                    distance = f"{length / np.pi}*sin({np.pi / length}*({distance}))"
                distances.append(distance)
                summands.append(f"({dv.name}_dist/{sigma})^2")
            function = f"height*exp(-0.5*({'+'.join(summands)}))"
            for dv, distance in zip(dynamical_variables, distances):
                function += f";\n{dv.name}_dist={distance}"
        else:
            function = f"bias({','.join(dv.name for dv in dynamical_variables)})"
        super().__init__(
            dynamical_variables,
            function,
            functools.partial(mm.CustomCompoundBondForce, len(dynamical_variables)),
        )
        if grid_sizes is None:
            self._bias_grid = None
            self.addPerBondParameter("height")
            for dv in dynamical_variables:
                self.addPerBondParameter(f"{dv.name}_center")
        else:
            self._bias_grid = SplineGrid(dynamical_variables, grid_sizes)
            self.addTabulatedFunction("bias", self._bias_grid)

    def _performAddKernel(
        self, context: mm.Context, center: t.Sequence[float], potential: float
    ) -> None:
        height = self._initial_height * np.exp(-potential / self._kb_delta_t)
        if self._bias_grid is None:
            self.addBond(self._dv_indices, [height, *center])
            context.reinitialize(preserveState=True)
        else:
            exponents = []
            for i, dv in enumerate(self._dvs):
                distances = self._bias_grid.getGridPoints(i) - center[i]
                if dv.isPeriodic():
                    length = dv.bounds.upper - dv.bounds.lower
                    distances = (length / np.pi) * np.sin((np.pi / length) * distances)
                    distances[-1] = distances[0]
                exponents.append(-0.5 * (distances / self._bandwidth[i]) ** 2)
            kernel = height * np.exp(
                exponents[0]
                if len(self._dvs) == 1
                else functools.reduce(np.add.outer, reversed(exponents))
            )
            self._bias_grid.addBias(kernel)
            self.updateParametersInContext(context)

    def initialize(self, context_dvs: t.Sequence[DynamicalVariable]) -> None:
        super().initialize(context_dvs)
        if self._bias_grid is not None:
            self.addBond(self._dv_indices, [])

    def getBias(self) -> np.ndarray:
        """
        Get the bias values on a spline grid.

        Returns
        -------
        np.ndarray
            The bias values on the spline grid.
        """
        return self._bias_grid.getBias()


class SplineGrid(mm.TabulatedFunction):
    """
    A grid used to interpolate a bias potential applied to dynamical variables.

    Parameters
    ----------
    dynamical_variables
        The dynamical variables subject to the bias potential.
    grid_sizes
        The number of points in each dimension of the grid used to interpolate the bias
        potential.

    Raises
    ------
    ValueError
        If the length of ``dynamical_variables`` is not between 1 and 3, if
        ``grid_sizes`` has a different length than ``dynamical_variables``, if any
        dynamical variable is not bounded, or if the dimensions of the grid are not
        all periodic or all non-periodic.

    Examples
    --------
    Create a 2D grid to interpolate a bias potential applied to a single extra degree
    of freedom:
    >>> import openxps as xps
    >>> from openmm import unit
    >>> import numpy as np
    >>> mass = 1.0 * unit.dalton
    >>> bounds = xps.bounds.Reflective(0, 1, unit.nanometers)
    >>> dynamical_variable = xps.DynamicalVariable("x", unit.nanometers, mass, bounds)
    >>> grid = xps.metadynamics.SplineGrid([dynamical_variable], [10])
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        dynamical_variables: t.Sequence[DynamicalVariable],
        grid_sizes: t.Sequence[int],
    ) -> None:
        num_dims = len(dynamical_variables)
        if not 1 <= num_dims <= MAX_DIMENSIONS:
            raise ValueError(
                f"Spline grid can only interpolate up to {MAX_DIMENSIONS}D functions"
            )
        if len(grid_sizes) != num_dims:
            raise ValueError("Number of grid sizes must match number of DVs")
        if any(dv.bounds is None for dv in dynamical_variables):
            raise ValueError("Spline interpolation requires bounded DVs")

        self._grid_sizes = np.array(grid_sizes)
        self._grid = [
            np.linspace(dv.bounds.lower, dv.bounds.upper, num=grid_size)
            for dv, grid_size in zip(dynamical_variables, grid_sizes)
        ]

        num_periodics = sum(dv.isPeriodic() for dv in dynamical_variables)
        if num_periodics not in [0, num_dims]:
            raise ValueError("Mixed periodic/non-periodic dimensions are not supported")

        self._widths = [] if num_dims == 1 else grid_sizes
        self._bias = np.zeros(grid_sizes)
        self._limits = []
        for dv in dynamical_variables:
            if dv.bounds is None:
                raise ValueError("Interpolation requires bounded DVs")
            self._limits.extend([dv.bounds.lower, dv.bounds.upper])
        periodic = num_periodics == num_dims
        self._table = getattr(mm, f"Continuous{num_dims}DFunction")(
            *self._widths, self._bias.flatten(), *self._limits, periodic
        )
        self.this = self._table.this

    def getGridPoints(self, dimension: int) -> np.ndarray:
        """
        Get the grid points along a specific dimension.

        Parameters
        ----------
        dimension
            The dimension for which the grid points will be returned.

        Returns
        -------
        np.ndarray
            The grid points along the specified dimension.
        """
        return self._grid[dimension]

    def getBias(self) -> np.ndarray:
        """
        Get the bias values on the spline grid.

        Returns
        -------
        np.ndarray
            The bias values on the spline grid.
        """
        return self._bias

    def addBias(self, bias: np.ndarray) -> None:
        """
        Add bias values to the grid.

        Parameters
        ----------
        bias
            The bias values to be added to the grid.
        """
        self._bias += bias
        self._table.setFunctionParameters(
            *self._widths, self._bias.flatten(), *self._limits
        )
