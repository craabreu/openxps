"""
.. module:: openxps.biasing
   :platform: Linux, Windows, macOS
   :synopsis: Biasing potentials applied to extra degrees of freedom.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import functools
import typing as t
from collections import namedtuple

import cvpack
import numpy as np
import openmm as mm
from openmm import unit as mmunit

from .extra_dof import ExtraDOF

KernelInfo = namedtuple("KernelInfo", ["height", "bandwidths", "center"])


class BiasingPotential(cvpack.OpenMMForceWrapper):
    """
    Abstract class for bias potentials applied to extra degrees of freedom.

    Parameters
    ----------
    extra_dofs
        The extra degrees of freedom subject to the bias potential.
    function
        The mathematical expression defining the bias potential.
    """

    def __init__(
        self,
        extra_dofs: t.Sequence[ExtraDOF],
        function: str,
    ) -> None:
        self._extra_dofs = tuple(extra_dofs)
        self._extra_dof_indices = None
        self._kernel_info = []
        expression = function
        for index, xdof in enumerate(extra_dofs, start=1):
            variable = f"x{index}"
            if xdof.bounds is not None:
                variable = xdof.bounds.leptonExpression(variable)
            expression += f";{xdof.name}={variable}"
        super().__init__(
            mm.CustomCompoundBondForce(len(extra_dofs), expression),
            mmunit.kilojoules_per_mole,
            name="biasing_potential",
        )

    def _performAddKernel(
        self,
        context: mm.Context,
        height: float,
        bandwidths: t.Sequence[float],
        center: t.Sequence[float],
    ) -> None:
        raise NotImplementedError(
            "Method _performAddKernel must be implemented by derived classes"
        )

    def addKernel(
        self,
        context: mm.Context,
        height: mmunit.Quantity,
        bandwidths: t.Sequence[mmunit.Quantity],
        center: t.Sequence[mmunit.Quantity],
    ) -> None:
        """
        Add a Gaussian kernel to the bias potential.

        Parameters
        ----------
        context
            The extension context where the kernel will be added.
        height
            The height of the kernel.
        bandwidths
            The bandwidths of the kernel.
        center
            The center of the kernel.
        """
        if self._extra_dof_indices is None:
            raise ValueError("Bias potential has not been initialized")
        height = height.value_in_unit(mmunit.kilojoule_per_mole)
        bandwidths = [
            quantity.value_in_unit(xdof.unit)
            for quantity, xdof in zip(bandwidths, self._extra_dofs)
        ]
        center = [
            quantity.value_in_unit(xdof.unit)
            for quantity, xdof in zip(center, self._extra_dofs)
        ]
        self._kernel_info.append(KernelInfo(height, bandwidths, center))
        self._performAddKernel(context, height, bandwidths, center)

    def getExtraDOFs(self) -> t.Tuple[ExtraDOF]:
        """
        Get the extra degrees of freedom subject to the bias potential.

        Returns
        -------
        Tuple[ExtraDOF]
            The extra degrees of freedom subject to the bias potential.
        """
        return self._extra_dofs

    def initialize(self, context_extra_dofs: t.Sequence[ExtraDOF]) -> None:
        """
        Initialize the bias potential for a specific context.

        Parameters
        ----------
        context_extra_dofs
            The extra degrees of freedom in the context.
        """
        try:
            self._extra_dof_indices = tuple(
                map(context_extra_dofs.index, self._extra_dofs)
            )
        except ValueError as error:
            raise ValueError(
                "Not all extra DOFs in the bias potential are present in the context"
            ) from error


class SplineGrid(mm.TabulatedFunction):
    """
    A grid used to interpolate a bias potential applied to extra degrees of freedom.

    Parameters
    ----------
    extra_dofs
        The extra degrees of freedom subject to the bias potential.
    grid_sizes
        The number of points in each dimension of the grid used to interpolate the bias
        potential.

    Raises
    ------
    ValueError
        If the length of ``extra_dofs`` is not between 1 and 3, if ``grid_sizes`` has
        a different length than ``extra_dofs``, if any extra degree of freedom is not
        bounded, or if the dimensions of the grid are not all periodic or all
        non-periodic.

    Examples
    --------
    Create a 2D grid to interpolate a bias potential applied to a single extra degree
    of freedom:
    >>> import openxps as xps
    >>> from openmm import unit
    >>> import numpy as np
    >>> mass = 1.0 * unit.dalton
    >>> bounds = xps.bounds.Reflective(0, 1, unit.nanometers)
    >>> extra_dof = xps.ExtraDOF("x", unit.nanometers, mass, bounds)
    >>> grid = xps.biasing.SplineGrid([extra_dof], [10])
    """

    def __init__(  # pylint: disable=super-init-not-called
        self, extra_dofs: t.Sequence[ExtraDOF], grid_sizes: t.Sequence[int]
    ) -> None:
        num_dims = len(extra_dofs)
        if not 1 <= num_dims <= 3:
            raise ValueError("Spline grid can only interpolate 1D, 2D, or 3D functions")
        if len(grid_sizes) != num_dims:
            raise ValueError("Number of grid sizes must match number of extra DOFs")
        if any(xdof.bounds is None for xdof in extra_dofs):
            raise ValueError("Spline interpolation requires bounded extra DOFs")

        self._grid_sizes = np.array(grid_sizes)
        self._grid = [
            np.linspace(xdof.bounds.lower, xdof.bounds.upper, num=grid_size)
            for xdof, grid_size in zip(extra_dofs, grid_sizes)
        ]

        num_periodics = sum(xdof.isPeriodic() for xdof in extra_dofs)
        if num_periodics not in [0, num_dims]:
            raise ValueError("Mixed periodic/non-periodic dimensions are not supported")

        self._widths = [] if num_dims == 1 else grid_sizes
        self._bias = np.zeros(grid_sizes)
        self._limits = []
        for xdof in extra_dofs:
            if xdof.bounds is None:
                raise ValueError("Interpolation requires bounded extra DOFs")
            self._limits.extend([xdof.bounds.lower, xdof.bounds.upper])
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


class SplineBiasingPotential(BiasingPotential):
    """
    A bias potential applied to extra degrees of freedom using a spline grid.

    Parameters
    ----------
    extra_dofs
        The extra degrees of freedom subject to the bias potential.
    grid_sizes
        The number of points in each dimension of the grid used to interpolate the bias
        potential.
    """

    def __init__(
        self,
        extra_dofs: t.Sequence[ExtraDOF],
        grid_sizes: t.Sequence[int],
    ) -> None:
        function = f"bias({','.join(xdof.name for xdof in extra_dofs)})"
        super().__init__(extra_dofs, function)
        self._bias_grid = SplineGrid(extra_dofs, grid_sizes)
        self.addTabulatedFunction("bias", self._bias_grid)

    def _performAddKernel(
        self,
        context: mm.Context,
        height: float,
        bandwidths: t.Sequence[float],
        center: t.Sequence[float],
    ) -> None:
        exponents = []
        for i, xdof in enumerate(self._extra_dofs):
            distances = self._bias_grid.getGridPoints(i) - center[i]
            if xdof.isPeriodic():
                length = xdof.bounds.upper - xdof.bounds.lower
                distances = (length / np.pi) * np.sin((np.pi / length) * distances)
                distances[-1] = distances[0]
            exponents.append(-0.5 * (distances / bandwidths[i]) ** 2)
        kernel = height * np.exp(
            exponents[0]
            if len(self._extra_dofs) == 1
            else functools.reduce(np.add.outer, reversed(exponents))
        )
        self._bias_grid.addBias(kernel)
        self.updateParametersInContext(context)

    def initialize(self, context_extra_dofs: t.Sequence[ExtraDOF]) -> None:
        super().initialize(context_extra_dofs)
        self.addBond(self._extra_dof_indices, [])

    def getBias(self) -> np.ndarray:
        """
        Get the bias values on a spline grid.

        Returns
        -------
        np.ndarray
            The bias values on the spline grid.
        """
        return self._bias_grid.getBias()
