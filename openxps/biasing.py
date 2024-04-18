"""
.. module:: openxps.biasing
   :platform: Linux, Windows, macOS
   :synopsis: Biasing potentials applied to extra degrees of freedom.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import cvpack
import numpy as np
import openmm as mm
from openmm import unit as mmunit

from .extra_dof import ExtraDOF


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

    def addKernel(
        self,
        context: mm.Context,
        height: mmunit.Quantity,
        bandwidths: t.Sequence[mmunit.Quantity],
        centers: t.Sequence[mmunit.Quantity],
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
        centers
            The centers of the kernel.
        """
        raise NotImplementedError(
            "Adding kernels to bias potentials is not supported by this class"
        )

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
        if not (0 < num_dims < 4 and len(grid_sizes) == num_dims):
            raise ValueError("Ill defined interpolation grid")
        self._limits = []
        for xdof in extra_dofs:
            if xdof.bounds is None:
                raise ValueError("Interpolation requires bounded extra DOFs")
            self._limits.extend([xdof.bounds.lower, xdof.bounds.upper])
        self._sizes = [] if num_dims == 1 else grid_sizes
        num_periodics = sum(xdof.isPeriodic() for xdof in extra_dofs)
        if num_periodics not in [0, num_dims]:
            raise ValueError("Mixed periodic/non-periodic dimensions are not supported")
        self._bias = np.zeros(grid_sizes, dtype=np.float64)
        self.this = getattr(mm, f"Continuous{num_dims}DFunction")(
            *self._sizes, self._bias.flatten(), *self._limits, num_periodics == num_dims
        ).this


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
        num_dims = len(extra_dofs)
        if not 1 <= num_dims <= 3:
            raise ValueError("Spline biasing potential requires 1 to 3 extra DOFs")
        if len(grid_sizes) != num_dims:
            raise ValueError("Grid sizes must match the number of extra DOFs")
        function = f"bias({','.join(xdof.name for xdof in extra_dofs)})"
        super().__init__(extra_dofs, function)
        self._bias_grid = SplineGrid(extra_dofs, grid_sizes)
        self.addTabulatedFunction("bias", self._bias_grid)

    def addKernel(
        self,
        context: mm.Context,
        height: mmunit.Quantity,
        bandwidths: t.Sequence[mmunit.Quantity],
        centers: t.Sequence[mmunit.Quantity],
    ) -> None:
        pass


class AdaptiveBiasingPotential(BiasingPotential):
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
    ) -> None:
        summands = []
        for xdof in extra_dofs:
            distance = f"({xdof.name}-{xdof.name}_center)"
            if xdof.isPeriodic():
                distance = f"(sin(pi*{distance})/pi)"  # von Mises kernel
            summands.append(f"({distance}/{xdof.name}_sigma)^2")
        function = f"height*exp(-0.5*({'+'.join(summands)}))"
        if any(xdof.isPeriodic() for xdof in extra_dofs):
            function += f";pi={np.pi}"
        super().__init__(extra_dofs, function)
        self.addPerBondParameter("height")
        for xdof in extra_dofs:
            self.addPerBondParameter(f"{xdof.name}_center")
        for xdof in extra_dofs:
            self.addPerBondParameter(f"{xdof.name}_sigma")

    def addKernel(
        self,
        context: mm.Context,
        height: mmunit.Quantity,
        bandwidths: t.Sequence[mmunit.Quantity],
        centers: t.Sequence[mmunit.Quantity],
    ) -> None:
        pass
