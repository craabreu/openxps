"""
.. module:: openxps.spline_grid
   :platform: Linux, Windows, macOS
   :synopsis: Grid to interpolate bias potentials applied to extra degrees of freedom.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import numpy as np
import openmm as mm

from .extra_dof import ExtraDOF


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
    >>> grid = xps.spline_grid.SplineGrid([extra_dof], [10])
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
