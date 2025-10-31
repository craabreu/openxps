"""
Circular boundary condition.

.. module:: openxps.bounds.circular
   :platform: Linux, MacOS, Windows
   :synopsis: Circular boundary condition

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import numpy as np
from openmm import unit as mmunit

from .periodic import PeriodicBounds


class CircularBounds(PeriodicBounds):
    """
    A circular boundary condition equivalent to PeriodicBounds(-π, π, radians).

    This is a convenience class for the common case of a periodic boundary condition
    spanning from -π to π radians, which corresponds to a full circle.

    Parameters
    ----------
    lower, upper, unit
        Optional arguments for internal use (e.g., during deserialization).
        If not provided, defaults to -π, π, radians respectively.

    Example
    -------
    >>> import openxps as xps
    >>> bounds = xps.CircularBounds()
    >>> bounds.lower == -3.141592653589793
    True
    >>> bounds.upper == 3.141592653589793
    True
    >>> bounds.unit == xps.CircularBounds().unit
    True
    """

    def __init__(
        self,
        lower: float | None = None,
        upper: float | None = None,
        unit: mmunit.Unit | None = None,
    ) -> None:
        if lower is None:
            lower = -np.pi
        if upper is None:
            upper = np.pi
        if unit is None:
            unit = mmunit.radians
        super().__init__(lower, upper, unit)


CircularBounds.registerTag("!openxps.bounds.CircularBounds")
