"""
Circular boundary condition.

.. module:: openxps.bounds.circular
   :platform: Linux, MacOS, Windows
   :synopsis: Circular boundary condition

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import math

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
    >>> bounds = xps.bounds.CircularBounds()
    >>> bounds.lower == -3.141592653589793
    True
    >>> bounds.upper == 3.141592653589793
    True
    >>> bounds.unit == xps.bounds.CircularBounds().unit
    True
    >>> from openmm import unit
    >>> bounds.convert(unit.degree)
    CircularBounds(lower=-180.0, upper=180.0, unit=deg)
    """

    def __init__(
        self,
        lower: float = -np.pi,
        upper: float = np.pi,
        unit: mmunit.Unit = mmunit.radians,
    ) -> None:
        super().__init__(lower, upper, unit)
        if not self.unit.is_compatible(mmunit.radians):
            raise ValueError("Circular bounds must have an angle unit.")
        factor = self.unit.conversion_factor_to(mmunit.radians)
        if not (
            math.isclose(self.lower * factor, -np.pi, rel_tol=1e-9)
            and math.isclose(self.upper * factor, np.pi, rel_tol=1e-9)
        ):
            raise ValueError("Circular bounds must span from -π to π radians.")


CircularBounds.registerTag("!openxps.bounds.CircularBounds")
