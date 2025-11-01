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
    >>> bounds = xps.bounds.CircularBounds()
    >>> bounds.lower == -3.141592653589793
    True
    >>> bounds.upper == 3.141592653589793
    True
    >>> bounds.unit == xps.bounds.CircularBounds().unit
    True
    """

    def __init__(self, *_) -> None:
        super().__init__(-np.pi, np.pi, mmunit.radians)


CircularBounds.registerTag("!openxps.bounds.CircularBounds")
