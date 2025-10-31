"""
.. module:: openxps.bounds
   :platform: Linux, MacOS, Windows
   :synopsis: Specification of boundary conditions for dynamical variables

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from .base import Bounds
from .circular import CircularBounds
from .no_bounds import NoBounds
from .periodic import PeriodicBounds
from .reflective import ReflectiveBounds

__all__ = [
    "PeriodicBounds",
    "ReflectiveBounds",
    "NoBounds",
    "CircularBounds",
    "CIRCULAR",
]

# Backward compatibility constant
CIRCULAR = CircularBounds()
