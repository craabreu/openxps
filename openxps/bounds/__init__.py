"""
.. module:: openxps.bounds
   :platform: Linux, MacOS, Windows
   :synopsis: Specification of boundary conditions for dynamical variables

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import numpy as np
from openmm import unit as mmunit

from .base import Bounds
from .no_bounds import NoBounds
from .periodic import PeriodicBounds
from .reflective import ReflectiveBounds

__all__ = ["PeriodicBounds", "ReflectiveBounds", "NoBounds", "CIRCULAR"]

CIRCULAR = PeriodicBounds(-np.pi, np.pi, mmunit.radians)
