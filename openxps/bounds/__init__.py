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
from .periodic import Periodic
from .reflective import Reflective

__all__ = ["Bounds", "Periodic", "Reflective", "NoBounds", "CIRCULAR"]

CIRCULAR = Periodic(-np.pi, np.pi, mmunit.radians)
