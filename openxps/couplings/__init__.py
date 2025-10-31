"""
.. module:: openxps.couplings
   :platform: Linux, MacOS, Windows
   :synopsis: Coupling between physical and extended phase-space systems.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from .base import Coupling, CouplingSum
from .collective_variable_coupling import CollectiveVariableCoupling
from .harmonic_coupling import HarmonicCoupling
from .inner_product_coupling import InnerProductCoupling

__all__ = [
    "CollectiveVariableCoupling",
    "HarmonicCoupling",
    "InnerProductCoupling",
]
