"""
No boundary condition.

.. module:: openxps.bounds.no_bounds
   :platform: Linux, MacOS, Windows
   :synopsis: No boundary condition

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from openmm import unit as mmunit

from .base import Bounds


class NoBounds(Bounds):
    """
    No boundary condition. The dynamical variable is allowed to take any value.

    Parameters
    ----------

    If it is not ``None``, its unit of measurement must be compatible with the dynamical
    variable's own unit.
    Example
    -------
    >>> import openxps as xps
    >>> import yaml
    >>> from openmm import unit
    >>> bounds = xps.PeriodicBounds(-180, 180, unit.degree)
    >>> print(bounds)
    PeriodicBounds(lower=-180, upper=180, unit=deg)
    >>> assert yaml.safe_load(yaml.safe_dump(bounds)) == bounds
    """

    def __init__(self, *_) -> None:
        super().__init__(0, 1, mmunit.dimensionless)

    def in_md_units(self) -> "NoBounds":
        return NoBounds()

    def leptonExpression(self, variable: str) -> str:
        return f"{variable}"

    def wrap(self, value: float, rate: float) -> tuple[float, float]:
        return value, rate
