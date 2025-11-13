"""
No boundary condition.

.. module:: openxps.bounds.no_bounds
   :platform: Linux, MacOS, Windows
   :synopsis: No boundary condition

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from .base import Bounds


class NoBounds(Bounds):
    """Boundary condition indicating no confinement for a dynamical variable.

    Parameters
    ----------
    lower
        A typical low value for the dynamical variable, not an actual bound.
    upper
        A typical high value for the dynamical variable, not an actual bound.
    unit
        The unity of measurement of the bounds. If the bounds do not have a unit, use
        ``dimensionless``.

    Example
    -------
    >>> import openxps as xps
    >>> import yaml
    >>> from openmm import unit
    >>> bounds = xps.NoBounds(0, 1, unit.dimensionless)
    >>> print(bounds)
    NoBounds(lower=0, upper=1, unit=dimensionless)
    >>> assert yaml.safe_load(yaml.safe_dump(bounds)) == bounds
    """

    def leptonExpression(self, variable: str) -> str:
        return f"{variable}"

    def wrap(self, value: float, rate: float) -> tuple[float, float]:
        return value, rate


NoBounds.registerTag("!openxps.bounds.NoBounds")
