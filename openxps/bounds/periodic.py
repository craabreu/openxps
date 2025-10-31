"""
Periodic boundary condition.

.. module:: openxps.bounds.periodic
   :platform: Linux, MacOS, Windows
   :synopsis: Periodic boundary condition

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from .base import Bounds


class PeriodicBounds(Bounds):
    """
    A periodic boundary condition. The dynamical variable is allowed to wrap
    around the upper and lower bounds.

    Parameters
    ----------
    lower
        The lower bound for the dynamical variable.
    upper
        The upper bound for the dynamical variable.
    unit
        The unity of measurement of the bounds. If the bounds do not have a unit, use
        ``dimensionless``.

    Example
    -------
    >>> import openxps as xps
    >>> import yaml
    >>> from openmm import unit
    >>> bounds = xps.bounds.PeriodicBounds(-180, 180, unit.degree)
    >>> print(bounds)
    PeriodicBounds(lower=-180, upper=180, unit=deg)
    >>> assert yaml.safe_load(yaml.safe_dump(bounds)) == bounds
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.period = self.upper - self.lower

    def leptonExpression(self, variable: str) -> str:
        scaled = f"scaled_{variable}"
        if self.lower == 0:
            shift = deshift = ""
        elif self.lower > 0:
            shift = f"-{self.lower}"
            deshift = f"+{self.lower}"
        else:
            shift = f"+{-self.lower}"
            deshift = f"-{-self.lower}"
        return (
            f"({scaled}-floor({scaled}))*{self.period}{deshift}"
            f";\n{scaled}=({variable}{shift})/{self.period}"
        )

    def wrap(self, value: float, rate: float) -> tuple[float, float]:
        return (value - self.lower) % self.period + self.lower, rate


PeriodicBounds.registerTag("!openxps.bounds.PeriodicBounds")
