"""
Reflective boundary condition.

.. module:: openxps.bounds.reflective
   :platform: Linux, MacOS, Windows
   :synopsis: Reflective boundary condition

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from .base import Bounds


class ReflectiveBounds(Bounds):
    """
    A reflective boundary condition. The dynamical variable collides elastically
    with the upper and lower bounds and is reflected back into the range.

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
    >>> from openmm import unit
    >>> bounds = xps.ReflectiveBounds(1, 10, unit.angstrom)
    >>> bounds == xps.ReflectiveBounds(0.1, 1, unit.nanometer)
    True
    >>> print(bounds)
    ReflectiveBounds(lower=1, upper=10, unit=A)
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.period = 2 * (self.upper - self.lower)

    def leptonExpression(self, variable: str) -> str:
        scaled = f"scaled_{variable}"
        wrapped = f"wrapped_{variable}"
        if self.lower == 0:
            shift = deshift = ""
        elif self.lower > 0:
            shift = f"-{self.lower}"
            deshift = f"+{self.lower}"
        else:
            shift = f"+{-self.lower}"
            deshift = f"-{-self.lower}"
        return (
            f"min({wrapped},1-{wrapped})*{self.period}{deshift}"
            f";\n{wrapped}={scaled}-floor({scaled})"
            f";\n{scaled}=({variable}{shift})/{self.period}"
        )

    def wrap(self, value: float, rate: float) -> tuple[float, float]:
        x = (value - self.lower) % self.period
        if x < self.period - x:
            return x + self.lower, rate
        return self.period - x + self.lower, -rate


ReflectiveBounds.registerTag("!openxps.bounds.ReflectiveBounds")
