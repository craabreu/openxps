"""
.. module:: openxps.bounds
   :platform: Linux, MacOS, Windows
   :synopsis: Specification of boundary conditions for extra degrees of freedom

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from dataclasses import dataclass
from numbers import Real

from openmm import unit as mmunit

from .serializable import Serializable
from .utils import preprocess_args


@dataclass(frozen=True, eq=False)
class Bounds(Serializable):
    """
    A boundary condition for an extra degree of freedom.

    Parameters
    ----------
    lower
        The lower bound for the extra degree of freedom.
    upper
        The upper bound for the extra degree of freedom.
    unit
        The unity of measurement of the bounds. If the bounds do not have a unit, use
        ``dimensionless``.
    """

    lower: Real
    upper: Real
    unit: mmunit.Unit

    def __post_init__(self) -> None:
        for kind in ("lower", "upper"):
            if not isinstance(getattr(self, kind), Real):
                raise TypeError(f"The {kind} bound must be a real number.")
        if self.lower >= self.upper:
            raise ValueError("The upper bound must be greater than the lower bound.")

    def __getstate__(self) -> t.Dict[str, t.Any]:
        return {"lower": self.lower, "upper": self.upper, "unit": self.unit}

    def __setstate__(self, keywords: t.Dict[str, t.Any]) -> None:
        self.__init__(**keywords)

    def __eq__(self, other: t.Any) -> bool:
        if not isinstance(other, Bounds):
            return False
        return (
            self.lower * self.unit == other.lower * other.unit
            and self.upper * self.unit == other.upper * other.unit
        )


Bounds.__init__ = preprocess_args(Bounds.__init__)


class Periodic(Bounds):
    """
    A periodic boundary condition. The extra degree of freedom is allowed to wrap
    around the upper and lower bounds.

    Parameters
    ----------
    lower
        The lower bound for the extra degree of freedom.
    upper
        The upper bound for the extra degree of freedom.
    unit
        The unity of measurement of the bounds. If the bounds do not have a unit, use
        ``dimensionless``.

    Example
    -------
    >>> import openxps as xps
    >>> import yaml
    >>> from openmm import unit
    >>> bounds = xps.bounds.Periodic(-180, 180, unit.degree)
    >>> print(bounds)
    Periodic(lower=-180, upper=180, unit=deg)
    >>> assert yaml.safe_load(yaml.safe_dump(bounds)) == bounds
    """


Periodic.register_tag("!openxps.bounds.Periodic")


class Reflective(Bounds):
    """
    A reflective boundary condition. The extra degree of freedom collides elastically
    with the upper and lower bounds and is reflected back into the range.

    Parameters
    ----------
    lower
        The lower bound for the extra degree of freedom.
    upper
        The upper bound for the extra degree of freedom.
    unit
        The unity of measurement of the bounds. If the bounds do not have a unit, use
        ``dimensionless``.

    Example
    -------
    >>> import openxps as xps
    >>> from openmm import unit
    >>> bounds = xps.bounds.Reflective(1, 10, unit.angstrom)
    >>> bounds == xps.bounds.Reflective(0.1, 1, unit.nanometer)
    True
    >>> print(bounds)
    Reflective(lower=1, upper=10, unit=A)
    """


Reflective.register_tag("!openxps.bounds.Reflective")
