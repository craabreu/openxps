"""
.. module:: path
   :platform: Linux, MacOS, Windows
   :synopsis: Specifications for boundary conditions in OpenXPS.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from dataclasses import dataclass

from openmm import unit as mmunit

from .serializable import Serializable
from .units import preprocess_units


@dataclass(frozen=True)
class Bounds(Serializable):
    """
    A boundary condition for a dynamical variable.

    Parameters
    ----------
    lower
        The lower bound for the dynamical variable.
    upper
        The upper bound for the dynamical variable.
    """

    lower: t.Union[mmunit.Quantity, float]
    upper: t.Union[mmunit.Quantity, float]

    def __post_init__(self) -> None:
        for kind in ("lower", "upper"):
            if not isinstance(getattr(self, kind), (mmunit.Quantity, float)):
                raise TypeError(f"The {kind} bound must be a Quantity or a float.")
        if self.lower >= self.upper:
            raise ValueError("The upper bound must be greater than the lower bound.")

    def __getstate__(self) -> t.Dict[str, t.Any]:
        return {"lower": self.lower, "upper": self.upper}

    def __setstate__(self, keywords: t.Dict[str, t.Any]) -> None:
        self.__init__(**keywords)


Bounds.__init__ = preprocess_units(Bounds.__init__)


class Periodic(Bounds):
    """
    A periodic boundary condition. The dynamical variable is allowed to wrap around the
    upper and lower bounds.

    Parameters
    ----------
    lower
        The lower bound for the dynamical variable.
    upper
        The upper bound for the dynamical variable.

    Example
    -------
    >>> import openxps as xps
    >>> import yaml
    >>> from openmm import unit
    >>> bounds = xps.bounds.Periodic(-180 * unit.degree, 180 * unit.degree)
    >>> print(bounds)
    Periodic(lower=-180 deg, upper=180 deg)
    >>> assert yaml.safe_load(yaml.safe_dump(bounds)) == bounds
    """


Periodic.register_tag("!openxps.bounds.Periodic")


class Reflective(Bounds):
    """
    A reflective boundary condition. The dynamical variable collides elastically with
    the upper and lower bounds and is reflected back into the range.

    Parameters
    ----------
    lower
        The lower bound for the dynamical variable.
    upper
        The upper bound for the dynamical variable.

    Example
    -------
    >>> import openxps as xps
    >>> from cvpack import unit
    >>> bounds = xps.bounds.Reflective(1 * unit.angstrom, 10 * unit.angstrom)
    >>> bounds == xps.bounds.Reflective(0.1 * unit.nanometer, 1 * unit.nanometer)
    True
    >>> print(bounds)
    Reflective(lower=1 A, upper=10 A)
    """


Reflective.register_tag("!openxps.bounds.Reflective")
