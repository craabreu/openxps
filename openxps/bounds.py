"""
.. module:: openxps.bounds
   :platform: Linux, MacOS, Windows
   :synopsis: Specification of boundary conditions for extra degrees of freedom

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from dataclasses import dataclass

import numpy as np
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

    lower: float
    upper: float
    unit: mmunit.Unit

    def __post_init__(self) -> None:
        for kind in ("lower", "upper"):
            if not isinstance(getattr(self, kind), (int, float)):
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

    def _wrap_float(self, value: float) -> float:
        raise NotImplementedError(
            "The method _wrap_float must be implemented in subclasses."
        )

    def convert_to(self, unit: mmunit.Unit) -> "Bounds":
        """
        Convert the bounds to a different unit of measurement.

        Parameters
        ----------
        unit
            The new unit of measurement for the bounds.

        Returns
        -------
        Bounds
            The bounds in the new unit of measurement.

        Example
        -------
        >>> import openxps as xps
        >>> from openmm import unit
        >>> bounds = xps.bounds.Periodic(-180, 180, unit.degree)
        >>> bounds.convert_to(unit.radian)
        Periodic(lower=-3.14159..., upper=3.14159..., unit=rad)
        """
        return type(self)(
            self.lower * self.unit / unit, self.upper * self.unit / unit, unit
        )

    def wrap(
        self, value: t.Union[mmunit.Quantity, float]
    ) -> t.Union[mmunit.Quantity, float]:
        """
        Wrap a value around the bounds.

        Parameters
        ----------
        value
            The value to be wrapped.

        Returns
        -------
        float
            The wrapped value.

        Example
        -------
        >>> import openxps as xps
        >>> from math import pi
        >>> from openmm import unit
        >>> bounds = xps.bounds.Periodic(-180, 180, unit.degree)
        >>> bounds.wrap(200)
        -160
        >>> bounds.wrap(-200 * unit.degree)
        Quantity(value=160, unit=degree)
        >>> bounds.wrap(2 * pi * unit.radian)
        Quantity(value=0.0, unit=radian)
        """
        if mmunit.is_quantity(value):
            wrapped = self._wrap_float(value / self.unit) * self.unit
            return wrapped.in_units_of(value.unit)
        return self._wrap_float(value)


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

    def __post_init__(self) -> None:
        super().__post_init__()
        self.period = self.upper - self.lower

    def _wrap_float(self, value: float) -> float:
        return (value - self.lower) % self.period + self.lower


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

    def __post_init__(self) -> None:
        super().__post_init__()
        self.period = 2 * (self.upper - self.lower)

    def _wrap_float(self, value: float) -> float:
        x = (value - self.lower) % self.period
        return np.minimum(x, self.period - x) + self.lower


Reflective.register_tag("!openxps.bounds.Reflective")


CIRCULAR = Periodic(-np.pi, np.pi, mmunit.radians)
