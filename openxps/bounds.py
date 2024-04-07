"""
.. module:: openxps.bounds
   :platform: Linux, MacOS, Windows
   :synopsis: Specification of boundary conditions for extra degrees of freedom

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from dataclasses import dataclass

import cvpack
import numpy as np
from cvpack.serialization import Serializable
from cvpack.units import Quantity
from openmm import unit as mmunit

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
    unit: cvpack.units.Unit

    def __post_init__(self) -> None:
        for kind in ("lower", "upper"):
            if not isinstance(getattr(self, kind), (int, float)):
                raise TypeError(f"The {kind} bound must be a real number.")
        if self.lower >= self.upper:
            raise ValueError("The upper bound must be greater than the lower bound.")
        if not mmunit.is_unit(self.unit):
            raise TypeError("The unit must be a valid OpenMM unit.")
        object.__setattr__(self, "unit", cvpack.units.Unit(self.unit))

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

    def convert(self, unit: mmunit.Unit) -> "Bounds":
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
        >>> bounds.convert(unit.radian)
        Periodic(lower=-3.14159..., upper=3.14159..., unit=rad)
        """
        factor = 1.0 * self.unit / unit
        if not isinstance(factor, float):
            raise ValueError("The unit must be compatible with the bounds unit.")
        return type(self)(factor * self.lower, factor * self.upper, unit)

    def asQuantity(self) -> mmunit.Quantity:
        """
        Return the bounds as a Quantity object.

        Returns
        -------
        Quantity
            The bounds as a Quantity object.

        Example
        -------
        >>> import openxps as xps
        >>> from openmm import unit
        >>> bounds = xps.bounds.Periodic(-180, 180, unit.degree)
        >>> bounds.asQuantity()
        (-180, 180) deg
        """
        return Quantity((self.lower, self.upper), self.unit)

    def leptonExpression(self, variable: str) -> str:
        """
        Return a lepton expression representing the transformation from an unwrapped
        variable to a wrapped value under the boundary conditions.

        Parameters
        ----------
        variable
            The name of the variable in the expression.
        Returns
        -------
        str
            A string representing the transformation.

        Example
        -------
        >>> import openxps as xps
        >>> from openmm import unit
        >>> periodic = xps.bounds.Periodic(-180, 180, unit.degree)
        >>> periodic.leptonExpression("x")
        'lb+T*(xs-floor(xs));xs=(x-lb)/T;lb=-180;T=360'
        >>> reflective = xps.bounds.Reflective(1, 10, unit.angstrom)
        >>> reflective.leptonExpression("y")
        'lb+T*min(ys,1-ys);ys=xs-floor(xs);xs=(y-lb)/T;lb=1;T=18'
        """
        raise NotImplementedError(
            "The method transformation must be implemented in subclasses."
        )

    def wrap(self, value: float, rate: float) -> t.Tuple[float, float]:
        """
        Wrap a value around the bounds and adjust its rate of change.

        Parameters
        ----------
        value
            The unwrapped value, in the same unit of measurement as the bounds.
        rate
            The rate of change of the unwrapped value, in the same unit of measurement
            as the bounds divided by a time unit.

        Returns
        -------
        float
            The wrapped value, in the same unit of measurement as the bounds.
        float
            The adjusted rate of change of the wrapped value, in the same unit of
            measurement as the original rate.
        """
        raise NotImplementedError("The method wrap must be implemented in subclasses.")


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

    def leptonExpression(self, variable: str) -> str:
        return (
            "lb+T*(xs-floor(xs))"
            f";xs=({variable}-lb)/T"
            f";lb={self.lower}"
            f";T={self.period}"
        )

    def wrap(self, value: float, rate: float) -> t.Tuple[float, float]:
        return (value - self.lower) % self.period + self.lower, rate


Periodic.registerTag("!openxps.bounds.Periodic")


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

    def leptonExpression(self, variable: str) -> str:
        return (
            f"lb+T*min(ys,1-ys)"
            f";ys=xs-floor(xs)"
            f";xs=({variable}-lb)/T"
            f";lb={self.lower}"
            f";T={self.period}"
        )

    def wrap(self, value: float, rate: float) -> t.Tuple[float, float]:
        x = (value - self.lower) % self.period
        if x < self.period - x:
            return x + self.lower, rate
        return self.period - x + self.lower, -rate


Reflective.registerTag("!openxps.bounds.Reflective")


CIRCULAR = Periodic(-np.pi, np.pi, mmunit.radians)
