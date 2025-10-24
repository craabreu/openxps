"""
.. module:: openxps.bounds
   :platform: Linux, MacOS, Windows
   :synopsis: Specification of boundary conditions for dynamical variables

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
    A boundary condition for a dynamical variable.

    Parameters
    ----------
    lower
        The lower bound for the dynamical variable.
    upper
        The upper bound for the dynamical variable.
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

    def __getstate__(self) -> dict[str, t.Any]:
        return {"lower": self.lower, "upper": self.upper, "unit": self.unit}

    def __setstate__(self, keywords: dict[str, t.Any]) -> None:
        self.__init__(**keywords)

    def __eq__(self, other: t.Any) -> bool:
        return (
            isinstance(other, Bounds)
            and self.lower * self.unit == other.lower * other.unit
            and self.upper * self.unit == other.upper * other.unit
        )

    def __hash__(self) -> int:
        unit, factor = self._md_unit_and_conversion_factor()
        return hash((self.lower * factor, self.upper * factor, unit))

    def _md_unit_and_conversion_factor(self) -> tuple[mmunit.Unit, float]:
        """
        Return the MD unit and conversion factor for the bounds.
        """
        unit = self.unit.in_unit_system(mmunit.md_unit_system)
        factor = self.unit.conversion_factor_to(unit)
        return unit, factor

    def in_md_units(self) -> "Bounds":
        """
        Return the bounds in the MD unit system.

        Example
        -------
        >>> import openxps as xps
        >>> from openmm import unit
        >>> bounds = xps.bounds.Periodic(-1.0, 1.0, unit.kilocalories_per_mole)
        >>> bounds.in_md_units()
        Periodic(lower=-4.184, upper=4.184, unit=nm**2 Da/(ps**2))
        """
        unit, factor = self._md_unit_and_conversion_factor()
        return self.__class__(self.lower * factor, self.upper * factor, unit)

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
        return self.__class__(factor * self.lower, factor * self.upper, unit)

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
        >>> print(periodic.leptonExpression("x"))
        (scaled_x-floor(scaled_x))*360-180;
        scaled_x=(x+180)/360
        >>> reflective = xps.bounds.Reflective(1, 10, unit.angstrom)
        >>> print(reflective.leptonExpression("y"))
        min(wrapped_y,1-wrapped_y)*18+1;
        wrapped_y=scaled_y-floor(scaled_y);
        scaled_y=(y-1)/18
        """
        raise NotImplementedError(
            "The method transformation must be implemented in subclasses."
        )

    def wrap(self, value: float, rate: float) -> tuple[float, float]:
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
    >>> bounds = xps.bounds.Periodic(-180, 180, unit.degree)
    >>> print(bounds)
    Periodic(lower=-180, upper=180, unit=deg)
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


Periodic.registerTag("!openxps.bounds.Periodic")


class Reflective(Bounds):
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


Reflective.registerTag("!openxps.bounds.Reflective")


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
    >>> bounds = xps.bounds.Periodic(-180, 180, unit.degree)
    >>> print(bounds)
    Periodic(lower=-180, upper=180, unit=deg)
    >>> assert yaml.safe_load(yaml.safe_dump(bounds)) == bounds
    """

    def __init__(self) -> None:
        super().__init__(-np.inf, np.inf, mmunit.dimensionless)

    def in_md_units(self) -> "NoBounds":
        return NoBounds()

    def leptonExpression(self, variable: str) -> str:
        return f"{variable}"

    def wrap(self, value: float, rate: float) -> tuple[float, float]:
        return value, rate


Periodic.registerTag("!openxps.bounds.Periodic")


CIRCULAR = Periodic(-np.pi, np.pi, mmunit.radians)
