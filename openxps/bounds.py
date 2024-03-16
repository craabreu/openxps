"""
.. module:: path
   :platform: Linux, MacOS, Windows
   :synopsis: Specifications for boundary conditions in OpenXPS.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import yaml
from cvpack import unit as mmunit


class Bounds(yaml.YAMLObject):
    """
    A boundary condition for a dynamical variable.

    Parameters
    ----------
    lower_bound
        The minimum value for the dynamical variable.
    upper_bound
        The maximum value for the dynamical variable.
    """

    @mmunit.convert_quantities
    def __init__(
        self,
        lower_bound: mmunit.ScalarQuantity,
        upper_bound: mmunit.ScalarQuantity,
    ) -> None:
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._range = self._upper_bound - self._lower_bound

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._lower_bound}, {self._upper_bound})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._lower_bound == other._lower_bound
            and self._upper_bound == other._upper_bound
        )

    def __getstate__(self) -> t.Dict[str, t.Any]:
        return {
            "version": 1,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }

    def __setstate__(self, kw: t.Dict[str, t.Any]) -> None:
        if kw.pop("version", 0) != 1:
            raise ValueError(f"Invalid version for {self.__class__.__name__}.")
        self.__init__(**kw)

    @property
    def lower_bound(self) -> mmunit.ScalarQuantity:
        """The minimum value for the dynamical variable."""
        return self._lower_bound

    @property
    def upper_bound(self) -> mmunit.ScalarQuantity:
        """The maximum value for the dynamical variable."""
        return self._upper_bound

    @property
    def range(self) -> mmunit.ScalarQuantity:
        """The range of the dynamical variable."""
        return self._range


class Periodic(Bounds):
    """
    A periodic boundary condition. The dynamical variable is allowed to wrap around the
    upper and lower bounds.

    Parameters
    ----------
    lower_bound
        The minimum value for the dynamical variable.
    upper_bound
        The maximum value for the dynamical variable.

    Example
    -------
    >>> import openxps as xps
    >>> from cvpack import unit
    >>> bounds = xps.bounds.Periodic(-180 * unit.degree, 180 * unit.degree)
    >>> print(bounds)
    Periodic(-3.14..., 3.14...)
    >>> bounds.range
    6.28318...
    """

    yaml_tag = "!openxps.bounds.Periodic"


yaml.SafeDumper.add_representer(Periodic, Periodic.to_yaml)
yaml.SafeLoader.add_constructor(Periodic.yaml_tag, Periodic.from_yaml)


class Reflective(Bounds):
    """
    A reflective boundary condition. The dynamical variable collides elastically with
    the upper and lower bounds and is reflected back into the range.

    Parameters
    ----------
    lower_bound
        The minimum value for the dynamical variable.
    upper_bound
        The maximum value for the dynamical variable.

    Example
    -------
    >>> import openxps as xps
    >>> from cvpack import unit
    >>> bounds = xps.bounds.Reflective(0.0 * unit.angstrom, 10 * unit.angstrom)
    >>> bounds == xps.bounds.Reflective(0.0 * unit.nanometer, 1.0 * unit.nanometer)
    True
    >>> print(bounds)
    Reflective(0.0, 1.0)
    """

    yaml_tag = "!openxps.bounds.Reflective"


yaml.SafeDumper.add_representer(Reflective, Reflective.to_yaml)
yaml.SafeLoader.add_constructor(Reflective.yaml_tag, Reflective.from_yaml)
