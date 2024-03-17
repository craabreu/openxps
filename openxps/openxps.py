"""
.. module:: openxps
   :platform: Linux, Windows, macOS
   :synopsis: Extended Phase-Space Simulations with OpenMM

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import numpy as np
import yaml
from cvpack import unit as mmunit

from .bounds import Bounds
from .utils import register_serializer


class DynamicalVariable(yaml.YAMLObject):
    """
    An OpenMM context parameter turned into a dynamical variable.

    Parameters
    ----------
    name
        The name of the context parameter to be turned into a dynamical variable.
    unit
        The unity of measurement of this dynamical variable. It must be compatible with
        the MD unit system (mass in ``dalton``, distance in ``nanometer``, time in
        ``picosecond``, temperature in ``kelvin``, energy in ``kilojoules_per_mol``,
        angle in ``radian``). If the collective variables does not have a unit, pass
        ``dimensionless``.
    mass
        The mass assigned to this dynamical variable, whose unit of measurement must be
        compatible with ``dalton*(nanometer/unit)**2``, where `unit` is the dynamical
        variable's own unit (see above).
    bounds
        The boundary condition to be applied to this dynamical variable. It must be an
        instance of `openxps.bounds.Periodic`, `openxps.bounds.Reflective`, or `None`
        (for unbounded variables).

    Example
    -------
        >>> import math
        >>> import openxps as xps
        >>> import yaml
        >>> from openmm import unit
        >>> dv = xps.DynamicalVariable(
        ...     "psi",
        ...     unit.radian,
        ...     3 * unit.dalton*(unit.nanometer/unit.radian)**2,
        ...     xps.bounds.Periodic(-math.pi, math.pi),
        ... )
        >>> dv
        <psi in Periodic(-3.14..., 3.14...), unit is rad, mass is 3 nm**2 Da/(rad**2)>
        >>> serialized_dv = yaml.safe_dump(dv)
        >>> print(serialized_dv)
        !openxps.DynamicalVariable
        bounds: !openxps.bounds.Periodic
          lower_bound: -3.14...
          upper_bound: 3.14...
          version: 1
        mass: 3
        name: psi
        unit: !cvpack.Unit
          description: radian
          version: 1
        version: 1
        <BLANKLINE>
        >>> yaml.safe_load(serialized_dv)
        <psi in Periodic(-3.14..., 3.14...), unit is rad, mass is 3 nm**2 Da/(rad**2)>
    """

    @mmunit.convert_quantities
    def __init__(
        self,
        name: str,
        unit: mmunit.Unit,
        mass: mmunit.ScalarQuantity,
        bounds: t.Union[Bounds, None],
    ) -> None:

        if not np.isclose(
            mmunit.Quantity(1.0, unit).value_in_unit_system(mmunit.md_unit_system),
            1.0,
        ):
            raise ValueError(f"Unit {unit} is not compatible with the MD unit system.")

        self._name = name
        self._unit = mmunit.SerializableUnit(unit)
        self._mass = mass
        self._bounds = bounds

        self._mass_unit = mmunit.SerializableUnit(
            mmunit.dalton * (mmunit.nanometer / unit) ** 2
        )

    def __repr__(self) -> str:
        return ", ".join(
            [
                f"<{self._name}" + f" in {self._bounds}" * bool(self._bounds),
                f"unit is {self._unit.get_symbol()}",
                f"mass is {self._mass} {self._mass_unit.get_symbol()}>",
            ]
        )

    def __getstate__(self) -> t.Dict[str, t.Any]:
        return {
            "version": 1,
            "name": self.name,
            "unit": self.unit,
            "mass": self.mass,
            "bounds": self.bounds,
        }

    def __setstate__(self, kw: t.Dict[str, t.Any]) -> None:
        if kw.pop("version", 0) != 1:
            raise ValueError(f"Invalid version for {self.__class__.__name__}.")
        self.__init__(**kw)

    @property
    def name(self) -> str:
        """The name of this dynamical variable."""
        return self._name

    @property
    def unit(self) -> mmunit.Unit:
        """The unity of measurement of this dynamical variable."""
        return self._unit

    @property
    def bounds(self) -> Bounds:
        """The boundary condition to be applied to this dynamical variable."""
        return self._bounds

    @property
    def mass(self) -> mmunit.SerializableQuantity:
        """The mass assigned to this auxiliary variable."""
        return self._mass


register_serializer(DynamicalVariable, "!openxps.DynamicalVariable")
