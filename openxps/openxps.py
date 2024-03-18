"""
.. module:: openxps
   :platform: Linux, Windows, macOS
   :synopsis: Extended Phase-Space Simulations with OpenMM

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from dataclasses import dataclass
from numbers import Real

from openmm import unit as mmunit

from .bounds import Bounds
from .serializable import Serializable
from .units import preprocess_units


@dataclass(frozen=True)
class DynamicalVariable(Serializable):
    """
    An OpenMM context parameter turned into a dynamical variable.

    Parameters
    ----------
    name
        The name of the context parameter to be turned into a dynamical variable.
    unit
        The unity of measurement of this dynamical variable. It must be compatible with
        the OpenMM's MD unit system (mass in ``dalton``, distance in ``nanometer``, time
        in ``picosecond``, temperature in ``kelvin``, energy in ``kilojoules_per_mol``,
        angle in ``radian``). If the collective variables does not have a unit, it must
        be set to ``dimensionless``.
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
        >>> import openxps as xps
        >>> import yaml
        >>> from openmm import unit
        >>> dv = xps.DynamicalVariable(
        ...     "psi",
        ...     unit.radian,
        ...     3 * unit.dalton*(unit.nanometer/unit.radian)**2,
        ...     xps.bounds.Periodic(-180*unit.degree, 180*unit.degree)
        ... )
        >>> dv
        DynamicalVariable(name='psi', unit=rad, mass=3 nm**2 Da/(rad**2), bounds=...)
        >>> assert yaml.safe_load(yaml.safe_dump(dv)) == dv
    """

    name: str
    unit: mmunit.Unit
    mass: t.Union[mmunit.Quantity, Real]
    bounds: t.Union[Bounds, None]

    def __post_init__(self) -> None:
        if self.unit != mmunit.md_unit_system.express_unit(self.unit):
            raise ValueError(
                f"Unit {self.unit} must be compatible with the MD unit system."
            )
        if isinstance(self.mass, mmunit.Quantity) and not self.mass.unit.is_compatible(
            mmunit.dalton * mmunit.nanometer**2 / self.unit**2
        ):
            raise ValueError("Provided mass has incompatible units.")

    def __getstate__(self) -> t.Dict[str, t.Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "mass": self.mass,
            "bounds": self.bounds,
        }

    def __setstate__(self, keywords: t.Dict[str, t.Any]) -> None:
        self.__init__(**keywords)


DynamicalVariable.__init__ = preprocess_units(DynamicalVariable.__init__)

DynamicalVariable.register_tag("!openxps.DynamicalVariable")
