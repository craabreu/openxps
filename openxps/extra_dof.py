"""
.. module:: openxps.extra_dof
   :platform: Linux, Windows, macOS
   :synopsis: Extra degrees of freedom for OpenXPS simulations

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from dataclasses import dataclass
from numbers import Real

from openmm import unit as mmunit

from .bounds import Bounds
from .serializable import Serializable
from .utils import preprocess_args


@dataclass(frozen=True)
class ExtraDOF(Serializable):
    """
    An OpenMM context parameter turned into a extra degree of freedom.

    Parameters
    ----------
    name
        The name of the context parameter to be turned into a extra degree of freedom.
    unit
        The unity of measurement of this extra degree of freedom. It must be compatible
        with OpenMM's MD unit system (mass in ``dalton``, distance in ``nanometer``,
        angle in ``radian``, time in ``picosecond``, temperature in ``kelvin``, energy
        in ``kilojoules_per_mol``). If the extra degree of freedom does not have a unit,
        use ``dimensionless``.
    mass
        The mass assigned to this extra degree of freedom, whose unit of measurement
        must be compatible with ``dalton*(nanometer/unit)**2``, where `unit` is the
        dynamical variable's own unit (see above).
    bounds
        The boundary condition to be applied to this extra degree of freedom. It must be
        an instance of `openxps.bounds.Periodic`, `openxps.bounds.Reflective`, or `None`
        (for unbounded variables).

    Example
    -------
    >>> import openxps as xps
    >>> import yaml
    >>> from openmm import unit
    >>> dv = xps.ExtraDOF(
    ...     "psi",
    ...     unit.radian,
    ...     3 * unit.dalton*(unit.nanometer/unit.radian)**2,
    ...     xps.bounds.Periodic(-180*unit.degree, 180*unit.degree)
    ... )
    >>> dv
    ExtraDOF(name='psi', unit=rad, mass=3 nm**2 Da/(rad**2), bounds=...)
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


ExtraDOF.__init__ = preprocess_args(ExtraDOF.__init__)

ExtraDOF.register_tag("!openxps.ExtraDOF")
