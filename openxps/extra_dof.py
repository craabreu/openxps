"""
.. module:: openxps.extra_dof
   :platform: Linux, Windows, macOS
   :synopsis: Extra degrees of freedom for XPS simulations

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from dataclasses import dataclass

import cvpack
import openmm as mm
from cvpack.serialization import Serializable
from cvpack.units import Quantity
from openmm import unit as mmunit

from .bounds import Bounds, Periodic
from .utils import preprocess_args


@dataclass(frozen=True)
class ExtraDOF(Serializable):
    """
    Extra degree of freedom for extended phase-space simulations with OpenMM.

    Parameters
    ----------
    name
        The name of the context parameter to be turned into an extra degree of
        freedom.
    unit
        The unity of measurement of this extra degree of freedom. It must be
        compatible with OpenMM's MD unit system (mass in ``dalton``, distance in
        ``nanometer``, angle in ``radian``, time in ``picosecond``, temperature in
        ``kelvin``, energy in ``kilojoules_per_mol``). If the extra degree of
        freedom does not have a unit, use ``dimensionless``.
    mass
        The mass assigned to this extra degree of freedom, whose unit of measurement
        must be compatible with ``dalton*(nanometer/unit)**2``, where ``unit`` is the
        extra degree of freedom's own unit (see above).
    bounds
        The boundary condition to be applied to this extra degree of freedom. It must
        be an instance of ``openxps.bounds.Periodic``, ``openxps.bounds.Reflective``,
        or ``None`` (for unbounded variables). If it is not ``None``, its unit of
        measurement must be compatible with the extra degree of freedom's own unit.

    Example
    -------
    >>> import openxps as xps
    >>> import yaml
    >>> from openmm import unit
    >>> dv = xps.ExtraDOF(
    ...     "psi",
    ...     unit.radian,
    ...     3 * unit.dalton*(unit.nanometer/unit.radian)**2,
    ...     xps.bounds.Periodic(-180, 180, unit.degree)
    ... )
    >>> dv
    ExtraDOF(name='psi', unit=rad, mass=3 nm**2 Da/(rad**2), bounds=...)
    >>> dv.bounds
    Periodic(lower=-3.14159..., upper=3.14159..., unit=rad)
    >>> assert yaml.safe_load(yaml.safe_dump(dv)) == dv
    """

    name: str
    unit: mmunit.Unit
    mass: mmunit.Quantity
    bounds: t.Union[Bounds, None]

    def __post_init__(self) -> None:
        if not mmunit.is_unit(self.unit):
            raise ValueError("The unit must be a valid OpenMM unit.")
        if (1 * self.unit).value_in_unit_system(mmunit.md_unit_system) != 1:
            raise ValueError(
                f"Unit {self.unit} is incompatible with OpenMM's MD unit system."
            )

        if not mmunit.is_quantity(self.mass):
            raise TypeError("Mass must be have units of measurement.")
        mass_unit = mmunit.dalton * (mmunit.nanometer / self.unit) ** 2
        if not self.mass.unit.is_compatible(mass_unit):
            raise TypeError(f"Mass units must be compatible with {mass_unit}.")
        object.__setattr__(self, "mass", Quantity(self.mass.in_units_of(mass_unit)))

        if isinstance(self.bounds, Bounds):
            if not self.bounds.unit.is_compatible(self.unit):
                raise ValueError("Provided bounds have incompatible units.")
            object.__setattr__(self, "bounds", self.bounds.convert(self.unit))
        elif self.bounds is not None:
            raise TypeError("The bounds must be an instance of Bounds or None.")

    def __getstate__(self) -> t.Dict[str, t.Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "mass": self.mass,
            "bounds": self.bounds,
        }

    def __setstate__(self, keywords: t.Dict[str, t.Any]) -> None:
        self.__init__(**keywords)

    def isPeriodic(self) -> bool:
        """
        Returns whether this extra degree of freedom is periodic.

        Returns
        -------
        bool
            ``True`` if this extra degree of freedom is periodic, ``False`` otherwise.

        Example
        -------
        >>> import openxps as xps
        >>> from openmm import unit
        >>> dv = xps.ExtraDOF(
        ...     "psi",
        ...     unit.radian,
        ...     3 * unit.dalton*(unit.nanometer/unit.radian)**2,
        ...     xps.bounds.Periodic(-180, 180, unit.degree)
        ... )
        >>> dv.isPeriodic()
        True
        """
        return isinstance(self.bounds, Periodic)

    def createCollectiveVariable(self, particle: int) -> cvpack.OpenMMForceWrapper:
        """
        Returns a :CVPack:`OpenMMForceWrapper` object associating this extra degree of
        freedom with the x coordinate of a particle in an OpenMM system.

        Parameters
        ----------
        particle
            The index of the particle whose x coordinate will be associated with this
            extra degree of freedom.

        Returns
        -------
        cvpack.OpenMMForceWrapper
            The collective variable object representing this extra degree of freedom.

        Example
        -------
        >>> import openxps as xps
        >>> from openmm import unit
        >>> xdof = xps.ExtraDOF(
        ...     "psi",
        ...     unit.radian,
        ...     3 * unit.dalton*(unit.nanometer/unit.radian)**2,
        ...     xps.bounds.Periodic(-180, 180, unit.degree)
        ... )
        >>> cv = xdof.createCollectiveVariable(0)
        """
        bounds = self.bounds
        if bounds is None:
            force = mm.CustomExternalForce("x")
        else:
            force = mm.CustomExternalForce(bounds.leptonExpression("x"))
            bounds = bounds.asQuantity() if self.isPeriodic() else None
        force.addParticle(particle, [])
        return cvpack.OpenMMForceWrapper(force, self.unit, bounds, self.name)


ExtraDOF.__init__ = preprocess_args(ExtraDOF.__init__)

ExtraDOF.registerTag("!openxps.ExtraDOF")
