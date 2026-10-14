"""
.. module:: openxps.dynamical_variable
   :platform: Linux, Windows, macOS
   :synopsis: Dynamical variables for XPS simulations

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from dataclasses import dataclass

import cvpack
import openmm as mm
import typing_extensions as te
from cvpack.serialization import Serializable
from cvpack.units import Quantity
from openmm import unit as mmunit

from .bounds import Bounds, PeriodicBounds
from .utils import preprocess_args


@dataclass(frozen=True)
class DynamicalVariable(Serializable):
    """
    Dynamical variable for extended phase-space simulations with OpenMM.

    Parameters
    ----------
    name
        The name of the context parameter to be turned into a dynamical variable.
        It must be an identifier, as in Python: a letter or an underscore followed
        by letters, digits or underscores, where letters may be non-ASCII (``λ``).
    unit
        The unit of measurement of this dynamical variable. If it does not have a
        unit, use ``openmm.unit.dimensionless``. Couplings convert dynamical
        variables to MD units (see :meth:`in_md_units`), so an extended phase-space
        context reports values in MD units and, as in OpenMM, reads numbers given
        without units in MD units too.
    mass
        The mass assigned to this dynamical variable, whose unit of measurement
        must be compatible with ``dalton*(nanometer/unit)**2``, where ``unit`` is the
        dynamical variable's own unit (see ``unit`` above).
    bounds
        The boundary condition to be applied to this dynamical variable. It must
        be an instance of :class:`PeriodicBounds`, :class:`ReflectiveBounds`, or
        :class:`NoBounds` (for unbounded variables, whose bounds are then typical
        values rather than actual limits). Its unit of measurement must be
        compatible with the dynamical variable's own unit.

    Example
    -------
    >>> import openxps as xps
    >>> import yaml
    >>> from openmm import unit
    >>> dv = xps.DynamicalVariable(
    ...     "psi",
    ...     unit.radian,
    ...     3 * unit.dalton*(unit.nanometer/unit.radian)**2,
    ...     xps.PeriodicBounds(-180, 180, unit.degree)
    ... )
    >>> dv
    DynamicalVariable(name='psi', unit=rad, mass=3 nm**2 Da/(rad**2), bounds=...)
    >>> dv.bounds
    PeriodicBounds(lower=-3.14159..., upper=3.14159..., unit=rad)
    >>> assert yaml.safe_load(yaml.safe_dump(dv)) == dv
    """

    name: str
    unit: mmunit.Unit
    mass: mmunit.Quantity
    bounds: Bounds

    def __post_init__(self) -> None:
        if not (isinstance(self.name, str) and self.name.isidentifier()):
            raise ValueError(
                f"Invalid name {self.name!r}: it must be an identifier, starting "
                "with a letter or an underscore and continuing with letters, digits "
                "or underscores."
            )
        if not mmunit.is_unit(self.unit):
            raise TypeError("The unit must be a valid OpenMM unit.")
        if not mmunit.is_quantity(self.mass):
            raise TypeError("Mass must have units of measurement.")
        mass_unit = mmunit.dalton * (mmunit.nanometer / self.unit) ** 2
        if not self.mass.unit.is_compatible(mass_unit):
            raise TypeError(f"Mass units must be compatible with {mass_unit}.")
        object.__setattr__(self, "mass", Quantity(self.mass.in_units_of(mass_unit)))

        if isinstance(self.bounds, Bounds):
            if not self.bounds.unit.is_compatible(self.unit):
                raise ValueError("Provided bounds have incompatible units.")
            object.__setattr__(self, "bounds", self.bounds.convert(self.unit))
        else:
            raise TypeError("The bounds must be an instance of Bounds.")

    def __getstate__(self) -> dict[str, t.Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "mass": self.mass,
            "bounds": self.bounds,
        }

    def __setstate__(self, keywords: dict[str, t.Any]) -> None:
        self.__init__(**keywords)

    def __hash__(self) -> int:
        return hash((self.name, str(self.unit), self.bounds))

    def in_md_units(self) -> "DynamicalVariable":
        """
        Return the dynamical variable in the MD unit system.
        """
        unit = self.unit.in_unit_system(mmunit.md_unit_system)
        mass_unit = mmunit.dalton * (mmunit.nanometer / self.unit) ** 2
        return DynamicalVariable(
            self.name,
            unit,
            self.mass.in_units_of(mass_unit),
            self.bounds.in_md_units(),
        )

    def isPeriodic(self) -> bool:
        """
        Returns whether this dynamical variable is periodic.

        Returns
        -------
        bool
            ``True`` if this dynamical variable is periodic, ``False`` otherwise.

        Example
        -------
        >>> import openxps as xps
        >>> from openmm import unit
        >>> dv = xps.DynamicalVariable(
        ...     "psi",
        ...     unit.radian,
        ...     3 * unit.dalton*(unit.nanometer/unit.radian)**2,
        ...     xps.PeriodicBounds(-180, 180, unit.degree)
        ... )
        >>> dv.isPeriodic()
        True
        """
        return isinstance(self.bounds, PeriodicBounds)

    def createCollectiveVariable(
        self, index: int, name: str | None = None
    ) -> cvpack.OpenMMForceWrapper:
        """
        Returns a :CVPack:`OpenMMForceWrapper` object associating this extra degree of
        freedom with the x coordinate of a particle in an OpenMM system.

        Parameters
        ----------
        index
            The index of the particle whose x coordinate will be associated with this
            dynamical variable.
        name
            The name of the context parameter to be used in the OpenMM system. If
            ``None``, the name of the dynamical variable will be used.

        Returns
        -------
        cvpack.OpenMMForceWrapper
            The collective variable object representing this dynamical variable.

        Example
        -------
        >>> import openxps as xps
        >>> from openmm import unit
        >>> dv = xps.DynamicalVariable(
        ...     "psi",
        ...     unit.radian,
        ...     3 * unit.dalton*(unit.nanometer/unit.radian)**2,
        ...     xps.PeriodicBounds(-180, 180, unit.degree)
        ... )
        >>> cv = dv.createCollectiveVariable(0)
        """
        force = mm.CustomExternalForce(self.bounds.leptonExpression("x"))
        bounds = self.bounds.asQuantity() if self.isPeriodic() else None
        force.addParticle(index, [])
        return cvpack.OpenMMForceWrapper(
            force,
            self.unit,
            bounds,
            self.name if name is None else name,
        )

    def _distanceToCV(self, other: cvpack.CollectiveVariable) -> str:
        name = other.getName()
        diff = f"{name}-{self.name}"
        if not other.getUnit().is_compatible(self.unit):
            raise ValueError("Incompatible units.")
        if other.getPeriodicBounds() is None and not self.isPeriodic():
            return f"({diff})"
        if self.isPeriodic() and other.getPeriodicBounds() == self.bounds.asQuantity():
            period = self.bounds.in_md_units().period
            return f"({diff}-{period}*floor(0.5+({diff})/{period}))"
        raise ValueError(
            "Incompatible boundary conditions: either both variables are nonperiodic "
            "or both have the same periodic bounds (same lower and upper limits)."
        )

    def _distanceToDV(self, other: te.Self) -> str:
        diff = f"{other.name}-{self.name}"
        if not other.unit.is_compatible(self.unit):
            raise ValueError("Incompatible units.")
        if not (self.isPeriodic() or other.isPeriodic()):
            return f"({diff})"
        if self.bounds != other.bounds:
            raise ValueError(
                "Incompatible boundary conditions: either both variables are "
                "nonperiodic or both have the same periodic bounds (same lower and "
                "upper limits)."
            )
        period = self.bounds.in_md_units().period
        return f"({diff}-{period}*floor(0.5+({diff})/{period}))"

    def distanceTo(self, other: cvpack.CollectiveVariable | te.Self) -> str:
        """
        Returns a Lepton expression representing the distance between this extra degree
        of freedom and another variable.

        Parameters
        ----------
        other
            The other variable to which the distance will be calculated.

        Returns
        -------
        str
            A string representing the distance between the two variables.

        Example
        -------
        >>> import openxps as xps
        >>> import pytest
        >>> from openmm import unit
        >>> dv = xps.DynamicalVariable(
        ...     "psi0",
        ...     unit.radian,
        ...     3 * unit.dalton*(unit.nanometer/unit.radian)**2,
        ...     xps.PeriodicBounds(-180, 180, unit.degree)
        ... )
        >>> psi = cvpack.Torsion(6, 8, 14, 16, name="psi")
        >>> dv.distanceTo(psi)
        '(psi-psi0-6.28318...*floor(0.5+(psi-psi0)/6.28318...))'

        """
        if isinstance(other, cvpack.CollectiveVariable):
            return self._distanceToCV(other)
        if type(other) is type(self):
            return self._distanceToDV(other)
        raise TypeError(f"Method distanceTo not implemented for type {type(other)}.")


DynamicalVariable.__init__ = preprocess_args(DynamicalVariable.__init__)

DynamicalVariable.registerTag("!openxps.DynamicalVariable")
