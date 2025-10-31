"""
Harmonic coupling.

.. module:: openxps.couplings.harmonic_coupling
   :platform: Linux, MacOS, Windows
   :synopsis: A harmonic coupling between a dynamical variable and a collective variable

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import cvpack
from openmm import unit as mmunit

from ..dynamical_variable import DynamicalVariable
from .collective_variable_coupling import CollectiveVariableCoupling


class HarmonicCoupling(CollectiveVariableCoupling):
    r"""A harmonic coupling between a dynamical variable and a collective variable.

    The coupling energy is given by:

    .. math::

        U = \frac{1}{2} \kappa \left(s - q({\bf r})\right)^2

    where :math:`s` is an extended dynamical variable, :math:`q({\bf r})` is a
    physical collective variable, and :math:`\kappa` is a coupling constant.

    Parameters
    ----------
    collective_variable
        The collective variable used in the coupling.
    dynamical_variable
        The dynamical variable used in the coupling.
    force_constant
        The force constant for the coupling.

    Examples
    --------
    >>> from copy import copy
    >>> import cvpack
    >>> import openxps as xps
    >>> from openmm import unit
    >>> dvmass = 3 * unit.dalton * (unit.nanometer / unit.radian)**2
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> phi_s = xps.DynamicalVariable(
    ...     "phi_s", unit.radian, dvmass, xps.CircularBounds()
    ... )
    >>> kappa = 1000 * unit.kilojoule_per_mole / unit.radian**2
    >>> xps.HarmonicCoupling(phi, phi_s, kappa)
    HarmonicCoupling("0.5*kappa_phi_phi_s*((phi-phi_s-6.28...*floor(...))^2)")
    """

    def __init__(
        self,
        collective_variable: cvpack.CollectiveVariable,
        dynamical_variable: DynamicalVariable,
        force_constant: mmunit.Quantity,
    ) -> None:
        self._validateArguments(collective_variable, dynamical_variable, force_constant)
        kappa = f"kappa_{collective_variable.getName()}_{dynamical_variable.name}"
        function = (
            f"0.5*{kappa}*({dynamical_variable.distanceTo(collective_variable)}^2)"
        )
        super().__init__(
            function=function,
            collective_variables=[collective_variable],
            dynamical_variables=[dynamical_variable],
            **{kappa: force_constant},
        )

    def _validateArguments(self, cv, dv, kappa):
        pair = f"{cv.getName()} and {dv.name}"
        if not cv.getUnit().is_compatible(dv.unit):
            raise ValueError(f"Incompatible units for {pair}.")
        if (dv.isPeriodic() == cv.getPeriodicBounds() is None) or (
            cv.getPeriodicBounds() != dv.bounds.asQuantity()
        ):
            raise ValueError(f"Incompatible periodicity for {pair}.")
        if mmunit.is_quantity(kappa) and not kappa.unit.is_compatible(
            mmunit.kilojoule_per_mole / dv.unit**2
        ):
            raise ValueError(f"Incompatible force constant units for {pair}.")


HarmonicCoupling.registerTag("!openxps.HarmonicCoupling")
