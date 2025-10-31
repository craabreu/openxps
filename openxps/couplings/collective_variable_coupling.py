"""
Collective variable coupling.

.. module:: openxps.couplings.collective_variable_coupling
   :platform: Linux, MacOS, Windows
   :synopsis: Coupling between dynamical variables and physical collective variables

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import cvpack
import openmm as mm
from openmm import _openmm as mmswig
from openmm import unit as mmunit

from openxps.utils import preprocess_args

from ..dynamical_variable import DynamicalVariable
from .base import Coupling


class CollectiveVariableCoupling(Coupling):
    """Coupling between dynamical variables and physical collective variables.

    This class uses a :CVPack:`MetaCollectiveVariable` to create a coupling defined by
    a mathematical expression involving physical collective variables and parameters.

    Parameters
    ----------
    function
        A mathematical expression that defines the coupling energy as a function of
        collective variables and parameters.
    collective_variables
        The physical collective variables used in the coupling function.
    dynamical_variables
        The extended dynamical variables used in the coupling function.
    **parameters
        Named parameters that appear in the mathematical expression.

    Examples
    --------
    >>> from copy import copy
    >>> import cvpack
    >>> import openxps as xps
    >>> from openmm import unit
    >>> from math import pi
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> mass = 3 * unit.dalton * (unit.nanometer / unit.radian)**2
    >>> phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    >>> xps.CollectiveVariableCoupling(
    ...     f"0.5*kappa*min(delta,{2*pi}-delta)^2; delta=abs(phi-phi0)",
    ...     [phi],
    ...     [phi0],
    ...     kappa=1000 * unit.kilojoules_per_mole / unit.radian**2,
    ... )
    CollectiveVariableCoupling("0.5*kappa*min(delta,6.28...-delta)^2; ...")
    """

    @preprocess_args
    def __init__(
        self,
        function: str,
        collective_variables: t.Iterable[cvpack.CollectiveVariable],
        dynamical_variables: t.Iterable[DynamicalVariable],
        **parameters: mmunit.Quantity,
    ) -> None:
        dv_names = {dv.name for dv in dynamical_variables}
        filtered_params = {k: v for k, v in parameters.items() if k not in dv_names}

        force = cvpack.MetaCollectiveVariable(
            function,
            collective_variables,
            unit=mmunit.kilojoule_per_mole,
            name="coupling",
            **filtered_params,
            **{dv.name: 0.0 * dv.unit for dv in dynamical_variables},
        )
        super().__init__([force], dynamical_variables)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self._forces[0].getEnergyFunction()}")'

    def _createFlippedForce(self) -> mm.CustomCVForce:
        force = self._forces[0]
        parameters = force.getParameterDefaultValues()
        dvs_to_flip = [dv for dv in self._dynamical_variables if dv.name in parameters]
        for dv in dvs_to_flip:
            parameters.pop(dv.name)
        parameters.update(
            {cv.getName(): 0.0 * cv.getUnit() for cv in force.getInnerVariables()}
        )
        return cvpack.MetaCollectiveVariable(
            force.getEnergyFunction(),
            [
                dv.createCollectiveVariable(self._dv_indices[dv.name])
                for dv in dvs_to_flip
            ],
            unit=mmunit.kilojoule_per_mole,
            name="coupling",
            **parameters,
        )

    def updateExtensionContext(
        self,
        physical_context: mm.Context,
        extension_context: mm.Context,
    ) -> None:
        force = self._forces[0]
        collective_variables = mmswig.CustomCVForce_getCollectiveVariableValues(
            force, physical_context
        )
        for index, value in enumerate(collective_variables):
            name = mmswig.CustomCVForce_getCollectiveVariableName(force, index)
            mmswig.Context_setParameter(extension_context, name, value)


CollectiveVariableCoupling.registerTag("!openxps.CollectiveVariableCoupling")
