"""
.. module:: openxps.bias_potential
   :platform: Linux, Windows, macOS
   :synopsis: Abstract class for bias potentials applied to dynamical variables.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import cvpack
import openmm as mm
from openmm import _openmm as mmswig
from openmm import unit as mmunit

from .dynamical_variable import DynamicalVariable


class BiasPotential(cvpack.OpenMMForceWrapper):
    """
    Abstract class for bias potentials applied to dynamical variables.

    Parameters
    ----------
    dynamical_variables
        The dynamical variables subject to the bias potential.
    function
        The mathematical expression defining the bias potential.
    force_constructor
        A function that constructs the OpenMM force object from the mathematical
        expression.
    """

    def __init__(
        self,
        dynamical_variables: t.Sequence[DynamicalVariable],
        function: str,
        force_constructor: t.Callable[[str], mm.Force],
    ) -> None:
        self._dvs = tuple(dynamical_variables)
        self._dv_indices = self._get_state_args = None
        expression = function
        for index, dv in enumerate(dynamical_variables, start=1):
            variable = f"x{index}"
            if dv.bounds is not None:
                variable = dv.bounds.leptonExpression(variable)
            expression += f";{dv.name}={variable}"
        super().__init__(
            force_constructor(expression),
            mmunit.kilojoules_per_mole,
            name="bias_potential",
        )

    def _performAddKernel(
        self, context: mm.Context, center: t.Sequence[float], potential: float
    ) -> None:
        raise NotImplementedError(
            "Method _performAddKernel must be implemented by derived classes"
        )

    def initialize(self, context_dvs: t.Sequence[DynamicalVariable]) -> None:
        """
        Initialize the bias potential for a specific context.

        Parameters
        ----------
        context_dvs
            All the dynamical variables in the extension context.
        """
        try:
            self._dv_indices = tuple(map(context_dvs.index, self._dvs))
        except ValueError as error:
            raise ValueError(
                "Not all DVs in the bias potential are present in the context"
            ) from error
        self._get_state_args = (
            mm.State.Positions | mm.State.Energy,
            False,
            1 << self.getForceGroup(),
        )

    def addKernel(self, context: mm.Context) -> None:
        """
        Add a Gaussian kernel to the bias potential.

        Parameters
        ----------
        context
            The OpenMM context object where the bias potential is applied.
        """
        if self._dv_indices is None:
            raise ValueError("Bias potential has not been initialized")

        state = mmswig.Context_getState(context, *self._get_state_args)
        positions = mmswig.State__getVectorAsVec3(state, mm.State.Positions)
        center = []
        for dv, index in zip(self._dvs, self._dv_indices):
            value = positions[index].x
            if dv.bounds is not None:
                value, _ = dv.bounds.wrap(value, 0)
            center.append(value)
        self._performAddKernel(context, center, mmswig.State_getPotentialEnergy(state))
