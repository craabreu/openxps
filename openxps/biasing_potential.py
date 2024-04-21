"""
.. module:: openxps.biasing_potential
   :platform: Linux, Windows, macOS
   :synopsis: Abstract class for bias potentials applied to extra degrees of freedom.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import cvpack
import openmm as mm
from openmm import _openmm as mmswig
from openmm import unit as mmunit

from .extra_dof import ExtraDOF


class BiasingPotential(cvpack.OpenMMForceWrapper):
    """
    Abstract class for bias potentials applied to extra degrees of freedom.

    Parameters
    ----------
    extra_dofs
        The extra degrees of freedom subject to the bias potential.
    function
        The mathematical expression defining the bias potential.
    force_constructor
        A function that constructs the OpenMM force object from the mathematical
        expression.
    """

    def __init__(
        self,
        extra_dofs: t.Sequence[ExtraDOF],
        function: str,
        force_constructor: t.Callable[[str], mm.Force],
    ) -> None:
        self._extra_dofs = tuple(extra_dofs)
        self._extra_dof_indices = self._get_state_args = None
        expression = function
        for index, xdof in enumerate(extra_dofs, start=1):
            variable = f"x{index}"
            if xdof.bounds is not None:
                variable = xdof.bounds.leptonExpression(variable)
            expression += f";{xdof.name}={variable}"
        super().__init__(
            force_constructor(expression),
            mmunit.kilojoules_per_mole,
            name="biasing_potential",
        )

    def _performAddKernel(
        self, context: mm.Context, center: t.Sequence[float], potential: float
    ) -> None:
        raise NotImplementedError(
            "Method _performAddKernel must be implemented by derived classes"
        )

    def initialize(self, context_extra_dofs: t.Sequence[ExtraDOF]) -> None:
        """
        Initialize the bias potential for a specific context.

        Parameters
        ----------
        context_extra_dofs
            All the extra degrees of freedom in the extension context.
        """
        try:
            self._extra_dof_indices = tuple(
                map(context_extra_dofs.index, self._extra_dofs)
            )
        except ValueError as error:
            raise ValueError(
                "Not all extra DOFs in the bias potential are present in the context"
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
        if self._extra_dof_indices is None:
            raise ValueError("Bias potential has not been initialized")

        state = mmswig.Context_getState(context, *self._get_state_args)
        positions = mmswig.State__getVectorAsVec3(state, mm.State.Positions)
        center = []
        for xdof, index in zip(self._extra_dofs, self._extra_dof_indices):
            value = positions[index].x
            if xdof.bounds is not None:
                value, _ = xdof.bounds.wrap(value, 0)
            center.append(value)
        self._performAddKernel(context, center, mmswig.State_getPotentialEnergy(state))
