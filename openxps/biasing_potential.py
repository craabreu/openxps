"""
.. module:: openxps.biasing_potential
   :platform: Linux, Windows, macOS
   :synopsis: Abstract class for bias potentials applied to extra degrees of freedom.

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import cvpack
import openmm as mm
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
        self._context = None
        self._extra_dof_indices = None
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

    def _performAddKernel(self, center: t.Sequence[float]) -> None:
        raise NotImplementedError(
            "Method _performAddKernel must be implemented by derived classes"
        )

    def initialize(self, context: mm.Context, extra_dofs: t.Sequence[ExtraDOF]) -> None:
        """
        Initialize the bias potential for a specific context.

        Parameters
        ----------
        context
            The extension context where the bias potential will be applied.
        extra_dofs
            All the extra degrees of freedom in the extension context.
        """
        try:
            self._extra_dof_indices = tuple(map(extra_dofs.index, self._extra_dofs))
        except ValueError as error:
            raise ValueError(
                "Not all extra DOFs in the bias potential are present in the context"
            ) from error
        self._context = context
        self.addToSystem(context.getSystem())
        context.reinitialize(preserveState=True)

    def addKernel(self, center: t.Sequence[mmunit.Quantity]) -> None:
        """
        Add a Gaussian kernel to the bias potential.

        Parameters
        ----------
        center
            The center vector of the kernel.
        """
        if self._extra_dof_indices is None:
            raise ValueError("Bias potential has not been initialized")
        center = [
            quantity.value_in_unit(xdof.unit)
            for quantity, xdof in zip(center, self._extra_dofs)
        ]
        self._performAddKernel(center)

    def getExtraDOFs(self) -> t.Tuple[ExtraDOF]:
        """
        Get the extra degrees of freedom subject to the bias potential.

        Returns
        -------
        Tuple[ExtraDOF]
            The extra degrees of freedom subject to the bias potential.
        """
        return self._extra_dofs
