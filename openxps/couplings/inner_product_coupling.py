"""
Inner product coupling.

.. module:: openxps.couplings.inner_product_coupling
   :platform: Linux, MacOS, Windows
   :synopsis: The inner product of two separable vector-valued functions

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import cvpack
import openmm as mm
from openmm import _openmm as mmswig
from openmm import unit as mmunit

from openxps.utils import Function, preprocess_args

from ..dynamical_variable import DynamicalVariable
from .base import Coupling


class InnerProductCoupling(Coupling):
    r"""Coupling defined by the inner product of two separable vector-valued functions.

    Use this class if the coupling energy is the inner product of a vector-valued
    function :math:`{\boldsymbol \lambda}` of the extended dynamical variables
    :math:`{\bf s}` and a vector-valued function :math:`{\bf u}` of the physical
    coordinates :math:`{\bf r}`, i.e.,

    .. math::

        U = {\boldsymbol \lambda}({\bf s}) \cdot {\bf u}({\bf r})
          = \sum_{i=1}^n \lambda_i({\bf s}) u_i({\bf r})

    where :math:`n` is the dimensionality of the vector-valued functions.

    Parameters
    ----------
    forces
        A sequence of :OpenMM:`Force` objects whose sum depends linearly on a vector of
        :math:`n` global context parameters.
    dynamical_variables
        The dynamical variables of the extended phase-space system.
    functions
        A dictionary defining global context parameters (keys) as mathematical functions
        (values) of the dynamical variables. It is not necessary to include identity
        functions (i.e., ``key=value``).
    **parameters
        Named parameters that appear in the mathematical expressions.

    Examples
    --------
    >>> from copy import copy
    >>> from collections import namedtuple
    >>> import cvpack
    >>> import openxps as xps
    >>> from openmm import unit
    >>> from math import pi
    >>> import openmm as mm
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()

    A function to remove Coulomb interactions:

    >>> Parameters = namedtuple(
    ...     "Parameters",
    ...     ["index", "charge", "sigma", "epsilon"],
    ... )
    >>> def remove_coulomb_interaction(
    ...     force: mm.NonbondedForce, p1: Parameters, p2: Parameters
    ... ) -> None:
    ...     force.addException(
    ...         p1.index,
    ...         p2.index,
    ...         0.0,
    ...         (p1.sigma + p2.sigma)/2,
    ...         (p1.epsilon * p2.epsilon).sqrt(),
    ...     )

    Remove carbonyl oxygen <=> amide hydrogen interactions:

    >>> nbforce = next(
    ...     f for f in model.system.getForces()
    ...     if isinstance(f, mm.NonbondedForce)
    ... )
    >>> O1 = Parameters(5, *nbforce.getParticleParameters(5))
    >>> H1 = Parameters(7, *nbforce.getParticleParameters(7))
    >>> O2 = Parameters(15, *nbforce.getParticleParameters(15))
    >>> H2 = Parameters(17, *nbforce.getParticleParameters(17))
    >>> remove_coulomb_interaction(nbforce, O1, H2)
    >>> remove_coulomb_interaction(nbforce, O2, H1)

    Add scaled Coulomb interactions:

    >>> force = mm.CustomBondForce(f"scaling*chargeProd/r")
    >>> _ = force.addGlobalParameter("scaling", 1.0)
    >>> _ = force.addEnergyParameterDerivative("scaling")
    >>> _ = force.addPerBondParameter("chargeProd")
    >>> _ = force.addBond(O1.index, H2.index, [O1.charge*H2.charge])
    >>> _ = force.addBond(O2.index, H1.index, [O2.charge*H1.charge])

    Create a coupling between the dynamical variable and the nonbonded force:

    >>> lambda_dv = xps.DynamicalVariable(
    ...     name="lambda",
    ...     unit=unit.dimensionless,
    ...     mass=1.0 * unit.dalton * unit.nanometer**2,
    ...     bounds=xps.bounds.ReflectiveBounds(0.0, 1.0, unit.dimensionless),
    ... )
    >>> coupling = xps.InnerProductCoupling(
    ...     [force],
    ...     [lambda_dv],
    ...     functions={"scaling": "(1-cos(alpha*lambda))/2"},
    ...     alpha=pi*unit.radian,
    ... )
    >>> context = xps.ExtendedSpaceContext(
    ...     xps.ExtendedSpaceSystem(model.system, coupling),
    ...     xps.LockstepIntegrator(mm.VerletIntegrator(1.0 * mmunit.femtosecond)),
    ...     mm.Platform.getPlatformByName("Reference"),
    ... )
    >>> context.setPositions(model.positions)
    >>> context.setDynamicalVariableValues([1.0])
    """

    @preprocess_args
    def __init__(
        self,
        forces: t.Iterable[mm.Force],
        dynamical_variables: t.Iterable[DynamicalVariable],
        functions: t.Optional[dict[str, str]] = None,
        **parameters: mmunit.Quantity,
    ) -> None:
        forces = [
            cvpack.OpenMMForceWrapper(
                force,
                mmunit.kilojoule_per_mole,
                name=force.getName(),
            )
            if not isinstance(force, cvpack.CollectiveVariable)
            else force
            for force in forces
        ]
        self._functions = [
            Function(name, expression, **parameters)
            for name, expression in (functions or {}).items()
        ]
        self._parameters = parameters
        super().__init__(forces, dynamical_variables)
        self._dynamic_parameters = self._getDynamicParameters()

    def __getstate__(self) -> dict[str, t.Any]:
        state = super().__getstate__()
        state["functions"] = self._functions
        state["parameters"] = self._parameters
        return state

    def __setstate__(self, state: dict[str, t.Any]) -> None:
        self._functions = state["functions"]
        self._parameters = state["parameters"]
        super().__setstate__(state)
        self._dynamic_parameters = self._getDynamicParameters()

    def _getDynamicParameters(self) -> set[str]:
        all_force_parameters = {
            force.getGlobalParameterName(index)
            for force in self._forces
            for index in range(force.getNumGlobalParameters())
        }
        function_names = {fn.getName() for fn in self._functions}
        if self._functions:
            function_variables = set.union(
                *(fn.getVariables() for fn in self._functions),
            )
        else:
            function_variables = set()

        if missing := function_names - all_force_parameters:
            raise ValueError(
                "These functions are not global parameters in the provided forces: "
                + ", ".join(missing)
            )

        all_dvs = {dv.name for dv in self._dynamical_variables}

        if functions_missing_dvs := [
            fn.getName() for fn in self._functions if not (fn.getVariables() & all_dvs)
        ]:
            raise ValueError(
                "These functions do not depend on any dynamical variables: "
                + ", ".join(functions_missing_dvs)
            )

        dvs_in_force_parameters = all_dvs & all_force_parameters
        dvs_in_function_variables = all_dvs & function_variables

        if dvs_in_both := dvs_in_function_variables & dvs_in_force_parameters:
            raise ValueError(
                "These dynamical variables are both function variables and global "
                f"context parameters: {', '.join(dvs_in_both)}"
            )
        if dvs_in_neither := all_dvs - (
            dvs_in_function_variables | dvs_in_force_parameters
        ):
            raise ValueError(
                "These dynamical variables are neither function variables nor global "
                "context parameters: " + ", ".join(dvs_in_neither)
            )

        dynamic_parameters = dvs_in_force_parameters | function_names
        for force in self._forces:
            force_parameters = {
                force.getGlobalParameterName(index)
                for index in range(force.getNumGlobalParameters())
            }
            has_derivatives = {
                force.getEnergyParameterDerivativeName(index)
                for index in range(force.getNumEnergyParameterDerivatives())
            }
            if missing := force_parameters & dynamic_parameters - has_derivatives:
                raise ValueError(
                    "The following parameters require a derivative and are present in "
                    f"force {force.getName()}, but no derivative was requested: "
                    + ", ".join(missing)
                )
        return dynamic_parameters

    @staticmethod
    def _derivativeName(parameter: str) -> str:
        return "derivative_with_respect_to_" + parameter

    def _createFlippedForce(self) -> mm.CustomCVForce:
        flipped_force = mm.CustomCVForce(
            "+".join(f"{p}*{self._derivativeName(p)}" for p in self._dynamic_parameters)
        )
        all_dvs = [dv.name for dv in self._dynamical_variables]
        for parameter in self._dynamic_parameters:
            flipped_force.addGlobalParameter(self._derivativeName(parameter), 0.0)
        for fn in self._functions:
            flipped_force.addCollectiveVariable(
                fn.getName(), fn.createCollectiveVariable(all_dvs)
            )
        for dv in self._dynamical_variables:
            if dv.name in self._dynamic_parameters:
                flipped_force.addCollectiveVariable(
                    dv.name, dv.createCollectiveVariable(self._dv_indices[dv.name])
                )
        return flipped_force

    def updateExtensionContext(
        self,
        physical_context: mm.Context,
        extension_context: mm.Context,
    ) -> None:
        state = mmswig.Context_getState(physical_context, mm.State.ParameterDerivatives)
        for name, value in mmswig.State_getEnergyParameterDerivatives(state).items():
            if name in self._dynamic_parameters:
                mmswig.Context_setParameter(
                    extension_context, self._derivativeName(name), value
                )


InnerProductCoupling.registerTag("!openxps.InnerProductCoupling")
