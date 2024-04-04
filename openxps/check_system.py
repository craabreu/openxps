"""
.. module:: openxps.check_system
   :platform: Linux, MacOS, Windows
   :synopsis: Function to check/modify a system for extended phase-space simulations.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from collections import defaultdict

import openmm as mm

from .extra_dof import ExtraDOF


def check_system(
    system: mm.System,
    extra_dofs: t.Tuple[ExtraDOF],
    add_missing_derivatives: bool = False,
) -> bool:
    """
    Checks and optionally modifies an :OpenMM:`System`, ensuring that it is properly
    configured for an extended phase-space (XPS) simulation with specified extra degrees
    of freedom (DOFs).

    Two main checks are performed:

    1. **Global Parameter Presence**: The system's potential energy must depend on all
       extra DOFs. This means that, for each extra DOF, a global parameter with the
       same name must have been added to at least one of the system's forces (via the
       ``addGlobalParameter`` method, available in many OpenMM force classes). An
       exception is raised if this condition is not met.

    2. **Parameter Derivative Requests**: Running an XPS simulation requires computing
       the derivative of the system's potential energy with respect to each extra DOF.
       Therefore, each force that depends on an extra DOF must include a request for
       the corresponding derivative (via the ``addEnergyParameterDerivative`` method).
       An exception is raised if this check fails and the add_missing_derivatives
       argument is set to False (the default).

    If the last check above fails but add_missing_derivatives is set to True, the
    function automatically adds all missing energy parameter derivatives, thus modifying
    the passed :OpenMM:`System` object in-place. No exception is raised in this case,
    but the function returns False to indicate that the system was not properly
    configured in the first place.

    .. note::

        If the passed :OpenMM:`System` instance belongs to an :OpenMM:`Context` and
        the function returns False, it will be necessary to `reinitialize`_ the context
        before the modifications take effect.

    .. _reinitialize:
        http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Context.html
        #openmm.openmm.Context.reinitialize

    Parameters
    ----------
    system
        The :OpenMM:`System` object to check and possibly modify.
    extra_dofs
        Extra degrees of freedom to include in the simulation, each represented by an
        :class:`~openxps.extra_dof.ExtraDOF` instance with a unique name.
    add_missing_derivatives
        If True, missing derivative requests are automatically added to the forces that
        depend on global parameters associated with extra DOFs.

    Returns
    -------
    bool
        True if the system was properly configured for an XPS simulation before the
        function was called, or False if missing derivative requests were added.

    Raises
    ------
    ValueError
        If there are missing energy parameter derivatives for any extra DOF and
        `add_missing_derivatives` is False.
    """
    dependent_forces = defaultdict(list)
    for index, force in enumerate(system.getForces()):
        if hasattr(force, "getNumGlobalParameters"):
            for i in range(force.getNumGlobalParameters()):
                dependent_forces[force.getGlobalParameterName(i)].append(index)

    missing_parameters = [
        xdof.name for xdof in extra_dofs if xdof.name not in dependent_forces
    ]
    if missing_parameters:
        raise ValueError(
            f"No forces depend on these global parameters: {missing_parameters}."
        )

    missing_derivatives = defaultdict(list)
    for xdof in extra_dofs:
        for index in dependent_forces[xdof.name]:
            force = system.getForce(index)
            if not any(
                force.getEnergyParameterDerivativeName(i) == xdof.name
                for i in range(force.getNumEnergyParameterDerivatives())
            ):
                missing_derivatives[index].append(xdof.name)

    if missing_derivatives:
        if not add_missing_derivatives:
            raise ValueError(
                "Missing derivative requests in system forces. "
                "Set add_missing_derivatives=True to add them automatically."
            )
        for index, names in missing_derivatives.items():
            force = system.getForce(index)
            for name in names:
                force.addEnergyParameterDerivative(name)
        return False

    return True
