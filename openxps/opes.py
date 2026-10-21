"""
.. module:: openxps.opes
   :platform: Linux, MacOS, Windows
   :synopsis: On-the-fly probability enhanced sampling of dynamical variables.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm as mm
from openmm import app as mmapp

from .bounds import NoBounds, ReflectiveBounds
from .context import ExtendedSpaceContext
from .metadynamics import ExtendedSpaceBiasVariable, _SimulationWrapper
from .system import ExtendedSpaceSystem


def _biasVariables(
    system: ExtendedSpaceSystem,
    variables: t.Sequence[ExtendedSpaceBiasVariable],
) -> tuple[list[mmapp.BiasVariable], bool]:
    """Bias variables on the extension particles, and whether any is reflective.

    Mirrors :class:`ExtendedSpaceMetadynamics`: each dynamical variable, in MD
    units, must belong to the system and have bounds, since the bias lives on a
    finite grid. ``bounded`` is True if any variable has reflective bounds; with
    unbounded variables rejected, the non-periodic variables are exactly the
    reflective ones.
    """
    system_dvs = system.getDynamicalVariables()
    bias_variables = []
    bounded = False
    for variable in variables:
        dv = variable.dynamical_variable.in_md_units()
        if dv not in system_dvs:
            raise ValueError(f"Dynamical variable {dv.name} not found in system")
        if isinstance(dv.bounds, NoBounds):
            raise ValueError(
                f"Dynamical variable {dv.name} has no bounds, but OPES needs a "
                "finite grid. Use ReflectiveBounds over the range the variable may "
                "explore."
            )
        bounded = bounded or isinstance(dv.bounds, ReflectiveBounds)
        index = system_dvs.index(dv)
        bias_variables.append(
            mmapp.BiasVariable(
                dv.createCollectiveVariable(index),
                dv.bounds.lower * dv.bounds.unit,
                dv.bounds.upper * dv.bounds.unit,
                variable.sigma,
                dv.isPeriodic(),
                variable.grid_width,
            )
        )
    return bias_variables, bounded


def _extensionSimulation(simulation):
    """A view of ``simulation`` whose context is the extension context.

    openmm-opes calls back into the sampler with the object it was given, so an
    already converted simulation is returned unchanged.
    """
    if isinstance(simulation, _SimulationWrapper):
        return simulation
    return _SimulationWrapper(simulation)


def _extensionContext(context: mm.Context) -> mm.Context:
    """The extension context of an :class:`ExtendedSpaceContext`, else ``context``."""
    if isinstance(context, ExtendedSpaceContext):
        return context.getExtensionContext()
    return context
