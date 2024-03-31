"""
.. module:: openxps.systems
   :platform: Linux, MacOS, Windows
   :synopsis: System classes for extended phase-space simulations with OpenMM.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from collections import defaultdict
from copy import copy

import openmm as mm
from openmm import _openmm as mmswig

from .extra_dof import ExtraDOF
from .serializable import Serializable


class PhysicalSystem(Serializable):
    """
    A physical system for extended phase-space simulations with OpenMM.

    Example
    -------
    >>> import openxps as xps
    >>> from math import pi
    >>> import cvpack
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> umbrella_potential = cvpack.MetaCollectiveVariable(
    ...     f"0.5*kappa*min(delta,{2*pi}-delta)^2; delta=abs(phi_cv-phi_dv)",
    ...     [cvpack.Torsion(6, 8, 14, 16, name="phi_cv")],
    ...     unit.kilojoule_per_mole,
    ...     kappa=1000 * unit.kilojoule_per_mole / unit.radian**2,
    ...     phi_dv=0*unit.radian,
    ... )
    >>> umbrella_potential.addToSystem(model.system)
    >>> phi_dv = xps.ExtraDOF(
    ...     "phi_dv",
    ...     unit.radian,
    ...     3 * unit.dalton*(unit.nanometer/unit.radian)**2,
    ...     xps.bounds.Periodic(-180, 180, unit.degree)
    ... )
    >>> xps_system = xps.PhysicalSystem(model.system, [phi_dv])
    >>> for dv in xps_system.extra_dofs:
    ...     print(dv)
    ExtraDOF(name='phi_dv', unit=rad, mass=3 nm**2 Da/(rad**2), ...)
    """

    thisown = property(
        lambda x: x.this.own(),
        lambda x, v: x.this.own(v),
        doc="The membership flag",
    )
    __swig_destroy__ = mmswig.delete_System

    def __init__(
        self,
        openmm_system: mm.System,
        extra_dofs: t.Iterable[ExtraDOF],
    ) -> None:
        mmswig.System_swiginit(self, copy(openmm_system).this)
        self.extra_dofs = tuple(extra_dofs)

        # List all forces that depend on each extra degree of freedom
        parameter_forces = defaultdict(list)
        for k in range(openmm_system.getNumForces()):
            force = mmswig.System_getForce(self, k)
            if hasattr(force, "getNumGlobalParameters"):
                for index in range(force.getNumGlobalParameters()):
                    parameter_forces[force.getGlobalParameterName(index)].append(force)

        # Check if all extra degrees of freedom are present in the system's forces
        absent_variables = [
            dv.name for dv in extra_dofs if dv.name not in parameter_forces
        ]
        if absent_variables:
            raise ValueError(
                "These global parameters are not present in the system's forces: "
                + ", ".join(absent_variables)
            )

        # Add missing derivatives with respect to the extra degrees of freedom
        for dv in extra_dofs:
            for force in parameter_forces[dv.name]:
                if not any(
                    force.getEnergyParameterDerivativeName(index) == dv.name
                    for index in range(force.getNumEnergyParameterDerivatives())
                ):
                    force.addEnergyParameterDerivative(dv.name)

    def __getstate__(self):
        return {
            "openmm_system": mm.XmlSerializer.serialize(self),
            "extra_dofs": self.extra_dofs,
        }

    def __setstate__(self, state):
        self.__init__(
            mm.XmlSerializer.deserialize(state["openmm_system"]),
            state["extra_dofs"],
        )


mmswig.System_swigregister(PhysicalSystem)
PhysicalSystem.register_tag("!openxps.PhysicalSystem")
