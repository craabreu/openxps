"""
.. module:: openxps
   :platform: Linux, Windows, macOS
   :synopsis: Extended Phase-Space Methods with OpenMM

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _CustomCVForce: http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html
.. _Force: http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Force.html

"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import openmm as mm
from openmm import unit

from openxps import utils
from openxps.utils import QuantityOrFloat, UnitOrStr


class CollectiveVariable:
    """
    A scalar-valued function of the particle coordinates, evaluated by means of an OpenMM
    Force_ object.

    Quoting OpenMM's CustomCVForce_ manual entry:

        "Each collective variable is defined by a Force object. The Force's potential energy is
        computed, and that becomes the value of the variable. This provides enormous flexibility
        in defining collective variables, especially by using custom forces. Anything that can
        be computed as a potential function can also be used as a collective variable."

    Parameters
    ----------
    id
        a valid identifier string.
    force
        an OpenMM Force_ object whose energy function is used to evaluate this collective variable.
    unit
        the unity of measurement of this collective variable.
    period
        the period of this collective variable, if it is periodic. If `period=None`, it will be
        considered as non-periodic.

    Raises
    ------
    TypeError
        if `period` has a unit of measurement that is incompatible with `unit`.

    ValueError
        if `id` is not a valid identifier string (like a Python variable name)

    Example
    -------
        >>> import openmm as mm
        >>> import openxps as xps
        >>> dihedral_angle = mm.CustomTorsionForce('theta')
        >>> index = dihedral_angle.addTorsion(0, 1, 2, 3, [])
        >>> xps.CollectiveVariable('psi', dihedral_angle, 'radians', 360*mm.unit.degrees)
        psi: CustomTorsionForce (period=6.283185307179586 rad)
        >>> distance = mm.CustomBondForce('10*r')
        >>> index = distance.addBond(1, 18, [])
        >>> xps.CollectiveVariable('distance', distance, 'angstroms', None)
        distance: CustomBondForce (non-periodic, unit=A)

    """
    def __init__(
        self,
        id: str,
        force: mm.Force,
        unit: UnitOrStr,
        period: Union[QuantityOrFloat, None],
    ) -> None:

        if not id.isidentifier():
            raise ValueError('Parameter id must be a valid Python variable name')
        self.id = id
        self.force = force
        self.unit = utils.str2unit(unit) if isinstance(unit, str) else unit
        self.period = utils.compatible_value(period, self.unit)

    def __repr__(self) -> str:

        description = f'{self.id}: {self.force.getName()}'
        if self.period is None:
            description += f' (non-periodic, unit={self.unit.get_symbol()})'
        else:
            description += f' (period={self.period} {self.unit.get_symbol()})'
        return description

    def __getstate__(self) -> Dict[str, Any]:

        return dict(
            id=self.id,
            force=self.force,
            unit=self.unit.get_name(),
            period=self.period,
        )

    def __setstate__(self, kw: Dict[str, Any]) -> None:

        self.__init__(**kw)

    def _create_context(
        self,
        system: mm.System,
        positions: List[mm.Vec3],
    ) -> mm.Context:

        system_copy = deepcopy(system)
        for force in system_copy.getForces():
            force.setForceGroup(0)
        force_copy = deepcopy(self.force)
        force_copy.setForceGroup(1)
        system_copy.addForce(force_copy)
        platform = mm.Platform.getPlatformByName('Reference')
        context = mm.Context(system_copy, mm.CustomIntegrator(0), platform)
        context.setPositions(positions)
        return context

    def evaluate(self, system: mm.System, positions: List[mm.Vec3]) -> QuantityOrFloat:
        """
        Computes the value of the collective variable for a given system and a given set of
        particle coordinates.

        Parameters
        ----------
        system
            the system for which the collective variable will be evaluated.
        positions
            a list whose size equals the number of particles in the system and which
            contains the coordinates of these particles.

        Example
        -------
            >>> import openxps as xps
            >>> from openmm import app
            >>> model = xps.AlanineDipeptideModel()
            >>> force_field = app.ForceField('amber03.xml')
            >>> system = force_field.createSystem(model.topology)
            >>> print(model.phi.evaluate(system, model.positions))
            3.141592653589793 rad
            >>> print(model.psi.evaluate(system, model.positions))
            3.141592653589793 rad

        """

        context = self._create_context(system, positions)
        energy = context.getState(getEnergy=True, groups={1}).getPotentialEnergy()
        value = energy.value_in_unit(unit.kilojoules_per_mole)
        if self.unit is not None:
            value *= self.unit/utils.stdval(1*self.unit)
        return value

    def effective_mass(self, system: mm.System, positions: List[mm.Vec3]) -> QuantityOrFloat:
        """
        Computes the effective mass of the collective variable for a given system and a given set of
        particle coordinates.

        The effective mass of a collective variable :math:`q(\\mathbf{r})` is defined as
        :cite:`Cuendet_2014`:

        .. math::
            m_\\mathrm{eff} = \\left(
                \\sum_{j=1}^N \\frac{1}{m_j} \\left\\|\\frac{dq}{d\\mathbf{r}_j}\\right\\|^2
            \\right)^{-1}

        Parameters
        ----------
        system
            the system for which the collective variable will be evaluated.
        positions
            a list whose size equals the number of particles in the system and which contains
            the coordinates of these particles.

        Example
        -------
            >>> import openxps as xps
            >>> from openmm import app
            >>> model = xps.AlanineDipeptideModel()
            >>> force_field = app.ForceField('amber03.xml')
            >>> system = force_field.createSystem(model.topology)
            >>> print(model.phi.effective_mass(system, model.positions))
            0.0479588726559707 nm**2 Da/(rad**2)
            >>> print(model.psi.effective_mass(system, model.positions))
            0.05115582071188152 nm**2 Da/(rad**2)

        """

        context = self._create_context(system, positions)
        get_masses = np.vectorize(lambda i: utils.stdval(system.getParticleMass(i)))
        masses = get_masses(np.arange(system.getNumParticles()))
        forces = context.getState(getForces=True, groups={1}).getForces(asNumpy=True)
        effective_mass = 1.0/np.einsum('ij,ij,i->', forces, forces, 1.0/masses)
        if self.unit is not None:
            factor = utils.stdval(1*self.unit)**2
            effective_mass *= factor*unit.dalton*(unit.nanometers/self.unit)**2
        return effective_mass


class AuxiliaryVariable:
    """
    An extended phase-space variable whose dynamics is coupled to that of one of more collective
    variables through a potential energy term.

    Parameters
    ----------
    id
        a valid identifier string.
    unit
        the unity of measurement of this auxiliary variable.
    boundary
        boundary conditions to be applied to this auxiliary variable. Valid options are "periodic"
        and "reflective".
    minval
        the minimum allowable value for this auxiliary variable.
    maxval
        the maximum allowable value for this auxiliary variable.
    mass
        the mass assigned to this auxiliary variable, whose unit of measurement must be
        compatible with `dalton*(nanometer/unit)**2`, where `unit` is the auxiliary variable's
        own unit (see above).

    Keyword Args
    ------------
    **parameters
        Names and values of additional parameters for this auxiliary variable.

    Example
    -------
        >>> import math
        >>> import openxps as xps
        >>> import openmm as mm
        >>> from openmm import unit
        >>> cv = xps.CollectiveVariable('psi', mm.CustomTorsionForce('theta'), 2*math.pi, 'radians')
        >>> index = cv.force.addTorsion(0, 1, 2, 3, [])
        >>> mass = 50*unit.dalton*(unit.nanometer/unit.radians)**2
        >>> limit = 180*unit.degrees
        >>> xps.AuxiliaryVariable('s_psi', 'radian', 'periodic', -limit, limit, mass)
        <s_psi in [-3.141592653589793 rad, 3.141592653589793 rad], periodic, m=50 nm**2 Da/(rad**2)>

    """
    def __init__(
        self,
        id: str,
        unit: UnitOrStr,
        boundary: str,
        minval: QuantityOrFloat,
        maxval: QuantityOrFloat,
        mass: QuantityOrFloat,
        **parameters,
    ) -> None:

        self.id = id
        self.unit = utils.str2unit(unit) if isinstance(unit, str) else unit
        self.boundary = boundary
        self.minval = utils.compatible_value(minval, self.unit)
        self.maxval = utils.compatible_value(maxval, self.unit)
        self._mass_unit = mm.unit.dalton*(mm.unit.nanometer/self.unit)**2
        self.mass = utils.compatible_value(mass, self._mass_unit)
        self._range = self.maxval - self.minval
        self.parameters = parameters

    def __repr__(self) -> str:

        minval = f'{self.minval} {self.unit.get_symbol()}'
        maxval = f'{self.maxval} {self.unit.get_symbol()}'
        mass = f'{self.mass} {self._mass_unit.get_symbol()}'
        return f'<{self.id} in [{minval}, {maxval}], {self.boundary}, m={mass}>'

    def __getstate__(self) -> Dict[str, Any]:

        return dict(
            id=self.id,
            minval=self.minval,
            maxval=self.maxval,
            mass=self.mass,
            boundary=self.boundary,
            unit=self.unit.get_name(),
            **self.parameters,
        )

    def __setstate__(self, kw) -> None:

        self.__init__(**kw)
