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
from typing import Any, Dict, List, Optional, Union

import numpy as np
import openmm as mm
from openmm import unit

from openxps.utils import QuantityOrFloat, UnitOrStr, stdval, str2unit


class CollectiveVariable:
    """A function of the particle coordinates, evaluated by means of an OpenMM Force_ object.

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
        an OpenMM Force_ object whose energy function is used to evaluate this CV.
    period
        the period of the CV, if it is periodic. If `period=None`, it will be considered as
        non-periodic.
    unit
        the unity of measurement of the CV.

    Raises
    ------
    TypeError
        if `period` has a unit of measurement that is incompatible with `unit`.

    Example
    -------
        >>> import openmm as mm
        >>> import openxps as xps
        >>> dihedral_angle = mm.CustomTorsionForce('theta')
        >>> index = dihedral_angle.addTorsion(0, 1, 2, 3, [])
        >>> xps.CollectiveVariable('psi', dihedral_angle, 360*mm.unit.degrees, 'radians')
        psi: CustomTorsionForce (period=6.283185307179586 radian)
        >>> distance = mm.CustomBondForce('10*r')
        >>> index = distance.addBond(1, 18, [])
        >>> xps.CollectiveVariable('distance', distance, None, 'angstroms')
        distance: CustomBondForce (non-periodic, unit=angstrom)

    """
    def __init__(
        self,
        id: str,
        force: mm.Force,
        period: QuantityOrFloat,
        unit: UnitOrStr,
    ) -> None:

        if not id.isidentifier():
            raise ValueError('Parameter id must be a valid Python variable name')
        self.id = id
        self.force = force
        self.period = stdval(period)
        self.unit = str2unit(unit) if isinstance(unit, str) else unit
        self.period = period.value_in_unit(self.unit) if mm.unit.is_quantity(period) else period

    def __repr__(self) -> str:

        description = f'{self.id}: {self.force.getName()}'
        if self.period is None:
            description += f' (non-periodic, unit={self.unit})'
        else:
            description += f' (period={self.period} {self.unit})'
        return description

    def __getstate__(self) -> Dict[str, Any]:

        return dict(
            id=self.id,
            force=self.force,
            period=self.period,
            unit=self.unit.get_name(),
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
            value *= self.unit/stdval(1*self.unit)
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
        get_masses = np.vectorize(lambda i: stdval(system.getParticleMass(i)))
        masses = get_masses(np.arange(system.getNumParticles()))
        forces = context.getState(getForces=True, groups={1}).getForces(asNumpy=True)
        effective_mass = 1.0/np.einsum('ij,ij,i->', forces, forces, 1.0/masses)
        if self.unit is not None:
            factor = stdval(1*self.unit)**2
            effective_mass *= factor*unit.dalton*(unit.nanometers/self.unit)**2
        return effective_mass


class AuxiliaryVariable:
    """
    An extended phase-space variable whose dynamics is coupled to that of one of more collective
    variables of a system.

    The coupling occurs in the form of a potential energy term that involves this extended
    phase-space variable and its associated collective variables.

    For a non-periodic variable, the default potential is a harmonic driving of the type:

    .. math::
        V(s, \\mathbf r) = \\frac{\\kappa}{2} [s - q(\\mathbf r)]^2

    where :math:`s` is the new dynamical variable, :math:`q(\\mathbf r)` is its associated
    collective variable, and :math:`kappa` is a force constant.

    For a periodic variable with period `L`, the default potential is:

    .. math::
        V(s, \\mathbf r) = \\frac{\\kappa}{2} \\min(|s-q(\\mathbf r)|, L-|s-q(\\mathbf r)|)^2

    Parameters
    ----------
    id
        a valid identifier string.
    min_value
        the minimum allowable value for this auxiliary variable.
    max_value
        the maximum allowable value for this auxiliary variable.
    periodic
        whether to apply periodic boundary conditions for the auxiliary variable. Otherwise,
        reflective boundary conditions will apply.
    mass
        the mass assigned to this auxiliary variable, whose unit of measurement must be
        compatible with `Da*(nm/Unit)**2`, where `Unit` is the auxiliary variable's unit.
    unit
        the unity of measurement of the collective variable. If this is `None`, then a
        numerical value is used based on OpenMM's default units.
    sigma
        the standard deviation. If this is `None`, then no bias will be considered.
    grid_size
        the grid size. If this is `None` and `sigma` is finite, then a convenient value will
        be automatically chosen.

    Keyword Args
    ------------
    **parameters
        Names and values of global parameters present in the algebraic expression defined as
        `potential` (see above).

    Example
    -------
        >>> import math
        >>> import openxps as xps
        >>> import openmm as mm
        >>> from openmm import unit
        >>> cv = xps.CollectiveVariable('psi', mm.CustomTorsionForce('theta'), 2*math.pi, 'radians')
        >>> cv.force.addTorsion(0, 1, 2, 3, [])
        0
        >>> mass = 50*unit.dalton*(unit.nanometer/unit.radians)**2
        >>> K = 1000*unit.kilojoules_per_mole/unit.radians**2
        >>> limit = 180*unit.degrees
        >>> xps.AuxiliaryVariable('s_psi', -limit, limit, True, mass, cv, K)
        <s_psi in [-3.141592653589793, 3.141592653589793], periodic, m=50>

    """
    def __init__(
        self,
        id: str,
        min_value: QuantityOrFloat,
        max_value: QuantityOrFloat,
        periodic: bool,
        mass: QuantityOrFloat,
        colvars: Union[CollectiveVariable, List[CollectiveVariable]],
        potential: Union[str, QuantityOrFloat],
        unit: Optional[UnitOrStr] = None,
        sigma: Optional[QuantityOrFloat] = None,
        grid_size: Optional[int] = None,
        **parameters,
    ) -> None:

        self.id = id
        self.min_value = stdval(min_value)
        self.max_value = stdval(max_value)
        self.mass = stdval(mass)
        self._range = self.max_value - self.min_value

        self.colvars = colvars if isinstance(colvars, (list, tuple)) else [colvars]

        if isinstance(potential, str):
            self.potential = potential
            self.parameters = {key: stdval(value) for key, value in parameters.items()}
        else:
            cv = self.colvars[0].id
            if periodic:
                self.potential = f'0.5*K_{cv}*min(d{cv},{self._range}-d{cv})^2'
                self.potential += f'; d{cv}=abs({cv}-{self.id})'
            else:
                self.potential = f'0.5*K_{cv}*({cv}-{self.id})^2'
            self.parameters = {f'K_{cv}': stdval(potential)}

        self.periodic = periodic

        self.unit = str2unit(unit) if isinstance(unit, str) else unit

        if sigma is None or sigma == 0.0:
            self.sigma = self.grid_size = None
        else:
            self.sigma = stdval(sigma)
            self._scaled_variance = (self.sigma/self._range)**2
            if grid_size is None:
                self.grid_size = int(np.ceil(5*self._range/self.sigma)) + 1
            else:
                self.grid_size = grid_size

    def __repr__(self) -> str:

        status = 'periodic' if self.periodic else 'non-periodic'
        return f'<{self.id} in [{self.min_value}, {self.max_value}], {status}, m={self.mass}>'

    def __getstate__(self) -> Dict[str, Any]:

        return dict(
            id=self.id,
            min_value=self.min_value,
            max_value=self.max_value,
            mass=self.mass,
            colvars=self.colvars,
            potential=self.potential,
            periodic=self.periodic,
            unit=self.unit,
            sigma=self.sigma,
            grid_size=self.grid_size,
            **self.parameters,
        )

    def __setstate__(self, kw) -> None:

        self.__init__(**kw)
