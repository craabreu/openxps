"""
.. module:: openxps
   :platform: Linux, Windows, macOS
   :synopsis: Extended Phase-Space Methods with OpenMM

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _CustomCVForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomCVForce.html
.. _Force: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Force.html

"""

from copy import deepcopy

import numpy as np
from simtk import openmm, unit


def _standardized(quantity):
    """
    Returns the numerical value of a quantity in a unit of measurement compatible with the
    Molecular Dynamics unit system (mass in Da, distance in nm, time in ps, temperature in K,
    energy in kJ/mol, angle in rad).

    """
    if unit.is_quantity(quantity):
        return quantity.value_in_unit_system(unit.md_unit_system)
    else:
        return quantity


class CollectiveVariable:
    """
    A function of the particle coordinates, evaluated by means of an OpenMM Force_ object.

    Quoting OpenMM's CustomCVForce_ manual entry:

        "Each collective variable is defined by a Force object. The Force's potential energy is
        computed, and that becomes the value of the variable. This provides enormous flexibility
        in defining collective variables, especially by using custom forces. Anything that can
        be computed as a potential function can also be used as a collective variable."

    Parameters
    ----------
        id : str
            A valid identifier string for this collective variable.
        force : openmm.Force
            An OpenMM Force_ object whose energy function is used to evaluate this collective
            variable.

    Keyword Arguments
    -----------------
        unit : unit.Unit, default=None
            The unity of measurement of the collective variable. If this is `None`, then a
            numerical value is used based on OpenMM's default units.

    Example
    -------
        >>> import openxps
        >>> from simtk import openmm, unit
        >>> dihedral_angle = openmm.CustomTorsionForce('theta')
        >>> cv = openxps.CollectiveVariable('psi', dihedral_angle, unit.radians)
        >>> cv.force.addTorsion(0, 1, 2, 3, [])
        0

    """
    def __init__(self, id, force, unit=None):
        if not id.isidentifier():
            raise ValueError('Parameter id must be a valid variable identifier')
        self.id = id
        self.force = force
        self.unit = unit

    def __repr__(self):
        return f'{self.id}: {self.force.__class__.__name__}'

    def _create_context(self, system, positions):
        system_copy = deepcopy(system)
        for force in system_copy.getForces():
            force.setForceGroup(0)
        force_copy = deepcopy(self.force)
        force_copy.setForceGroup(1)
        system_copy.addForce(force_copy)
        platform = openmm.Platform.getPlatformByName('Reference')
        context = openmm.Context(system_copy, openmm.CustomIntegrator(0), platform)
        context.setPositions(positions)
        return context

    def evaluate(self, system, positions):
        """
        Computes the value of the collective variable for a given system and a given set of
        particle coordinates.

        Parameters
        ----------
            system : openmm.System
                The system for which the collective variable will be evaluated.
            positions : list of openmm.Vec3
                A list whose size equals the number of particles in the system and which contains
                the coordinates of these particles.

        Returns
        -------
            float or unit.Quantity

        Example
        -------
            >>> import openxps
            >>> from simtk import unit
            >>> model = openxps.AlanineDipeptideModel()
            >>> model.phi.evaluate(model.system, model.positions)
            Quantity(value=3.141592653589793, unit=radian)
            >>> model.psi.evaluate(model.system, model.positions)
            Quantity(value=3.141592653589793, unit=radian)

        """
        context = self._create_context(system, positions)
        energy = context.getState(getEnergy=True, groups={1}).getPotentialEnergy()
        value = energy.value_in_unit(unit.kilojoules_per_mole)
        if self.unit is not None:
            value *= self.unit/_standardized(1*self.unit)
        return value

    def effective_mass(self, system, positions):
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
            system : openmm.System
                The system for which the collective variable will be evaluated.
            positions : list of openmm.Vec3
                A list whose size equals the number of particles in the system and which contains
                the coordinates of these particles.

        Returns
        -------
            float or unit.Quantity

        Example
        -------
            >>> import openxps
            >>> from simtk import unit
            >>> model = openxps.AlanineDipeptideModel()
            >>> model.phi.effective_mass(model.system, model.positions)
            Quantity(value=0.0479588726559707, unit=nanometer**2*dalton/(radian**2))
            >>> model.psi.effective_mass(model.system, model.positions)
            Quantity(value=0.05115582071188152, unit=nanometer**2*dalton/(radian**2))

        """
        context = self._create_context(system, positions)
        forces = _standardized(context.getState(getForces=True, groups={1}).getForces(asNumpy=True))
        denom = sum(f.dot(f)/_standardized(system.getParticleMass(i)) for i, f in enumerate(forces))
        effective_mass = 1.0/float(denom)
        if self.unit is not None:
            factor = _standardized(1*self.unit)**2
            effective_mass *= factor*unit.dalton*(unit.nanometers/self.unit)**2
        return effective_mass


class ExtendedSpaceVariable:
    """
    An extended phase-space variable, whose dynamics is coupled to that of one of more collective
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
        id : str
            A valid identifier string for this dynamical variable.
        min_value : float or unit.Quantity
            The minimum allowable value for this dynamical variable.
        max_value : float or unit.Quantity
            The maximum allowable value for this dynamical variable.
        periodic : bool
            Whether the collective variable is periodic with period `L=max_value-min_value`.
        mass : float or unit.Quantity
            The mass assigned to this dynamical variable, whose unit of measurement must be
            compatible with `unit.dalton*(unit.nanometers/X)**2`, where `X` is the unit of
            measurement of the dynamical variable itself.
        colvars : :class:`~openxps.openxps.CollectiveVariable` or list thereof
            Either a single colective variable or a list.
        potential : float or unit.Quantity or str
            Either the value of the force constant of a harmonic driving potential or an algebraic
            expression giving the energy of the system as a function of this dynamical variable and
            its associated collective variable. Such expression can also contain a set of global
            parameters, whose values must be passed as keyword arguments (see below).

    Keyword Args
    ------------
        unit : unit.Unit, default=None
            The unity of measurement of the collective variable. If this is `None`, then a
            numerical value is used based on OpenMM's default units.
        sigma : float or unit.Quantity, default=None
            The standard deviation. If this is `None`, then no bias will be considered.
        grid_size : int, default=None
            The grid size. If this is `None` and `sigma` is finite, then a convenient value will
            be automatically chosen.
        **parameters
            Names and values of global parameters present in the algebraic expression defined as
            `potential` (see above).

    Example
    -------
        >>> import openxps
        >>> from simtk import openmm, unit
        >>> cv = openxps.CollectiveVariable('psi', openmm.CustomTorsionForce('theta'))
        >>> cv.force.addTorsion(0, 1, 2, 3, [])
        0
        >>> mass = 50*unit.dalton*(unit.nanometer/unit.radians)**2
        >>> K = 1000*unit.kilojoules_per_mole/unit.radians**2
        >>> limit = 180*unit.degrees
        >>> openxps.ExtendedSpaceVariable('s_psi', -limit, limit, True, mass, cv, K)
        <s_psi in [-3.141592653589793, 3.141592653589793], periodic, m=50>

    """
    def __init__(self, id, min_value, max_value, periodic, mass, colvars, potential,
                 sigma=None, grid_size=None, **parameters):
        self.id = id
        self.min_value = _standardized(min_value)
        self.max_value = _standardized(max_value)
        self.mass = _standardized(mass)
        self._range = self.max_value - self.min_value

        self.colvars = colvars if isinstance(colvars, (list, tuple)) else [colvars]

        if isinstance(potential, str):
            self.potential = potential
            self.parameters = {key: _standardized(value) for key, value in parameters.items()}
        else:
            cv = self.colvars[0].id
            if periodic:
                self.potential = f'0.5*K_{cv}*min(d{cv},{self._range}-d{cv})^2'
                self.potential += f'; d{cv}=abs({cv}-{self.id})'
            else:
                self.potential = f'0.5*K_{cv}*({cv}-{self.id})^2'
            self.parameters = {f'K_{cv}': _standardized(potential)}

        self.periodic = periodic

        if sigma is None or sigma == 0.0:
            self.sigma = self.grid_size = None
        else:
            self.sigma = _standardized(sigma)
            self._scaled_variance = (self.sigma/self._range)**2
            if grid_size is None:
                self.grid_size = int(np.ceil(5*self._range/self.sigma)) + 1
            else:
                self.grid_size = grid_size

    def __repr__(self):
        status = 'periodic' if self.periodic else 'non-periodic'
        return f'<{self.id} in [{self.min_value}, {self.max_value}], {status}, m={self.mass}>'

    def __getstate__(self):
        return dict(
            id=self.id,
            min_value=self.min_value,
            max_value=self.max_value,
            mass=self.mass,
            colvars=self.colvars,
            potential=self.potential,
            periodic=self.periodic,
            sigma=self.sigma,
            grid_size=self.grid_size,
            **self.parameters,
        )

    def __setstate__(self, kw):
        self.__init__(**kw)
