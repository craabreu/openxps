"""
.. module:: openxps
   :platform: Linux, Windows, macOS
   :synopsis: Extended Phase-Space Methods with OpenMM

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _CustomCVForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomCVForce.html
.. _Force: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Force.html

"""

from copy import deepcopy

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

    Example
    -------
        >>> import openxps
        >>> from simtk import openmm, unit
        >>> cv = openxps.CollectiveVariable('psi', openmm.CustomTorsionForce('theta'))
        >>> cv.force.addTorsion(0, 1, 2, 3, [])
        0

    """
    def __init__(self, id, force):
        if not id.isidentifier():
            raise ValueError('Parameter id must be a valid variable identifier')
        self.id = id
        self.force = force

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

    def evaluate(self, system, positions, cv_unit=None):
        """
        Computes the value of the collective variable for a given system and a given set of particle
        coordinates.

        Parameters
        ----------
            system : openmm.System
                The system for which the collective variable will be evaluated.
            positions : list of openmm.Vec3
                A list whose size equals the number of particles in the system and which contains
                the coordinates of these particles.

        Keyword Args
        ------------
            cv_unit : unit.Unit, default=None
                The unity of measurement of the collective variable. If this is `None`, then a
                numerical value is returned based on the OpenMM default units.

        Returns
        -------
            float or unit.Quantity

        Example
        -------
            >>> import openxps
            >>> from simtk import unit
            >>> model = openxps.AlanineDipeptideModel()
            >>> model.phi.evaluate(model.system, model.positions)
            3.141592653589793
            >>> model.psi.evaluate(model.system, model.positions)
            3.141592653589793

        """
        context = self._create_context(system, positions)
        energy = context.getState(getEnergy=True, groups={1}).getPotentialEnergy()
        value = energy.value_in_unit(unit.kilojoules_per_mole)
        if cv_unit is not None:
            value *= cv_unit/_standardized(1*cv_unit)
        return value

    def effective_mass(self, system, positions, cv_unit=None):
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

        Keyword Args
        ------------
            cv_unit : unit.Unit, default=None
                The unity of measurement of the collective variable. If this is `None`, then a
                numerical value is returned based on the OpenMM default units.

        Returns
        -------
            float or unit.Quantity

        Example
        -------
            >>> import openxps
            >>> from simtk import unit
            >>> model = openxps.AlanineDipeptideModel()
            >>> model.phi.effective_mass(model.system, model.positions)
            0.0479588726559707
            >>> model.psi.effective_mass(model.system, model.positions)
            0.05115582071188152

        """
        context = self._create_context(system, positions)
        forces = _standardized(context.getState(getForces=True, groups={1}).getForces(asNumpy=True))
        denom = sum(f.dot(f)/_standardized(system.getParticleMass(i)) for i, f in enumerate(forces))
        effective_mass = 1.0/float(denom)
        if cv_unit is not None:
            factor = _standardized(1*cv_unit)**2
            effective_mass *= factor*unit.dalton*(unit.nanometers/cv_unit)**2
        return effective_mass
