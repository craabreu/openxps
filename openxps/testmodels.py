"""
.. module:: testmodels
   :platform: Unix, Windows, macOS
   :synopsis: Unified Free Energy Dynamics Test Model Systems

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _System: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html
.. _Topology: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.app.topology.Topology.html

"""

import os
from typing import Optional

import openmm
from openmm import app

import openxps


class AlanineDipeptideModel:
    """
    A model consisting of a single alanine-dipeptide molecule, either in a vacuum or solvated
    in explicit water.

    Keyword Args
        water_model : str, default=None
            The water model to be used if the alanine dipeptide is supposed to be solvated.
            Available options are "spce", "tip3p", "tip4pew", and "tip5p"
        number : int, default=500
            The number of water molecules to be added if `water` is not `None`

    Attributes
    ----------
        topology : openmm.app.Topology
            The topology of the alanine dipeptide model
        positions : list[openmm.Vec3]
            The positions of all alanine dipeptide and water (if any) atoms
        phi : `openxps.CollectiveVariable`
            The Ramachandran dihedral angle :math:`\\phi` of the alanine dipeptide molecule
        psi : `openxps.CollectiveVariable`
            The Ramachandran dihedral angle :math:`\\psi` of the alanine dipeptide molecule

    Examples
    --------
        >>> import openxps
        >>> model = openxps.AlanineDipeptideModel()
        >>> model.phi
        phi: CustomTorsionForce in radian

        >>> import openxps
        >>> model = openxps.AlanineDipeptideModel(water_model='tip3p')
        >>> model.topology.getNumAtoms()
        1522

    """

    def __init__(
        self,
        force_field: str = 'amber03',
        water_model: Optional[str] = None,
        number: Optional[int] = 500,
    ):
        pdb = app.PDBFile(os.path.join(openxps.__path__[0], 'data', 'alanine-dipeptide.pdb'))
        if water_model is None:
            self.topology = pdb.topology
            self.positions = pdb.positions
        else:
            force_field = app.ForceField('amber03.xml', f'{water_model}.xml')
            modeller = app.Modeller(pdb.topology, pdb.positions)
            modeller.addSolvent(force_field, model=water_model, numAdded=number)
            self.topology = modeller.topology
            self.positions = modeller.positions
        atoms = [(a.name, a.residue.name) for a in self.topology.atoms()]
        phi_atoms = [('C', 'ACE'), ('N', 'ALA'), ('CA', 'ALA'), ('C', 'ALA')]
        self.phi = openxps.CollectiveVariable('phi', openmm.CustomTorsionForce('theta'), 'radians')
        self.phi.force.addTorsion(*[atoms.index(i) for i in phi_atoms], [])
        psi_atoms = [('N', 'ALA'), ('CA', 'ALA'), ('C', 'ALA'), ('N', 'NME')]
        self.psi = openxps.CollectiveVariable('psi', openmm.CustomTorsionForce('theta'), 'radians')
        self.psi.force.addTorsion(*[atoms.index(i) for i in psi_atoms], [])
