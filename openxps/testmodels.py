"""
.. module:: testmodels
   :platform: Unix, Windows, macOS
   :synopsis: Unified Free Energy Dynamics Test Model Systems

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _Topology: http://docs.openmm.org/latest/api-python/generated/openmm.app.topology.Topology.html

"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import openmm as mm
from openmm import app

import openxps as xps


class AlanineDipeptideModel:
    """
    A model consisting of a single alanine-dipeptide molecule, either in a vacuum or solvated
    in explicit water.

    Parameters
    ----------
        water_model
            the water model to be used. If this is `None`, then the alanine dipeptide will be
            simulated in a vacuum. Available options are "spce", "tip3p", "tip4pew", and "tip5p".
        number
            the number of water molecules to be added if `water_model` is not `None`.

    Attributes
    ----------
        topology : :class:`openmm.app.Topology`
            The Topology_ of the alanine dipeptide model
        positions : list[:class:`openmm.Vec3`]
            The positions of all alanine dipeptide and water (if any) atoms
        phi : :class:`~openxps.openxps.CollectiveVariable`
            The Ramachandran dihedral angle :math:`\\phi` of the alanine dipeptide molecule
        psi : :class:`~openxps.openxps.CollectiveVariable`
            The Ramachandran dihedral angle :math:`\\psi` of the alanine dipeptide molecule

    Examples
    --------
        >>> import openxps as xps
        >>> model = xps.AlanineDipeptideModel()
        >>> model.phi
        phi: CustomTorsionForce (period=6.283185307179586 rad)

        >>> import openxps as xps
        >>> model = xps.AlanineDipeptideModel(water_model='tip3p')
        >>> model.topology.getNumAtoms()
        1522

    """

    def __init__(
        self,
        water_model: Optional[str] = None,
        number: Optional[int] = 500,
    ) -> None:

        source = app.PDBFile(os.path.join(xps.__path__[0], 'data', 'alanine-dipeptide.pdb'))
        if water_model is not None:
            force_field = app.ForceField('amber03.xml', f'{water_model}.xml')
            source = app.Modeller(source.topology, source.positions)
            source.addSolvent(force_field, model=water_model, numAdded=number)
        self.topology = source.topology
        self.positions = source.positions
        all_atoms = [(a.name, a.residue.name) for a in self.topology.atoms()]
        dihedrals = dict(
            phi=[('C', 'ACE'), ('N', 'ALA'), ('CA', 'ALA'), ('C', 'ALA')],
            psi=[('N', 'ALA'), ('CA', 'ALA'), ('C', 'ALA'), ('N', 'NME')]
        )
        for name, atoms in dihedrals.items():
            angle = mm.CustomTorsionForce('theta')
            angle.addTorsion(*[all_atoms.index(i) for i in atoms], [])
            setattr(self, name, xps.CollectiveVariable(name, angle, 'radians', 2*np.pi))
