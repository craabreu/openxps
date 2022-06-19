"""
Unit and regression test for the openxps package.

"""

# Import package, test suite, and other packages as needed
import io
import os
import sys
import tempfile

import numpy as np
import openmm as mm
import pytest
from openmm import unit

import openxps as xps


def test_openxps_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "openxps" in sys.modules


def test_collective_variable_serialization():
    torsion = mm.CustomTorsionForce('theta')
    torsion.addTorsion(4, 6, 8, 14, [])
    phi = xps.CollectiveVariable('phi', torsion, 360*unit.degrees, unit='unit.radians')
    pipe = io.StringIO()
    xps.serialize(phi, pipe)
    pipe.seek(0)
    new = xps.deserialize(pipe)
    assert new.__repr__() == phi.__repr__()


def test_collective_variable_exception():
    torsion = mm.CustomTorsionForce('theta')
    with pytest.raises(TypeError):
        xps.CollectiveVariable('phi', torsion, 1*unit.angstrom, unit='radians')


def test_extended_space_variable_serialization():
    # Extended-space variable
    model = xps.AlanineDipeptideModel()
    old = xps.AuxiliaryVariable('s_phi', -np.pi, np.pi, True, 1.0, model.phi, 1.0)
    pipe = io.StringIO()
    xps.serialize(old, pipe)
    pipe.seek(0)
    print(pipe.getvalue())
    new = xps.deserialize(pipe)
    assert new.__repr__() == old.__repr__()


def test_serialization_to_file():
    torsion = mm.CustomTorsionForce('theta')
    torsion.addTorsion(4, 6, 8, 14, [])
    phi = xps.CollectiveVariable('phi', torsion, 360*unit.degrees, unit='unit.radians')
    file = tempfile.NamedTemporaryFile(delete=False)
    file.close()
    xps.serialize(phi, file.name)
    new = xps.deserialize(file.name)
    os.remove(file.name)
    assert new.__repr__() == phi.__repr__()
