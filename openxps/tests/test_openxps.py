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
    phi = xps.CollectiveVariable('phi', torsion, 'unit.radians', 360*unit.degrees)
    pipe = io.StringIO()
    xps.serialize(phi, pipe)
    pipe.seek(0)
    new = xps.deserialize(pipe)
    assert new.__repr__() == phi.__repr__()


def test_collective_variable_exceptions():
    torsion = mm.CustomTorsionForce('theta')
    with pytest.raises(TypeError):
        xps.CollectiveVariable('phi', torsion, 'radians', 1*unit.angstrom)
    with pytest.raises(ValueError):
        xps.CollectiveVariable('1phi', torsion, 'radians', 1*unit.angstrom)


def test_auxiliary_variable_serialization():
    old = xps.AuxiliaryVariable('s_phi', 'radians', 'periodic', -np.pi, np.pi, 1.0)
    pipe = io.StringIO()
    xps.serialize(old, pipe)
    pipe.seek(0)
    new = xps.deserialize(pipe)
    assert new.__repr__() == old.__repr__()


def test_auxiliary_variable_exceptions():
    with pytest.raises(TypeError):
        xps.AuxiliaryVariable('s_phi', 'radians', 'periodic', -np.pi, np.pi, 1.0*unit.angstrom)


def test_serialization_to_file():
    torsion = mm.CustomTorsionForce('theta')
    torsion.addTorsion(4, 6, 8, 14, [])
    phi = xps.CollectiveVariable('phi', torsion, 'unit.radians', 360*unit.degrees)
    file = tempfile.NamedTemporaryFile(delete=False)
    file.close()
    xps.serialize(phi, file.name)
    new = xps.deserialize(file.name)
    os.remove(file.name)
    assert new.__repr__() == phi.__repr__()
