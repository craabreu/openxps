"""
Unit and regression test for the openxps package.
"""

# Import package, test suite, and other packages as needed
import io
# import pytest
import openxps
import openmm
import os
import sys
import tempfile

import numpy as np


def test_openxps_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "openxps" in sys.modules


def test_collective_variable_serialization():
    torsion = openmm.CustomTorsionForce('theta')
    torsion.addTorsion(4, 6, 8, 14, [])
    phi = openxps.CollectiveVariable('phi', torsion, 'unit.radians')
    pipe = io.StringIO()
    openxps.serialize(phi, pipe)
    pipe.seek(0)
    new = openxps.deserialize(pipe)
    assert new.__repr__() == phi.__repr__()


def test_extended_space_variable_serialization():
    # Extended-space variable
    model = openxps.AlanineDipeptideModel()
    old = openxps.ExtendedSpaceVariable('s_phi', -np.pi, np.pi, True, 1.0, model.phi, 1.0)
    pipe = io.StringIO()
    openxps.serialize(old, pipe)
    pipe.seek(0)
    print(pipe.getvalue())
    new = openxps.deserialize(pipe)
    assert new.__repr__() == old.__repr__()


def test_serialization_to_file():
    torsion = openmm.CustomTorsionForce('theta')
    torsion.addTorsion(4, 6, 8, 14, [])
    phi = openxps.CollectiveVariable('phi', torsion, 'unit.radians')
    file = tempfile.NamedTemporaryFile(delete=False)
    file.close()
    openxps.serialize(phi, file.name)
    new = openxps.deserialize(file.name)
    os.remove(file.name)
    assert new.__repr__() == phi.__repr__()
