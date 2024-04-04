"""
Unit tests for the check_system function in openxps.check_system.
"""

import openmm as mm
import pytest
from openmm import unit as mmunit
from openmmtools import testsystems

from openxps.bounds import CIRCULAR
from openxps.check_system import check_system
from openxps.extra_dof import ExtraDOF


def create_system_with_extra_dof(extra_dof_name):
    """
    Helper function to create a fresh OpenMM system for each test,
    with a CustomTorsionForce that depends on the provided global parameter.
    """
    model = testsystems.AlanineDipeptideVacuum()
    dihedral = mm.CustomTorsionForce(f"0.5 * k * (theta - {extra_dof_name})^2")
    dihedral.addPerTorsionParameter("k")
    dihedral.addTorsion(6, 8, 14, 16, [1000.0])
    dihedral.addGlobalParameter(extra_dof_name, 0.0)
    model.system.addForce(dihedral)
    return model.system


def create_extra_dof(name="phi_dv"):
    """Helper function to create an ExtraDOF object."""
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    return ExtraDOF(name=name, unit=mmunit.radian, mass=mass, bounds=CIRCULAR)


def test_check_system_without_modifications():
    """
    Test check_system on a system that already includes the necessary global parameter
    and derivative for the provided ExtraDOF.
    """

    extra_dof = create_extra_dof()
    system = create_system_with_extra_dof(extra_dof.name)
    force = system.getForce(system.getNumForces() - 1)
    force.addEnergyParameterDerivative(extra_dof.name)

    # Expect True: system is already correctly configured
    assert check_system(system, (extra_dof,), add_missing_derivatives=False) is True


def test_check_system_with_missing_derivatives_autocorrect():
    """
    Test check_system on a system missing derivative requests for the ExtraDOF, with
    add_missing_derivatives set to True to autocorrect this.
    """
    extra_dof = create_extra_dof()
    system = create_system_with_extra_dof(extra_dof.name)

    # Expect False: missing derivative requests were added
    assert check_system(system, (extra_dof,), add_missing_derivatives=True) is False


def test_check_system_raises_for_missing_global_parameter():
    """
    Test check_system raises an exception when a global parameter for the ExtraDOF is
    missing.
    """
    extra_dof = create_extra_dof()
    model = testsystems.AlanineDipeptideVacuum()

    with pytest.raises(ValueError) as excinfo:
        check_system(model.system, (extra_dof,), add_missing_derivatives=False)
    assert "No forces depend on these global parameters" in str(excinfo.value)


def test_check_system_raises_for_missing_derivatives():
    """
    Test check_system raises an exception when derivative requests for the ExtraDOF are
    missing.
    """
    extra_dof = create_extra_dof()
    system = create_system_with_extra_dof(extra_dof.name)

    with pytest.raises(ValueError) as excinfo:
        check_system(system, (extra_dof,), add_missing_derivatives=False)
    assert "Missing derivative requests in system forces" in str(excinfo.value)
