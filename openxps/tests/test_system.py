"""
Unit tests for the openxps.system module.
"""

import cvpack
import numpy as np
from openmm import unit as mmunit
from openmmtools import testsystems

from openxps import DynamicalVariable, ExtendedSpaceSystem
from openxps.bounds import CIRCULAR, Reflective


def create_dvs():
    """Helper function to create a DynamicalVariable object."""
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    kwargs = {"unit": mmunit.nanometer, "mass": 1 * mmunit.dalton}
    return [
        DynamicalVariable(name="phi0", unit=mmunit.radian, mass=mass, bounds=CIRCULAR),
        DynamicalVariable(name="x0", bounds=None, **kwargs),
        DynamicalVariable(
            name="y0", bounds=Reflective(-1, 1, mmunit.nanometer), **kwargs
        ),
    ]


def create_coupling_potential(
    phi0=180 * mmunit.degrees, unit=mmunit.kilojoule_per_mole
):
    """Helper function to create a MetaCollectiveVariable object."""
    kwargs = {
        "kappa": 1000 * mmunit.kilojoule_per_mole / mmunit.radians**2,
        "alpha": 0.01 * mmunit.kilojoule_per_mole / mmunit.nanometer**2,
        "x0": 1 * mmunit.nanometer,
        "y0": 1 * mmunit.nanometer,
    }
    if phi0 is not None:
        kwargs["phi0"] = phi0
    return cvpack.MetaCollectiveVariable(
        f"0.5*kappa*min(delta_phi,{2 * np.pi}-delta_phi)^2+alpha*(x0-y0)^2"
        "; delta_phi=abs(phi-phi0)",
        [cvpack.Torsion(6, 8, 14, 16, name="phi")],
        unit,
        **kwargs,
    )


def test_get_coupling_potential():
    """Test getCouplingPotential returns the correct potential."""
    model = testsystems.AlanineDipeptideVacuum()
    coupling_potential = create_coupling_potential()
    dvs = create_dvs()

    system = ExtendedSpaceSystem(dvs, coupling_potential, model.system)

    retrieved_potential = system.getCouplingPotential()
    assert retrieved_potential is coupling_potential
    assert (
        retrieved_potential.getEnergyFunction()
        == coupling_potential.getEnergyFunction()
    )
