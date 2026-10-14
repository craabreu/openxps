"""
Physical-consistency tests for InnerProductCoupling.

The central invariant is that the force the extension system applies to each
dynamical variable equals minus the derivative of the physical potential energy
with respect to that variable.
"""

import math

import cvpack
import numpy as np
import openmm as mm
import pytest
from openmm import unit as mmunit

import openxps as xps
from openxps.bounds import NoBounds, PeriodicBounds, ReflectiveBounds
from openxps.couplings import CouplingSum

REFERENCE = mm.Platform.getPlatformByName("Reference")
DIMENSIONLESS = mmunit.dimensionless


def _dv(name, bounds=None):
    return xps.DynamicalVariable(
        name,
        DIMENSIONLESS,
        1 * mmunit.dalton * mmunit.nanometer**2,
        bounds or ReflectiveBounds(0, 1, DIMENSIONLESS),
    )


def _bond(expression, parameters):
    force = mm.CustomBondForce(expression)
    for parameter in parameters:
        force.addGlobalParameter(parameter, 1.0)
        force.addEnergyParameterDerivative(parameter)
    force.addBond(0, 1, [])
    return force


def _two_particles():
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    return system


def _context(coupling, system=None):
    context = xps.ExtendedSpaceContext(
        xps.ExtendedSpaceSystem(system or _two_particles(), coupling),
        xps.LockstepIntegrator(mm.VerletIntegrator(0.001)),
        REFERENCE,
    )
    context.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(0.7, 0, 0)])
    return context


def _physical_energy(context, values):
    context.setDynamicalVariableValues(values)
    context.synchronize()
    return context.getState(getEnergy=True).getPotentialEnergy()._value


def _ipc(expression, parameters, dvs, functions=None, **kwargs):
    return xps.InnerProductCoupling(
        [_bond(expression, parameters)], dvs, functions=functions, **kwargs
    )


CASES = {
    "direct parameter": (lambda: _ipc("lam*10*r^2", ["lam"], [_dv("lam")]), [0.3]),
    "function": (
        lambda: _ipc(
            "scaling*10*r^2",
            ["scaling"],
            [_dv("lam")],
            {"scaling": "(1-cos(pi*lam))/2"},
            pi=math.pi,
        ),
        [0.3],
    ),
    "direct parameter beyond reflective bound": (
        lambda: _ipc("lam*10*r^2", ["lam"], [_dv("lam")]),
        [1.2],
    ),
    "function beyond reflective bound": (
        lambda: _ipc("scaling*10*r^2", ["scaling"], [_dv("lam")], {"scaling": "lam^2"}),
        [1.2],
    ),
    "function beyond periodic bound": (
        lambda: _ipc(
            "scaling*10*r^2",
            ["scaling"],
            [_dv("th", PeriodicBounds(0, 1, DIMENSIONLESS))],
            {"scaling": "th^2"},
        ),
        [1.3],
    ),
    "function of two variables": (
        lambda: _ipc(
            "scaling*10*r^2", ["scaling"], [_dv("a"), _dv("b")], {"scaling": "a*b^2"}
        ),
        [0.3, 0.6],
    ),
    "two parameters in one force": (
        lambda: _ipc("a*10*r^2 + b*3*r", ["a", "b"], [_dv("a"), _dv("b")]),
        [0.3, 0.6],
    ),
    "variable both direct and inside a function": (
        lambda: xps.InnerProductCoupling(
            [_bond("lam*10*r^2", ["lam"]), _bond("b*3*r", ["b"])],
            [_dv("lam")],
            functions={"b": "lam^2"},
        ),
        [0.3],
    ),
    "nonlinear in the parameter": (
        lambda: _ipc("lam^2*10*r^2", ["lam"], [_dv("lam")]),
        [0.3],
    ),
    "sum sharing a function": (
        lambda: (
            _ipc("scaling*10*r^2", ["scaling"], [_dv("lam")], {"scaling": "lam^2"})
            + _ipc("scaling*3*r", ["scaling"], [_dv("lam")], {"scaling": "lam^2"})
        ),
        [0.3],
    ),
    "sum sharing a direct parameter": (
        lambda: (
            _ipc("lam*10*r^2", ["lam"], [_dv("lam")])
            + _ipc("lam*3*r", ["lam"], [_dv("lam")])
        ),
        [0.3],
    ),
    "sum sharing a variable through different parameters": (
        lambda: (
            _ipc("a*10*r^2", ["a"], [_dv("lam")], {"a": "lam"})
            + _ipc("b*3*r", ["b"], [_dv("lam")], {"b": "lam^2"})
        ),
        [0.3],
    ),
    "sum with a direct and a function use": (
        lambda: (
            _ipc("lam*10*r^2", ["lam"], [_dv("lam")])
            + _ipc("b*3*r", ["b"], [_dv("lam")], {"b": "lam^2"})
        ),
        [0.3],
    ),
    "sum with a harmonic coupling": (
        lambda: (
            _ipc("lam*10*r^2", ["lam"], [_dv("lam")])
            + xps.HarmonicCoupling(
                cvpack.Distance(0, 1, name="d"),
                xps.DynamicalVariable(
                    "d0",
                    mmunit.nanometer,
                    10 * mmunit.dalton,
                    NoBounds(0, 1, mmunit.nanometer),
                ),
                100 * mmunit.kilojoule_per_mole / mmunit.nanometer**2,
            )
        ),
        [0.7, 0.3],
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_extension_force_is_minus_physical_energy_gradient(case):
    """The extension force on each variable is -dU/ds of the physical system."""
    factory, values = CASES[case]
    context = _context(factory())
    _physical_energy(context, values)
    state = context.getExtensionContext().getState(getForces=True)
    forces = np.array([force[0] for force in state.getForces()._value])
    h = 1e-5
    gradient = []
    for index in range(len(values)):
        up, down = list(values), list(values)
        up[index] += h
        down[index] -= h
        gradient.append(
            (_physical_energy(context, up) - _physical_energy(context, down)) / (2 * h)
        )
    np.testing.assert_allclose(forces, -np.array(gradient), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("case", [c for c in CASES if "nonlinear" not in c])
def test_extension_energy_equals_coupling_energy_when_linear(case):
    """For couplings linear in their parameters, both energies agree."""
    factory, values = CASES[case]
    coupling = factory()
    system = _two_particles()
    context = _context(coupling, system)
    physical = _physical_energy(context, values)
    extension = context.getExtensionContext().getState(getEnergy=True)
    assert extension.getPotentialEnergy()._value == pytest.approx(physical, rel=1e-6)


@pytest.mark.parametrize(
    ("bounds", "raw", "wrapped"),
    [
        (ReflectiveBounds(0, 1, DIMENSIONLESS), 1.2, 0.8),
        (ReflectiveBounds(0, 1, DIMENSIONLESS), -0.3, 0.3),
        (PeriodicBounds(0, 1, DIMENSIONLESS), 1.3, 0.3),
        (PeriodicBounds(0, 1, DIMENSIONLESS), -0.25, 0.75),
    ],
)
def test_functions_receive_wrapped_variables(bounds, raw, wrapped):
    coupling = _ipc(
        "scaling*10*r^2", ["scaling"], [_dv("s", bounds)], {"scaling": "s^2"}
    )
    context = _context(coupling)
    _physical_energy(context, [raw])
    assert context.getParameter("scaling") == pytest.approx(wrapped**2)


def test_sum_merges_inner_product_couplings():
    first = _ipc("scaling*10*r^2", ["scaling"], [_dv("lam")], {"scaling": "lam^2"})
    second = _ipc("scaling*3*r", ["scaling"], [_dv("lam")], {"scaling": "lam^2"})
    total = first + second
    assert isinstance(total, CouplingSum)
    (merged,) = total.getCouplings()
    assert isinstance(merged, xps.InnerProductCoupling)
    assert len(merged.getForces()) == 2
    assert first.getForceGroups() == set() and second.getForceGroups() == set()


def test_sum_rejects_conflicting_function_definitions():
    first = _ipc("scaling*10*r^2", ["scaling"], [_dv("lam")], {"scaling": "lam"})
    second = _ipc("scaling*3*r", ["scaling"], [_dv("lam")], {"scaling": "lam^2"})
    with pytest.raises(ValueError, match='function "scaling" has conflicting'):
        _ = first + second


def test_sum_rejects_conflicting_parameter_values():
    first = _ipc("a*r", ["a"], [_dv("lam")], {"a": "k*lam"}, k=1.0)
    second = _ipc("b*r", ["b"], [_dv("lam")], {"b": "k*lam^2"}, k=2.0)
    with pytest.raises(ValueError, match='parameter "k" has conflicting'):
        _ = first + second


def test_sum_rejects_conflicting_dynamical_variables():
    first = _ipc("scaling*10*r^2", ["scaling"], [_dv("lam")], {"scaling": "lam"})
    other_lam = _dv("lam", ReflectiveBounds(0.0, 2.0, mmunit.dimensionless))
    second = _ipc("scaling*3*r", ["scaling"], [other_lam], {"scaling": "lam"})
    with pytest.raises(ValueError, match='variable "lam" has conflicting'):
        _ = first + second


def test_force_outside_coupling_using_its_parameter_is_rejected():
    system = _two_particles()
    system.addForce(_bond("scaling*3*r", ["scaling"]))
    coupling = _ipc("scaling*10*r^2", ["scaling"], [_dv("lam")], {"scaling": "lam"})
    with pytest.raises(ValueError, match="not part of the coupling.*scaling"):
        xps.ExtendedSpaceSystem(system, coupling)
    assert system.getNumForces() == 1


def test_force_without_global_parameters_is_rejected():
    with pytest.raises(TypeError, match="does not support global parameters"):
        xps.InnerProductCoupling([mm.HarmonicBondForce()], [_dv("lam")])


def test_function_named_as_dynamical_variable_is_rejected():
    with pytest.raises(ValueError, match="same names as dynamical variables"):
        xps.InnerProductCoupling(
            [_bond("lam*r", ["lam"])], [_dv("lam")], functions={"lam": "lam^2"}
        )


def _oscillator_context(integrator, bounds):
    system = mm.System()
    system.addParticle(10.0)
    system.addParticle(10.0)
    bond = mm.HarmonicBondForce()
    bond.addBond(0, 1, 0.5, 1000.0)
    system.addForce(bond)
    lam = xps.DynamicalVariable(
        "lam", DIMENSIONLESS, 0.5 * mmunit.dalton * mmunit.nanometer**2, bounds
    )
    coupling = _ipc(
        "scaling*50*r^2", ["scaling"], [lam], {"scaling": "(1-cos(3*lam))/2"}
    )
    context = xps.ExtendedSpaceContext(
        xps.ExtendedSpaceSystem(system, coupling), integrator, REFERENCE
    )
    context.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(0.6, 0, 0)])
    context.setVelocities([mm.Vec3(0, 0, 0), mm.Vec3(0, 0, 0)])
    context.setDynamicalVariableValues([0.5])
    context.setDynamicalVariableVelocities([3.0])
    return context


@pytest.mark.parametrize(
    "integrator",
    [
        lambda: xps.LockstepIntegrator(mm.VerletIntegrator(0.0005)),
        lambda: xps.SplitIntegrator(
            xps.integrators.SymmetricVerletIntegrator(0.0005),
            xps.integrators.SymmetricVerletIntegrator(0.00025),
        ),
    ],
    ids=["lockstep", "split"],
)
@pytest.mark.parametrize(
    "bounds",
    [NoBounds(0, 1, DIMENSIONLESS), ReflectiveBounds(0, 1, DIMENSIONLESS)],
    ids=["unbounded", "reflective"],
)
def test_dynamics_conserve_energy(integrator, bounds):
    """Total energy of physical and extension systems is conserved."""
    context = _oscillator_context(integrator(), bounds)
    extension = context.getExtensionContext()
    energies = []
    for _ in range(50):
        context.getIntegrator().step(100)
        physical = context.getState(getEnergy=True)
        kinetic = extension.getState(getEnergy=True).getKineticEnergy()
        energies.append(
            (
                physical.getPotentialEnergy() + physical.getKineticEnergy() + kinetic
            )._value
        )
    assert np.ptp(energies) < 1e-3 * np.mean(np.abs(energies))
