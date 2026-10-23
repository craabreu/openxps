"""
Physical-consistency tests for InnerProductCoupling.

The central invariant is that the force the extension system applies to each
dynamical variable equals minus the derivative of the physical potential energy
with respect to that variable.
"""

import math
from copy import copy, deepcopy

import cvpack
import numpy as np
import openmm as mm
import pytest
import yaml
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
    "nonlinear embedded function": (
        lambda: _ipc(
            "scaling*10*r^2*(1+gamma^2) + scaling*exp(-gamma)*r",
            ["scaling", "gamma"],
            [_dv("lam")],
            {"scaling": "(1-cos(pi*lam))/2"},
            embedded_functions={"gamma": "sin(lam)^2+lam"},
            pi=math.pi,
        ),
        [0.3],
    ),
    "nonlinear embedded function with the form of a prefactor": (
        lambda: _ipc(
            "a*10*r^2*(1+b^2)",
            ["a", "b"],
            [_dv("lam")],
            {"a": "lam^2"},
            embedded_functions={"b": "lam^2"},
        ),
        [0.3],
    ),
    "nonlinear embedded identity": (
        lambda: _ipc(
            "gamma*10*r^2*(1+gamma^2)",
            ["gamma"],
            [_dv("lam")],
            embedded_functions={"gamma": "lam"},
        ),
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


def _derivative_requests(monkeypatch, nan_without_energy=False):
    """Record getState calls that request derivatives, optionally faking a force
    that returns NaN derivatives unless the energy is also requested."""
    requests = []
    get_state = mm.Context.getState

    class NaNDerivatives:
        def __init__(self, state):
            self._state = state

        def getEnergyParameterDerivatives(self):
            derivatives = self._state.getEnergyParameterDerivatives()
            return dict.fromkeys(derivatives, math.nan)

    def spy(self, *args, **kwargs):
        state = get_state(self, *args, **kwargs)
        if not kwargs.get("getParameterDerivatives"):
            return state
        requests.append(bool(kwargs.get("getEnergy")))
        if nan_without_energy and not kwargs.get("getEnergy"):
            return NaNDerivatives(state)
        return state

    monkeypatch.setattr(mm.Context, "getState", spy)
    return requests


def _scaling_context():
    return _context(
        _ipc("scaling*10*r^2", ["scaling"], [_dv("lam")], {"scaling": "lam"})
    )


def test_energy_is_not_requested_for_ordinary_forces(monkeypatch):
    context = _scaling_context()
    requests = _derivative_requests(monkeypatch)
    context.synchronize()
    context.synchronize()
    assert requests == [False, False]


def test_energy_is_requested_after_nan_derivatives(monkeypatch):
    """Forces such as SlicedNonbondedForce give NaN derivatives without the energy."""
    context = _scaling_context()
    name = "derivative_with_respect_to_scaling"
    expected = context.getExtensionContext().getParameter(name)
    requests = _derivative_requests(monkeypatch, nan_without_energy=True)
    context.synchronize()
    assert requests == [False, True]
    assert context.getExtensionContext().getParameter(name) == pytest.approx(expected)
    context.synchronize()
    assert requests == [False, True, True]


def test_sliced_nonbonded_force_scaling_derivative():
    """SlicedNonbondedForce returns NaN derivatives unless the energy is requested."""
    nbs = pytest.importorskip("nonbondedslicing")
    nonbonded = mm.NonbondedForce()
    nonbonded.addParticle(1.0, 0.3, 0.0)
    nonbonded.addParticle(-1.0, 0.3, 0.0)
    force = nbs.SlicedNonbondedForce(nonbonded, 2)
    force.setParticleSubset(1, 1)
    force.addGlobalParameter("scaling", 1.0)
    force.addScalingParameter("scaling", 0, 1, True, False)
    force.addEnergyParameterDerivative("scaling")
    coupling = xps.InnerProductCoupling([force], [_dv("scaling")])
    context = _context(coupling)
    context.setDynamicalVariableValues([0.5])
    state = context.getExtensionContext().getState(getForces=True)
    coulomb = -138.935456 / 0.7
    assert state.getForces()._value[0][0] == pytest.approx(-coulomb, rel=1e-6)


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


def test_sum_order_survives_copy_and_yaml():
    """Merging interleaved inner product couplings does not reorder the variables."""
    b = xps.DynamicalVariable(
        "b", mmunit.nanometer, 1 * mmunit.dalton, NoBounds(0, 1, mmunit.nanometer)
    )
    harmonic = xps.HarmonicCoupling(
        cvpack.Distance(0, 1, name="d"),
        b,
        100 * mmunit.kilojoule_per_mole / mmunit.nanometer**2,
    )
    total = _ipc("a*r", ["a"], [_dv("a")]) + harmonic + _ipc("c*r^2", ["c"], [_dv("c")])
    clones = {
        "copy": copy(total),
        "deepcopy": deepcopy(total),
        "yaml": yaml.safe_load(yaml.safe_dump(total)),
    }
    for kind, coupling in {"original": total, **clones}.items():
        names = [dv.name for dv in coupling.getDynamicalVariables()]
        assert names == ["a", "b", "c"], kind
    nested = deepcopy(total) + _ipc("e*r", ["e"], [_dv("e")])
    assert [dv.name for dv in nested.getDynamicalVariables()] == ["a", "b", "c", "e"]


@pytest.mark.parametrize("name", ["λ", "λ_vdw"])
def test_non_ascii_dv_names(name):
    """Non-ASCII identifiers work as parameters and in functions."""
    direct = _context(_ipc(f"{name}*10*r^2", [name], [_dv(name)]))
    assert _physical_energy(direct, [0.3]) == pytest.approx(0.3 * 10 * 0.7**2)
    function = _context(
        _ipc("s*10*r^2", ["s"], [_dv(name)], {"s": f"{name}^2"}),
    )
    assert _physical_energy(function, [0.3]) == pytest.approx(0.3**2 * 10 * 0.7**2)


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


def _embedded_coupling():
    return _ipc(
        "scaling*10*r^2*(1+gamma^2)",
        ["scaling", "gamma"],
        [_dv("lam")],
        {"scaling": "lam"},
        embedded_functions={"gamma": "lam^2"},
    )


def test_setting_variables_refreshes_derivatives():
    """Without an explicit synchronization, the extension force is up to date."""
    context = _context(_embedded_coupling())
    context.setDynamicalVariableValues([0.3])
    state = context.getExtensionContext().getState(getForces=True)
    force = state.getForces()._value[0][0]
    h = 1e-5
    gradient = (
        _physical_energy(context, [0.3 + h]) - _physical_energy(context, [0.3 - h])
    ) / (2 * h)
    assert force == pytest.approx(-gradient, rel=1e-5)


@pytest.mark.parametrize(
    "factory",
    [
        _embedded_coupling,
        lambda: _ipc(
            "gamma*r^2*(1+gamma^2)",
            ["gamma"],
            [_dv("lam")],
            embedded_functions={"gamma": "lam"},
        ),
        lambda: _ipc("a*r", ["a"], [_dv("a")]) + _embedded_coupling(),
    ],
    ids=["embedded function", "embedded identity", "sum"],
)
def test_embedded_functions_require_lockstep(factory):
    coupling = factory()
    assert coupling.requiresLockstep()
    system = xps.ExtendedSpaceSystem(_two_particles(), coupling)
    integrator = xps.SplitIntegrator(
        xps.integrators.SymmetricVerletIntegrator(0.001),
        xps.integrators.SymmetricVerletIntegrator(0.0005),
    )
    with pytest.raises(ValueError, match="require a LockstepIntegrator"):
        xps.ExtendedSpaceContext(system, integrator, REFERENCE)


def test_linear_coupling_does_not_require_lockstep():
    assert not _ipc("lam^2*10*r^2", ["lam"], [_dv("lam")]).requiresLockstep()


def test_embedded_functions_survive_copy_and_yaml():
    total = _embedded_coupling() + _ipc(
        "r^2*exp(-delta)", ["delta"], [_dv("lam")], embedded_functions={"delta": "lam"}
    )
    for coupling in (
        total,
        copy(total),
        deepcopy(total),
        yaml.safe_load(yaml.safe_dump(total)),
    ):
        (merged,) = coupling.getCouplings()
        assert merged.requiresLockstep()
        names = sorted(fn.getName() for fn in merged._embedded_functions)
        assert names == ["delta", "gamma"]


def test_sum_rejects_parameter_embedded_in_only_one_coupling():
    linear = _ipc("scaling*3*r", ["scaling"], [_dv("lam")], {"scaling": "lam^2"})
    embedded = _ipc(
        "r^2*exp(-scaling)",
        ["scaling"],
        [_dv("lam")],
        embedded_functions={"scaling": "lam^2"},
    )
    with pytest.raises(ValueError, match='"scaling" is embedded in some'):
        _ = linear + embedded


def test_parameter_both_function_and_embedded_function_is_rejected():
    with pytest.raises(ValueError, match="both functions and embedded functions"):
        _ipc(
            "scaling*r",
            ["scaling"],
            [_dv("lam")],
            {"scaling": "lam"},
            embedded_functions={"scaling": "lam"},
        )


def test_function_named_as_dynamical_variable_is_rejected():
    with pytest.raises(ValueError, match="same names as dynamical variables"):
        xps.InnerProductCoupling(
            [_bond("lam*r", ["lam"])], [_dv("lam")], functions={"lam": "lam^2"}
        )


def _oscillator_context(integrator, bounds, embedded=False):
    system = mm.System()
    system.addParticle(10.0)
    system.addParticle(10.0)
    bond = mm.HarmonicBondForce()
    bond.addBond(0, 1, 0.5, 1000.0)
    system.addForce(bond)
    lam = xps.DynamicalVariable(
        "lam", DIMENSIONLESS, 0.5 * mmunit.dalton * mmunit.nanometer**2, bounds
    )
    if embedded:
        coupling = _ipc(
            "scaling*50*(r-gamma)^2",
            ["scaling", "gamma"],
            [lam],
            {"scaling": "(1-cos(3*lam))/2"},
            embedded_functions={"gamma": "0.1*lam^2"},
        )
    else:
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
    _assert_energy_conserved(_oscillator_context(integrator(), bounds))


@pytest.mark.parametrize(
    "bounds",
    [NoBounds(0, 1, DIMENSIONLESS), ReflectiveBounds(0, 1, DIMENSIONLESS)],
    ids=["unbounded", "reflective"],
)
def test_lockstep_dynamics_with_embedded_functions_conserve_energy(bounds):
    integrator = xps.LockstepIntegrator(mm.VerletIntegrator(0.0005))
    _assert_energy_conserved(_oscillator_context(integrator, bounds, embedded=True))


def _assert_energy_conserved(context):
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
