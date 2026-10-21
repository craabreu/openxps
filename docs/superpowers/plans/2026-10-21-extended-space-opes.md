# Extended-space OPES Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ExtendedSpaceOPES` and `ExtendedSpaceCombinedOPES` to openxps, OPES samplers on dynamical variables built on openmm-opes the way `ExtendedSpaceMetadynamics` is built on `openmm.app.Metadynamics`.

**Architecture:** Two subclasses of `openmm_opes.OPES` and `openmm_opes.CombinedOPES` in a new module `openxps/opes.py`. Their constructors turn `ExtendedSpaceBiasVariable`s into `openmm.app.BiasVariable`s on the extension particles and hand the extension system to the parent; the methods that take a simulation or a context convert the openxps objects to the extension ones and delegate. Everything else is inherited.

**Tech Stack:** Python ≥ 3.10, OpenMM ≥ 8.4, openmm-opes ≥ 0.2.0 (numpy, scipy), cvpack, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-21-extended-space-opes-design.md`

## Global Constraints

- `openmm-opes>=0.2.0` is a required dependency (pip and conda, channel `mdtools`).
- The constructors take `system`, `variables` and `**kwargs`; every other argument is forwarded to the parent by keyword, unchanged. `bounded` is derived and must not be passed (`TypeError`).
- Kernels are openmm-opes's, Gaussian by default and minimum-image on periodic variables; `kernelShape` is forwarded. `ExtendedSpaceMetadynamics` is not modified.
- Docstrings document `system` and `variables` only and link to https://craabreu.github.io/openmm-opes/latest/api/opes/ and https://craabreu.github.io/openmm-opes/latest/api/combined-opes/ for the rest.
- Tests run on the Reference platform. Commands assume the shared checkout `/data3/craabreu/github_repos/openxps`, the env `/data3/craabreu/envs/lambda-metad` (which has openmm-opes 0.2.0 installed editable from `/data3/craabreu/github_repos/openmm-opes`) and `PYTHONPATH=.:/data3/craabreu/github_repos/cvpack`. Shorthand used below:
  `PY=/data3/craabreu/envs/lambda-metad/bin/python; export PYTHONPATH=.:/data3/craabreu/github_repos/cvpack; RUFF=/data3/craabreu/envs/openxps-test/bin/ruff`
  and `pytest` means `$PY -m pytest -q -o addopts="" -p no:cacheprovider`.
- Commits end with the attribution trailer in force for the session (currently `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and the `Claude-Session:` line).
- Ask the user before creating the branch (shared checkout vs. worktree) and before opening the PR.

## Review Focus

1. A user who passes `bounded=` expects a clear error, not a silent override — Task 2 tests `TypeError`.
2. A dynamical variable with `CircularBounds` is periodic in radians; the `BiasVariable` must get radian limits even when the variable was declared in degrees — Task 2 tests the grid limits of a degree-declared variable.
3. `getBiasEnergy`/`updateContext` called with the raw extension context (as openmm-opes does internally) must behave exactly as with the `ExtendedSpaceContext` — Task 2 and Task 3 test both inputs.
4. Two samplers on the same extension system must land in different force groups and run together — Task 4 tests `stepTogether`.
5. A periodic and a reflective variable together are rejected by openmm-opes; the user should see openmm-opes's message, not a crash deeper down — Task 2 tests the `ValueError`.

---

### Task 1: Dependency, module skeleton and helpers

**Files:**
- Modify: `pyproject.toml` (dependencies list)
- Modify: `devtools/conda-recipes/anaconda/meta.yaml` (`run:` list)
- Modify: `devtools/conda-envs/test_env.yaml`, `docs_env.yaml`, `deployment_env.yaml`, `lint_env.yaml`
- Modify: `docs/conf.py` (extlinks)
- Create: `openxps/opes.py`
- Test: `openxps/tests/test_opes.py`

**Interfaces:**
- Produces: `openxps.opes._biasVariables(system: ExtendedSpaceSystem, variables: Sequence[ExtendedSpaceBiasVariable]) -> tuple[list[openmm.app.BiasVariable], bool]` (the bias variables and the derived `bounded`); `openxps.opes._extensionSimulation(simulation)`; `openxps.opes._extensionContext(context)`.

- [ ] **Step 1: Declare the dependency**

In `pyproject.toml`, add `"openmm-opes>=0.2.0",` to `dependencies` after `"openmm>=8.4",`. In `devtools/conda-recipes/anaconda/meta.yaml`, add `    - openmm-opes >=0.2.0` after `    - openmm >=8.4` under `run:`. In each of the four `devtools/conda-envs/*_env.yaml`, add `  - openmm-opes` right after the `  - openmm` line (the `mdtools` channel is already listed in `docs`, `deployment` and `lint`; add `  - mdtools` under `channels:` of `test_env.yaml` if it is missing, keeping `conda-forge` first).

- [ ] **Step 2: Write the failing tests for the helpers**

Create `openxps/tests/test_opes.py`:

```python
"""Tests for the extended-space OPES samplers."""

import cvpack
import numpy as np
import openmm
import openmm_opes
import pytest
from openmm import app, unit
from openmmtools import testsystems

import openxps as xps
from openxps.opes import _biasVariables, _extensionContext, _extensionSimulation

REFERENCE = openmm.Platform.getPlatformByName("Reference")
KELVIN = unit.kelvin
MASS = 3 * unit.dalton * (unit.nanometer / unit.radian) ** 2


def _circular_dv(name="phi0"):
    return xps.DynamicalVariable(name, unit.radian, MASS, xps.CircularBounds())


def _reflective_dv(name="s", lower=0.0, upper=1.0):
    return xps.DynamicalVariable(
        name,
        unit.dimensionless,
        1 * unit.dalton * unit.nanometer**2,
        xps.ReflectiveBounds(lower, upper, unit.dimensionless),
    )


def _alanine_system(dv, cv=None):
    """Alanine dipeptide with a harmonic coupling between a torsion and ``dv``."""
    model = testsystems.AlanineDipeptideVacuum()
    cv = cv or cvpack.Torsion(6, 8, 14, 16, name="phi")
    kappa = 1000 * unit.kilojoules_per_mole / dv.unit**2
    return model, xps.ExtendedSpaceSystem(model.system, xps.HarmonicCoupling(cv, dv, kappa))


def test_bias_variables_follow_the_dynamical_variables():
    dv = _circular_dv()
    _, system = _alanine_system(dv)
    variable = xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree, grid_width=73)
    (bias_variable,), bounded = _biasVariables(system, [variable])
    assert isinstance(bias_variable, app.BiasVariable)
    assert bias_variable.periodic
    assert not bounded
    assert bias_variable.minValue == pytest.approx(-np.pi)
    assert bias_variable.maxValue == pytest.approx(np.pi)
    assert bias_variable.biasWidth == pytest.approx(np.radians(18))
    assert bias_variable.gridWidth == 73


def test_reflective_variable_sets_bounded():
    dv = _reflective_dv()
    _, system = _alanine_system(dv, cv=cvpack.Distance(0, 1, name="d"))
    variable = xps.ExtendedSpaceBiasVariable(dv, 0.05 * unit.dimensionless)
    (bias_variable,), bounded = _biasVariables(system, [variable])
    assert bounded
    assert not bias_variable.periodic
    assert (bias_variable.minValue, bias_variable.maxValue) == (0.0, 1.0)


def test_unknown_and_unbounded_variables_are_rejected():
    dv = _circular_dv()
    _, system = _alanine_system(dv)
    other = xps.ExtendedSpaceBiasVariable(_circular_dv("psi0"), 18 * unit.degree)
    with pytest.raises(ValueError, match="not found"):
        _biasVariables(system, [other])
    free = xps.DynamicalVariable(
        "free", unit.nanometer, 1 * unit.dalton, xps.NoBounds(0, 1, unit.nanometer)
    )
    _, free_system = _alanine_system(free, cv=cvpack.Distance(0, 1, name="d"))
    with pytest.raises(ValueError, match="no bounds"):
        _biasVariables(
            free_system, [xps.ExtendedSpaceBiasVariable(free, 0.1 * unit.nanometer)]
        )


def test_conversions_pass_converted_objects_through():
    dv = _circular_dv()
    model, system = _alanine_system(dv)
    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        system,
        xps.LockstepIntegrator(openmm.VerletIntegrator(1 * unit.femtosecond)),
        REFERENCE,
    )
    wrapped = _extensionSimulation(simulation)
    assert wrapped.context is simulation.context.getExtensionContext()
    assert _extensionSimulation(wrapped) is wrapped
    extension = _extensionContext(simulation.context)
    assert extension is simulation.context.getExtensionContext()
    assert _extensionContext(extension) is extension
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest openxps/tests/test_opes.py`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'openxps.opes'`.

- [ ] **Step 4: Write the module skeleton**

Create `openxps/opes.py`:

```python
"""
.. module:: openxps.opes
   :platform: Linux, MacOS, Windows
   :synopsis: On-the-fly probability enhanced sampling of dynamical variables.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

import openmm as mm
from openmm import app as mmapp

from .bounds import NoBounds, ReflectiveBounds
from .context import ExtendedSpaceContext
from .metadynamics import ExtendedSpaceBiasVariable, _SimulationWrapper
from .system import ExtendedSpaceSystem


def _biasVariables(
    system: ExtendedSpaceSystem,
    variables: t.Sequence[ExtendedSpaceBiasVariable],
) -> tuple[list[mmapp.BiasVariable], bool]:
    """Bias variables on the extension particles, and whether any is reflective.

    Mirrors :class:`ExtendedSpaceMetadynamics`: each dynamical variable, in MD
    units, must belong to the system and have bounds, since the bias lives on a
    finite grid. ``bounded`` is True if any variable has reflective bounds; with
    unbounded variables rejected, the non-periodic variables are exactly the
    reflective ones.
    """
    system_dvs = system.getDynamicalVariables()
    bias_variables = []
    bounded = False
    for variable in variables:
        dv = variable.dynamical_variable.in_md_units()
        if dv not in system_dvs:
            raise ValueError(f"Dynamical variable {dv.name} not found in system")
        if isinstance(dv.bounds, NoBounds):
            raise ValueError(
                f"Dynamical variable {dv.name} has no bounds, but OPES needs a "
                "finite grid. Use ReflectiveBounds over the range the variable may "
                "explore."
            )
        bounded = bounded or isinstance(dv.bounds, ReflectiveBounds)
        index = system_dvs.index(dv)
        bias_variables.append(
            mmapp.BiasVariable(
                dv.createCollectiveVariable(index),
                dv.bounds.lower * dv.bounds.unit,
                dv.bounds.upper * dv.bounds.unit,
                variable.sigma,
                dv.isPeriodic(),
                variable.grid_width,
            )
        )
    return bias_variables, bounded


def _extensionSimulation(simulation):
    """A view of ``simulation`` whose context is the extension context.

    openmm-opes calls back into the sampler with the object it was given, so an
    already converted simulation is returned unchanged.
    """
    if isinstance(simulation, _SimulationWrapper):
        return simulation
    return _SimulationWrapper(simulation)


def _extensionContext(context: mm.Context) -> mm.Context:
    """The extension context of an :class:`ExtendedSpaceContext`, else ``context``."""
    if isinstance(context, ExtendedSpaceContext):
        return context.getExtensionContext()
    return context
```

Note: `openmm.app.BiasVariable` takes `(force, minValue, maxValue, biasWidth, periodic, gridWidth)` and converts quantities to MD units itself, which is what `ExtendedSpaceMetadynamics` relies on; `dv.bounds.unit` is already the MD unit after `in_md_units()`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest openxps/tests/test_opes.py`
Expected: 4 passed.

- [ ] **Step 6: Add the openmm-opes extlink**

In `docs/conf.py`, inside the `extlinks` dict after the `"CVPack"` entry, add:

```python
    "OpenMMOPES": (
        "https://craabreu.github.io/openmm-opes/latest/api/%s/",
        "openmm_opes.%s",
    ),
```

so that docstrings can write `:OpenMMOPES:`opes`` and `:OpenMMOPES:`combined-opes``.

- [ ] **Step 7: Lint and commit**

Run: `$RUFF check openxps && $RUFF format --check openxps`
Expected: `All checks passed!` and `N files already formatted`.

```bash
git add pyproject.toml devtools docs/conf.py openxps/opes.py openxps/tests/test_opes.py
git commit -m "Add openmm-opes as a dependency and the extended-space OPES helpers"
```

---

### Task 2: `ExtendedSpaceOPES`

**Files:**
- Modify: `openxps/opes.py`
- Modify: `openxps/__init__.py`
- Test: `openxps/tests/test_opes.py`

**Interfaces:**
- Consumes: `_biasVariables`, `_extensionSimulation`, `_extensionContext` from Task 1.
- Produces: `class ExtendedSpaceOPES(openmm_opes.OPES)` with `__init__(self, system, variables, **kwargs)`, overrides `step(simulation, steps)`, `getCollectiveVariables(simulation)`, `getBiasEnergy(context)`, `updateContext(context)`, and `@staticmethod stepTogether(simulation, samplers, steps)`.

- [ ] **Step 1: Write the failing tests**

Append to `openxps/tests/test_opes.py`:

```python
def _opes(system, variables, **kwargs):
    settings = dict(
        temperature=300 * KELVIN,
        barrier=20 * unit.kilojoules_per_mole,
        frequency=10,
        varianceFrequency=None,
    )
    settings.update(kwargs)
    return xps.ExtendedSpaceOPES(system, variables, **settings)


def _simulation(model, system, seed=1):
    integrator = openmm.LangevinMiddleIntegrator(
        300 * KELVIN, 1 / unit.picosecond, 2 * unit.femtosecond
    )
    integrator.setRandomNumberSeed(seed)
    simulation = xps.ExtendedSpaceSimulation(
        model.topology, system, xps.LockstepIntegrator(integrator), REFERENCE
    )
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * KELVIN, seed)
    simulation.context.setDynamicalVariableValues([180 * unit.degree])
    simulation.context.setDynamicalVariableVelocitiesToTemperature(300 * KELVIN, seed)
    return simulation


def test_opes_adds_one_force_to_the_extension_system():
    dv = _circular_dv()
    _, system = _alanine_system(dv)
    before = system.getExtensionSystem().getNumForces()
    sampler = _opes(system, [xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)])
    extension = system.getExtensionSystem()
    assert extension.getNumForces() == before + 1
    assert isinstance(sampler, openmm_opes.OPES)
    assert sampler.variables[0].periodic
    assert not sampler.bounded


def test_opes_rejects_bounded_and_forwards_other_keywords():
    dv = _reflective_dv()
    _, system = _alanine_system(dv, cv=cvpack.Distance(0, 1, name="d"))
    variable = xps.ExtendedSpaceBiasVariable(dv, 0.05 * unit.dimensionless)
    with pytest.raises(TypeError, match="bounded"):
        _opes(system, [variable], bounded=True)
    sampler = _opes(system, [variable], exploreMode=True, kernelShape="compact")
    assert sampler.bounded
    assert sampler.exploreMode
    with pytest.raises(ValueError, match="frequency"):
        _opes(system, [variable], frequency=0)


def test_opes_rejects_mixed_periodicity():
    phi0, s = _circular_dv(), _reflective_dv()
    model = testsystems.AlanineDipeptideVacuum()
    coupling = xps.HarmonicCoupling(
        cvpack.Torsion(6, 8, 14, 16, name="phi"),
        phi0,
        1000 * unit.kilojoules_per_mole / unit.radian**2,
    ) + xps.HarmonicCoupling(
        cvpack.Distance(0, 1, name="d"), s, 1000 * unit.kilojoules_per_mole
    )
    system = xps.ExtendedSpaceSystem(model.system, coupling)
    with pytest.raises(ValueError, match="mixed periodic"):
        _opes(
            system,
            [
                xps.ExtendedSpaceBiasVariable(phi0, 18 * unit.degree),
                xps.ExtendedSpaceBiasVariable(s, 0.05 * unit.dimensionless),
            ],
        )


def test_opes_step_and_collective_variables():
    dv = _circular_dv()
    model, system = _alanine_system(dv)
    sampler = _opes(system, [xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)])
    simulation = _simulation(model, system)
    sampler.step(simulation, 50)
    assert simulation.currentStep == 50
    assert sampler.getNumKernels() >= 1
    (value,) = sampler.getCollectiveVariables(simulation)
    (dv_value,) = simulation.context.getDynamicalVariableValues()
    assert value == pytest.approx(dv_value.value_in_unit(unit.radian), abs=1e-6)
    assert sampler.getFreeEnergy().shape == (sampler.variables[0].gridWidth,)


def test_opes_methods_accept_openxps_and_extension_objects():
    dv = _circular_dv()
    model, system = _alanine_system(dv)
    sampler = _opes(system, [xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)])
    simulation = _simulation(model, system)
    sampler.step(simulation, 20)
    context = simulation.context
    extension = context.getExtensionContext()
    assert sampler.getBiasEnergy(context) == sampler.getBiasEnergy(extension)
    assert sampler.getCollectiveVariables(simulation) == sampler.getCollectiveVariables(
        _extensionSimulation(simulation)
    )
    sampler.updateContext(context)  # must not raise
    sampler.updateContext(extension)
    assert sampler.getBiasEnergy(context) == sampler.getBiasEnergy(extension)


class _Replay:
    """A stand-alone 'simulation' of an extension system that replays recorded values.

    openmm-opes needs only ``context``, ``currentStep`` and ``step``.
    """

    def __init__(self, context, values, frequency):
        self.context = context
        self.currentStep = 0
        self._values = values
        self._frequency = frequency

    def step(self, steps):
        self.currentStep += steps
        if self.currentStep % self._frequency == 0:
            value = self._values[self.currentStep // self._frequency - 1]
            self.context.setPositions([openmm.Vec3(value, 0, 0)])


def test_opes_matches_plain_openmm_opes_on_the_extension_system():
    """The adapter deposits what a plain OPES would deposit on the extension system."""
    dv = _circular_dv()
    model, system = _alanine_system(dv)
    plain_extension = openmm.XmlSerializer.clone(system.getExtensionSystem())
    frequency = 10
    sampler = _opes(
        system, [xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)], frequency=frequency
    )
    simulation = _simulation(model, system)
    values = []
    for _ in range(8):
        sampler.step(simulation, frequency)
        (value,) = simulation.context.getDynamicalVariableValues()
        values.append(value.value_in_unit(unit.radian))

    (bias_variable,), bounded = _biasVariables(
        xps.ExtendedSpaceSystem(
            testsystems.AlanineDipeptideVacuum().system,
            xps.HarmonicCoupling(
                cvpack.Torsion(6, 8, 14, 16, name="phi"),
                dv,
                1000 * unit.kilojoules_per_mole / unit.radian**2,
            ),
        ),
        [xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)],
    )
    plain = openmm_opes.OPES(
        plain_extension,
        [bias_variable],
        temperature=300 * KELVIN,
        barrier=20 * unit.kilojoules_per_mole,
        frequency=frequency,
        varianceFrequency=None,
        bounded=bounded,
    )
    context = openmm.Context(
        plain_extension, openmm.VerletIntegrator(1 * unit.femtosecond), REFERENCE
    )
    context.setPositions([openmm.Vec3(0, 0, 0)])
    replay = _Replay(context, values, frequency)
    plain.step(replay, frequency * len(values))

    assert plain.getNumKernels() == sampler.getNumKernels()
    np.testing.assert_allclose(plain.getBias(), sampler.getBias(), rtol=1e-10)
    np.testing.assert_allclose(
        plain.getFreeEnergy(), sampler.getFreeEnergy(), rtol=1e-10
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest openxps/tests/test_opes.py -k opes`
Expected: FAIL with `AttributeError: module 'openxps' has no attribute 'ExtendedSpaceOPES'` (the helper tests still pass).

- [ ] **Step 3: Implement the class**

Append to `openxps/opes.py` (add `import openmm_opes` with the imports):

```python
class ExtendedSpaceOPES(openmm_opes.OPES):
    """On-the-fly probability enhanced sampling (OPES) of dynamical variables.

    A bias potential is applied to dynamical variables of an
    :class:`~openxps.ExtendedSpaceSystem` to enhance the sampling of the physical
    coordinates coupled to them. This is :OpenMMOPES:`opes` acting on the extension
    system: OPES and OPES-explore (``exploreMode=True``), multiple walkers and all
    other options are those of openmm-opes, documented there.

    Parameters
    ----------
    system
        The :class:`~openxps.ExtendedSpaceSystem` to be simulated. The bias force
        is added to its extension system when this object is created, so it must
        be created before the :class:`~openxps.ExtendedSpaceSimulation` or
        :class:`~openxps.ExtendedSpaceContext` that uses the system.
    variables
        The dynamical variables to bias and their kernel widths, as
        :class:`~openxps.ExtendedSpaceBiasVariable` objects. Variables with
        :class:`~openxps.bounds.NoBounds` are rejected, since the bias is stored on
        a finite grid. Periodic and reflective variables cannot be mixed.
    **kwargs
        The remaining arguments of :OpenMMOPES:`opes` (``temperature``,
        ``barrier``, ``frequency``, ``varianceFrequency``, ...), except ``bounded``,
        which follows from the variables' bounds.

    .. note::
        Kernels are those of openmm-opes: Gaussian by default and minimum-image on
        periodic variables, whereas :class:`~openxps.ExtendedSpaceMetadynamics`
        uses von Mises kernels. ``sigma`` keeps its openmm-opes meaning: with
        ``varianceFrequency=None``, the standard deviation of the unbiased
        distribution.

    .. note::
        ``extraBias`` entries are evaluated in the extension context, so they must
        be forces of the extension system, such as other samplers on it.

    Example
    -------
    >>> import openxps as xps
    >>> from openmm import unit
    >>> import cvpack
    >>> import openmm
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> mass = 3 * unit.dalton*(unit.nanometer/unit.radian)**2
    >>> phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    >>> kappa = 1000 * unit.kilojoules_per_mole / unit.radian**2
    >>> system = xps.ExtendedSpaceSystem(
    ...     model.system, xps.HarmonicCoupling(phi, phi0, kappa)
    ... )
    >>> opes = xps.ExtendedSpaceOPES(
    ...     system,
    ...     [xps.ExtendedSpaceBiasVariable(phi0, 18 * unit.degrees)],
    ...     temperature=300 * unit.kelvin,
    ...     barrier=20 * unit.kilojoules_per_mole,
    ...     frequency=100,
    ...     varianceFrequency=None,
    ... )
    >>> simulation = xps.ExtendedSpaceSimulation(
    ...     model.topology,
    ...     system,
    ...     xps.LockstepIntegrator(
    ...         openmm.LangevinMiddleIntegrator(
    ...             300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond
    ...         )
    ...     ),
    ...     openmm.Platform.getPlatformByName("Reference"),
    ... )
    >>> context = simulation.context
    >>> context.setPositions(model.positions)
    >>> context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    >>> context.setDynamicalVariableValues([180 * unit.degree])
    >>> context.setDynamicalVariableVelocitiesToTemperature(300 * unit.kelvin, 1234)
    >>> opes.step(simulation, 200)
    >>> opes.getNumKernels()
    2
    """

    def __init__(
        self,
        system: ExtendedSpaceSystem,
        variables: t.Sequence[ExtendedSpaceBiasVariable],
        **kwargs,
    ) -> None:
        if "bounded" in kwargs:
            raise TypeError(
                "bounded is set from the bounds of the dynamical variables; "
                "do not pass it"
            )
        bias_variables, bounded = _biasVariables(system, variables)
        super().__init__(
            system.getExtensionSystem(), bias_variables, bounded=bounded, **kwargs
        )

    def step(self, simulation, steps: int) -> None:
        """Advance an :class:`~openxps.ExtendedSpaceSimulation` by ``steps`` steps."""
        super().step(_extensionSimulation(simulation), steps)

    @staticmethod
    def stepTogether(simulation, samplers, steps: int) -> None:
        """Advance a simulation biased by several extended-space samplers at once.

        See :OpenMMOPES:`opes`. The samplers may be
        :class:`ExtendedSpaceOPES` or :class:`ExtendedSpaceCombinedOPES` objects
        acting on the same extension system.
        """
        openmm_opes.OPES.stepTogether(_extensionSimulation(simulation), samplers, steps)

    def getCollectiveVariables(self, simulation):
        """Current values of the biased dynamical variables, in MD units."""
        return super().getCollectiveVariables(_extensionSimulation(simulation))

    def getBiasEnergy(self, context: mm.Context):
        """The bias energy applied in an :class:`~openxps.ExtendedSpaceContext`."""
        return super().getBiasEnergy(_extensionContext(context))

    def updateContext(self, context: mm.Context) -> None:
        """Push the current bias into an :class:`~openxps.ExtendedSpaceContext`."""
        super().updateContext(_extensionContext(context))
```

The doctest's `getNumKernels()` value must be checked when the test is first run (200 steps at frequency 100 give 2 depositions with a fixed variance); adjust the expected number to what the run prints if openmm-opes merges them.

Export the class: in `openxps/__init__.py`, add `from .opes import ExtendedSpaceOPES  # noqa: F401` after the metadynamics import, and `"ExtendedSpaceOPES",` to `__all__` after `"ExtendedSpaceBiasVariable",`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest openxps/tests/test_opes.py openxps/opes.py --doctest-modules`
Expected: all passed (10 tests + 1 doctest). If the equivalence test fails only on `getNumKernels`, check that `_Replay` sets the position at every deposition step; the kernels must match exactly because the plain sampler sees the same positions and the same bias.

- [ ] **Step 5: Lint and commit**

Run: `$RUFF check openxps && $RUFF format --check openxps`
Expected: clean.

```bash
git add openxps/opes.py openxps/__init__.py openxps/tests/test_opes.py
git commit -m "Add ExtendedSpaceOPES on top of openmm-opes"
```

---

### Task 3: `ExtendedSpaceCombinedOPES`

**Files:**
- Modify: `openxps/opes.py`
- Modify: `openxps/__init__.py`
- Test: `openxps/tests/test_opes.py`

**Interfaces:**
- Consumes: the helpers of Task 1 and `_opes`, `_simulation`, `_Replay` from the test file.
- Produces: `class ExtendedSpaceCombinedOPES(openmm_opes.CombinedOPES)` with the same overrides as Task 2 plus `getComponentBiasEnergies(context)`.

- [ ] **Step 1: Write the failing tests**

Append to `openxps/tests/test_opes.py`:

```python
def _combined(system, variables, **kwargs):
    settings = dict(
        temperature=300 * KELVIN,
        barrier=20 * unit.kilojoules_per_mole,
        exploreBarrier=10 * unit.kilojoules_per_mole,
        frequency=10,
        varianceFrequency=None,
    )
    settings.update(kwargs)
    return xps.ExtendedSpaceCombinedOPES(system, variables, **settings)


def test_combined_opes_adds_one_force_and_rejects_bounded():
    dv = _circular_dv()
    _, system = _alanine_system(dv)
    variable = xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)
    before = system.getExtensionSystem().getNumForces()
    sampler = _combined(system, [variable])
    assert system.getExtensionSystem().getNumForces() == before + 1
    assert isinstance(sampler, openmm_opes.CombinedOPES)
    with pytest.raises(TypeError, match="bounded"):
        _combined(system, [variable], bounded=False)


def test_combined_opes_step_and_component_energies():
    dv = _circular_dv()
    model, system = _alanine_system(dv)
    sampler = _combined(system, [xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)])
    simulation = _simulation(model, system)
    sampler.step(simulation, 50)
    assert simulation.currentStep == 50
    assert all(n >= 1 for n in sampler.getNumKernels())
    context = simulation.context
    extension = context.getExtensionContext()
    opes, explore = sampler.getComponentBiasEnergies(context)
    assert (opes, explore) == sampler.getComponentBiasEnergies(extension)
    total = sampler.getBiasEnergy(context)
    assert (opes + explore).value_in_unit(unit.kilojoules_per_mole) == pytest.approx(
        total.value_in_unit(unit.kilojoules_per_mole), abs=1e-9
    )
    assert sampler.getCollectiveVariables(simulation) == sampler.getCollectiveVariables(
        _extensionSimulation(simulation)
    )
    sampler.updateContext(context)
    sampler.updateContext(extension)
    assert sampler.getFreeEnergy().shape == (sampler.variables[0].gridWidth,)


def test_combined_opes_matches_plain_combined_opes():
    dv = _circular_dv()
    model, system = _alanine_system(dv)
    plain_extension = openmm.XmlSerializer.clone(system.getExtensionSystem())
    frequency = 10
    sampler = _combined(
        system, [xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)], frequency=frequency
    )
    simulation = _simulation(model, system)
    values = []
    for _ in range(8):
        sampler.step(simulation, frequency)
        (value,) = simulation.context.getDynamicalVariableValues()
        values.append(value.value_in_unit(unit.radian))
    (bias_variable,), bounded = _biasVariables(
        xps.ExtendedSpaceSystem(
            testsystems.AlanineDipeptideVacuum().system,
            xps.HarmonicCoupling(
                cvpack.Torsion(6, 8, 14, 16, name="phi"),
                dv,
                1000 * unit.kilojoules_per_mole / unit.radian**2,
            ),
        ),
        [xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)],
    )
    plain = openmm_opes.CombinedOPES(
        plain_extension,
        [bias_variable],
        temperature=300 * KELVIN,
        barrier=20 * unit.kilojoules_per_mole,
        exploreBarrier=10 * unit.kilojoules_per_mole,
        frequency=frequency,
        varianceFrequency=None,
        bounded=bounded,
    )
    context = openmm.Context(
        plain_extension, openmm.VerletIntegrator(1 * unit.femtosecond), REFERENCE
    )
    context.setPositions([openmm.Vec3(0, 0, 0)])
    plain.step(_Replay(context, values, frequency), frequency * len(values))
    assert plain.getNumKernels() == sampler.getNumKernels()
    np.testing.assert_allclose(plain.getBias(), sampler.getBias(), rtol=1e-10)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest openxps/tests/test_opes.py -k combined`
Expected: FAIL with `AttributeError: module 'openxps' has no attribute 'ExtendedSpaceCombinedOPES'`.

- [ ] **Step 3: Implement the class**

Append to `openxps/opes.py`:

```python
class ExtendedSpaceCombinedOPES(openmm_opes.CombinedOPES):
    """OPES and OPES-explore combined (OPES_COM) on dynamical variables.

    This is :OpenMMOPES:`combined-opes` acting on the extension system of an
    :class:`~openxps.ExtendedSpaceSystem`: both components deposit kernels on the
    same dynamical variables, and the total bias is their sum. See
    :class:`ExtendedSpaceOPES` for how the variables are handled; the same notes
    on kernels and ``extraBias`` apply.

    Parameters
    ----------
    system
        The :class:`~openxps.ExtendedSpaceSystem` to be simulated. The bias force
        is added to its extension system when this object is created, so it must
        be created before the simulation or context that uses the system.
    variables
        The dynamical variables to bias and their kernel widths, as
        :class:`~openxps.ExtendedSpaceBiasVariable` objects.
    **kwargs
        The remaining arguments of :OpenMMOPES:`combined-opes` (``temperature``,
        ``barrier``, ``exploreBarrier``, ``frequency``, ``varianceFrequency``, ...),
        except ``bounded``, which follows from the variables' bounds.

    Example
    -------
    >>> import openxps as xps
    >>> from openmm import unit
    >>> import cvpack
    >>> import openmm
    >>> from openmmtools import testsystems
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    >>> mass = 3 * unit.dalton*(unit.nanometer/unit.radian)**2
    >>> phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    >>> kappa = 1000 * unit.kilojoules_per_mole / unit.radian**2
    >>> system = xps.ExtendedSpaceSystem(
    ...     model.system, xps.HarmonicCoupling(phi, phi0, kappa)
    ... )
    >>> opes = xps.ExtendedSpaceCombinedOPES(
    ...     system,
    ...     [xps.ExtendedSpaceBiasVariable(phi0, 18 * unit.degrees)],
    ...     temperature=300 * unit.kelvin,
    ...     barrier=20 * unit.kilojoules_per_mole,
    ...     exploreBarrier=10 * unit.kilojoules_per_mole,
    ...     frequency=100,
    ...     varianceFrequency=None,
    ... )
    >>> simulation = xps.ExtendedSpaceSimulation(
    ...     model.topology,
    ...     system,
    ...     xps.LockstepIntegrator(
    ...         openmm.LangevinMiddleIntegrator(
    ...             300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond
    ...         )
    ...     ),
    ...     openmm.Platform.getPlatformByName("Reference"),
    ... )
    >>> context = simulation.context
    >>> context.setPositions(model.positions)
    >>> context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    >>> context.setDynamicalVariableValues([180 * unit.degree])
    >>> context.setDynamicalVariableVelocitiesToTemperature(300 * unit.kelvin, 1234)
    >>> opes.step(simulation, 200)
    >>> opes.getNumKernels()
    (2, 2)
    """

    def __init__(
        self,
        system: ExtendedSpaceSystem,
        variables: t.Sequence[ExtendedSpaceBiasVariable],
        **kwargs,
    ) -> None:
        if "bounded" in kwargs:
            raise TypeError(
                "bounded is set from the bounds of the dynamical variables; "
                "do not pass it"
            )
        bias_variables, bounded = _biasVariables(system, variables)
        super().__init__(
            system.getExtensionSystem(), bias_variables, bounded=bounded, **kwargs
        )

    def step(self, simulation, steps: int) -> None:
        """Advance an :class:`~openxps.ExtendedSpaceSimulation` by ``steps`` steps."""
        super().step(_extensionSimulation(simulation), steps)

    stepTogether = staticmethod(ExtendedSpaceOPES.stepTogether)

    def getCollectiveVariables(self, simulation):
        """Current values of the biased dynamical variables, in MD units."""
        return super().getCollectiveVariables(_extensionSimulation(simulation))

    def getBiasEnergy(self, context: mm.Context):
        """The total bias energy applied in an :class:`~openxps.ExtendedSpaceContext`."""
        return super().getBiasEnergy(_extensionContext(context))

    def getComponentBiasEnergies(self, context: mm.Context):
        """The OPES and OPES-explore bias energies in an :class:`~openxps.ExtendedSpaceContext`."""
        return super().getComponentBiasEnergies(_extensionContext(context))

    def updateContext(self, context: mm.Context) -> None:
        """Push the current biases into an :class:`~openxps.ExtendedSpaceContext`."""
        super().updateContext(_extensionContext(context))
```

Check the doctest's `getNumKernels()` tuple on the first run as in Task 2. Export the class in `openxps/__init__.py` (`from .opes import ExtendedSpaceCombinedOPES, ExtendedSpaceOPES` and `"ExtendedSpaceCombinedOPES",` in `__all__`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest openxps/tests/test_opes.py openxps/opes.py --doctest-modules`
Expected: all passed.

- [ ] **Step 5: Lint and commit**

Run: `$RUFF check openxps && $RUFF format --check openxps`
Expected: clean.

```bash
git add openxps/opes.py openxps/__init__.py openxps/tests/test_opes.py
git commit -m "Add ExtendedSpaceCombinedOPES"
```

---

### Task 4: Stepping together, state and walkers

**Files:**
- Test: `openxps/tests/test_opes.py`
- Modify: `openxps/opes.py` only if a test uncovers a defect.

**Interfaces:**
- Consumes: `ExtendedSpaceOPES.stepTogether`, `getState`/`setState`, `biasDir`/`walkerId`/`saveFrequency` (inherited).

- [ ] **Step 1: Write the tests**

Append to `openxps/tests/test_opes.py`:

```python
def test_samplers_step_together_on_their_own_schedules():
    dv = _circular_dv()
    model, system = _alanine_system(dv)
    variable = xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)
    opes = _opes(system, [variable], frequency=10)
    combined = _combined(system, [variable], frequency=20)
    groups = {opes._force.getForceGroup(), combined._force.getForceGroup()}
    assert len(groups) == 2
    simulation = _simulation(model, system)
    xps.ExtendedSpaceOPES.stepTogether(simulation, [opes, combined], 60)
    assert simulation.currentStep == 60
    assert opes.getNumKernels() >= 1
    assert all(n >= 1 for n in combined.getNumKernels())
    # Each sampler's bias is in its own group, so they add up in the extension context
    extension = simulation.context.getExtensionContext()
    both = extension.getState(getEnergy=True, groups=groups).getPotentialEnergy()
    sum_of_parts = opes.getBiasEnergy(simulation.context) + opes.barrier
    sum_of_parts += combined.getBiasEnergy(simulation.context)
    sum_of_parts += combined.barrier + combined.exploreBarrier
    assert both.value_in_unit(unit.kilojoules_per_mole) == pytest.approx(
        sum_of_parts.value_in_unit(unit.kilojoules_per_mole), abs=1e-9
    )


def test_state_round_trip_restores_the_bias():
    dv = _circular_dv()
    model, system = _alanine_system(dv)
    variable = xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)
    sampler = _opes(system, [variable])
    sampler.step(_simulation(model, system), 40)
    state = sampler.getState()
    _, fresh_system = _alanine_system(dv)
    restored = _opes(fresh_system, [variable])
    restored.setState(state)
    assert restored.getNumKernels() == sampler.getNumKernels()
    np.testing.assert_allclose(restored.getBias(), sampler.getBias())


def test_two_walkers_share_a_bias(tmp_path):
    dv = _circular_dv()
    walkers = []
    for walker_id in (1, 2):
        model, system = _alanine_system(dv)
        sampler = _opes(
            system,
            [xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree)],
            saveFrequency=20,
            biasDir=str(tmp_path),
            walkerId=walker_id,
        )
        walkers.append((sampler, _simulation(model, system, seed=walker_id)))
    for sampler, simulation in walkers:
        sampler.step(simulation, 40)
    for sampler, simulation in walkers:
        sampler.step(simulation, 20)  # reads the other walker's file
    first, second = (sampler for sampler, _ in walkers)
    assert first.getNumKernels() > 1 and second.getNumKernels() > 1
    assert first.getNumKernels() == second.getNumKernels()
```

- [ ] **Step 2: Run the tests**

Run: `pytest openxps/tests/test_opes.py`
Expected: all passed. If `test_two_walkers_share_a_bias` fails on the kernel counts, read openmm-opes's `_syncWithDisk` to see when peers are folded in (it happens at save steps after the own state is written) and adjust the stepping so that each walker syncs once after the other has written; the assertion that the counts are equal after the final syncs stays.

Note: the combined sampler's `getBiasEnergy` subtracts both barriers (see `CombinedOPES.getBiasEnergy`); the first test adds them back to compare with the raw group energies. If the parent subtracts something else, print both values once and fix the test's bookkeeping, not the class.

- [ ] **Step 3: Commit**

```bash
git add openxps/tests/test_opes.py
git commit -m "Test stepping extended-space OPES samplers together, state and walkers"
```

---

### Task 5: Changelog, docs check and full suite

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `README.md` (feature list, if it lists metadynamics)

- [ ] **Step 1: Changelog**

Under `## Unreleased`, add a `### New` section (or extend it if one exists) with:

```markdown
- `ExtendedSpaceOPES` and `ExtendedSpaceCombinedOPES` bias dynamical variables with
  on-the-fly probability enhanced sampling, built on the openmm-opes package the way
  `ExtendedSpaceMetadynamics` is built on OpenMM's metadynamics. OPES, OPES-explore
  and their combination (OPES_COM), several samplers stepped together, multiple
  walkers and checkpoints are those of openmm-opes, now a dependency.
```

If `README.md` lists `ExtendedSpaceMetadynamics` among the features, add the OPES classes next to it in one sentence.

- [ ] **Step 2: Run the whole suite, doctests included, and the linter**

Run: `pytest openxps --doctest-modules && $RUFF check openxps && $RUFF format --check openxps`
Expected: all passed, clean. The API pages are generated by `docs/conf.py` from the classes exported in `openxps`, so no `.rst` is needed; if the docs environment is available, `cd docs && make html` must build without a warning about `openxps.opes`.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md README.md
git commit -m "Document the extended-space OPES samplers"
```

---

### Task 6: First use case — Gabriela's λ-OPES run with `ExtendedSpaceCombinedOPES`

**Files:**
- Create: `/data3/craabreu/github_repos/lambda-dynamics/run_lambda_opes.py` (outside the openxps repository; no commit in openxps)
- Modify: `/data3/craabreu/github_repos/lambda-dynamics/README.md` (one table row)

**Interfaces:**
- Consumes: `lambda_setup.build("cv")` (Gabriela's `CollectiveVariableCoupling` on θ with `PeriodicBounds(0, 2)`), `lambda_setup.integrator(dt)`, `xps.ExtendedSpaceCombinedOPES`.

- [ ] **Step 1: Write the script**

```python
"""
Gabriela's lambda-OPES run on quinone in water with a single ExtendedSpaceCombinedOPES
(OPES_COM) on theta, instead of a conventional OPES and an OPES-explore stepped
side by side. Barriers as in her script: 126 kJ/mol (OPES) and 21 kJ/mol (explore).
Usage: python run_lambda_opes.py [--time <ps>] [--coupling cv|inner]
"""

import argparse
import time

import numpy as np
import openmm as mm
import openxps as xps
from openmm import unit

import lambda_setup

parser = argparse.ArgumentParser()
parser.add_argument("--time", type=float, default=50.0, help="production time in ps")
parser.add_argument("--coupling", choices=["cv", "inner"], default="cv")
parser.add_argument("--platform", default="CUDA")
args = parser.parse_args()

T = lambda_setup.TEMPERATURE
dt = 1 * unit.femtosecond
topology, positions, system, theta, basis = lambda_setup.build(args.coupling)
opes = xps.ExtendedSpaceCombinedOPES(
    system,
    [xps.ExtendedSpaceBiasVariable(theta, 0.10 * unit.dimensionless, grid_width=101)],
    temperature=T,
    barrier=126.0 * unit.kilojoules_per_mole,
    exploreBarrier=21.0 * unit.kilojoules_per_mole,
    frequency=400,
    varianceFrequency=None,
    biasFactor=50.8,
    exploreBiasFactor=8.5,
)
simulation = xps.ExtendedSpaceSimulation(
    topology,
    system,
    lambda_setup.integrator(dt),
    mm.Platform.getPlatformByName(args.platform),
)
context = simulation.context
context.setPositions(positions)
context.setVelocitiesToTemperature(T, 1)
context.setDynamicalVariableValues([0.0])
context.setDynamicalVariableVelocitiesToTemperature(T, 1)

steps = int(round(args.time / dt.value_in_unit(unit.picosecond)))
chunk = 4000
start = time.perf_counter()
for done in range(0, steps, chunk):
    opes.step(simulation, min(chunk, steps - done))
    (value,) = context.getDynamicalVariableValues()
    opes_kernels, explore_kernels = opes.getNumKernels()
    print(
        f"{simulation.currentStep:8d} steps  theta = {value / value.unit:6.3f}"
        f"  kernels = {opes_kernels}/{explore_kernels}"
        f"  c(t) = {opes.getCt().value_in_unit(unit.kilojoules_per_mole):8.3f} kJ/mol",
        flush=True,
    )
elapsed = time.perf_counter() - start
print(f"{steps * 1e-3 / elapsed * 86400:.0f} ns/day")
grid = np.linspace(0, 2, 101)
free_energy = opes.getFreeEnergy()
np.savetxt("lambda_opes_fes.csv", np.column_stack([grid, free_energy]), delimiter=",",
           header="theta,free_energy_kJ_per_mol", comments="")
print("free energy difference between lambda = 1 and lambda = 0 (kJ/mol):",
      f"{free_energy[50] - free_energy[0]:.2f}")
```

If `opes.getCt()` returns a plain float in openmm-opes 0.2.0, drop the `value_in_unit` call; check `CombinedOPES.getCt`'s return type first.

- [ ] **Step 2: Run it briefly on the GPU**

Run (from `/data3/craabreu/github_repos/lambda-dynamics`, with the env from its README: `CUDA_VISIBLE_DEVICES=1`, `OPENMM_PLUGIN_DIR=/data3/craabreu/scratch/nbs-bench-plugins`, `PYTHONPATH=/data3/craabreu/github_repos/openxps:/data3/craabreu/github_repos/cvpack`): `$PY run_lambda_opes.py --time 20`
Expected: the script prints progress lines, kernel counts that grow over time, finite `c(t)` values and a ns/day figure in the hundreds; `lambda_opes_fes.csv` is written. θ should leave its initial value within the run (the bias factor 50.8 makes the first kernels wide).

- [ ] **Step 3: Record it**

Add to the Scripts table of `lambda-dynamics/README.md`:

```markdown
| `run_lambda_opes.py` | Gabriela's λ-OPES setup with one `ExtendedSpaceCombinedOPES` (OPES_COM) on θ; writes `lambda_opes_fes.csv`. `--time <ps> --coupling cv\|inner`. |
```

No commit: `lambda-dynamics` is not a git repository.

---

## Self-review notes

- Spec coverage: API and bounds mapping (Tasks 1–3), overrides and `stepTogether` (Tasks 2–4), packaging and docs (Tasks 1, 5), the six test groups (Tasks 1–4 and the doctests in 2–3), the first use case (Task 6), limitations in docstrings (Tasks 2–3), mixed periodicity (Task 2 test).
- Review Focus items 1–5 each have a test: `TypeError` on `bounded` (Task 2 and 3), radian grid for a degree-declared width (Task 1 asserts `biasWidth == radians(18)` and the ±π limits), both context kinds (Tasks 2–3), two groups and `stepTogether` (Task 4), mixed periodicity message (Task 2).
- Names are consistent: `_biasVariables`, `_extensionSimulation`, `_extensionContext`, `ExtendedSpaceOPES`, `ExtendedSpaceCombinedOPES`, `_opes`, `_combined`, `_simulation`, `_Replay`.
