# Code Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve all 55 findings of the 2026-10-10 code review of OpenXPS through seven themed pull requests, each one independently reviewable, tested, and mergeable into `main`.

**Architecture:** Each PR is a branch off `main` grouping the findings that touch one subsystem and whose fixes interact: packaging/CI, bounds and dynamical-variable semantics, couplings and context synchronization, integrator infrastructure, the GGMT thermostat algorithm, reporting/metadynamics, and the regression module. Every task is test-first: write the failing test that reproduces the finding, make it pass with the smallest change, commit. Doctests in docstrings are part of the test suite (`--doctest-modules`), so docstring examples are updated whenever behavior changes.

**Tech Stack:** Python ≥ 3.10, OpenMM ≥ 7.7 (CustomIntegrator, Lepton expressions), cvpack, NumPy, PyTorch + Lightning (regression only), pytest with `--doctest-modules`, ruff, GitHub Actions with micromamba.

**Spec:** `CODE_REVIEW.md` on branch `code-review-notes` (commit `875470e`), which is not meant to be merged into `main`. Finding IDs (H1–H9, M1–M24, L1–L22) below refer to that document.

## Global Constraints

- Public API stays camelCase for OpenMM-facing methods (`setStepSize`, `getDynamicalVariableValues`), snake_case for everything else; no renames of existing public names.
- No new runtime dependencies beyond declaring the ones already imported (`pandas`, `typing-extensions`); `scipy` is **not** available in tests.
- Code must run on OpenMM 7.7 through 8.3 (no OpenMM APIs newer than 7.7) and on the `Reference` platform, which every test uses.
- Ruff config in `pyproject.toml` is authoritative: line length 88, `PLR0913` max 6 args (use `# noqa: PLR0913` as the codebase does), `PLR2004` allowed only in tests.
- Every test lives under `openxps/tests/` and runs with `pytest openxps/tests/<file>.py -p no:cacheprovider`; the full suite is `pytest` (which includes doctests). A task is done only when the full suite passes.
- Docstring examples are doctests with `ELLIPSIS`; when an integrator's computation steps change, its docstring listing changes too.
- Commit messages follow the repository style: a short imperative first line (e.g. `Fix generated property setters`), no scope prefixes.
- Branch names: `fix/<theme>` or `feat/<theme>`; one PR per branch; each PR description lists the finding IDs it closes.

## Review Focus

Inputs the spec implies but no finding's own test exercises; each gets a test in the owning task.

1. **A dynamical variable left at rest under a CSVR extension integrator** (velocities never set): the first step must not produce NaN; the DV must stay at rest and a documented warning must say velocities should be initialized. → Task 4.6.
2. **A non-periodic DV declared in ångströms coupled harmonically to a cvpack `Distance` CV in nanometers**: the coupling energy must be computed with both quantities in nanometers, with no silent factor of 10. → Task 2.4.
3. **`setParameter` on a physical-only parameter** (one that does not exist in the extension context, e.g. a force-field parameter): must set the physical value and must not raise. → Task 3.3.
4. **`fit()` called twice on the same `ForceMatchingRegressor`**: the second fit must start from fresh centers and the learning curve must be the second fit's. → Task 7.5.
5. **A `CouplingSum` built from a coupling that is also used on its own**: using the standalone coupling afterwards must behave exactly as before the sum was built (no leaked indices). → Task 3.6.

## PR Map and Order

| Order | Branch | Theme | Findings closed |
|---|---|---|---|
| 1 | `fix/packaging-ci` | Packaging and CI | H1, M21, M22, L18, L19, L20, L21, L22 |
| 2 | `fix/bounds-dv-semantics` | Bounds and dynamical variables | H8, M6, M7, M8, M9, L8, L9, L14 (DV part), M18 (changelog) |
| 3 | `fix/couplings-context-sync` | Couplings and context synchronization | H5, H6, H7, M1, M12, M14, M20, L7, L11, L16 |
| 4 | `fix/integrators` | Integrator infrastructure | H4, M2, M10, M11, M15, L2, L3, L4, L5, L15, M23 (integrator tests) |
| 5 | `feat/ggmt-coulomb-splitting` | GGMT thermostat algorithm | M24 |
| 6 | `fix/reporting-metadynamics` | Extension writer and metadynamics | M3, M16, L6, L14 (writer part), L17, M23 (metadynamics/writer tests) |
| 7 | `fix/regression` | Regression module | H2, H3, H9, M4, M5, M13, M17, M18 (regression part), M19, L1, L10, L12, L13 |

PRs 1–4 should be merged in order because later ones build on earlier semantics (PR 3 relies on PR 2's bounds equality; PR 5 relies on PR 4's `setStepSize` hook). PRs 6 and 7 are independent of each other and of PR 5 and can be opened any time after PR 2.

Each PR starts with:

```bash
git checkout main && git pull
git checkout -b <branch>
```

and ends with the full suite green (`pytest`), `ruff check openxps` and `ruff format --check openxps` clean, and a PR whose description lists the closed finding IDs.

---

## PR 1 — `fix/packaging-ci` (H1, M21, M22, L18–L22)

**Files:**
- Modify: `pyproject.toml`
- Modify: `devtools/conda-recipes/anaconda/meta.yaml`
- Modify: `devtools/conda-envs/test_env.yaml`, `devtools/conda-envs/docs_env.yaml`, `devtools/conda-envs/deployment_env.yaml`
- Modify: `.github/workflows/Linux.yaml`, `.github/workflows/Docs.yaml`, `.github/workflows/Anaconda.yaml`
- Create: `CHANGELOG.md`
- Test: `openxps/tests/test_openxps.py`

**Interfaces:**
- Produces: `CHANGELOG.md` with an `## Unreleased` section that PRs 2 and 7 append to.

### Task 1.1: Declare runtime and test dependencies, drop Python 3.9

- [ ] **Step 1: Write the failing test**

Append to `openxps/tests/test_openxps.py`:

```python
from importlib import metadata


def test_declared_runtime_dependencies():
    """Every third-party package imported at import time is declared."""
    requires = [r.split(";")[0].split(" ")[0].lower() for r in metadata.requires("openxps")]
    for package in ("cvpack", "lightning", "numpy", "openmm", "pandas", "pyyaml",
                    "torch", "typing-extensions"):
        assert package in requires, f"{package} is not declared in pyproject.toml"


def test_declared_python_floor():
    """PEP 604 annotations at import time require Python >= 3.10."""
    requires_python = metadata.metadata("openxps")["Requires-Python"]
    assert requires_python == ">=3.10"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pip install -e . --no-deps && pytest openxps/tests/test_openxps.py -p no:cacheprovider -k declared`
Expected: FAIL — `pandas is not declared` and `'>=3.9' != '>=3.10'`

- [ ] **Step 3: Edit `pyproject.toml`**

Replace the classifiers, `requires-python`, `dependencies`, the `test` extra, and `package-data` blocks with:

```toml
classifiers = [
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]
requires-python = ">=3.10"
# Declare any run-time dependencies that should be installed with the package.
dependencies = [
    "cvpack",
    "lightning",
    "numpy",
    "openmm",
    "pandas",
    "pyyaml",
    "torch",
    "typing-extensions",
]
```

```toml
[project.optional-dependencies]
test = [
  "pytest>=7.0",
  "pytest-cov",
  "openmmtools",
]
```

```toml
[tool.setuptools.package-data]
openxps = [
    "data/*.pdb"
]
```

and in `[tool.pytest.ini_options]` add the XML report that CI uploads:

```toml
addopts = "-v -ra --doctest-modules --cov=openxps --cov-report=term --cov-report=html --cov-report=xml"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e . --no-deps && pytest openxps/tests/test_openxps.py -p no:cacheprovider -k declared`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml openxps/tests/test_openxps.py
git commit -m "Declare pandas and typing-extensions, require Python 3.10"
```

### Task 1.2: Fix the conda recipe and environment files

- [ ] **Step 1: Edit `devtools/conda-recipes/anaconda/meta.yaml`**

Replace the `run:` list and the `summary:` line:

```yaml
  run:
    - python >=3.10
    - numpy
    - openmm
    - cvpack
    - pyyaml
    - pandas
    - typing-extensions
    - pytorch
    - lightning
```

```yaml
  summary: Extended Phase-Space Simulations with OpenMM
```

- [ ] **Step 2: Add `pandas` to the three environment files**

In `devtools/conda-envs/test_env.yaml`, `docs_env.yaml`, and `deployment_env.yaml`, add `  - pandas` immediately after the `  - pyyaml` line. In `deployment_env.yaml`, also add `  - typing-extensions` after it and delete the line `  - boa` (deprecated; `conda build` is used instead, see Task 1.3).

- [ ] **Step 3: Verify the recipe renders**

Run: `OPENXPS_VERSION=v0.0.0 conda render devtools/conda-recipes/anaconda 2>&1 | grep -A12 "run:"`
Expected: the nine run requirements listed above, and `summary: Extended Phase-Space Simulations with OpenMM`. (If `conda render` is unavailable locally, `grep -c "typing-extensions" devtools/conda-recipes/anaconda/meta.yaml` must print `1`.)

- [ ] **Step 4: Commit**

```bash
git add devtools
git commit -m "Complete conda recipe run requirements"
```

### Task 1.3: Make CI test what it claims

- [ ] **Step 1: Pin OpenMM in the Linux matrix**

In `.github/workflows/Linux.yaml`, change the micromamba `create-args` to:

```yaml
          create-args: >-
            python=${{ matrix.python-version }}
            openmm=${{ matrix.openmm-version }}
```

- [ ] **Step 2: Upload a coverage file that exists, with a supported action**

Replace the CodeCov step at the end of `Linux.yaml` with:

```yaml
      - name: CodeCov
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml
          flags: unittests
          name: codecov-${{ matrix.os }}-py${{ matrix.python-version }}
```

(`coverage.xml` is produced by the `--cov-report=xml` option added in Task 1.1.)

- [ ] **Step 3: Remove the duplicated steps in `Docs.yaml`**

Delete the second `Additional info about the build` step and the second `Install package` step (the ones that follow the first `Install package` step), so that the sequence is: checkout → info → setup-micromamba → Install package → Run tests → Build Documentation → …

- [ ] **Step 4: Replace the deprecated `mambabuild`**

In `.github/workflows/Anaconda.yaml`, change

```
          conda mambabuild . --no-anaconda-upload -c conda-forge --output-folder $outdir
```

to

```
          conda build . --no-anaconda-upload -c conda-forge --output-folder $outdir
```

- [ ] **Step 5: Validate the workflow files**

Run: `python -c "import yaml, glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yaml')]; print('ok')"`
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add .github
git commit -m "Pin OpenMM in CI matrix and fix coverage upload"
```

### Task 1.4: Start a changelog

- [ ] **Step 1: Create `CHANGELOG.md`**

```markdown
# Changelog

## Unreleased

### Changed
- Python 3.9 is no longer supported; the minimum version is 3.10.

### Fixed
- `pandas` and `typing-extensions` are now declared as runtime dependencies.
- The conda recipe declares every runtime dependency.
- The Linux CI matrix now installs the OpenMM version it names.
```

- [ ] **Step 2: Commit and open the PR**

```bash
git add CHANGELOG.md
git commit -m "Add changelog"
git push -u origin fix/packaging-ci
```

PR title: `Fix packaging metadata and CI`. Body: closes H1, M21, M22, L18, L19, L20, L21, L22.

---

## PR 2 — `fix/bounds-dv-semantics` (H8, M6, M7, M8, M9, L8, L9, L14-DV, M18)

**Files:**
- Modify: `openxps/bounds/base.py`
- Modify: `openxps/bounds/no_bounds.py`, `openxps/bounds/periodic.py`, `openxps/bounds/reflective.py`
- Modify: `openxps/dynamical_variable.py`
- Modify: `openxps/couplings/harmonic_coupling.py`
- Modify: `openxps/tests/test_bounds.py`, `openxps/tests/test_dynamical_variables.py`, `openxps/tests/test_coupling.py`, `openxps/tests/test_context.py`, `openxps/tests/test_regression.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Produces: `Bounds.kind` (class attribute, `str`): `"periodic"`, `"reflective"`, or `"none"`. `Bounds.__eq__` requires equal `kind`. `DynamicalVariable.__hash__` exists. `DynamicalVariable.distanceTo` always emits MD-unit periods. PR 3 relies on `Bounds.__eq__` for `CouplingSum` conflict detection.

### Task 2.1: Bounds equality respects the subclass and is symmetric (M8)

- [ ] **Step 1: Write the failing tests**

Append to `openxps/tests/test_bounds.py`:

```python
def test_bounds_equality_requires_same_kind():
    """NoBounds and PeriodicBounds with the same range are not equal."""
    from openxps.bounds import NoBounds

    assert NoBounds(-np.pi, np.pi, mmunit.radian) != CircularBounds()
    assert PeriodicBounds(-np.pi, np.pi, mmunit.radian) == CircularBounds()
    assert ReflectiveBounds(0, 1, mmunit.nanometer) != PeriodicBounds(
        0, 1, mmunit.nanometer
    )


def test_bounds_equality_is_symmetric_across_units():
    """Equality after unit conversion holds in both directions."""
    a = ReflectiveBounds(12.11, 57.66, mmunit.angstrom)
    b = ReflectiveBounds(1.211, 5.766, mmunit.nanometer)
    assert a == b
    assert b == a
    assert hash(a) == hash(b)
    assert ReflectiveBounds(3, 10, mmunit.angstrom) == ReflectiveBounds(
        0.3, 1, mmunit.nanometer
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest openxps/tests/test_bounds.py -p no:cacheprovider -k "same_kind or symmetric"`
Expected: FAIL — `NoBounds(...) != CircularBounds()` is `False`; `b == a` is `False`.

- [ ] **Step 3: Implement**

In `openxps/bounds/base.py`, add `import math` and the `kind` attribute, and replace `__eq__`/`__hash__`:

```python
@dataclass(frozen=True, eq=False)
class Bounds(Serializable):
    ...
    lower: float
    upper: float
    unit: cvpack.units.Unit

    #: The kind of boundary condition; subclasses override it.
    kind: t.ClassVar[str] = "bounds"
```

```python
    def __eq__(self, other: t.Any) -> bool:
        if not isinstance(other, Bounds) or self.kind != other.kind:
            return False
        _, self_factor = self._md_unit_and_conversion_factor()
        other_md_unit, other_factor = other._md_unit_and_conversion_factor()
        if not self.unit.is_compatible(other_md_unit):
            return False
        return math.isclose(
            self.lower * self_factor, other.lower * other_factor, rel_tol=1e-9
        ) and math.isclose(
            self.upper * self_factor, other.upper * other_factor, rel_tol=1e-9
        )

    def __hash__(self) -> int:
        unit, _ = self._md_unit_and_conversion_factor()
        return hash((self.kind, str(unit)))
```

Set the class attribute in each subclass: `kind: t.ClassVar[str] = "none"` in `NoBounds`, `"periodic"` in `PeriodicBounds` (inherited by `CircularBounds`), `"reflective"` in `ReflectiveBounds`. Add `import typing as t` where missing.

- [ ] **Step 4: Run the bounds tests**

Run: `pytest openxps/tests/test_bounds.py openxps/bounds -p no:cacheprovider`
Expected: PASS (including the existing `test_bounds_equality`, which compares two base `Bounds` of kind `"bounds"`).

- [ ] **Step 5: Commit**

```bash
git add openxps/bounds openxps/tests/test_bounds.py
git commit -m "Make bounds equality kind-aware and symmetric"
```

### Task 2.2: Validate bound values (L8)

- [ ] **Step 1: Write the failing test**

Append to `openxps/tests/test_bounds.py`:

```python
def test_bounds_accept_numpy_scalars_and_reject_non_finite():
    """numpy scalars are real numbers; bools, NaN and inf are not valid bounds."""
    bounds = PeriodicBounds(np.int64(0), np.float32(1), mmunit.nanometer)
    assert bounds.lower == 0 and bounds.upper == pytest.approx(1)
    for lower, upper in ((True, 2), (np.nan, 1), (0, np.inf)):
        with pytest.raises((TypeError, ValueError)):
            PeriodicBounds(lower, upper, mmunit.nanometer)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest openxps/tests/test_bounds.py -p no:cacheprovider -k numpy_scalars`
Expected: FAIL — `TypeError: The lower bound must be a real number.` on `np.int64(0)`.

- [ ] **Step 3: Implement**

In `Bounds.__post_init__`, add `import numbers` at the top of the module and replace the type check loop with:

```python
        for kind in ("lower", "upper"):
            value = getattr(self, kind)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"The {kind} bound must be a real number.")
            if not math.isfinite(value):
                raise ValueError(f"The {kind} bound must be finite.")
            object.__setattr__(self, kind, float(value))
```

- [ ] **Step 4: Run tests**

Run: `pytest openxps/tests/test_bounds.py openxps/bounds -p no:cacheprovider`
Expected: PASS. Note: `float(value)` changes the repr of integer bounds from `lower=0` to `lower=0.0`; update the doctests in `no_bounds.py` (`NoBounds(lower=0.0, upper=1.0, unit=dimensionless)`), `periodic.py` (`PeriodicBounds(lower=-180.0, upper=180.0, unit=deg)`), and `reflective.py` (`ReflectiveBounds(lower=1.0, upper=10.0, unit=A)`) accordingly, and `base.py` (`asQuantity` → `(-180.0, 180.0) deg`).

- [ ] **Step 5: Commit**

```bash
git add openxps/bounds openxps/tests/test_bounds.py
git commit -m "Validate bound values as finite real numbers"
```

### Task 2.3: NoBounds goes through unit validation and conversion (M6)

- [ ] **Step 1: Write the failing test**

Append to `openxps/tests/test_dynamical_variables.py`:

```python
def test_dv_no_bounds_units_are_checked_and_converted():
    """NoBounds values are typical values, so they must be in the DV's unit."""
    mass = 1 * mmunit.dalton
    with pytest.raises(ValueError, match="incompatible units"):
        DynamicalVariable(
            "x", mmunit.nanometer, mass, NoBounds(-180, 180, mmunit.degree)
        )
    dv = DynamicalVariable("x", mmunit.nanometer, mass, NoBounds(1, 10, mmunit.angstrom))
    assert dv.bounds.unit == mmunit.nanometer
    assert dv.bounds.lower == pytest.approx(0.1)
    assert dv.bounds.upper == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest openxps/tests/test_dynamical_variables.py -p no:cacheprovider -k no_bounds_units`
Expected: FAIL — no exception raised for the degree bounds.

- [ ] **Step 3: Implement**

In `openxps/dynamical_variable.py`, delete the two lines

```python
        if isinstance(self.bounds, NoBounds):
            return
```

from `__post_init__`, remove `NoBounds` from the import line (it becomes unused), and change the `bounds` parameter docstring to:

```
    bounds
        The boundary condition to be applied to this dynamical variable. It must
        be an instance of :class:`PeriodicBounds`, :class:`ReflectiveBounds`, or
        :class:`NoBounds` (for unbounded variables, whose bounds are then typical
        values rather than actual limits). Its unit of measurement must be
        compatible with the dynamical variable's own unit.
```

- [ ] **Step 4: Fix the tests that relied on the loophole**

These helpers pair dimensionless `NoBounds` with nm/rad variables; change them:

- `openxps/tests/test_context.py`, `create_dvs()`: `bounds=NoBounds(0, 1, mmunit.dimensionless)` → `bounds=NoBounds(0, 1, mmunit.nanometer)`.
- `openxps/tests/test_regression.py`, `create_test_dynamical_variables_mixed()`: `NoBounds(0, 1, mmunit.dimensionless)` → `NoBounds(0, 1, mmunit.nanometer)`.
- `openxps/tests/test_coupling.py`, `test_harmonic_coupling_incompatible_periodicity_nonperiodic_dv`: `NoBounds(0, 1, mmunit.dimensionless)` → `NoBounds(0, 1, mmunit.radian)`.
- `openxps/tests/test_dynamical_variables.py`, every `NoBounds(0, 1, mmunit.dimensionless)` paired with a radian DV → `NoBounds(0, 1, mmunit.radian)`.

Run `grep -rn "NoBounds(0, 1, mmunit.dimensionless)" openxps/tests` and fix any remaining occurrence whose DV unit is not dimensionless.

- [ ] **Step 5: Run tests**

Run: `pytest openxps/tests -p no:cacheprovider`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add openxps/dynamical_variable.py openxps/tests
git commit -m "Validate and convert NoBounds units"
```

### Task 2.4: Distance expressions use MD units and consistent periods (H8, M9)

- [ ] **Step 1: Write the failing tests**

Append to `openxps/tests/test_dynamical_variables.py`:

```python
def test_distance_to_cv_uses_md_units_for_degree_dv():
    """A degree DV coupled to a radian torsion must wrap with period 2*pi."""
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.degree) ** 2
    dv = DynamicalVariable(
        "phi0", mmunit.degree, mass, PeriodicBounds(-180, 180, mmunit.degree)
    )
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    expression = dv.distanceTo(phi)
    assert "360" not in expression
    assert "6.28318" in expression


def test_distance_to_dv_requires_matching_bounds():
    """Periodic DVs with different periods cannot be compared."""
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    a = DynamicalVariable("a", mmunit.radian, mass, PeriodicBounds(-1, 1, mmunit.radian))
    b = DynamicalVariable("b", mmunit.radian, mass, CircularBounds())
    with pytest.raises(ValueError, match="Incompatible boundary conditions"):
        a.distanceTo(b)
    with pytest.raises(ValueError, match="Incompatible boundary conditions"):
        b.distanceTo(a)
    c = DynamicalVariable(
        "c", mmunit.degree,
        3 * mmunit.dalton * (mmunit.nanometer / mmunit.degree) ** 2,
        PeriodicBounds(-180, 180, mmunit.degree),
    )
    assert "6.28318" in b.distanceTo(c)
    x = DynamicalVariable("x", mmunit.nanometer, 1 * mmunit.dalton, NoBounds(0, 1, mmunit.nanometer))
    with pytest.raises(ValueError, match="Incompatible units"):
        x.distanceTo(a)
```

Add `from openxps.bounds import CircularBounds` to the test module imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest openxps/tests/test_dynamical_variables.py -p no:cacheprovider -k "md_units or matching_bounds"`
Expected: FAIL — `"360" in expression`; `a.distanceTo(b)` returns an expression instead of raising.

- [ ] **Step 3: Implement**

Replace `_distanceToCV` and `_distanceToDV` in `openxps/dynamical_variable.py`:

```python
    def _distanceToCV(self, other: cvpack.CollectiveVariable) -> str:
        name = other.getName()
        diff = f"{name}-{self.name}"
        if not other.getUnit().is_compatible(self.unit):
            raise ValueError("Incompatible units.")
        if other.getPeriodicBounds() is None and not self.isPeriodic():
            return f"({diff})"
        if self.isPeriodic() and other.getPeriodicBounds() == self.bounds.asQuantity():
            period = self.bounds.in_md_units().period
            return f"({diff}-{period}*floor(0.5+({diff})/{period}))"
        raise ValueError("Incompatible boundary conditions.")

    def _distanceToDV(self, other: te.Self) -> str:
        diff = f"{other.name}-{self.name}"
        if not other.unit.is_compatible(self.unit):
            raise ValueError("Incompatible units.")
        if not (self.isPeriodic() or other.isPeriodic()):
            return f"({diff})"
        if self.bounds != other.bounds:
            raise ValueError("Incompatible boundary conditions.")
        period = self.bounds.in_md_units().period
        return f"({diff}-{period}*floor(0.5+({diff})/{period}))"
```

(`self.bounds != other.bounds` uses the kind-aware, unit-converting equality from Task 2.1, so a periodic/non-periodic mix or a period mismatch both raise.)

- [ ] **Step 4: Add the Review-Focus test for an ångström non-periodic DV (item 2)**

Append to `openxps/tests/test_coupling.py`:

```python
def test_harmonic_coupling_nonperiodic_angstrom_dv_energy():
    """A non-periodic DV in angstrom couples to a distance CV in nanometers."""
    model = create_test_system()
    distance = cvpack.Distance(0, 1, name="d")
    d0 = DynamicalVariable(
        "d0", mmunit.angstrom, 10 * mmunit.dalton * (mmunit.nanometer / mmunit.angstrom) ** 2,
        ReflectiveBounds(0, 20, mmunit.angstrom),
    )
    kappa = 100 * mmunit.kilojoule_per_mole / mmunit.angstrom**2
    coupling = HarmonicCoupling(distance, d0, kappa)
    system = ExtendedSpaceSystem(model.system, coupling)
    context = mm.Context(
        system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(model.positions)
    actual = distance.getValue(context).value_in_unit(mmunit.nanometer)
    context.setParameter("d0", actual + 0.1)  # 0.1 nm = 1 angstrom away
    energy = coupling.getForce(0).getValue(context).value_in_unit(
        mmunit.kilojoule_per_mole
    )
    assert energy == pytest.approx(0.5 * 100 * 1.0**2, rel=1e-6)
```

This passes only after Task 2.5 removes the spurious periodicity rejection; keep it and run it there.

- [ ] **Step 5: Run the DV tests**

Run: `pytest openxps/tests/test_dynamical_variables.py openxps/dynamical_variable.py -p no:cacheprovider`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add openxps/dynamical_variable.py openxps/tests/test_dynamical_variables.py openxps/tests/test_coupling.py
git commit -m "Emit distance expressions in MD units and check bound compatibility"
```

### Task 2.5: HarmonicCoupling accepts non-periodic pairs (M7)

- [ ] **Step 1: Write the failing test**

Append to `openxps/tests/test_coupling.py`:

```python
def test_harmonic_coupling_nonperiodic_pair_is_accepted():
    """A non-periodic CV with a non-periodic DV is a valid harmonic coupling."""
    distance = cvpack.Distance(0, 1, name="d")
    mass = 10 * mmunit.dalton
    kappa = 1000 * mmunit.kilojoule_per_mole / mmunit.nanometer**2
    for bounds in (NoBounds(0, 1, mmunit.nanometer), ReflectiveBounds(0, 1, mmunit.nanometer)):
        d0 = DynamicalVariable("d0", mmunit.nanometer, mass, bounds)
        coupling = HarmonicCoupling(distance, d0, kappa)
        assert "floor" not in coupling.getForce(0).getEnergyFunction()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest openxps/tests/test_coupling.py -p no:cacheprovider -k "nonperiodic_pair or angstrom_dv"`
Expected: FAIL — `ValueError: Incompatible periodicity for d and d0.`

- [ ] **Step 3: Implement**

Replace `_validateArguments` in `openxps/couplings/harmonic_coupling.py`:

```python
    def _validateArguments(self, cv, dv, kappa):
        pair = f"{cv.getName()} and {dv.name}"
        if not cv.getUnit().is_compatible(dv.unit):
            raise ValueError(f"Incompatible units for {pair}.")
        cv_periodic = cv.getPeriodicBounds() is not None
        if cv_periodic != dv.isPeriodic() or (
            cv_periodic and cv.getPeriodicBounds() != dv.bounds.asQuantity()
        ):
            raise ValueError(f"Incompatible periodicity for {pair}.")
        if mmunit.is_quantity(kappa) and not kappa.unit.is_compatible(
            mmunit.kilojoule_per_mole / dv.unit**2
        ):
            raise ValueError(f"Incompatible force constant units for {pair}.")
```

- [ ] **Step 4: Run the coupling tests**

Run: `pytest openxps/tests/test_coupling.py -p no:cacheprovider`
Expected: PASS, including the pre-existing `test_harmonic_coupling_incompatible_periodicity_nonperiodic_dv` (periodic CV + non-periodic DV still raises).

- [ ] **Step 5: Commit**

```bash
git add openxps/couplings/harmonic_coupling.py openxps/tests/test_coupling.py
git commit -m "Accept non-periodic pairs in HarmonicCoupling"
```

### Task 2.6: Hashable DynamicalVariable and dead-branch removal (L9, L14)

- [ ] **Step 1: Write the failing test**

Append to `openxps/tests/test_dynamical_variables.py`:

```python
def test_dv_is_hashable_and_hash_matches_equality():
    mass = 3 * mmunit.dalton * (mmunit.nanometer / mmunit.radian) ** 2
    a = DynamicalVariable("phi", mmunit.radian, mass, PeriodicBounds(-180, 180, mmunit.degree))
    b = DynamicalVariable("phi", mmunit.radian, mass, CircularBounds())
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest openxps/tests/test_dynamical_variables.py -p no:cacheprovider -k hashable`
Expected: FAIL — `TypeError: unhashable type: 'Quantity'`

- [ ] **Step 3: Implement**

In `openxps/dynamical_variable.py`, add inside the class body (after `__setstate__`):

```python
    def __hash__(self) -> int:
        return hash((self.name, str(self.unit), self.bounds))
```

and simplify `createCollectiveVariable` by deleting the unreachable `None` branch:

```python
        force = mm.CustomExternalForce(self.bounds.leptonExpression("x"))
        bounds = self.bounds.asQuantity() if self.isPeriodic() else None
        force.addParticle(index, [])
        return cvpack.OpenMMForceWrapper(
            force,
            self.unit,
            bounds,
            self.name if name is None else name,
        )
```

- [ ] **Step 4: Run the full suite**

Run: `pytest -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Record the API change and open the PR**

Append under `## Unreleased` → `### Changed` in `CHANGELOG.md`:

```markdown
- `NoBounds` now requires `lower`, `upper` and `unit`, which are typical values of the
  variable used to initialize regression kernels; `DynamicalVariable` requires `bounds`.
- `NoBounds` units are validated and converted like other bounds.
- `Bounds` equality distinguishes periodic, reflective and unbounded kinds.
- `DynamicalVariable.distanceTo` raises for mismatched periods or incompatible units.
```

```bash
git add openxps/dynamical_variable.py openxps/tests/test_dynamical_variables.py CHANGELOG.md
git commit -m "Make DynamicalVariable hashable"
git push -u origin fix/bounds-dv-semantics
```

PR title: `Fix bounds and dynamical variable semantics`. Body: closes H8, M6, M7, M8, M9, L8, L9, part of L14 and M18.

---

## PR 3 — `fix/couplings-context-sync` (H5, H6, H7, M1, M12, M14, M20, L7, L11, L16)

**Files:**
- Modify: `openxps/utils.py` (`Function`)
- Modify: `openxps/couplings/base.py`, `openxps/couplings/collective_variable_coupling.py`, `openxps/couplings/inner_product_coupling.py`
- Modify: `openxps/context.py`, `openxps/simulation.py`
- Modify: `openxps/tests/test_coupling.py`, `openxps/tests/test_context.py`, `openxps/tests/test_simulation.py`, `openxps/tests/test_system.py`

**Interfaces:**
- Produces: `Function.createCollectiveVariable(variable_indices: Mapping[str, int])` (was a list); `Coupling.addToExtensionSystem(system, dv_indices: Mapping[str, int] | None = None)`; `Coupling._createFlippedForce(dv_indices: Mapping[str, int])` (abstract hook, now takes the mapping); `ExtendedSpaceContext.synchronize() -> None`; `InnerProductCoupling.getForceGroup() -> int | None`.
- Consumes: `Bounds.__eq__` from PR 2 (used by `CouplingSum` to detect conflicting DV definitions).

### Task 3.1: Function dependency parsing and index mapping (L7, prerequisite of H5)

- [ ] **Step 1: Write the failing tests**

Create `openxps/tests/test_utils.py`:

```python
"""
Unit tests for openxps.utils.Function.
"""

import openmm as mm
import pytest

from openxps.utils import Function


def test_function_ignores_lepton_definitions():
    """Names defined after ';' are local definitions, not variables."""
    fn = Function("s", "(1-cos(u))/2; u=alpha*lam", alpha=3.0)
    assert fn.getVariables() == {"lam"}
    assert set(fn.getParameters()) == {"alpha"}


def test_function_collective_variable_uses_index_mapping():
    """Particle indices come from the mapping, not from list positions."""
    fn = Function("g", "x*y")
    cv = fn.createCollectiveVariable({"a": 0, "x": 2, "y": 1})
    system = mm.System()
    for _ in range(3):
        system.addParticle(1.0)
    cv.addToSystem(system)
    context = mm.Context(system, mm.VerletIntegrator(0.0))
    context.setPositions([mm.Vec3(9, 0, 0), mm.Vec3(3, 0, 0), mm.Vec3(2, 0, 0)])
    energy = context.getState(getEnergy=True).getPotentialEnergy()._value
    assert energy == pytest.approx(6.0)


def test_function_collective_variable_unknown_variable():
    fn = Function("g", "x*typo")
    with pytest.raises(ValueError, match="unknown variables"):
        fn.createCollectiveVariable({"x": 0})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest openxps/tests/test_utils.py -p no:cacheprovider`
Expected: FAIL — `fn.getVariables() == {"lam", "u"}`; `TypeError: list indices must be integers` for the dict argument.

- [ ] **Step 3: Implement**

In `openxps/utils.py`, replace `_parseDependencies` and `createCollectiveVariable`:

```python
    def _parseDependencies(
        self,
        given_parameters: dict[str, mmunit.Quantity],
    ) -> tuple[set[str], set[str]]:
        given_parameters = set(given_parameters.keys())
        main, *definitions = self._expression.split(";")
        defined = {definition.partition("=")[0].strip() for definition in definitions}
        identifiers = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", self._expression))
        dependencies = identifiers - LEPTON_FUNCTIONS - defined
        return dependencies - given_parameters, dependencies & given_parameters
```

```python
    def createCollectiveVariable(
        self,
        variable_indices: t.Mapping[str, int],
    ) -> cvpack.AtomicFunction:
        """
        Create a collective variable from the function.

        Parameters
        ----------
        variable_indices
            A mapping from the name of each variable of the function to the index of
            the particle whose x coordinate represents it.

        Returns
        -------
        cvpack.AtomicFunction
            The collective variable object.

        Examples
        --------
        >>> from openxps.utils import Function
        >>> import openmm as mm
        >>> from math import exp, cos, pi
        >>> fn = Function("g", "exp(-x)*cos(2*pi*y)", pi=3.14159)
        >>> cv = fn.createCollectiveVariable({"x": 0, "y": 1})
        >>> system = mm.System()
        >>> for _ in range(2):
        ...     _ = system.addParticle(1.0)
        >>> cv.addToSystem(system)
        >>> context = mm.Context(system, mm.VerletIntegrator(0.0))
        >>> x, y = 2, 1
        >>> exp(-x)*cos(2*pi*y)
        0.13533528...
        >>> context.setPositions([mm.Vec3(x, 0.0, 0.0), mm.Vec3(y, 0.0, 0.0)])
        >>> context.getState(getEnergy=True).getPotentialEnergy()
        0.13533528... kJ/mol
        """
        variables = sorted(self._variables)
        if unknown := set(variables) - set(variable_indices):
            raise ValueError(
                f"Function {self._name} depends on unknown variables: "
                + ", ".join(sorted(unknown))
            )
        return cvpack.AtomicFunction(
            function=";".join(
                [self._expression]
                + [f"{variable}=x{i + 1}" for i, variable in enumerate(variables)]
            ),
            unit=mmunit.kilojoule_per_mole,
            groups=[[variable_indices[variable] for variable in variables]],
            name=self._name,
            **self._parameters,
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest openxps/tests/test_utils.py openxps/utils.py -p no:cacheprovider`
Expected: PASS (doctest included)

- [ ] **Step 5: Commit**

```bash
git add openxps/utils.py openxps/tests/test_utils.py
git commit -m "Parse Lepton definitions and map function variables by name"
```

### Task 3.2: Pass DV indices down instead of mutating child couplings (H5, M12)

- [ ] **Step 1: Write the failing tests**

Append to `openxps/tests/test_coupling.py`:

```python
def _two_particle_system():
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    return system


def _distance_coupling(dv_name="a"):
    dv = DynamicalVariable(dv_name, mmunit.nanometer, 10 * mmunit.dalton,
                           NoBounds(0, 1, mmunit.nanometer))
    return CollectiveVariableCoupling(
        f"0.5*k*(d-{dv_name})^2", [cvpack.Distance(0, 1, name="d")], [dv],
        k=100 * mmunit.kilojoule_per_mole / mmunit.nanometer**2,
    )


def _scaling_coupling():
    force = mm.CustomBondForce("scaling*r^2")
    force.addGlobalParameter("scaling", 1.0)
    force.addEnergyParameterDerivative("scaling")
    force.addBond(0, 1, [])
    lam = DynamicalVariable("lam", mmunit.dimensionless, 1 * mmunit.dalton * mmunit.nanometer**2,
                            ReflectiveBounds(0, 1, mmunit.dimensionless))
    return InnerProductCoupling([force], [lam], functions={"scaling": "lam^2"})


def test_inner_product_function_in_sum_reads_its_own_particle():
    """H5: the function CV must read particle 'lam' (index 1), not 'a' (index 0)."""
    from openxps import ExtendedSpaceContext, LockstepIntegrator

    coupling = _distance_coupling() + _scaling_coupling()
    assert [dv.name for dv in coupling.getDynamicalVariables()] == ["a", "lam"]
    context = ExtendedSpaceContext(
        ExtendedSpaceSystem(_two_particle_system(), coupling),
        LockstepIntegrator(mm.VerletIntegrator(0.001)),
        mm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(0.5, 0, 0)])
    context.setDynamicalVariableValues([0.3 * mmunit.nanometer, 0.7])
    assert context.getParameter("scaling") == pytest.approx(0.49)


def test_coupling_sum_does_not_mutate_children():
    """M12 and Review Focus 5: a child keeps its own indices after being summed."""
    cb = _distance_coupling("b")
    ca = _distance_coupling("a")
    _ = cb + ca  # cb's DV would be index 1 inside the sum
    assert cb._dv_indices == {"b": 0}
    system = ExtendedSpaceSystem(_two_particle_system(), cb)
    assert system.getExtensionSystem().getNumParticles() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest openxps/tests/test_coupling.py -p no:cacheprovider -k "own_particle or mutate_children"`
Expected: FAIL — `scaling == 0.09`; `cb._dv_indices == {"b": 1}`.

- [ ] **Step 3: Implement in `openxps/couplings/base.py`**

Delete `_updateDynamicalVariableIndices` and `_broadcastDynamicalVariableIndices` (and the call to the latter in `CouplingSum.__init__`). Change the hook and the public method:

```python
    def _createFlippedForce(self, dv_indices: t.Mapping[str, int]) -> mm.Force | None:
        return None
```

```python
    def addToExtensionSystem(
        self, system: mm.System, dv_indices: t.Optional[t.Mapping[str, int]] = None
    ) -> None:
        """
        Add the flipped version of this coupling to the extension system.

        The flipped force replaces dynamical variable parameters with collective
        variables that represent the dynamical variables as particles. Physical
        collective variables become parameters set to zero.

        Parameters
        ----------
        system
            The extension system to which the flipped coupling should be added.
        dv_indices
            A mapping from dynamical variable names to extension particle indices.
            If ``None``, the dynamical variables of this coupling are assumed to be
            the only ones in the system, in their own order.

        Examples
        --------
        (keep the existing example unchanged)
        """
        if dv_indices is None:
            dv_indices = self._dv_indices
        else:
            dv_indices = {name: dv_indices[name] for name in self._dv_indices}
        self._flipped_force = self._createFlippedForce(dv_indices)
        self._addForceToSystem(self._flipped_force, system)
```

In `CouplingSum`:

```python
    def addToExtensionSystem(
        self, system: mm.System, dv_indices: t.Optional[t.Mapping[str, int]] = None
    ) -> None:
        if dv_indices is None:
            dv_indices = self._dv_indices
        for coupling in self._couplings:
            coupling.addToExtensionSystem(system, dv_indices)
```

Also fix the `addToPhysicalSystem` docstring (L16): delete its `Raises` section (the method is implemented), and change the return annotation of `_checkGlobalParameters` to `-> None`.

- [ ] **Step 4: Update the two subclasses**

`openxps/couplings/collective_variable_coupling.py`:

```python
    def _createFlippedForce(self, dv_indices: t.Mapping[str, int]) -> mm.CustomCVForce:
        force = self._forces[0]
        parameters = force.getParameterDefaultValues()
        dvs_to_flip = [dv for dv in self._dynamical_variables if dv.name in parameters]
        for dv in dvs_to_flip:
            parameters.pop(dv.name)
        parameters.update(
            {cv.getName(): 0.0 * cv.getUnit() for cv in force.getInnerVariables()}
        )
        return cvpack.MetaCollectiveVariable(
            force.getEnergyFunction(),
            [dv.createCollectiveVariable(dv_indices[dv.name]) for dv in dvs_to_flip],
            unit=mmunit.kilojoule_per_mole,
            name="coupling",
            **parameters,
        )
```

`openxps/couplings/inner_product_coupling.py`:

```python
    def _createFlippedForce(self, dv_indices: t.Mapping[str, int]) -> mm.CustomCVForce:
        flipped_force = mm.CustomCVForce(
            "+".join(f"{p}*{self._derivativeName(p)}" for p in self._dynamic_parameters)
        )
        for parameter in self._dynamic_parameters:
            flipped_force.addGlobalParameter(self._derivativeName(parameter), 0.0)
        for fn in self._functions:
            flipped_force.addCollectiveVariable(
                fn.getName(), fn.createCollectiveVariable(dv_indices)
            )
        for dv in self._dynamical_variables:
            if dv.name in self._dynamic_parameters:
                flipped_force.addCollectiveVariable(
                    dv.name, dv.createCollectiveVariable(dv_indices[dv.name])
                )
        return flipped_force
```

and in `_getDynamicParameters`, right after `all_dvs = {...}`, add the clear error for typos:

```python
        for fn in self._functions:
            if unknown := fn.getVariables() - all_dvs:
                raise ValueError(
                    f"Function {fn.getName()} depends on names that are neither "
                    "dynamical variables nor given parameters: "
                    + ", ".join(sorted(unknown))
                )
```

- [ ] **Step 5: Run the coupling and system tests**

Run: `pytest openxps/tests/test_coupling.py openxps/tests/test_system.py openxps/couplings -p no:cacheprovider`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add openxps/couplings openxps/tests/test_coupling.py
git commit -m "Pass dynamical variable indices to flipped forces instead of mutating couplings"
```

### Task 3.3: Keep the two contexts synchronized (H6, H7, M1, M14)

- [ ] **Step 1: Write the failing tests**

Append to `openxps/tests/test_context.py`:

```python
def test_get_dv_values_for_function_only_dv():
    """H6: a DV that appears only inside functions= is not a physical parameter."""
    force = mm.CustomBondForce("scaling*r^2")
    force.addGlobalParameter("scaling", 1.0)
    force.addEnergyParameterDerivative("scaling")
    force.addBond(0, 1, [])
    lam = DynamicalVariable("lam", mmunit.dimensionless, 1 * mmunit.dalton * mmunit.nanometer**2,
                            ReflectiveBounds(0, 1, mmunit.dimensionless))
    from openxps import InnerProductCoupling

    coupling = InnerProductCoupling([force], [lam], functions={"scaling": "lam^2"})
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    context = ExtendedSpaceContext(
        ExtendedSpaceSystem(system, coupling),
        LockstepIntegrator(mm.VerletIntegrator(0.001)),
        mm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(0.5, 0, 0)])
    context.setDynamicalVariableValues([0.7])
    (value,) = context.getDynamicalVariableValues()
    assert value.value_in_unit(mmunit.dimensionless) == pytest.approx(0.7)


def test_set_parameter_updates_extension_context():
    """H7 and Review Focus 3: coupling parameters live in both contexts."""
    model = testsystems.AlanineDipeptideVacuum()
    context = create_extended_context(model)
    context.setPositions(model.positions)
    context.setParameter("kappa", 2000.0)
    assert context.getExtensionContext().getParameter("kappa") == pytest.approx(2000.0)
    # a physical-only parameter must still be settable without error
    system = context.getSystem()
    system.getForce(0).addGlobalParameter("physical_only", 1.0)
    context.reinitialize(preserveState=True)
    context.setParameter("physical_only", 3.0)
    assert context.getParameter("physical_only") == pytest.approx(3.0)


def test_set_state_and_reinitialize_resync():
    """M1: operations that bypass setPositions keep the extension in sync."""
    model = testsystems.AlanineDipeptideVacuum()
    context = create_extended_context(model)
    context.setPositions(model.positions)
    context.setDynamicalVariableValues([0.5, 0.3, -0.2])
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    extension = context.getExtensionContext()
    before = extension.getParameter("phi")
    state = context.getState(getPositions=True)
    shifted = state.getPositions(asNumpy=True) * 1.02
    context.setState(context.getState(getPositions=True))  # same positions: no change
    assert extension.getParameter("phi") == pytest.approx(before)
    context.setPositions(shifted)
    expected = phi.getValue(context).value_in_unit(mmunit.radian)
    context.setState(state)  # back to the original positions
    assert extension.getParameter("phi") == pytest.approx(before)
    context.setPositions(shifted)
    assert extension.getParameter("phi") == pytest.approx(expected)
    context.reinitialize(preserveState=True)
    assert extension.getParameter("phi") == pytest.approx(expected)
    assert context.getParameter("phi0") == pytest.approx(0.5)


def test_set_dv_values_wrong_length():
    """M14: too many or too few values raise instead of being silently truncated."""
    model = testsystems.AlanineDipeptideVacuum()
    context = create_extended_context(model)
    context.setPositions(model.positions)
    with pytest.raises(ValueError, match="3 dynamical variables"):
        context.setDynamicalVariableValues([0.1, 99.0])
    with pytest.raises(ValueError, match="3 dynamical variables"):
        context.setDynamicalVariableVelocities([0.1, 0.2, 0.3, 0.4])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest openxps/tests/test_context.py -p no:cacheprovider -k "function_only or updates_extension or resync or wrong_length"`
Expected: FAIL — `OpenMMException: invalid parameter name: lam`; extension `kappa` still 1000; `phi` stale after `setState`; no `ValueError` for the wrong lengths.

- [ ] **Step 3: Implement in `openxps/context.py`**

Add after `getIntegrator`:

```python
    def synchronize(self) -> None:
        """
        Push the current dynamical variable values to the physical context and the
        current collective variable values to the extension context.

        This is called automatically by the methods of this class that change the
        state of either context. Call it manually after changing the physical state
        through a channel that bypasses this class (e.g. an :OpenMM:`LocalEnergyMinimizer`
        applied directly to the context).
        """
        self._coupling.updatePhysicalContext(self, self._extension_context)
        self._coupling.updateExtensionContext(self, self._extension_context)
```

Replace `setParameter`'s Notes and body:

```python
    def setParameter(self, name: str, value: mmunit.Quantity) -> None:
        """
        Set the value of a global parameter defined by a Force object in the System.

        Notes
        -----
        Parameters that represent dynamical variables are protected and cannot be set
        through this method; use :meth:`setDynamicalVariableValues` instead. A
        parameter that also exists in the extension system (e.g. a coupling constant)
        is updated in both contexts.

        Parameters
        ----------
        name
            The name of the parameter to set.
        value
            The value of the parameter.
        """
        if name in self._coupling.getProtectedParameters():
            raise ValueError(
                f'Cannot manually set the parameter "{name}". This parameter is '
                "set automatically via setDynamicalVariableValues."
            )
        super().setParameter(name, value)
        if name in self._extension_context.getParameters():
            self._extension_context.setParameter(name, value)
```

Add the three overrides after `setPositions`:

```python
    def setState(self, state: mm.State) -> None:
        """
        Set the state of the physical context and synchronize the extension context.
        """
        super().setState(state)
        self._coupling.updateExtensionContext(self, self._extension_context)

    def setPeriodicBoxVectors(self, a: mm.Vec3, b: mm.Vec3, c: mm.Vec3) -> None:
        """
        Set the periodic box vectors and synchronize the extension context.
        """
        super().setPeriodicBoxVectors(a, b, c)
        self._coupling.updateExtensionContext(self, self._extension_context)

    def reinitialize(self, preserveState: bool = False) -> None:
        """
        Reinitialize both contexts and synchronize them.
        """
        super().reinitialize(preserveState)
        self._extension_context.reinitialize(preserveState)
        self.synchronize()
```

Replace the body of `setDynamicalVariableValues`:

```python
        values = list(values)
        if len(values) != len(self._dvs):
            raise ValueError(
                f"Expected values for {len(self._dvs)} dynamical variables, "
                f"got {len(values)}."
            )
        positions = []
        for dv, quantity in zip(self._dvs, values):
            if mmunit.is_quantity(quantity):
                value = quantity.value_in_unit(dv.unit)
            else:
                value = quantity
            positions.append(mm.Vec3(value, 0, 0))
        self._extension_context.setPositions(positions)
        self._coupling.updatePhysicalContext(self, self._extension_context)
```

Replace `getDynamicalVariableValues`:

```python
    def getDynamicalVariableValues(self) -> tuple[mmunit.Quantity]:
        """
        Get the values of the dynamical variables, wrapped into their bounds.

        Returns
        -------
        t.Tuple[mmunit.Quantity]
            A tuple containing the values of the dynamical variables.
        """
        state = mmswig.Context_getState(self._extension_context, mm.State.Positions)
        positions = mmswig.State__getVectorAsVec3(state, mm.State.Positions)
        values = []
        for i, dv in enumerate(self._dvs):
            value, _ = dv.bounds.wrap(positions[i].x, 0)
            values.append(value * dv.unit)
        return tuple(values)
```

In `setDynamicalVariableVelocities`, change the docstring line to `A sequence of quantities containing the velocities of the dynamical variables.` and add the length check right after `velocities = list(velocities)`:

```python
        if len(velocities) != len(self._dvs):
            raise ValueError(
                f"Expected velocities for {len(self._dvs)} dynamical variables, "
                f"got {len(velocities)}."
            )
```

- [ ] **Step 4: Run the context tests**

Run: `pytest openxps/tests/test_context.py openxps/context.py -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openxps/context.py openxps/tests/test_context.py
git commit -m "Keep physical and extension contexts synchronized"
```

### Task 3.4: Simulation-level synchronization and state files (M1, L11)

- [ ] **Step 1: Write the failing tests**

Append to `openxps/tests/test_simulation.py`:

```python
def test_minimize_energy_resyncs_extension():
    """M1: after minimization the extension context sees the new CV values."""
    model, _, coupling, integrator = create_test_system()
    simulation = xps.ExtendedSpaceSimulation(
        model.topology, xps.ExtendedSpaceSystem(model.system, coupling),
        xps.LockstepIntegrator(integrator), mm.Platform.getPlatformByName("Reference"),
    )
    simulation.context.setPositions(model.positions)
    simulation.context.setDynamicalVariableValues([3.0])
    simulation.minimizeEnergy(maxIterations=50)
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    expected = phi.getValue(simulation.context).value_in_unit(mmunit.radian)
    actual = simulation.context.getExtensionContext().getParameter("phi")
    assert actual == pytest.approx(expected)


def test_save_and_load_state_roundtrip_dvs(tmp_path):
    """M1: saveState/loadState carry the dynamical variables."""
    model, _, coupling, integrator = create_test_system()
    simulation = xps.ExtendedSpaceSimulation(
        model.topology, xps.ExtendedSpaceSystem(model.system, coupling),
        xps.LockstepIntegrator(integrator), mm.Platform.getPlatformByName("Reference"),
    )
    simulation.context.setPositions(model.positions)
    simulation.context.setDynamicalVariableValues([2.5])
    simulation.context.setDynamicalVariableVelocities([0.4])
    path = str(tmp_path / "state.xml")
    simulation.saveState(path)
    simulation.context.setDynamicalVariableValues([-1.0])
    simulation.context.setDynamicalVariableVelocities([0.0])
    simulation.loadState(path)
    (value,) = simulation.context.getDynamicalVariableValues()
    (rate,) = simulation.context.getDynamicalVariableVelocities()
    assert value.value_in_unit(mmunit.radian) == pytest.approx(2.5)
    assert rate.value_in_unit(mmunit.radian / mmunit.picosecond) == pytest.approx(0.4)
    assert simulation.context.getParameter("phi0") == pytest.approx(2.5)


def test_state_argument_accepts_filename(tmp_path):
    """L11: like OpenMM's Simulation, state= may be a file name."""
    model, _, coupling, integrator = create_test_system()
    simulation = xps.ExtendedSpaceSimulation(
        model.topology, xps.ExtendedSpaceSystem(model.system, coupling),
        xps.LockstepIntegrator(integrator), mm.Platform.getPlatformByName("Reference"),
    )
    simulation.context.setPositions(model.positions)
    simulation.context.setDynamicalVariableValues([1.0])
    path = str(tmp_path / "state.xml")
    simulation.saveState(path)
    model2, _, coupling2, integrator2 = create_test_system()
    restored = xps.ExtendedSpaceSimulation(
        model2.topology, xps.ExtendedSpaceSystem(model2.system, coupling2),
        xps.LockstepIntegrator(integrator2), mm.Platform.getPlatformByName("Reference"),
        state=path,
    )
    (value,) = restored.context.getDynamicalVariableValues()
    assert value.value_in_unit(mmunit.radian) == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest openxps/tests/test_simulation.py -p no:cacheprovider -k "resyncs or roundtrip_dvs or accepts_filename"`
Expected: FAIL — extension `phi` stale after minimization; DV value `-1.0` after `loadState`; `TypeError` for `state=path`.

- [ ] **Step 3: Implement in `openxps/simulation.py`**

Add `from .utils import STRING_SEPARATOR` and replace the `state` handling and add the overrides:

```python
        if state is not None:
            self.loadState(state) if isinstance(state, str) else self.context.setState(state)
```

```python
    def minimizeEnergy(
        self,
        tolerance: mmunit.Quantity = 10 * mmunit.kilojoules_per_mole / mmunit.nanometer,
        maxIterations: int = 0,
        reporter: t.Optional[mmapp.MinimizationReporter] = None,
    ) -> None:
        """
        Perform a local energy minimization of the physical system and synchronize
        the extension context afterwards. See :OpenMM:`Simulation` for the arguments.
        """
        if reporter is None:
            super().minimizeEnergy(tolerance, maxIterations)
        else:
            super().minimizeEnergy(tolerance, maxIterations, reporter)
        self.context.synchronize()

    def saveState(self, file: t.Union[str, t.IO]) -> None:
        """
        Save the state of both the physical and the extension context to a file.

        The file holds two XML-serialized :OpenMM:`State` objects separated by a
        form-feed marker, so it is not readable by :OpenMM:`Simulation.loadState`.
        """
        physical = self.context.getState(
            getPositions=True, getVelocities=True, getParameters=True,
            getIntegratorParameters=True, enforcePeriodicBox=self._usesPBC,
        )
        extension = self.context.getExtensionContext().getState(
            getPositions=True, getVelocities=True, getParameters=True,
        )
        content = (
            mm.XmlSerializer.serialize(physical)
            + STRING_SEPARATOR
            + mm.XmlSerializer.serialize(extension)
        )
        if isinstance(file, str):
            with open(file, "w", encoding="utf-8") as stream:
                stream.write(content)
        else:
            file.write(content)

    def loadState(self, file: t.Union[str, t.IO]) -> None:
        """
        Load a state written by :meth:`saveState` and synchronize both contexts.
        """
        if isinstance(file, str):
            with open(file, encoding="utf-8") as stream:
                content = stream.read()
        else:
            content = file.read()
        physical, _, extension = content.partition(STRING_SEPARATOR)
        self.context.setState(mm.XmlSerializer.deserialize(physical))
        if extension:
            self.context.getExtensionContext().setState(
                mm.XmlSerializer.deserialize(extension)
            )
        self.context.synchronize()
```

Add `from openmm import unit as mmunit` to the imports and change the `state` docstring to: `If specified, either an :OpenMM:`State` object or the name of a file written by :meth:`saveState`; the simulation state is set from it after initialization.` (`MinimizationReporter` exists since OpenMM 8.1; guard the annotation with a string: `reporter: t.Optional["mmapp.MinimizationReporter"] = None`.)

- [ ] **Step 4: Run the simulation tests**

Run: `pytest openxps/tests/test_simulation.py openxps/simulation.py -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openxps/simulation.py openxps/tests/test_simulation.py
git commit -m "Synchronize contexts after minimization and state loading"
```

### Task 3.5: Restrict the parameter-derivative evaluation to the coupling's force group (M20)

- [ ] **Step 1: Write the failing test**

Append to `openxps/tests/test_coupling.py`:

```python
def test_inner_product_coupling_uses_dedicated_force_group():
    """M20: derivative evaluation touches only the coupling forces."""
    model = create_test_system()
    coupling = _scaling_coupling()
    groups_before = {f.getForceGroup() for f in model.system.getForces()}
    system = ExtendedSpaceSystem(model.system, coupling)
    group = coupling.getForceGroup()
    assert group is not None and group not in groups_before
    assert all(f.getForceGroup() == group for f in coupling.getForces())
    from openxps import ExtendedSpaceContext, LockstepIntegrator

    context = ExtendedSpaceContext(
        system, LockstepIntegrator(mm.VerletIntegrator(0.001)),
        mm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(model.positions)
    context.setDynamicalVariableValues([0.5])
    full = context.getState(getParameterDerivatives=True).getEnergyParameterDerivatives()
    partial = context.getState(
        getParameterDerivatives=True, groups=1 << group
    ).getEnergyParameterDerivatives()
    assert partial["scaling"] == pytest.approx(full["scaling"])
    assert context.getExtensionContext().getParameter(
        "derivative_with_respect_to_scaling"
    ) == pytest.approx(full["scaling"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest openxps/tests/test_coupling.py -p no:cacheprovider -k dedicated_force_group`
Expected: FAIL — `AttributeError: 'InnerProductCoupling' object has no attribute 'getForceGroup'`

- [ ] **Step 3: Implement in `openxps/couplings/inner_product_coupling.py`**

Initialize `self._force_group = None` in `__init__` (before `super().__init__`) and in `__setstate__` (`self._force_group = state.get("force_group")`); add `state["force_group"] = self._force_group` in `__getstate__`. Add:

```python
    def getForceGroup(self) -> t.Optional[int]:
        """Get the force group assigned to the coupling forces, or None if the coupling
        has not been added to a physical system yet."""
        return self._force_group

    def addToPhysicalSystem(self, system: mm.System) -> None:
        used = {system.getForce(i).getForceGroup() for i in range(system.getNumForces())}
        self._force_group = next(g for g in range(31, -1, -1) if g not in used)
        for force in self._forces:
            force.setForceGroup(self._force_group)
        super().addToPhysicalSystem(system)

    def updateExtensionContext(
        self,
        physical_context: mm.Context,
        extension_context: mm.Context,
    ) -> None:
        groups = -1 if self._force_group is None else 1 << self._force_group
        state = mmswig.Context_getState(
            physical_context, mm.State.ParameterDerivatives, False, groups
        )
        for name, value in mmswig.State_getEnergyParameterDerivatives(state).items():
            if name in self._dynamic_parameters:
                mmswig.Context_setParameter(
                    extension_context, self._derivativeName(name), value
                )
```

- [ ] **Step 4: Run tests**

Run: `pytest openxps/tests/test_coupling.py openxps/tests/test_extension_writer.py -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openxps/couplings/inner_product_coupling.py openxps/tests/test_coupling.py
git commit -m "Evaluate parameter derivatives only for the coupling force group"
```

### Task 3.6: Full suite, docstrings, PR

- [ ] **Step 1: Fix the remaining docstring mismatches (L16)**

In `openxps/context.py`, the `setDynamicalVariableVelocities` docstring now says "A sequence" (done in Task 3.3). In `openxps/couplings/base.py`, confirm the `addToPhysicalSystem` `Raises` section is gone and `_checkGlobalParameters` is annotated `-> None` (Task 3.2).

- [ ] **Step 2: Run the full suite and linters**

Run: `pytest -p no:cacheprovider && ruff check openxps && ruff format --check openxps`
Expected: all PASS

- [ ] **Step 3: Commit and open the PR**

```bash
git push -u origin fix/couplings-context-sync
```

PR title: `Fix coupling index mapping and context synchronization`. Body: closes H5, H6, H7, M1, M12, M14, M20, L7, L11, L16.

---

## PR 4 — `fix/integrators` (H4, M2, M10, M11, M15, L2, L3, L4, L5, L15, M23-integrators)

**Files:**
- Modify: `openxps/integrators/utils.py`, `openxps/integrators/csvr.py`, `openxps/integrators/massive_ggmt.py`, `openxps/integrators/symmetric_langevin.py`
- Modify: `openxps/integrator.py`
- Modify: `openxps/tests/test_integrator.py`
- Create: `openxps/tests/test_thermostats.py`

**Interfaces:**
- Produces: `IntegratorMixin.setStepSize(size)` (calls `_update_global_variables` when present); `ExtendedSpaceIntegrator.__getstate__() -> dict` (was `str`); `ExtendedSpaceIntegrator._assume_valid: bool`; `CSVRIntegrator.setRandomNumberSeed(seed)` with OpenMM semantics (0 = random). PR 5 relies on the `setStepSize` hook.

### Task 4.1: Generated setters, step-size hook and repr (H4, M2-part, L15)

- [ ] **Step 1: Write the failing tests**

Create `openxps/tests/test_thermostats.py`:

```python
"""
Unit tests for the integrators in openxps.integrators.
"""

import copy
import pickle

import numpy as np
import openmm as mm
import pytest
from openmm import unit as mmunit

from openxps import integrators

KELVIN = mmunit.kelvin
PS = mmunit.picosecond
FS = mmunit.femtosecond


def _ideal_gas(num_particles=100, mass=10.0):
    system = mm.System()
    for _ in range(num_particles):
        system.addParticle(mass)
    return system


def _context(system, integrator, seed=1):
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(np.zeros((system.getNumParticles(), 3)))
    context.setVelocitiesToTemperature(300 * KELVIN, seed)
    return context


@pytest.mark.parametrize(
    "factory",
    [
        lambda: integrators.SymmetricLangevinIntegrator(300 * KELVIN, 1 / PS, 1 * FS),
        lambda: integrators.CSVRIntegrator(300 * KELVIN, 10 / PS, 1 * FS),
        lambda: integrators.MassiveGGMTIntegrator(300 * KELVIN, 0.1 * PS, 1 * FS),
        lambda: integrators.RegulatedNHLIntegrator(300 * KELVIN, 0.1 * PS, 1 * FS),
    ],
)
def test_property_setters_update_global_variables(factory):
    """H4: set<Property> must not crash and must refresh the integrator globals."""
    integrator = factory()
    integrator.setTemperature(310 * KELVIN)
    assert integrator.getTemperature() == 310 * KELVIN
    kt = (mmunit.MOLAR_GAS_CONSTANT_R * 310 * KELVIN).value_in_unit(
        mmunit.kilojoule_per_mole
    )
    assert integrator.getGlobalVariableByName("kT") == pytest.approx(kt)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: integrators.SymmetricLangevinIntegrator(300 * KELVIN, 10 / PS, 4 * FS),
        lambda: integrators.MassiveGGMTIntegrator(300 * KELVIN, 0.1 * PS, 4 * FS, stochastic=True),
        lambda: integrators.RegulatedNHLIntegrator(300 * KELVIN, 0.1 * PS, 4 * FS),
    ],
)
def test_set_step_size_refreshes_friction_coefficients(factory):
    """M2: a = exp(-gamma dt) must follow the step size."""
    integrator = factory()
    gamma = integrator.getFrictionCoefficient().value_in_unit(1 / PS)
    integrator.setStepSize(1 * FS)
    assert integrator.getGlobalVariableByName("a") == pytest.approx(np.exp(-gamma * 0.001))


def test_repr_has_no_blank_line_without_per_dof_variables():
    """L15"""
    integrator = integrators.RegulatedNHLIntegrator(300 * KELVIN, 0.1 * PS, 1 * FS)
    text = repr(integrator)
    assert not text.startswith("  \n")
    assert "Per-dof variables:\n  v1" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest openxps/tests/test_thermostats.py -p no:cacheprovider -k "setters or refreshes or blank_line"`
Expected: FAIL — `TypeError: set_value() missing 1 required positional argument: 'value'`; stale `a` for GGMT/NHL.

- [ ] **Step 3: Implement in `openxps/integrators/utils.py`**

Fix the setter and add the hook to the mixin:

```python
    @preprocess_args
    def set_and_update(self, value: t.Union[mmunit.Quantity, float]) -> None:
        set_value(self, value)
        if hasattr(self, "_update_global_variables"):
            self._update_global_variables()
```

In `IntegratorMixin`, add:

```python
    def setStepSize(self, size: t.Union[mmunit.Quantity, float]) -> None:
        """Set the step size and refresh any step-size-dependent global variables."""
        super().setStepSize(size)
        if hasattr(self, "_update_global_variables"):
            self._update_global_variables()
```

and fix `__repr__` so the per-dof line is only added when there are per-dof variables:

```python
        if self.getNumPerDofVariables() > 0:
            readable_lines.append("Per-dof variables:")
            per_dof = [
                self.getPerDofVariableName(index)
                for index in range(self.getNumPerDofVariables())
            ]
            readable_lines.append("  " + ", ".join(per_dof))
        if self.getNumGlobalVariables() > 0:
            readable_lines.append("Global variables:")
```

Delete the now-redundant `setStepSize` override from `openxps/integrators/symmetric_langevin.py` (the mixin provides it, and `IntegratorMixin` precedes `mm.CustomIntegrator` in every integrator's MRO).

- [ ] **Step 4: Run tests**

Run: `pytest openxps/tests/test_thermostats.py openxps/integrators -p no:cacheprovider`
Expected: PASS (doctests included)

- [ ] **Step 5: Commit**

```bash
git add openxps/integrators openxps/tests/test_thermostats.py
git commit -m "Fix generated property setters and refresh globals on setStepSize"
```

### Task 4.2: Extended-space step-size changes go through the Python API (M2)

- [ ] **Step 1: Write the failing tests**

Append to `openxps/tests/test_integrator.py` (and add `import math` to its imports):

```python
def test_split_default_extension_integrator_has_consistent_coefficients():
    """M2: the halved extension step must come with a recomputed friction factor."""
    physical = integrators.SymmetricLangevinIntegrator(
        300 * mmunit.kelvin, 10 / mmunit.picosecond, 4 * mmunit.femtosecond
    )
    split = SplitIntegrator(physical)
    extension = split.getExtensionIntegrator()
    assert mmswig.Integrator_getStepSize(extension) == pytest.approx(0.002)
    assert extension.getGlobalVariableByName("a") == pytest.approx(math.exp(-10 * 0.002))
    split.setStepSize(8 * mmunit.femtosecond)
    assert physical.getGlobalVariableByName("a") == pytest.approx(math.exp(-10 * 0.008))
    assert extension.getGlobalVariableByName("a") == pytest.approx(math.exp(-10 * 0.004))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest openxps/tests/test_integrator.py -p no:cacheprovider -k consistent_coefficients`
Expected: FAIL — extension `a == exp(-0.04)` (computed for 4 fs).

- [ ] **Step 3: Implement in `openxps/integrator.py`**

In `ExtendedSpaceIntegrator.setStepSize`, replace the two `mmswig.Integrator_setStepSize(...)` calls with:

```python
        self._physical_integrator.setStepSize(
            factor * mmswig.Integrator_getStepSize(self._physical_integrator)
        )
        self._extension_integrator.setStepSize(
            factor * mmswig.Integrator_getStepSize(self._extension_integrator)
        )
```

In `SplitIntegrator.__init__`, replace `mmswig.Integrator_setStepSize(extension_integrator, extension_step_size)` with `extension_integrator.setStepSize(extension_step_size)`.

- [ ] **Step 4: Run tests**

Run: `pytest openxps/tests/test_integrator.py -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openxps/integrator.py openxps/tests/test_integrator.py
git commit -m "Change step sizes through the Python integrator API"
```

### Task 4.3: Copying and pickling preserve integrator classes and flags (M10, M15)

- [ ] **Step 1: Write the failing tests**

Append to `openxps/tests/test_thermostats.py`:

```python
def test_pickled_extended_integrator_keeps_subclasses():
    """M10: CSVR's Python-level step() must survive a pickle round trip."""
    from openxps import LockstepIntegrator

    csvr = integrators.CSVRIntegrator(300 * KELVIN, 10 / PS, 1 * FS, forceFirst=True)
    original = LockstepIntegrator(mm.VerletIntegrator(1 * FS), csvr)
    restored = pickle.loads(pickle.dumps(original))
    assert isinstance(restored.getExtensionIntegrator(), integrators.CSVRIntegrator)
    assert isinstance(restored.getPhysicalIntegrator(), mm.VerletIntegrator)
    extension = restored.getExtensionIntegrator()
    extension.registerWithSystem(_ideal_gas(4), True)
    context = _context(_ideal_gas(4), extension)
    before = extension.getGlobalVariableByName("sumRsq")
    extension.step(1)
    assert extension.getGlobalVariableByName("sumRsq") != before


def test_deepcopy_keeps_assume_flags():
    """M15"""
    from openxps import LockstepIntegrator, SplitIntegrator
    from openxps.tests.test_integrator import create_non_force_first_integrator

    custom = create_non_force_first_integrator(1 * FS)
    lockstep = LockstepIntegrator(custom, assume_force_first=True)
    clone = copy.deepcopy(lockstep)
    assert isinstance(clone, LockstepIntegrator)
    split = SplitIntegrator(
        mm.LangevinIntegrator(300 * KELVIN, 1 / PS, 2 * FS), assume_symmetric=True
    )
    assert isinstance(copy.copy(split), SplitIntegrator)
```

and update the two existing state tests in `openxps/tests/test_integrator.py` (`test_getstate_setstate_lockstep`, `test_getstate_setstate_split`): change `assert isinstance(state, str)` to `assert isinstance(state, dict)`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest openxps/tests/test_thermostats.py openxps/tests/test_integrator.py -p no:cacheprovider -k "keeps_subclasses or assume_flags or getstate"`
Expected: FAIL — restored extension is `CustomIntegrator`; `deepcopy` raises `ValueError: The integrators must follow a force-first scheme`.

- [ ] **Step 3: Implement in `openxps/integrator.py`**

Add two module-level helpers after the known-integrator tuples:

```python
def _integrator_state(integrator: mm.Integrator) -> tuple[type, str, dict]:
    """Capture class, XML and Python attributes of an integrator."""
    attributes = {k: v for k, v in vars(integrator).items() if k != "this"}
    return type(integrator), mm.XmlSerializer.serialize(integrator), attributes


def _restore_integrator(cls: type, xml: str, attributes: dict) -> mm.Integrator:
    """Rebuild an integrator captured by :func:`_integrator_state`."""
    integrator = cls.__new__(cls)
    integrator.this = mm.XmlSerializer.deserialize(xml).this
    integrator.__dict__.update(attributes)
    return integrator
```

Replace `__init__`, `__copy__`, `__getstate__`, `__setstate__` of `ExtendedSpaceIntegrator`:

```python
    def __init__(
        self,
        physical_integrator: mm.Integrator,
        extension_integrator: mm.Integrator,
    ) -> None:
        self._physical_integrator = physical_integrator
        self._extension_integrator = extension_integrator
        self._assume_valid = getattr(self, "_assume_valid", False)
        self._initialize()

    def __copy__(self) -> "ExtendedSpaceIntegrator":
        """Return an unconfigured copy of the integrator."""
        clone = self.__class__.__new__(self.__class__)
        clone._physical_integrator = self._physical_integrator.__copy__()
        clone._extension_integrator = self._extension_integrator.__copy__()
        clone._assume_valid = self._assume_valid
        clone._initialize()
        return clone

    def __deepcopy__(self, memo: dict) -> "ExtendedSpaceIntegrator":
        return self.__copy__()

    def __getstate__(self) -> dict:
        """Get the state of the integrator as a picklable dictionary."""
        return {
            "physical": _integrator_state(self._physical_integrator),
            "extension": _integrator_state(self._extension_integrator),
            "assume_valid": self._assume_valid,
        }

    def __setstate__(self, state: dict) -> None:
        """Set the state of the integrator from a dictionary."""
        self._physical_integrator = _restore_integrator(*state["physical"])
        self._extension_integrator = _restore_integrator(*state["extension"])
        self._assume_valid = state["assume_valid"]
        self._initialize()
```

In `LockstepIntegrator.__init__` add `self._assume_valid = assume_force_first` as the first statement; in `SplitIntegrator.__init__` add `self._assume_valid = assume_symmetric` as the first statement. Remove the now-unused `STRING_SEPARATOR` import from `integrator.py`.

- [ ] **Step 4: Run tests**

Run: `pytest openxps/tests/test_thermostats.py openxps/tests/test_integrator.py openxps/tests/test_context.py openxps/tests/test_simulation.py -p no:cacheprovider`
Expected: PASS (checkpoint tests in `test_context.py`/`test_simulation.py` still pass; they use `mmswig` checkpoints, not integrator pickling).

- [ ] **Step 5: Commit**

```bash
git add openxps/integrator.py openxps/tests
git commit -m "Preserve integrator classes and validation flags when copying or pickling"
```

### Task 4.4: Random-number hygiene (L2, L3, L4)

- [ ] **Step 1: Write the failing tests**

Append to `openxps/tests/test_thermostats.py`:

```python
def test_default_extension_integrator_uses_a_different_seed():
    """L2: a copied extension integrator must not replay the physical noise."""
    from openxps import LockstepIntegrator

    langevin = mm.LangevinMiddleIntegrator(300 * KELVIN, 1 / PS, 1 * FS)
    langevin.setRandomNumberSeed(42)
    lockstep = LockstepIntegrator(langevin)
    assert lockstep.getExtensionIntegrator().getRandomNumberSeed() == 43
    csvr = integrators.CSVRIntegrator(300 * KELVIN, 10 / PS, 1 * FS, forceFirst=True)
    lockstep = LockstepIntegrator(csvr)
    assert csvr._rng.random() != lockstep.getExtensionIntegrator()._rng.random()


def test_csvr_seed_follows_openmm_conventions():
    """L3: getRandomNumberSeed returns what was set; 0 means random."""
    a = integrators.CSVRIntegrator(300 * KELVIN, 10 / PS, 1 * FS)
    b = integrators.CSVRIntegrator(300 * KELVIN, 10 / PS, 1 * FS)
    for integrator in (a, b):
        integrator.registerWithSystem(_ideal_gas(), False)
        integrator.setRandomNumberSeed(5)
    assert a.getRandomNumberSeed() == 5
    np.testing.assert_allclose(
        a._sums_of_squared_gaussians(3), b._sums_of_squared_gaussians(3)
    )
    a.setRandomNumberSeed(0)
    b.setRandomNumberSeed(0)
    assert not np.allclose(a._sums_of_squared_gaussians(3), b._sums_of_squared_gaussians(3))


def test_csvr_step_before_register_raises_clear_error():
    """L4"""
    integrator = integrators.CSVRIntegrator(300 * KELVIN, 10 / PS, 1 * FS)
    _context(_ideal_gas(2), integrator)
    with pytest.raises(ValueError, match="registerWithSystem"):
        integrator.step(1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest openxps/tests/test_thermostats.py -p no:cacheprovider -k "different_seed or openmm_conventions or clear_error"`
Expected: FAIL — extension seed is `42`; `getRandomNumberSeed() == 595007472`; `TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'`.

- [ ] **Step 3: Implement**

In `openxps/integrator.py`, add a helper after `_restore_integrator`:

```python
def _decorrelate_random_seed(physical: mm.Integrator, extension: mm.Integrator) -> None:
    """Give a copied extension integrator a seed different from the physical one."""
    if hasattr(physical, "getRandomNumberSeed"):
        seed = physical.getRandomNumberSeed()
        extension.setRandomNumberSeed(0 if seed == 0 else seed + 1)
```

and call it right after each `extension_integrator = deepcopy(physical_integrator)` in `LockstepIntegrator.__init__` and `SplitIntegrator.__init__`:

```python
            _decorrelate_random_seed(physical_integrator, extension_integrator)
```

In `openxps/integrators/csvr.py`:

```python
    def _sums_of_squared_gaussians(self, num_steps: int) -> np.ndarray:
        num_dof = self.getNumDegreesOfFreedom()
        sumRsq = 2.0 * self._rng.standard_gamma((num_dof - 1) // 2, num_steps)
        if num_dof % 2 == 0:
            sumRsq += self._rng.standard_normal(num_steps) ** 2
        return sumRsq
```

```python
    def setRandomNumberSeed(self, seed: int) -> None:
        """
        This method overrides the :class:`openmm.CustomIntegrator` method to also seed
        the generator used to draw the sum of squared Gaussian variates. Following the
        OpenMM convention, a seed of 0 selects a random, non-reproducible seed.

        .. note::
            The state of this generator is not part of a context checkpoint, so a
            CSVR simulation restarted from a checkpoint reproduces the thermostat
            noise only up to the point where the checkpoint was written.

        Parameters
        ----------
        seed
            The seed to use for the random number generators.
        """
        super().setRandomNumberSeed(seed)
        self._rng = np.random.default_rng(None if seed == 0 else seed)
```

- [ ] **Step 4: Run tests**

Run: `pytest openxps/tests/test_thermostats.py openxps/tests/test_integrator.py openxps/integrators -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openxps/integrator.py openxps/integrators/csvr.py openxps/tests/test_thermostats.py
git commit -m "Decorrelate copied integrator seeds and follow OpenMM seed conventions in CSVR"
```

### Task 4.5: GGMT time-constant unit (L5)

- [ ] **Step 1: Write the failing test**

Append to `openxps/tests/test_thermostats.py`:

```python
def test_ggmt_time_constant_unit():
    """L5"""
    integrator = integrators.MassiveGGMTIntegrator(300 * KELVIN, 0.1, 1 * FS)
    assert integrator.getTimeConstant() == 0.1 * PS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest openxps/tests/test_thermostats.py -p no:cacheprovider -k time_constant_unit`
Expected: FAIL — `0.1 /ps != 0.1 ps`

- [ ] **Step 3: Implement**

In `openxps/integrators/massive_ggmt.py`, change `@add_property("time constant", 1 / mmunit.picosecond)` to `@add_property("time constant", mmunit.picosecond)`.

- [ ] **Step 4: Run tests and commit**

Run: `pytest openxps/tests/test_thermostats.py openxps/integrators/massive_ggmt.py -p no:cacheprovider`
Expected: PASS

```bash
git add openxps/integrators/massive_ggmt.py openxps/tests/test_thermostats.py
git commit -m "Declare the GGMT time constant in picoseconds"
```

### Task 4.6: CSVR with zero kinetic energy (M11, Review Focus 1)

- [ ] **Step 1: Write the failing test**

Append to `openxps/tests/test_thermostats.py`:

```python
def test_csvr_at_rest_does_not_produce_nan():
    """M11 / Review Focus 1: a system at rest under zero force stays at rest."""
    integrator = integrators.CSVRIntegrator(300 * KELVIN, 10 / PS, 1 * FS, forceFirst=True)
    system = _ideal_gas(3)
    integrator.registerWithSystem(system, True)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(np.zeros((3, 3)))
    context.setVelocities(np.zeros((3, 3)))
    integrator.step(5)
    velocities = context.getState(getVelocities=True).getVelocities(asNumpy=True)._value
    assert np.all(np.isfinite(velocities))
    assert np.all(velocities == 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest openxps/tests/test_thermostats.py -p no:cacheprovider -k at_rest`
Expected: FAIL — velocities are NaN.

- [ ] **Step 3: Implement**

In `openxps/integrators/csvr.py`, change `_add_rescaling`:

```python
    def _add_rescaling(self, fraction: float) -> None:
        self.addComputeSum("mvv", "m*v*v")
        self.addComputeGlobal(
            "scaling_factor",
            "select(mvv, sqrt(A + BC*(R1^2 + sumRsq) + 2*sqrt(A*BC)*R1), 1)"
            "; R1 = gaussian"
            "; BC = (1 - A)*kT/mvv"
            f"; A = exp(-{fraction}*dt*friction)",
        )
        self.addComputePerDof("v", "v*scaling_factor")
```

Update the class docstring: step 9 of the listed computation steps becomes

```
       9: scaling_factor <- select(mvv, sqrt(A + BC*(R1^2 + sumRsq) + 2*sqrt(A*BC)*R1), 1); R1 = ...
```

and add this note after the existing `.. warning::` block:

```
    .. note::
        A system with zero kinetic energy has no velocity direction to rescale, so it
        is left at rest. Initialize velocities (e.g. with
        ``setVelocitiesToTemperature`` or
        :meth:`~openxps.ExtendedSpaceContext.setDynamicalVariableVelocitiesToTemperature`)
        before starting a simulation.
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest openxps/tests/test_thermostats.py openxps/integrators/csvr.py -p no:cacheprovider`
Expected: PASS

```bash
git add openxps/integrators/csvr.py openxps/tests/test_thermostats.py
git commit -m "Leave a system at rest under CSVR instead of producing NaN"
```

### Task 4.7: Thermostat sampling tests (M23)

- [ ] **Step 1: Write the tests**

Append to `openxps/tests/test_thermostats.py`:

```python
def _mean_kinetic_energy_per_dof(integrator, num_particles=100, equil=500, prod=1500):
    system = _ideal_gas(num_particles)
    if isinstance(integrator, integrators.CSVRIntegrator):
        integrator.registerWithSystem(system, True)
    context = _context(system, integrator)
    integrator.step(equil)
    total = 0.0
    for _ in range(prod // 10):
        integrator.step(10)
        total += context.getState(getEnergy=True).getKineticEnergy()._value
    return total / (prod // 10) / (3 * num_particles)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: integrators.SymmetricLangevinIntegrator(300 * KELVIN, 10 / PS, 2 * FS),
        lambda: integrators.CSVRIntegrator(300 * KELVIN, 10 / PS, 2 * FS),
        lambda: integrators.MassiveGGMTIntegrator(300 * KELVIN, 0.1 * PS, 2 * FS),
        lambda: integrators.MassiveGGMTIntegrator(300 * KELVIN, 0.1 * PS, 2 * FS, stochastic=True),
        lambda: integrators.RegulatedNHLIntegrator(300 * KELVIN, 0.1 * PS, 2 * FS),
    ],
)
def test_thermostats_reproduce_target_temperature(factory):
    """Each thermostat gives <K>/dof = kT/2 within 5% on an ideal gas."""
    integrator = factory()
    integrator.setRandomNumberSeed(7)
    kt = (mmunit.MOLAR_GAS_CONSTANT_R * 300 * KELVIN).value_in_unit(
        mmunit.kilojoule_per_mole
    )
    assert _mean_kinetic_energy_per_dof(integrator) == pytest.approx(0.5 * kt, rel=0.05)
```

- [ ] **Step 2: Run the tests**

Run: `pytest openxps/tests/test_thermostats.py -p no:cacheprovider -k target_temperature`
Expected: PASS for all five (Regulated NHL samples a regulated, not canonical, kinetic energy; if its mean falls outside 5%, widen its tolerance to 10% and say so in the test's docstring, because the regulated kinetic energy `0.5*m*v*c*tanh(v/c)` with the default regulation parameter is what `getKineticEnergy` reports and it equipartitions to kT/2 only in the limit of large regulation parameter).

- [ ] **Step 3: Run the full suite and linters, then open the PR**

Run: `pytest -p no:cacheprovider && ruff check openxps && ruff format --check openxps`
Expected: all PASS

```bash
git add openxps/tests/test_thermostats.py
git commit -m "Add thermostat sampling tests"
git push -u origin fix/integrators
```

PR title: `Fix integrator setters, step-size handling, pickling and CSVR edge cases`. Body: closes H4, M2, M10, M11, M15, L2, L3, L4, L5, L15 and the integrator part of M23.

---

## PR 5 — `feat/ggmt-coulomb-splitting` (M24)

**Files:**
- Modify: `openxps/integrators/massive_ggmt.py`
- Modify: `openxps/tests/test_thermostats.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `IntegratorMixin.setStepSize` hook (PR 4) so `_update_global_variables` runs on step-size changes.
- Produces: `MassiveGGMTIntegrator(..., newtonIterations: int = 5)` keyword; per-DOF variables `v1, v2, z, w, E, s, zmin, th, T1` (the `v2lb` variable is removed).

Background (see `CODE_REVIEW.md`, section "GGMT stabilization" and the manuscript at `github.com/craabreu/ggmt_integrator`, file `coulomb.tex`): with `z = sqrt(3)*kT/(m*v^2)` and `w = (2/sqrt(3))*kT*v2`, the cubic-friction/`v2` sub-dynamics is Hamiltonian with `H = w^2/2 + c*(z + 1/z)`, `c = (2/sqrt(3))*kT^3*invQ2 = sqrt(3)/(4*tau^2)`. Splitting `H = [w^2/2 + c/z] + c*z`, the bracket is a 1-D repulsive Coulomb problem solved exactly by `z = zmin*cosh(th)^2`, `w = s*tanh(th)`, `s*t/zmin = M(th) - M(th0)`, `M(th) = th + sinh(2*th)/2`; the linear part is a constant kick on `w`. This replaces the `v2lb` reflection, which does not prevent `z` from reaching zero.

### Task 5.1: Replace the `v2lb` reflection with the Coulomb splitting

- [ ] **Step 1: Write the failing tests**

Append to `openxps/tests/test_thermostats.py`:

```python
def _harmonic_oscillators(num_particles, mass=1.0, k=100.0):
    system = mm.System()
    force = mm.CustomExternalForce("0.5*k*(x^2+y^2+z^2)")
    force.addGlobalParameter("k", k)
    for i in range(num_particles):
        system.addParticle(mass)
        force.addParticle(i, [])
    system.addForce(force)
    return system


def _platform():
    names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    return mm.Platform.getPlatformByName("CPU" if "CPU" in names else "Reference")


def test_ggmt_has_no_square_root_problem_at_large_steps():
    """M24: dt = tau/5 must not produce NaN (the old scheme failed at tau/10)."""
    integrator = integrators.MassiveGGMTIntegrator(300 * KELVIN, 0.1 * PS, 20 * FS)
    system = _harmonic_oscillators(200)
    context = mm.Context(system, integrator, _platform())
    rng = np.random.default_rng(0)
    kt = (mmunit.MOLAR_GAS_CONSTANT_R * 300 * KELVIN).value_in_unit(mmunit.kilojoule_per_mole)
    context.setPositions(rng.normal(0, np.sqrt(kt / 100.0), (200, 3)))
    context.setVelocitiesToTemperature(300 * KELVIN, 1)
    integrator.step(5000)
    velocities = context.getState(getVelocities=True).getVelocities(asNumpy=True)._value
    assert np.all(np.isfinite(velocities))


def test_ggmt_fourth_moment_at_one_tenth_of_tau():
    """M24: <(m v^2)^2>/(kT)^2 = 3 within 5% at dt = tau/10 on harmonic oscillators."""
    integrator = integrators.MassiveGGMTIntegrator(300 * KELVIN, 0.1 * PS, 10 * FS)
    integrator.setRandomNumberSeed(3)
    num = 300
    system = _harmonic_oscillators(num)
    context = mm.Context(system, integrator, _platform())
    rng = np.random.default_rng(0)
    kt = (mmunit.MOLAR_GAS_CONSTANT_R * 300 * KELVIN).value_in_unit(mmunit.kilojoule_per_mole)
    context.setPositions(rng.normal(0, np.sqrt(kt / 100.0), (num, 3)))
    context.setVelocitiesToTemperature(300 * KELVIN, 1)
    integrator.step(2000)
    second = fourth = 0.0
    samples = 400
    for _ in range(samples):
        integrator.step(10)
        v = context.getState(getVelocities=True).getVelocities(asNumpy=True)._value
        kin = v**2 / kt  # m = 1 Da
        second += kin.mean()
        fourth += (kin**2).mean()
    assert second / samples == pytest.approx(1.0, rel=0.03)
    assert fourth / samples == pytest.approx(3.0, rel=0.05)


def _reference_ggmt_step(v, v1, v2, kt, inv_q, inv_q2, dt, substeps=200):
    """One thermostat step (no forces) with the (z, w) sub-dynamics integrated by RK4."""
    c = (2 / np.sqrt(3)) * kt**3 * inv_q2

    def zw_flow(v, v2, h):
        z = np.sqrt(3) * kt / (v * v)
        w = (2 / np.sqrt(3)) * kt * v2
        f = lambda z, w: (w, c * (1 / z**2 - 1))
        hh = h / substeps
        for _ in range(substeps):
            k1 = f(z, w)
            k2 = f(z + 0.5 * hh * k1[0], w + 0.5 * hh * k1[1])
            k3 = f(z + 0.5 * hh * k2[0], w + 0.5 * hh * k2[1])
            k4 = f(z + hh * k3[0], w + hh * k3[1])
            z += hh / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            w += hh / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return np.sign(v) * np.sqrt(np.sqrt(3) * kt / z), (np.sqrt(3) / 2) * w / kt

    v1 += 0.5 * dt * (v * v - kt) * inv_q
    v, v2 = zw_flow(v, v2, 0.5 * dt)
    v *= np.exp(-dt * (v1 + kt * v2))
    v, v2 = zw_flow(v, v2, 0.5 * dt)
    v1 += 0.5 * dt * (v * v - kt) * inv_q
    return v, v1, v2


def test_ggmt_step_matches_reference_sub_dynamics():
    """The Lepton transcription of the Coulomb splitting reproduces the exact
    (z, w) flow to second order in the step size."""
    dt = 0.0002  # ps
    integrator = integrators.MassiveGGMTIntegrator(300 * KELVIN, 0.1 * PS, dt * PS)
    system = _ideal_gas(1, mass=1.0)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions([mm.Vec3(0, 0, 0)])
    v0, v1_0, v2_0 = 2.5, 0.3, -1.7
    context.setVelocities([mm.Vec3(v0, -v0, 0.5 * v0)])
    integrator.setPerDofVariableByName("v1", [mm.Vec3(v1_0, v1_0, v1_0)])
    integrator.setPerDofVariableByName("v2", [mm.Vec3(v2_0, v2_0, v2_0)])
    kt = integrator.getGlobalVariableByName("kT")
    inv_q = integrator.getGlobalVariableByName("invQ")
    inv_q2 = integrator.getGlobalVariableByName("invQ2")
    integrator.step(1)
    v = context.getState(getVelocities=True).getVelocities(asNumpy=True)._value[0]
    v2 = integrator.getPerDofVariableByName("v2")[0]
    for component, start in ((0, v0), (1, -v0), (2, 0.5 * v0)):
        v_ref, _, v2_ref = _reference_ggmt_step(start, v1_0, v2_0, kt, inv_q, inv_q2, dt)
        assert v[component] == pytest.approx(v_ref, rel=1e-6)
        assert v2[component] == pytest.approx(v2_ref, rel=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest openxps/tests/test_thermostats.py -p no:cacheprovider -k "square_root or fourth_moment or reference_sub"`
Expected: FAIL — NaN at 20 fs (OpenMMException "Particle coordinate is NaN") and/or the fourth moment off; the reference test may pass or fail depending on how the `v2lb` reflection fires.

- [ ] **Step 3: Implement in `openxps/integrators/massive_ggmt.py`**

Add the module constant after the imports:

```python
#: c = sqrt(3)/(4 tau^2) written in terms of the global variables of the integrator.
_COULOMB_STRENGTH = "(2/sqrt(3))*kT^3*invQ2"
```

Add the keyword argument and attribute in `__init__` (signature and first lines):

```python
    @preprocess_args
    def __init__(  # noqa: PLR0913
        self,
        temperature: mmunit.Quantity,
        timeConstant: mmunit.Quantity,
        stepSize: mmunit.Quantity,
        *,
        stochastic: bool = False,
        bathLoops: int = 1,
        forceFirst: bool = False,
        newtonIterations: int = 5,
    ) -> None:
        if bathLoops < 1:
            raise ValueError("The number of bath loops must be at least 1.")
        if newtonIterations < 1:
            raise ValueError("The number of Newton iterations must be at least 1.")
        super().__init__(stepSize)
        self._forceFirst = forceFirst
        self._newton_iterations = newtonIterations
```

(keep the rest of `__init__` as it is). In `_add_variables`, replace `self.addPerDofVariable("v2lb", 0)` with:

```python
        for name in ("z", "w", "E", "s", "zmin", "th", "T1"):
            self.addPerDofVariable(name, 0)
```

Replace `_add_v2_boost`:

```python
    def _add_v2_boost(self, fraction: float) -> None:
        """Propagate the joint dynamics of the cubic friction and v2 over a time
        fraction*dt by the Coulomb splitting: a half kick of the constant part of the
        force on v2, the exact flow of the repulsive Coulomb Hamiltonian
        w^2/2 + c/z, and another half kick. The time equation of the Coulomb flow,
        th + sinh(2 th)/2 = M, is solved by Newton iterations started from the upper
        bound asinh(2|M|)/2, which converge monotonically."""
        add = self.addComputePerDof
        c = _COULOMB_STRENGTH
        h = f"({fraction}*dt)"
        const_kick = f"v2 - {fraction / 2}*dt*kT^2*invQ2"
        add("v2", const_kick)
        # p = 0 is a fixed point of the exact flow; the floor only avoids 0*inf
        add("z", "sqrt(3)*kT/max(m*v*v, 1e-30*kT)")
        add("w", "(2/sqrt(3))*kT*v2")
        add("E", f"0.5*w*w + c/z; c={c}")
        add("zmin", f"c/E; c={c}")
        add("s", "sqrt(2*E)")
        # th0 = asinh(q), q = w*sqrt(z/(2c)), written to avoid cancellation for q < 0
        add("th", f"(2*step(q) - 1)*log(abs(q) + sqrt(q*q + 1)); q = w*sqrt(z/(2*c)); c={c}")
        add("T1", f"th + 0.5*sinh(2*th) + s*{h}/zmin")
        add("th", "0.5*log(q + sqrt(q*q + 1)); q = 2*abs(T1)")
        for _ in range(self._newton_iterations):
            add("th", "th - (th + 0.5*sinh(2*th) - abs(T1))/(1 + cosh(2*th))")
        add("th", "(2*step(T1) - 1)*th")
        add("v", "v*sqrt(z/znew); znew = zmin*cosh(th)^2")
        add("w", "s*tanh(th)")
        add("v2", "(sqrt(3)/2)*w/kT")
        add("v2", const_kick)
```

Update the class docstring: add to the `Keyword Arguments` section

```
    newtonIterations
        The number of Newton iterations used to solve the time equation of the
        Coulomb sub-step (see Notes). The default is sufficient for double precision.
```

add a `Notes` section before `Example`:

```
    Notes
    -----
    The cubic friction term of the GGMT equations has a finite-time singularity when
    propagated in isolation, which shows up as a negative argument under a square
    root in Trotter-factorized integrators. This implementation propagates the cubic
    friction together with the evolution of the fourth-moment thermostat momentum,
    whose joint dynamics is Hamiltonian in suitable variables, by splitting that
    Hamiltonian into an exactly solvable repulsive Coulomb problem and a constant
    kick. The resulting step is free of the singularity for any step size, so the
    usable step size is limited by the remaining factors of the integrator rather
    than by the thermostat.
```

and replace the listed per-DOF variables and computation steps of the symmetric example with (the doctest uses `ELLIPSIS`):

```
    >>> integrator
    Per-dof variables:
      v1, v2, z, w, E, s, zmin, th, T1
    Global variables:
      kT = 2.494338...
      invQ = 40.090...
      invQ2 = 2.416...
    Computation steps:
       0: allow forces to update the context state
       1: v <- v + 0.5*dt*f/m
       2: x <- x + 0.5*dt*v
       3: v1 <- v1 + 0.5*dt*(m*v^2 - kT)*invQ
       4: v2 <- v2 - 0.25*dt*kT^2*invQ2
       5: z <- sqrt(3)*kT/max(m*v*v, 1e-30*kT)
       ...
      23: v <- v*exp(-1*dt*(v1 + kT*v2))
       ...
      43: v1 <- v1 + 0.5*dt*(m*v^2 - kT)*invQ
      44: x <- x + 0.5*dt*v
      45: v <- v + 0.5*dt*f/m
    >>> integrator.getNumComputations()
    46
```

- [ ] **Step 4: Run the tests**

Run: `pytest openxps/tests/test_thermostats.py openxps/integrators/massive_ggmt.py -p no:cacheprovider`
Expected: PASS. If the doctest's step numbering differs, print `repr(integrator)` and align the listed lines (the structure is: 4 header steps, 19 steps per `_add_v2_boost`, the scaling step, 19 more, then 3 closing steps = 46).

- [ ] **Step 5: Full suite, changelog, PR**

Append under `## Unreleased` → `### Changed` in `CHANGELOG.md`:

```markdown
- `MassiveGGMTIntegrator` propagates the fourth-moment thermostat sub-dynamics with a
  singularity-free Coulomb splitting instead of a reflective bound on `v2`; the
  per-DOF variable `v2lb` is gone and `newtonIterations` was added.
```

Run: `pytest -p no:cacheprovider && ruff check openxps && ruff format --check openxps`

```bash
git add openxps/integrators/massive_ggmt.py openxps/tests/test_thermostats.py CHANGELOG.md
git commit -m "Replace the GGMT v2 reflection with a Coulomb splitting of the sub-dynamics"
git push -u origin feat/ggmt-coulomb-splitting
```

PR title: `Singularity-free GGMT thermostat via Coulomb splitting`. Body: closes M24; link the manuscript repository.

---

## PR 6 — `fix/reporting-metadynamics` (M3, M16, L6, L14-writer, L17, M23-reporting)

**Files:**
- Modify: `openxps/extension_writer.py`, `openxps/metadynamics.py`
- Modify: `openxps/tests/test_extension_writer.py`
- Create: `openxps/tests/test_metadynamics.py`

**Interfaces:**
- Produces: `ExtensionWriter` getters raise `RuntimeError` before `initialize()`; `ExtensionWriter.getCollectiveVariableHeaders()`/`getEffectiveMassHeaders()` work regardless of the output flags.

### Task 6.1: Kinetic energy and temperature from the x velocities only (M3)

- [ ] **Step 1: Write the failing test**

Append to `openxps/tests/test_extension_writer.py`:

```python
def test_extension_writer_temperature_is_physical():
    """M3: the reported temperature uses one degree of freedom per DV."""
    model = testsystems.AlanineDipeptideVacuum()
    mass = 3 * unit.dalton * (unit.nanometer / unit.radian) ** 2
    phi0 = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    phi = cvpack.Torsion(6, 8, 14, 16, name="phi")
    kappa = 1000 * unit.kilojoules_per_mole / unit.radian**2
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 10 / unit.picosecond, 4 * unit.femtosecond
    )
    integrator.setRandomNumberSeed(1234)
    simulation = xps.ExtendedSpaceSimulation(
        model.topology,
        xps.ExtendedSpaceSystem(model.system, xps.HarmonicCoupling(phi, phi0, kappa)),
        xps.LockstepIntegrator(integrator),
        openmm.Platform.getPlatformByName("Reference"),
    )
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    simulation.context.setDynamicalVariableValues([180 * unit.degree])
    simulation.context.setDynamicalVariableVelocitiesToTemperature(300 * unit.kelvin, 1234)
    writer = xps.ExtensionWriter(kinetic=True, temperature=True)
    writer.initialize(simulation)
    simulation.step(1000)
    temperatures = []
    for _ in range(1000):
        simulation.step(10)
        kinetic, temperature = writer.getValues(simulation)
        temperatures.append(temperature)
        rate = simulation.context.getDynamicalVariableVelocities()[0]
        v = rate.value_in_unit(unit.radian / unit.picosecond)
        assert kinetic == pytest.approx(0.5 * 3 * v * v, rel=1e-6)
    assert abs(np.mean(temperatures) - 300) < 80  # 1 DOF: wide but unbiased
```

Add `import numpy as np` and `import pytest` to the test module imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest openxps/tests/test_extension_writer.py -p no:cacheprovider -k is_physical`
Expected: FAIL — `kinetic` includes y/z components (not equal to `0.5*m*v_x^2`), and the mean temperature is near 900 K.

- [ ] **Step 3: Implement in `openxps/extension_writer.py`**

Add `import numpy as np`. In `__init__` replace `self._needs_energy = kinetic or temperature` with `self._needs_velocities = kinetic or temperature` and add `self._masses = np.empty(0)`. In `initialize`, after `self._dv_objects = ...`, add:

```python
        self._masses = np.array([dv.mass / dv.mass.unit for dv in self._dv_objects])
```

In `getValues`, request velocities instead of energy and compute the kinetic energy from the x components:

```python
        state = extension_context.getState(
            velocities=self._needs_velocities,
            positions=self._needs_positions,
            forces=self._needs_forces,
        )
        if self._needs_velocities:
            velocities = mmswig.State__getVectorAsVec3(state, mm.State.Velocities)
            rates = np.array([v.x for v in velocities])
            kinetic_energy = 0.5 * float(np.dot(self._masses, rates * rates))
```

(The DV masses are stored in `dalton*(nm/unit)^2` and the rates in `unit/ps`, so the product is in kJ/mol.) Update the `kinetic` and `temperature` parameter docstrings to say "computed from the dynamical variables' own velocities (one degree of freedom per variable)".

- [ ] **Step 4: Run tests and commit**

Run: `pytest openxps/tests/test_extension_writer.py openxps/extension_writer.py -p no:cacheprovider`
Expected: PASS

```bash
git add openxps/extension_writer.py openxps/tests/test_extension_writer.py
git commit -m "Report extension kinetic energy and temperature from the dynamical variables only"
```

### Task 6.2: Header getters that do not depend on output flags (M16, L14)

- [ ] **Step 1: Write the failing tests**

Append to `openxps/tests/test_extension_writer.py`:

```python
def test_extension_writer_headers_independent_of_flags():
    """M16: getters work after initialize() regardless of which columns are enabled."""
    simulation = create_test_simulation()
    writer = xps.ExtensionWriter(dynamical_variables=True)
    with pytest.raises(RuntimeError, match="initialize"):
        writer.getCollectiveVariableHeaders()
    writer.initialize(simulation)
    assert writer.getDynamicalVariableHeaders() == ["phi0 (rad)"]
    assert writer.getCollectiveVariableHeaders() == ["phi (rad)"]
    assert writer.getEffectiveMassHeaders() == ["emass[phi] (nm**2 Da/(rad**2))"]
    assert writer.getHeaders() == ["phi0 (rad)"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest openxps/tests/test_extension_writer.py -p no:cacheprovider -k independent_of_flags`
Expected: FAIL — no `RuntimeError`; `getCollectiveVariableHeaders() == []`.

- [ ] **Step 3: Implement**

In `__init__`, add `self._initialized = False` and `self._mass_units = []`. Rewrite `initialize` so that every list is built unconditionally:

```python
    def initialize(self, simulation: ExtendedSpaceSimulation) -> None:
        context = simulation.extended_space_context
        coupling = context.getSystem().getCoupling()
        self._dv_objects = coupling.getDynamicalVariables()
        self._masses = np.array([dv.mass / dv.mass.unit for dv in self._dv_objects])
        kb = mmunit.MOLAR_GAS_CONSTANT_R.value_in_unit(
            mmunit.kilojoules_per_mole / mmunit.kelvin
        )
        self._temp_factor = 2 / (len(self._dv_objects) * kb)
        self._meta_cvs = [
            force
            for force in coupling.getForces()
            if isinstance(force, cvpack.MetaCollectiveVariable)
        ]
        self._mass_units = [
            mmunit.dalton * (mmunit.nanometer / cv.getUnit()) ** 2
            for meta_cv in self._meta_cvs
            for cv in meta_cv.getInnerVariables()
        ]
        dv_names = {dv.name for dv in self._dv_objects}
        self._function_names = [
            parameter
            for parameter in coupling.getProtectedParameters()
            if parameter not in dv_names
        ]
        self._initialized = True

    def _checkInitialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("The writer must be initialized with a simulation first.")
```

Call `self._checkInitialized()` as the first statement of `getDynamicalVariableHeaders`, `getForceHeaders`, `getCollectiveVariableHeaders`, `getEffectiveMassHeaders`, and `getHeaders`. Rewrite `getEffectiveMassHeaders` and the effective-mass branch of `getValues` to use the cache:

```python
    def getEffectiveMassHeaders(self) -> list[str]:
        self._checkInitialized()
        names = [cv.getName() for meta_cv in self._meta_cvs for cv in meta_cv.getInnerVariables()]
        return [
            f"emass[{name}] ({mass_unit.get_symbol()})"
            for name, mass_unit in zip(names, self._mass_units)
        ]
```

```python
        if self._effective_masses:
            masses = [
                mass
                for meta_cv in self._meta_cvs
                for mass in meta_cv.getInnerEffectiveMasses(context).values()
            ]
            for mass, mass_unit in zip(masses, self._mass_units):
                values.append(mass.value_in_unit(mass_unit))
```

Remove the `# noqa: PLR0912` comments from `getHeaders` and `getValues` (run `ruff check openxps/extension_writer.py`; if PLR0912 still fires on `getValues`, keep that one comment).

- [ ] **Step 4: Run tests and commit**

Run: `pytest openxps/tests/test_extension_writer.py openxps/extension_writer.py -p no:cacheprovider && ruff check openxps/extension_writer.py`
Expected: PASS

```bash
git add openxps/extension_writer.py openxps/tests/test_extension_writer.py
git commit -m "Build extension writer headers independently of output flags"
```

### Task 6.3: Metadynamics validation, docs and tests (L6, L17, M23)

- [ ] **Step 1: Write the failing tests**

Create `openxps/tests/test_metadynamics.py`:

```python
"""
Unit tests for the openxps.metadynamics module.
"""

import cvpack
import openmm
import pytest
import yaml
from openmm import unit
from openmmtools import testsystems

import openxps as xps


def _system_with(dv, cv=None):
    model = testsystems.AlanineDipeptideVacuum()
    cv = cv or cvpack.Torsion(6, 8, 14, 16, name="phi")
    kappa = 1000 * unit.kilojoules_per_mole / dv.unit**2
    coupling = xps.HarmonicCoupling(cv, dv, kappa)
    return model, xps.ExtendedSpaceSystem(model.system, coupling)


def _metadynamics(system, variable, **kwargs):
    return xps.ExtendedSpaceMetadynamics(
        system=system, variables=[variable], temperature=300 * unit.kelvin,
        biasFactor=5, height=2 * unit.kilojoule_per_mole, frequency=10, **kwargs,
    )


def test_bias_variable_rejects_non_positive_sigma():
    """L6"""
    mass = 3 * unit.dalton * (unit.nanometer / unit.radian) ** 2
    dv = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    with pytest.raises(ValueError, match="sigma must be positive"):
        xps.ExtendedSpaceBiasVariable(dv, -18 * unit.degree)
    with pytest.raises(ValueError, match="grid_width must be greater than 1"):
        xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree, grid_width=1)


def test_bias_variable_yaml_roundtrip():
    mass = 3 * unit.dalton * (unit.nanometer / unit.radian) ** 2
    dv = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    variable = xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree, grid_width=50)
    assert yaml.safe_load(yaml.safe_dump(variable)) == variable


def test_metadynamics_with_degree_dv_uses_radian_grid():
    mass = 3 * unit.dalton * (unit.nanometer / unit.degree) ** 2
    dv = xps.DynamicalVariable("phi0", unit.degree, mass, xps.PeriodicBounds(-180, 180, unit.degree))
    _, system = _system_with(dv)
    meta = _metadynamics(system, xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree))
    (variable,) = meta.variables
    assert variable.minValue == pytest.approx(-3.14159, rel=1e-5)
    assert variable.maxValue == pytest.approx(3.14159, rel=1e-5)
    assert variable.periodic


def test_metadynamics_with_reflective_dv():
    mass = 10 * unit.dalton
    dv = xps.DynamicalVariable("d0", unit.nanometer, mass, xps.ReflectiveBounds(0.1, 1.0, unit.nanometer))
    _, system = _system_with(dv, cvpack.Distance(0, 1, name="d"))
    meta = _metadynamics(system, xps.ExtendedSpaceBiasVariable(dv, 0.05 * unit.nanometer))
    (variable,) = meta.variables
    assert not variable.periodic


def test_metadynamics_rejects_unknown_and_unbounded_dvs():
    mass = 3 * unit.dalton * (unit.nanometer / unit.radian) ** 2
    dv = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    _, system = _system_with(dv)
    other = xps.DynamicalVariable("psi0", unit.radian, mass, xps.CircularBounds())
    with pytest.raises(ValueError, match="not found in system"):
        _metadynamics(system, xps.ExtendedSpaceBiasVariable(other, 18 * unit.degree))
    free = xps.DynamicalVariable("x0", unit.nanometer, 1 * unit.dalton, xps.NoBounds(0, 1, unit.nanometer))
    _, free_system = _system_with(free, cvpack.Distance(0, 1, name="d"))
    with pytest.raises(ValueError, match="has no bounds"):
        _metadynamics(free_system, xps.ExtendedSpaceBiasVariable(free, 0.1 * unit.nanometer))


def test_metadynamics_step_and_collective_variables():
    model = testsystems.AlanineDipeptideVacuum()
    mass = 3 * unit.dalton * (unit.nanometer / unit.radian) ** 2
    dv = xps.DynamicalVariable("phi0", unit.radian, mass, xps.CircularBounds())
    _, system = _system_with(dv)
    meta = _metadynamics(system, xps.ExtendedSpaceBiasVariable(dv, 18 * unit.degree))
    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    integrator.setRandomNumberSeed(1)
    simulation = xps.ExtendedSpaceSimulation(
        model.topology, system, xps.LockstepIntegrator(integrator),
        openmm.Platform.getPlatformByName("Reference"),
    )
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    simulation.context.setDynamicalVariableValues([180 * unit.degree])
    simulation.context.setDynamicalVariableVelocitiesToTemperature(300 * unit.kelvin, 1)
    meta.step(simulation, 50)
    (value,) = meta.getCollectiveVariables(simulation)
    (dv_value,) = simulation.context.getDynamicalVariableValues()
    assert value == pytest.approx(dv_value.value_in_unit(unit.radian), abs=1e-6)
    assert meta.getFreeEnergy().shape == (meta.variables[0].gridWidth,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest openxps/tests/test_metadynamics.py -p no:cacheprovider`
Expected: FAIL on `rejects_non_positive_sigma` (no validation); the others should pass or reveal real defects — fix any that fail for a reason other than missing validation before moving on.

- [ ] **Step 3: Implement in `openxps/metadynamics.py`**

Extend `ExtendedSpaceBiasVariable.__post_init__`:

```python
    def __post_init__(self) -> None:
        if not mmunit.is_quantity(self.sigma):
            raise TypeError("sigma must be a quantity")
        if not self.sigma.unit.is_compatible(self.dynamical_variable.unit):
            raise ValueError(
                "sigma must be compatible with the dynamical variable's unit"
            )
        if self.sigma / self.sigma.unit <= 0:
            raise ValueError("sigma must be positive")
        if not (self.grid_width is None or isinstance(self.grid_width, int)):
            raise TypeError("grid_width must be an integer or None")
        if self.grid_width is not None and self.grid_width <= 1:
            raise ValueError("grid_width must be greater than 1")
```

Add to the `ExtendedSpaceMetadynamics` class docstring, after the parameter list and before `Example`:

```
    .. note::
        The bias force is added to the extension system of ``system`` when this
        object is created, so it must be created **before** the
        :class:`~openxps.ExtendedSpaceSimulation` or
        :class:`~openxps.ExtendedSpaceContext` that uses the system.
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest openxps/tests/test_metadynamics.py openxps/metadynamics.py -p no:cacheprovider`
Expected: PASS

```bash
git add openxps/metadynamics.py openxps/tests/test_metadynamics.py
git commit -m "Validate bias variables and add metadynamics tests"
```

- [ ] **Step 5: Full suite and PR**

Run: `pytest -p no:cacheprovider && ruff check openxps && ruff format --check openxps`

```bash
git push -u origin fix/reporting-metadynamics
```

PR title: `Fix extension writer energies and add metadynamics tests`. Body: closes M3, M16, L6, L17, writer part of L14 and reporting part of M23.

---

## PR 7 — `fix/regression` (H2, H3, H9, M4, M5, M13, M17, M18-regression, M19, L1, L10, L12, L13)

**Files:**
- Modify: `openxps/regression.py`
- Modify: `openxps/tests/test_regression.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Produces: `ForceMatchingRegressor(..., devices: int | str = 1)`; `ForceMatchingRegressor.get_learning_curve()` returns a copy with columns `epoch`, `train_loss` and (when validating) `val_loss`, and raises `RuntimeError` before `fit`; `RBFPotential._length_over_pi`/`_periodic` are buffers; `GradMatch.configure_optimizers()` returns an `AdamW` with two parameter groups; module-level `_LearningCurveRecorder(pl.Callback)`.

### Task 7.1: Kernel initialization and constant buffers (M5, L13)

- [ ] **Step 1: Write the failing tests**

Append to `openxps/tests/test_regression.py`, inside `class TestRBFPotential`:

```python
    def test_centers_start_inside_bounds(self):
        """M5: centers are drawn uniformly in [lower, upper]."""
        dvs = create_test_dynamical_variables_mixed()
        torch.manual_seed(0)
        potential = RBFPotential(dvs, M=2000)
        centers = potential.c.detach().numpy()
        for column, dv in zip(centers.T, dvs):
            assert column.min() >= dv.bounds.lower
            assert column.max() <= dv.bounds.upper
            assert abs(column.mean() - 0.5 * (dv.bounds.lower + dv.bounds.upper)) < 0.05 * dv.bounds.length

    def test_constants_are_buffers(self):
        """L13"""
        potential = RBFPotential(create_test_dynamical_variables(), M=5)
        buffers = dict(potential.named_buffers())
        assert set(buffers) == {"_length_over_pi", "_periodic"}
        assert {name for name, _ in potential.named_parameters()} == {"c", "logsig", "w"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest openxps/tests/test_regression.py -p no:cacheprovider -k "inside_bounds or are_buffers"`
Expected: FAIL — centers far outside bounds; buffers set is empty.

- [ ] **Step 3: Implement in `openxps/regression.py`**

In `RBFPotential.__init__`:

```python
            random_points.append(torch.rand(M) * dv.bounds.length + dv.bounds.lower)
```

and replace the two `nn.Parameter(..., requires_grad=False)` assignments with:

```python
        self.register_buffer(
            "_length_over_pi",
            torch.tensor(np.array(lengths) / np.pi, dtype=torch.float32)[None, None, :],
        )
        self.register_buffer(
            "_periodic", torch.tensor(periodic, dtype=torch.bool)[None, None, :]
        )
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest openxps/tests/test_regression.py -p no:cacheprovider -k TestRBFPotential`
Expected: PASS

```bash
git add openxps/regression.py openxps/tests/test_regression.py
git commit -m "Initialize RBF centers uniformly and store constants as buffers"
```

### Task 7.2: Optimizer parameter groups and checkpoint hyperparameters (L1, H2 prerequisite)

- [ ] **Step 1: Write the failing tests**

Append inside `class TestGradMatch` (create the class if `test_regression.py` has none; it exists):

```python
    def test_weight_decay_applies_only_to_weights(self):
        """L1"""
        model = GradMatch(create_test_dynamical_variables(), M=4)
        optimizer = model.configure_optimizers()
        groups = {id(p): g["weight_decay"] for g in optimizer.param_groups for p in g["params"]}
        assert groups[id(model.f.w)] == pytest.approx(1e-4)
        assert groups[id(model.f.c)] == 0.0
        assert groups[id(model.f.logsig)] == 0.0

    def test_hyperparameters_exclude_dynamical_variables(self):
        """H2: checkpoints must be loadable with weights_only=True."""
        model = GradMatch(create_test_dynamical_variables(), M=4)
        assert "dynamical_variables" not in model.hparams
        assert set(model.hparams) == {"M", "lr", "wd"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest openxps/tests/test_regression.py -p no:cacheprovider -k "only_to_weights or exclude_dynamical"`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
    def __init__(
        self,
        dynamical_variables: t.Sequence[DynamicalVariable],
        M: int = 256,
        lr: float = 2e-3,
        wd: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["dynamical_variables"])
        self.f = RBFPotential(dynamical_variables, M)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [
                {"params": [self.f.w], "weight_decay": self.hparams.wd},
                {"params": [self.f.c, self.f.logsig], "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
        )
```

- [ ] **Step 4: Run tests and commit**

Run: `pytest openxps/tests/test_regression.py -p no:cacheprovider -k TestGradMatch`
Expected: PASS

```bash
git add openxps/regression.py openxps/tests/test_regression.py
git commit -m "Restrict weight decay to kernel weights and keep checkpoints plain"
```

### Task 7.3: Learning curve recorder (H3, M17, L12)

- [ ] **Step 1: Write the failing tests**

Append to `openxps/tests/test_regression.py` (module level):

```python
class _FakeTrainer:
    def __init__(self, epoch, metrics, sanity_checking=False):
        self.current_epoch = epoch
        self.callback_metrics = metrics
        self.sanity_checking = sanity_checking


def test_learning_curve_recorder_tolerates_nan_validation_loss():
    """H3: a NaN val_loss must not break the epoch alignment."""
    from openxps.regression import _LearningCurveRecorder

    recorder = _LearningCurveRecorder()
    recorder.on_validation_epoch_end(_FakeTrainer(0, {"val_loss": torch.tensor(9.0)}, True), None)
    for epoch, val in enumerate((0.5, float("nan"), 0.3)):
        recorder.on_validation_epoch_end(_FakeTrainer(epoch, {"val_loss": torch.tensor(val)}), None)
        recorder.on_train_epoch_end(_FakeTrainer(epoch, {"train_loss": torch.tensor(1.0 / (epoch + 1))}), None)
    frame = recorder.to_frame()
    assert list(frame.columns) == ["epoch", "val_loss", "train_loss"]
    assert list(frame.epoch) == [0, 1, 2]
    assert np.isnan(frame.val_loss[1])
    assert frame.train_loss[2] == pytest.approx(1 / 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest openxps/tests/test_regression.py -p no:cacheprovider -k recorder`
Expected: FAIL — `ImportError: cannot import name '_LearningCurveRecorder'`

- [ ] **Step 3: Implement**

Add to `openxps/regression.py` after `GradMatch`:

```python
class _LearningCurveRecorder(pl.Callback):
    """Collect per-epoch training and validation losses keyed by epoch."""

    def __init__(self) -> None:
        super().__init__()
        self._records: dict[int, dict[str, float]] = {}

    def _record(self, trainer: pl.Trainer, key: str) -> None:
        value = trainer.callback_metrics.get(key)
        if value is not None:
            epoch = trainer.current_epoch
            record = self._records.setdefault(epoch, {"epoch": epoch})
            record[key] = float(value)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._record(trainer, "train_loss")

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not trainer.sanity_checking:
            self._record(trainer, "val_loss")

    def to_frame(self) -> pd.DataFrame:
        """Return the recorded losses as a data frame with one row per epoch."""
        return pd.DataFrame(list(self._records.values()))
```

- [ ] **Step 4: Run test and commit**

Run: `pytest openxps/tests/test_regression.py -p no:cacheprovider -k recorder`
Expected: PASS

```bash
git add openxps/regression.py openxps/tests/test_regression.py
git commit -m "Record the learning curve with a callback keyed by epoch"
```

### Task 7.4: Rewrite `fit` (H2, H3, H9, M13, L10, L12) and the accessors (M4, M17, M19)

- [ ] **Step 1: Write the failing tests**

Append to `openxps/tests/test_regression.py`, inside `class TestForceMatchingRegressor`:

```python
    def test_fit_with_validation_restores_best_checkpoint(self):
        """H2: works on torch >= 2.6 / lightning >= 2.6 without unpickling DVs."""
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(dvs, num_kernels=5, num_epochs=3, validation_fraction=0.2)
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 2)) * 0.5
        forces = -rng.standard_normal((100, 2)) * 0.1
        regressor.fit(positions, forces, seed=1)
        curve = regressor.get_learning_curve()
        assert list(curve.epoch) == [0, 1, 2]
        assert {"train_loss", "val_loss"} <= set(curve.columns)
        assert np.all(np.isfinite(regressor.predict(positions)))

    def test_devices_default_to_one(self):
        """H9: never launch DDP implicitly."""
        regressor = ForceMatchingRegressor(create_test_dynamical_variables(), num_kernels=5)
        assert regressor._devices == 1

    def test_fit_rejects_mismatched_shapes(self):
        """M13"""
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(dvs, num_kernels=5, num_epochs=1)
        with pytest.raises(ValueError, match=r"positions must have shape \(N, 2\)"):
            regressor.fit(np.zeros((10, 3)), np.zeros((10, 3)))
        with pytest.raises(ValueError, match="forces must have the same shape"):
            regressor.fit(np.zeros((10, 2)), np.zeros(10))

    def test_fit_twice_resets_model_and_curve(self):
        """M17 and Review Focus 4"""
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(dvs, num_kernels=5, num_epochs=2, validation_fraction=0.0)
        rng = np.random.default_rng(0)
        positions = rng.standard_normal((40, 2)) * 0.5
        forces = -rng.standard_normal((40, 2)) * 0.1
        with pytest.raises(RuntimeError, match="not been fitted"):
            regressor.get_learning_curve()
        regressor.fit(positions, forces, seed=1)
        first_centers, _, _ = regressor.get_parameters()
        first_curve = regressor.get_learning_curve()
        regressor._num_epochs = 1
        regressor.fit(positions, forces, seed=2)
        second_centers, _, _ = regressor.get_parameters()
        assert len(regressor.get_learning_curve()) == 1
        assert len(first_curve) == 2  # the returned frame was a copy
        assert not np.allclose(first_centers, second_centers)

    def test_get_parameters_does_not_change_predictions(self):
        """M4"""
        mass = 3 * mmunit.dalton
        dv = xps.DynamicalVariable("r", mmunit.nanometer, mass, xps.ReflectiveBounds(0, 1, mmunit.nanometer))
        regressor = ForceMatchingRegressor([dv], num_kernels=20, num_epochs=1, validation_fraction=0.0)
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 1, (50, 1))
        forces = -positions
        regressor.fit(positions, forces, seed=1)
        before = regressor.predict(positions)
        centers, _, _ = regressor.get_parameters()
        assert np.all((centers >= 0) & (centers <= 1))
        np.testing.assert_allclose(regressor.predict(positions), before)

    def test_fit_does_not_touch_global_numpy_rng(self):
        """L10"""
        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(dvs, num_kernels=5, num_epochs=1, validation_fraction=0.0)
        np.random.seed(123)
        expected = np.random.rand()
        np.random.seed(123)
        regressor.fit(np.zeros((20, 2)), np.zeros((20, 2)), seed=None)
        assert np.random.rand() == expected

    def test_fit_emits_no_checkpoint_directory_warning(self):
        """L12"""
        import warnings

        dvs = create_test_dynamical_variables()
        regressor = ForceMatchingRegressor(dvs, num_kernels=5, num_epochs=1, validation_fraction=0.2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            regressor.fit(np.zeros((20, 2)), np.zeros((20, 2)), seed=1)
        assert not [w for w in caught if "exists and is not empty" in str(w.message)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest openxps/tests/test_regression.py -p no:cacheprovider -k "restores_best or default_to_one or mismatched or twice_resets or change_predictions or global_numpy or directory_warning"`
Expected: FAIL (UnpicklingError or FileNotFoundError on multi-GPU hosts; `AttributeError: _devices`; no ValueError; `None` curve; changed predictions; global RNG state consumed; warning emitted).

- [ ] **Step 3: Implement**

In `ForceMatchingRegressor.__init__`, add the keyword `devices: int | str = 1` after `accelerator` (keep `# noqa: PLR0913`), store `self._devices = devices`, and document it:

```
    devices
        The number of devices (or ``"auto"``) handed to the Lightning trainer. The
        default of ``1`` avoids launching distributed training on multi-GPU hosts.
```

Add the unit note to the class docstring, right after the first paragraph of `Parameters` → `dynamical_variables`:

```
    dynamical_variables
        The dynamical variables to be used in the potential. They are converted to
        the MD unit system, so ``positions`` and ``forces`` passed to :meth:`fit` and
        :meth:`predict` must be expressed in MD units (nanometers, radians, or
        dimensionless, and kJ/mol per such unit for forces).
```

Change `_as_numpy` to return an independent array:

```python
    @staticmethod
    def _as_numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().clone().numpy()
```

Add the shape check and rewrite `fit`:

```python
    def _validate_shapes(self, X: torch.Tensor, G: torch.Tensor) -> None:
        d = len(self._dynamical_variables)
        if X.ndim != 2 or X.shape[1] != d:
            raise ValueError(
                f"positions must have shape (N, {d}), got {tuple(X.shape)}"
            )
        if G.shape != X.shape:
            raise ValueError(
                f"forces must have the same shape as positions {tuple(X.shape)}, "
                f"got {tuple(G.shape)}"
            )

    def fit(
        self, positions: ArrayLike, forces: ArrayLike, *, seed: t.Optional[int] = None
    ) -> None:
        """Fit the potential regressor to the given positions and forces.

        Parameters
        ----------
        positions
            An array of shape (N, d) with the positions in MD units.
        forces
            An array of shape (N, d) with the forces in MD units.

        Keyword Arguments
        -----------------
        seed
            The seed used to split the data and initialize the kernels. If None, a
            random seed is used. The global NumPy random state is left untouched;
            the global PyTorch seed is set.
        """
        rng = np.random.default_rng(seed)
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))

        X = torch.as_tensor(np.asarray(positions, dtype=np.float32))
        G = -torch.as_tensor(np.asarray(forces, dtype=np.float32))
        self._validate_shapes(X, G)
        self._metrics = None
        train_dl, val_dl = self._create_dataloaders(X, G)

        self._model = GradMatch(self._dynamical_variables, self._M, self._lr, self._wd)
        recorder = _LearningCurveRecorder()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer_kwargs = {
                "max_epochs": self._num_epochs,
                "accelerator": self._accelerator,
                "devices": self._devices,
                "logger": False,
                "enable_progress_bar": True,
                "enable_model_summary": True,
                "callbacks": [recorder],
            }

            if val_dl is None:
                trainer_kwargs["enable_checkpointing"] = False
                trainer = pl.Trainer(**trainer_kwargs)
                trainer.fit(self._model, train_dl)
            else:
                checkpoint, early_stopping = self._create_callbacks(
                    os.path.join(tmp_dir, "checkpoints")
                )
                trainer_kwargs["callbacks"] += [checkpoint, early_stopping]
                trainer = pl.Trainer(**trainer_kwargs)
                trainer.fit(self._model, train_dl, val_dl)
                if checkpoint.best_model_path:
                    state = torch.load(
                        checkpoint.best_model_path, map_location="cpu", weights_only=True
                    )
                    self._model.load_state_dict(state["state_dict"])

        self._metrics = recorder.to_frame()
```

Rewrite `get_learning_curve`:

```python
    def get_learning_curve(self) -> pd.DataFrame:
        """Get the learning curve of the last fit.

        Returns
        -------
        pd.DataFrame
            A copy of the per-epoch losses, with columns ``epoch``, ``train_loss``
            and, when a validation set was used, ``val_loss`` (``NaN`` for epochs
            whose validation loss was not finite).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self._metrics is None:
            raise RuntimeError("The model has not been fitted yet.")
        return self._metrics.copy()
```

Remove the now-unused `pd.read_csv` path and the `CSVLogger` usage (and `import os` stays, it is used for the checkpoint path).

- [ ] **Step 4: Run the regression tests**

Run: `pytest openxps/tests/test_regression.py openxps/regression.py -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Changelog, full suite, PR**

Append under `## Unreleased` → `### Changed` in `CHANGELOG.md`:

```markdown
- `ForceMatchingRegressor`: the `initial_bandwidth` keyword was removed (the initial
  bandwidth is a quarter of each variable's range); `devices` was added and defaults
  to 1; `get_learning_curve()` returns a copy and raises before the first fit;
  positions and forces must be given in MD units.
```

and under `### Fixed`:

```markdown
- `ForceMatchingRegressor.fit` works with torch >= 2.6 / lightning >= 2.6, tolerates
  non-finite validation losses, validates input shapes, no longer reseeds NumPy's
  global generator, and `get_parameters()` no longer alters the fitted model.
```

Run: `pytest -p no:cacheprovider && ruff check openxps && ruff format --check openxps`

```bash
git add openxps/regression.py openxps/tests/test_regression.py CHANGELOG.md
git commit -m "Make ForceMatchingRegressor.fit robust and its accessors side-effect free"
git push -u origin fix/regression
```

PR title: `Fix ForceMatchingRegressor training and accessors`. Body: closes H2, H3, H9, M4, M5, M13, M17, M19, L1, L10, L12, L13 and the regression part of M18.

---

## Self-Review

**Spec coverage.** Every finding in `CODE_REVIEW.md` maps to a task: H1/M21/M22/L18–L22 → PR 1; H8/M6/M7/M8/M9/L8/L9/L14(DV) → PR 2; H5/H6/H7/M1/M12/M14/M20/L7/L11/L16 → PR 3; H4/M2/M10/M11/M15/L2–L5/L15 → PR 4; M24 → PR 5; M3/M16/L6/L14(writer)/L17 → PR 6; H2/H3/H9/M4/M5/M13/M17/M19/L1/L10/L12/L13 → PR 7; M18 is split between the changelog entries of PRs 2 and 7; M23's test gaps are covered by Task 4.7 (thermostats, setters, step size, pickling), Task 3.2 (IPC-in-sum), Task 2.5 (non-periodic harmonic coupling), Task 2.4 (`distanceTo` with non-radian units), Task 6.1 (writer temperature), Task 6.3 (metadynamics), and Task 5.1 (reference integration of the GGMT sub-dynamics). The one M23 item not covered is a `SplitIntegrator`/`LockstepIntegrator` comparison against a reference integration of a coupled toy system; it is deferred because it needs a new toy-system fixture and does not reproduce any finding.

**Placeholder scan.** No TBD/TODO; every code step shows the code; the only "adjust if it differs" instructions (doctest step numbering in Task 5.1, NHL tolerance in Task 4.7) state the exact expected structure and the exact fallback.

**Type consistency.** `Function.createCollectiveVariable(variable_indices)` (Task 3.1) is what `InnerProductCoupling._createFlippedForce(dv_indices)` (Task 3.2) calls; `Coupling.addToExtensionSystem(system, dv_indices=None)` is what `CouplingSum.addToExtensionSystem` forwards; `ExtendedSpaceContext.synchronize()` (Task 3.3) is what `ExtendedSpaceSimulation.minimizeEnergy`/`loadState` (Task 3.4) call; `_LearningCurveRecorder.to_frame()` (Task 7.3) is what `fit` (Task 7.4) stores; `IntegratorMixin.setStepSize` (Task 4.1) is what `ExtendedSpaceIntegrator.setStepSize` (Task 4.2) relies on; `Bounds.kind` (Task 2.1) is what `__eq__` and `__hash__` use and what `_distanceToDV` (Task 2.4) relies on through `!=`.

**Review Focus.** Item 1 → Task 4.6 (`test_csvr_at_rest_does_not_produce_nan`); item 2 → Task 2.4 (`test_harmonic_coupling_nonperiodic_angstrom_dv_energy`, run in Task 2.5); item 3 → Task 3.3 (`test_set_parameter_updates_extension_context`, second half); item 4 → Task 7.4 (`test_fit_twice_resets_model_and_curve`); item 5 → Task 3.2 (`test_coupling_sum_does_not_mutate_children`).
