# OpenXPS Code Review — Findings and Fix Tracker

Review date: 2026-10-10 · Base commit: `995b794` (Improve Regression Module, #40)

Scope: the whole `openxps/` package plus packaging (`pyproject.toml`, `setup.cfg`,
`MANIFEST.in`, `devtools/`) and CI (`.github/workflows/`). The first pass covered only
commit `995b794`; the second pass covered the full package in three parts (integrators;
couplings/context/system/simulation; bounds/DVs/metadynamics/utils/writer/regression/packaging).

**Repro status** — ✅ reproduced with a script · 📖 found by reading the code.
Reproductions ran in scratch environments owned by this account: OpenMM 8.6.1 + cvpack 0.12.2
(torch stubbed) for integrators/couplings; torch 2.6.0 + Lightning 2.6.5 (the `chemeleon` env
plus an OpenMM overlay) for regression; a fresh Python 3.9.23 + OpenMM 8.3.1 env for M21.
Machine: 2× RTX 4090. Nothing in the repository was modified apart from this file.

Severity: **High** = crash or silently wrong physics in normal use · **Medium** = wrong
results in specific setups, broken API, or packaging/CI that misleads · **Low** = validation,
docs, cleanup.

---

## High

- [ ] **H1. `import openxps` fails on a clean install — undeclared runtime dependencies** ✅
  `openxps/regression.py:15`, `pyproject.toml`, `devtools/conda-recipes/anaconda/meta.yaml`
  `regression.py` (imported from `__init__.py`) needs `pandas`, which is not declared anywhere.
  `typing_extensions` (`dynamical_variable.py:15`) is also undeclared; torch only pulls it in transitively.
  The conda recipe's run requirements also omit `cvpack`, `pyyaml`, `typing-extensions` and `pandas`, so its
  `test: imports: openxps` step fails. The recipe's `summary` was copied from cvpack.
  *Fix:* declare all four in `pyproject.toml` and `meta.yaml`, and correct the summary.

- [ ] **H2. Every `fit()` with a validation set crashes on recent torch/Lightning** ✅
  `openxps/regression.py:340`
  `GradMatch.load_from_checkpoint` unpickles hparams that contain `DynamicalVariable`. Lightning ≥ 2.6
  with torch ≥ 2.6 defaults to `weights_only=True`, which raises `UnpicklingError: Unsupported global`.
  *Fix:* pass `weights_only=False`, or reload only the `state_dict` into a freshly built `GradMatch`.

- [ ] **H3. A NaN `val_loss` makes `fit()` raise after training finishes** ✅
  `openxps/regression.py:349`
  The train and val rows are filtered with `.notna()` separately and then paired by position, so a NaN
  epoch gives them different lengths → `ValueError: Length of values ... does not match`.
  *Fix:* group by `epoch`, or collect metrics in a callback keyed by epoch.

- [ ] **H4. All generated property setters crash** ✅
  `openxps/integrators/utils.py:147`
  `set_and_update` calls `set_value(value)` without passing `self`. So `setTemperature`,
  `setFrictionCoefficient` and `setTimeConstant` raise `TypeError` on SymmetricLangevin, CSVR, MassiveGGMT
  and RegulatedNHL, even though the docstrings tell users to call them.
  *Fix:* `set_value(self, value)`, plus tests that check the global variables are updated.

- [ ] **H5. `InnerProductCoupling` functions read the wrong extension particle inside a `CouplingSum`** ✅
  `openxps/couplings/inner_product_coupling.py:238-244`, `openxps/utils.py:246`
  `fn.createCollectiveVariable(all_dvs)` maps variables to particles by their position in the coupling's
  *local* DV list, ignoring `self._dv_indices`, which `CouplingSum` rewrites to the global order.
  *Failure:* `CVCoupling([a]) + InnerProductCoupling([f], [lam], functions={"scaling": "lam^2"})` with
  a=0.3, lam=0.7 gives scaling = 0.09 instead of 0.49, and the dU/dλ force lands on particle `a`.
  *Fix:* pass the index map, e.g. `groups=[[self._dv_indices[v] for v in variables]]`. Add a test with the
  IPC's DVs not in first position.

- [ ] **H6. `getDynamicalVariableValues()` crashes for DVs that appear only inside `functions=`** ✅
  `openxps/context.py:208`
  It reads `getParameter(dv.name)` from the *physical* context. The class's own docstring example
  (`functions={"scaling": ...lambda...}`) raises `OpenMMException: invalid parameter name: lambda`.
  *Fix:* read the extension-context positions and wrap them, as `getDynamicalVariableVelocities` already does.

- [ ] **H7. `ExtendedSpaceContext.setParameter` does not update the extension context** ✅
  `openxps/context.py:144-165`
  Coupling parameters such as `kappa` exist in both contexts, but only the physical copy is changed.
  *Failure:* after `setParameter("k", 1000)` the physical force is 300 while the force on the DV stays at -30.
  Momentum is no longer conserved and the DV samples the wrong distribution.
  *Fix:* also set the parameter on `_extension_context` when it exists there. Strengthen
  `test_set_parameter_for_regular_param`.

- [ ] **H8. The harmonic coupling never wraps periodic DVs that are not in radians** ✅
  `openxps/dynamical_variable.py:191-192` (`_distanceToCV`), reached via `couplings/harmonic_coupling.py:65`
  The minimum-image period is written in the DV's own unit (360) but evaluated in MD units (radians).
  *Failure:* a degree DV with phi=-179°, phi0=+179° gives an energy of 19520 kJ/mol instead of 0.609.
  *Fix:* build the expression from `self.in_md_units()`, or convert the period. Add a test with a degree DV
  across the ±180° seam.

- [ ] **H9. `fit()` crashes and re-runs the user's script on multi-GPU machines** ✅
  `openxps/regression.py` (`trainer_kwargs`: `accelerator="auto"`, `devices="auto"`)
  With more than one visible GPU, Lightning switches to DDP and relaunches the whole calling script in a
  second process. Each process makes its own `TemporaryDirectory`, so rank 1 fails with
  `FileNotFoundError: .../best-epoch=01-....ckpt`, since that path exists only in rank 0's directory.
  Every line of the user's script before `fit()` also runs twice. This still happens after H2 is fixed.
  *Failure:* any `fit()` with validation on this 2-GPU machine (`CUDA_VISIBLE_DEVICES=0` avoids it).
  *Fix:* default to `devices=1` and expose `devices` as an option. If DDP is wanted, use a shared,
  rank-0-created checkpoint directory and reload the model on every rank.

## Medium

### Physics / correctness

- [ ] **M1. The two contexts go out of sync after state changes that bypass `setPositions`** ✅
  `openxps/context.py` (only `setPositions` is overridden, lines 167-177), `openxps/simulation.py:112-113`
  `setState`, `reinitialize`, `setPeriodicBoxVectors`, `LocalEnergyMinimizer` / `Simulation.minimizeEnergy`,
  `Simulation(state=...)` and `loadState` all leave stale CV values or parameters in the other context.
  *Failure:* after `minimizeEnergy`, the extension `cvx` still holds the pre-minimization value.
  `saveState`/`loadState` also drop DV positions and velocities.
  *Fix:* override these methods to resync, or sync lazily at the start of `step`.

- [ ] **M2. Changing the step size leaves dt-dependent thermostat coefficients stale** ✅
  `openxps/integrator.py:184-191, 406-408`; `regulated_nhl.py`, `massive_ggmt.py` (no `setStepSize` override)
  `mmswig.Integrator_setStepSize` bypasses the Python overrides, so `a = exp(-γ·dt)` keeps its old value.
  *Failure:* `SplitIntegrator(SymmetricLangevin(γ=10/ps, 4 fs))` runs the extension with dt=2 fs but `a`
  computed for 4 fs, i.e. double the friction. The sampled distribution is still correct, but the dynamics are wrong.
  *Fix:* add `setStepSize` → `_update_global_variables` in `IntegratorMixin`, and call the Python method
  instead of the raw SWIG one.

- [ ] **M3. ExtensionWriter reports inflated extension temperature and kinetic energy** ✅
  `openxps/extension_writer.py:136-141, 235, 244`
  It assumes 1 degree of freedom per DV, but `getKineticEnergy` also includes the y/z velocities that
  Langevin thermostats keep thermalized. *Failure:* a 300 K run reports about 833 K.
  *Fix:* compute KE from the x-velocities only, with masses cached in `initialize`.

- [ ] **M4. `get_parameters()` modifies the fitted model** ✅
  `openxps/regression.py:387`
  It wraps centers in place on a numpy view that shares memory with the CPU parameter. With
  `ReflectiveBounds`, `predict` output shifted by up to 0.39 afterwards.
  *Fix:* `.clone()` before converting to numpy.

- [ ] **M5. RBF centers are initialized from a normal distribution instead of a uniform one** ✅
  `openxps/regression.py:60`
  `torch.randn(M)*length + lower` → only 34% of centers start in [lower, upper], and about 27% are
  effectively dead kernels given the new σ = length/4. *Fix:* `torch.rand`.

- [ ] **M6. `NoBounds` skips the unit check and conversion in `DynamicalVariable.__post_init__`** ✅
  `openxps/dynamical_variable.py:80`
  `NoBounds` now carries values that the RBF uses, so `NoBounds(-180, 180, degree)` on a DV in nm is
  accepted silently. Some tests pair `NoBounds(0, 1, dimensionless)` with nm/rad DVs, and the docstring is stale.
  *Fix:* remove the early return, then fix the tests and docstring.

- [ ] **M7. `HarmonicCoupling` rejects every valid non-periodic DV/CV pair** ✅
  `openxps/couplings/harmonic_coupling.py:79`
  `cv.getPeriodicBounds() != dv.bounds.asQuantity()` is always True when the DV is non-periodic.
  Confirmed for `cvpack.Distance` with both `NoBounds(0, 1, nm)` and `ReflectiveBounds(0, 1, nm)`.
  *Fix:* compare periodicity first, then compare bounds only when both are periodic. Add a test.

- [ ] **M8. `Bounds.__eq__` ignores the subclass and is asymmetric** ✅
  `openxps/bounds/base.py:59-64`
  `NoBounds(-π, π, rad) == CircularBounds()` → True, so DV lookups in metadynamics can match a DV with
  different periodicity. Each side is converted to the *other's* unit and compared as exact floats:
  `ReflectiveBounds(12.11, 57.66, Å) == ReflectiveBounds(1.211, 5.766, nm)` is True one way and False the other.
  *Fix:* compare `type(self) is type(other)`, compare both sides in MD units with `math.isclose`, and keep
  `__hash__` consistent.

- [ ] **M9. `_distanceToDV` uses `self`'s period without checking that it matches `other`'s** ✅
  `openxps/dynamical_variable.py:199-201`
  For `PeriodicBounds(-1, 1, rad)` vs `CircularBounds()`, `A.distanceTo(B)` uses period 2 while
  `B.distanceTo(A)` uses 2π. Degree vs radian mixes are wrong in the same way.
  *Fix:* require equal bounds and compatible units, otherwise raise.

- [ ] **M10. Pickling an `ExtendedSpaceIntegrator` loses the Python integrator subclasses** ✅
  `openxps/integrator.py:87-92`
  `XmlSerializer.deserialize` returns a plain `CustomIntegrator`, so CSVR's `step()` override disappears
  (`sumRsq` stays frozen at 0, giving the wrong ensemble) and the `registerWithSystem` checks are skipped.
  *Fix:* pickle the sub-integrators with Python pickle so their class survives. Add a test that checks the
  type after a round trip.

- [ ] **M11. CSVR produces NaN when the kinetic energy is zero** ✅
  `openxps/integrators/csvr.py:163-169`
  `BC = (1-A)*kT/mvv` → ∞ → NaN velocities. This happens on the first step whenever DVs start at rest
  under zero force, which is common.
  *Fix:* guard with `select(mvv, ...)` or handle mvv = 0 separately.

- [ ] **M12. `CouplingSum` overwrites its child couplings' DV indices** ✅
  `openxps/couplings/base.py:103-116, 382-384`
  A coupling shared by two sums, or reused after being summed, silently gets the wrong indices
  (`s1 = cb+ca; s2 = cb+cc` → physical `b` reads `a`'s value) or raises `Illegal particle index`.
  *Fix:* copy the children, or pass the index map into `addToExtensionSystem`.

- [ ] **M13. `fit()` does not validate input shapes** ✅
  `openxps/regression.py:145-147, 312-313`
  For d=1, forces of shape (N,) against positions of shape (N,1) broadcast to (B,B), so training fits
  the wrong objective silently.
  *Fix:* check `G.shape == X.shape`, `X.ndim == 2` and `X.shape[1] == len(dvs)`.

### API

- [ ] **M14. `setDynamicalVariableValues`/`Velocities` accept the wrong number of values** ✅
  `openxps/context.py:189-196`
  `zip` silently drops extra values, and too few give an opaque OpenMM error. *Fix:* check the length.

- [ ] **M15. `copy`/`deepcopy` of an integrator built with `assume_force_first`/`assume_symmetric` raises** ✅
  `openxps/integrator.py:72-77`
  The flag isn't passed through `__copy__`. *Fix:* store the flag and pass it on.

- [ ] **M16. New header getters return `[]` silently** ✅
  `openxps/extension_writer.py:179`
  `getCollectiveVariableHeaders()`/`getEffectiveMassHeaders()` return `[]` unless the matching output flag
  was set, and all getters return `[]` before `initialize()`.
  *Fix:* build `_meta_cvs` regardless of the flags, or raise before `initialize()`.

- [ ] **M17. `get_learning_curve()` can return a stale curve or `None`** ✅
  `openxps/regression.py:345`
  `_metrics` is never reset in `fit()`, and the method returns the internal DataFrame by reference.
  *Fix:* reset it at the start of `fit()`, return a copy, and raise before the first fit.

- [ ] **M18. Breaking API changes with no deprecation path** ✅
  `openxps/regression.py:226`, `bounds/no_bounds.py`, `dynamical_variable.py`
  `initial_bandwidth=` was removed, `NoBounds()` now requires arguments, and `DynamicalVariable` now
  requires `bounds`. *Fix:* deprecation shims, or a changelog entry and version bump.

- [ ] **M19. `ForceMatchingRegressor` silently expects MD units** ✅
  `openxps/regression.py:235-237`
  DVs are converted to MD units internally, so data given in a degree DV's own unit produces a
  meaningless fit. *Fix:* document the requirement, or convert the inputs.

### Efficiency

- [ ] **M20. `InnerProductCoupling` evaluates the full physical force field an extra time every step** ✅
  `openxps/couplings/inner_product_coupling.py:257`
  `getState(ParameterDerivatives)` runs all force groups. Measured on 8000 LJ particles (CPU): one
  `step(1)` takes 2.1 ms, the all-groups `getState` takes 1.45 ms (about +70% per step), and a
  coupling-group-only `getState` takes 0.52 ms.
  *Fix:* put the coupling forces in their own force group and pass `groups=1<<g`.

### Packaging / CI

- [ ] **M21. The claim of Python 3.9 support is false** ✅
  `pyproject.toml:21-29` vs `dynamical_variable.py:204`, `regression.py:256`, `couplings/base.py:93`
  PEP 604 `X | Y` annotations are evaluated at import time, so `import openxps` fails on 3.9.
  Confirmed on Python 3.9.23: `couplings/base.py:93 ... TypeError: unsupported operand type(s) for |`.
  *Fix:* require Python ≥ 3.10, or add `from __future__ import annotations`.

- [ ] **M22. The Linux CI matrix never installs the OpenMM version it names** 📖
  `.github/workflows/Linux.yaml`
  `create-args` only passes `python=...`, so the jobs labeled 7.7/8.1/8.2/8.3 all test whatever OpenMM
  version the solver picks. *Fix:* add `openmm=${{ matrix.openmm-version }}`, as MacOS/Windows already do.

### Test coverage

- [ ] **M23. Major untested areas** 📖
  - CSVR, MassiveGGMT and RegulatedNHL have no unit tests (doctests only), and no statistical test
    checks temperature or KE variance for any thermostat.
  - Property setters (H4), `setStepSize` coefficient updates (M2) and the class type after pickling (M10)
    are untested.
  - Nothing compares `SplitIntegrator`/`LockstepIntegrator` against a reference integration of a toy system.
  - `metadynamics.py` has no unit tests: degree DVs, `ReflectiveBounds`, error paths and the YAML round
    trip are all untested.
  - Nothing tests `distanceTo` with non-radian units (H8), whether ExtensionWriter's temperature value is
    physically right (M3), or that `leptonExpression` agrees with `wrap`.
  - The IPC-in-sum case (H5) and non-periodic `HarmonicCoupling` (M7) are untested.

## Low

### Validation

- [ ] **L1.** ✅ `openxps/regression.py:141` — AdamW weight decay also applies to `logsig` and the centers,
  pulling them toward the absolute values 1 and 0 regardless of the variable's scale. Restrict it to `w`.
- [ ] **L2.** ✅ `openxps/integrator.py:266, 406` — the default extension integrator is a `deepcopy` of the
  physical one, so the two share an RNG seed (and CSVR's numpy `_rng` state) and draw correlated noise.
  Reseed it after copying.
- [ ] **L3.** ✅ `openxps/integrators/csvr.py:199-210` — `setRandomNumberSeed(0)` is deterministic, whereas
  OpenMM uses 0 to mean "pick a random seed". `getRandomNumberSeed()` returns a derived value, and the numpy
  RNG state is not in checkpoints.
- [ ] **L4.** ✅ `openxps/integrators/csvr.py:172-176, 212-224` — `step()` before `registerWithSystem` gives
  an opaque `NoneType - int` error.
- [ ] **L5.** ✅ `openxps/integrators/massive_ggmt.py:19` — the time constant is declared in `1/ps`; it should
  be `ps`.
- [ ] **L6.** ✅ `openxps/metadynamics.py:69-77` — `sigma` is not checked to be > 0, so a later OpenMM error
  is obscure.
- [ ] **L7.** ✅ `openxps/utils.py:171-186` — the `missing` check in `Function._parseDependencies` can never
  fire. Unknown names such as typos, and Lepton intermediate definitions (`; u=...`), become "variables" and
  later cause a cryptic `'u' is not in list`.
- [ ] **L8.** ✅ `openxps/bounds/base.py:43-47` — the type check rejects numpy scalars but accepts bool, NaN
  and inf.
- [ ] **L9.** `openxps/dynamical_variable.py:24` — `DynamicalVariable` is unhashable (its `Quantity` field),
  so DVs can't be set members or dict keys.
- [ ] **L10.** `openxps/regression.py:309-310` — `np.random.seed(seed)` overwrites the caller's global RNG
  state, even when `seed=None`. Use a local Generator.

### API

- [ ] **L11.** `openxps/simulation.py:95, 112-113` — `state=` takes a `State` object, whereas OpenMM's
  `Simulation` takes an XML filename. Accept both.

### Cleanup / efficiency

- [ ] **L12.** ✅ `openxps/regression.py:335` — the checkpoint directory is the same as the CSVLogger
  directory, so Lightning warns on every fit. Use a `checkpoints/` subdirectory.
- [ ] **L13.** ✅ `openxps/regression.py:70` — `_length_over_pi` and `_periodic` are `nn.Parameter`s that never
  get gradients. Make them buffers.
- [ ] **L14.** `openxps/extension_writer.py:204, 262` — the effective-mass unit formula is duplicated and
  should be cached in `initialize()`. The `# noqa: PLR0912` is stale, and the `bounds is None` branch in
  `DynamicalVariable.createCollectiveVariable` is unreachable.
- [ ] **L15.** ✅ `openxps/integrators/utils.py:34-38` — `__repr__` adds a blank line when there are no
  per-DOF variables.

### Docs

- [ ] **L16.** Docstrings that don't match the code:
  - `context.py:148-151` says DV parameters are wrapped; they actually raise.
  - `context.py:217-220` says the argument is "a dictionary"; it is a sequence.
  - `couplings/base.py:201-204` says it raises `NotImplementedError`; it is implemented.
  - `couplings/base.py:78` has a return annotation that doesn't match.
- [ ] **L17.** `openxps/metadynamics.py` — the docs don't say that the bias must be created before the
  context.

### Packaging / CI

- [ ] **L18.** Linux CI uploads a `coverage.xml` that is never generated (no `--cov-report=xml`), using the
  deprecated `codecov-action@v1`.
- [ ] **L19.** The `test` extra lacks `pytest-cov`, but `addopts` uses `--cov`.
- [ ] **L20.** `package-data` lists `py.typed` (missing) and `data/*.csv` (`data/` only has a `.pdb`).
- [ ] **L21.** `conda mambabuild`/boa is deprecated (`Anaconda.yaml`, `deployment_env.yaml`). Switch to
  `conda build` or rattler-build.
- [ ] **L22.** `Docs.yaml` runs the "Install package" step twice.

---

## Checked and found correct

- CSVR scaling formula and chi-square sampling, including N=1 and odd/even N
- Degree-of-freedom counting
- GGMT noise amplitudes
- NHL regulated equipartition term and noise amplitude
- B^n A B^n ordering and context syncing in `SplitIntegrator.step`
- Lockstep force-first assumption
- InnerProductCoupling flipped-force energy and sign
- DV wrapping in the Lepton expressions
- Unit stripping in system/context
- How checkpoints are split
- Metadynamics bias deposition and grid for radian and degree DVs

Not checked: M22 (CI matrix) is confirmed in the workflow YAML but was not run on CI.

## GGMT stabilization (`7d770b1`): analysis and tests

**Theory.** Take the sub-flow that the `v2` kicks and the cubic velocity step approximate:
dK/dt = −(2/3)·v2·K² and dv2/dt = (K²/3 − kT²)/Q2, with K = m v². It conserves

  I = v2² + (K + e²/K)/Q2, with e = √3·kT.

`v2lb = −sqrt(v2² + (K + e²/K − 2e)·invQ2)` is the most negative `v2` reachable on that level set
(where K = e). Reflecting at `v2lb` therefore makes the discrete step respect a bound that the exact
flow obeys. The idea is sound, with three caveats:

1. **Not volume-preserving.** `v2lb` depends on `v2`, so the reflected branch has Jacobian
   2·v2/v2lb − 1, which is not ±1 in general. GGMT's invariant measure assumes a known phase-space
   compressibility, so every reflection adds a small sampling bias. It is rare at small dt/τ.
2. **Stale bound.** The second reflection in each `_add_v2_boost` reuses the `v2lb` computed before
   the cubic step changed K. That matches the exact flow, which conserves I, but not the discrete one.
3. **Does not guard the actual failure mode.** The bound does not depend on dt, so it cannot keep
   `1 + (2/3)·f·dt·K·v2` positive inside the cubic step's `sqrt`.

**Sampling test.** 2000 3D harmonic oscillators, ω = 10/ps, τ = 0.1 ps; exact values are
⟨mv²⟩/kT = 1, ⟨(mv²)²⟩/kT² = 3 and ⟨kx²⟩/kT = 1.

| dt = 5 fs     | ⟨mv²⟩/kT | ⟨(mv²)²⟩/kT² | ⟨kx²⟩/kT |
|---------------|----------|--------------|----------|
| new, determ.  | 0.9993   | 3.0021       | 0.9905   |
| old, determ.  | NaN      | —            | —        |
| new, stoch.   | 0.9996   | 3.0070       | 0.9990   |
| old, stoch.   | 0.9977   | 2.9895       | 0.9999   |

No bias is visible at this dt/τ (no error bars computed).

**Stability scan** (Y = survives 20000 steps):

| τ / dt     | 2 fs    | 5 fs        | 10 fs   | 20 fs       |
|------------|---------|-------------|---------|-------------|
| 0.1 ps     | all Y   | old-det NaN | all NaN | all NaN     |
| 0.5 ps     | all Y   | all Y       | all Y   | old-det NaN |
| 1.0 ps     | all Y   | all Y       | all Y   | all Y       |

The stabilization extends the deterministic range, but all variants blow up once dt/τ ≳ 0.1, even
though Verlet alone is very stable at those dt (ω·dt = 0.2). In the diagnosed blow-up (dt = 20 fs,
τ = 0.1 ps) the culprit DOF had K = 5.5 kT, v2·kT·τ = −2.2 and a cubic-step `sqrt` argument of
0.20 before its last step, i.e. caveat 3.

**Resolution: the Coulomb splitting.** In the variables z = √3·kT/(m v²) and w = (2/√3)·kT·v2, the
(K, v2) sub-flow is Hamiltonian with H = w²/2 + c(z + 1/z), c = √3/(4τ²). Splitting
H = [w²/2 + c/z] + c·z, the first part is a 1D repulsive Coulomb problem, solvable exactly via a
hyperbolic Kepler equation (θ + ½ sinh 2θ = M, 5 Newton iterations), and never reaches z = 0; the
second is a constant kick. The composition is symplectic in (z, w), reversible and z > 0 for any dt.
Full derivation, standalone OpenMM implementation and benchmark: `github.com/craabreu/ggmt_integrator`
(`coulomb.tex`, `scripts/ggmt_integrators.py`).

Benchmark (6000 DOF, τ = 0.1 ps, 200 ps, 95% CI from 20 blocks), ⟨(mv²)²⟩/kT² (exact 3):

| dt        | Trotter   | walls at turning points | Coulomb splitting |
|-----------|-----------|-------------------------|-------------------|
| 5 fs det  | 3.004(2)  | 2.945(2)                | 3.001(3)          |
| 10 fs det | NaN       | 2.850(6)                | 3.006(6)          |
| 20 fs det | NaN       | 2.682(6)                | 2.982(7)          |
| 10 fs sto | NaN       | 2.908(6)                | 3.002(7)          |
| 20 fs sto | NaN       | 2.849(8)                | 3.006(9)          |

The Coulomb splitting also survived dt = 50 fs at τ = 0.1 and 0.5 ps. Cost: 3.4–3.5× the Trotter
step when forces are negligible (about a dozen extra per-DOF ops per half step).

- [ ] **M24. Replace the `v2lb` reflection in `MassiveGGMTIntegrator` with the Coulomb splitting** ✅
  `openxps/integrators/massive_ggmt.py:184-200` (`_add_v2_boost`)
  *Fix:* port `_add_coulomb_step` from `ggmt_integrator/scripts/ggmt_integrators.py` (guard
  `m*v*v` with a floor and write asinh as `sign(q)·log(|q| + sqrt(q²+1))` to avoid cancellation).
  Remove the `v2lb` per-DOF variable. Add a sampling regression test on harmonic oscillators that checks
  ⟨(mv²)²⟩/kT² = 3 within tolerance at dt/τ = 0.1, and document that the outer Verlet/scaling factors,
  not the thermostat, now set the time-step limit.

## Suggested fix order

1. **Installation and crashes:** H1, M21, H2, H9, H3, H4.
2. **Silently wrong physics:** H5, H7, H8, M1, M2, M3, M11, M12, then M10.
3. **Regression correctness:** M4, M5, M13, M17, L1.
4. **Bounds and DV semantics:** M6, M7, M8, M9 (these overlap, so fix them together).
5. **Tests:** add tests alongside each fix, then fill in the rest of M23.
6. **Low items**, grouped by file.
