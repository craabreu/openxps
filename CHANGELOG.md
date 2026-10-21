# Changelog

## Unreleased

### Changed
- PyTorch, Lightning and pandas are no longer installed with openxps. Only
  `ForceMatchingRegressor` uses them; install them with `pip install
  openxps[regression]` or from conda-forge. `import openxps` no longer loads them,
  which makes it about 2 seconds faster.
- When the physical integrator of a `SplitIntegrator` takes the substeps, the
  extension context is updated after the second half of a step only in the last step
  of a call to `step`. The update only serves observers, since the next step redoes
  it before the extension system moves again. With an `InnerProductCoupling`, this
  halves the force evaluations spent on its energy derivatives.

### New
- `ExtendedSpaceSystem` warns when a `CustomNonbondedForce` in the physical system uses
  a long-range correction and depends on a parameter the coupling changes at every
  step. OpenMM then recomputes the correction on the CPU at every step, which in a
  1,500-atom system took longer than the step itself.

### Fixed
- `InnerProductCoupling` works with forces that compute parameter derivatives only
  along with the energy, such as `SlicedNonbondedForce` from openmm-nonbonded-slicing,
  which gave NaN forces on the dynamical variables (#68).
- `ForceMatchingRegressor.fit` no longer shows Lightning warnings that don't apply
  to it, about data loader workers, a missing validation set, SLURM, and Lightning's
  use of a deprecated PyTorch API (#70).

## 0.2.0

This release rewrites most of the package, so code written for 0.1.0 needs to be
ported. In 0.1.0, an `ExtendedSpaceContext` wrapped an existing OpenMM context and
took extra degrees of freedom, a coupling meta-collective variable and an optional
biasing potential. Since 0.2.0, an `ExtendedSpaceSystem` combines a physical system
with a coupling, and an `ExtendedSpaceContext` or `ExtendedSpaceSimulation` is built
from that system and an extended-space integrator.

### Upgrading from 0.1.0
- `ExtraDOF` is now `DynamicalVariable`. Its `bounds` argument can no longer be
  `None`; use `NoBounds` for unbounded variables.
- The `Periodic` and `Reflective` bounds are now `PeriodicBounds` and
  `ReflectiveBounds`, and the new `CircularBounds` and `NoBounds` join them in the
  `openxps.bounds` module.
- `MetadynamicsBias` and the `biasing_potential` module are replaced by
  `ExtendedSpaceMetadynamics` and `ExtendedSpaceBiasVariable`.
- Couplings are built with `HarmonicCoupling`, `CollectiveVariableCoupling` or
  `InnerProductCoupling`, and can be summed, instead of being given as a
  meta-collective variable.
- `ExtensionWriter` no longer takes a context. It is added to a CVPack
  `StateDataReporter` of an `ExtendedSpaceSimulation`, and its `potential` and
  `total` columns are gone.

### New in 0.2.0
- `ExtendedSpaceSystem` and `ExtendedSpaceSimulation`.
- `LockstepIntegrator` and `SplitIntegrator` advance the physical and extension
  systems together, the latter with different step sizes.
- `SymmetricVerletIntegrator`, `SymmetricLangevinIntegrator`, `CSVRIntegrator`,
  `MassiveGGMTIntegrator` and `RegulatedNHLIntegrator`.
- `ForceMatchingRegressor` fits free energy surfaces to mean forces with radial
  basis functions.
- `ExtensionWriter` can also report dynamical variables, forces, collective
  variables, effective masses and coupling functions.
- Example notebooks for alanine dipeptide.

The entries below describe changes made during the 0.2.0 development cycle.

### Changed
- Python 3.9 is no longer supported; the minimum version is 3.10.
- OpenMM 8.4 or newer is required.
- `NoBounds` now requires `lower`, `upper` and `unit`, which are typical values of the
  variable used to initialize regression kernels; `DynamicalVariable` requires `bounds`.
- `NoBounds` units are validated and converted like other bounds.
- `Bounds` equality distinguishes periodic, reflective and unbounded kinds.
- `DynamicalVariable.distanceTo` raises for mismatched periods or incompatible units.
- `Function.createCollectiveVariable` takes a mapping from variable names to particle
  indices instead of a list of names.
- `ExtendedSpaceSimulation.saveState` writes the states of both the physical and the
  extension contexts; `loadState` and the `state` argument accept such files as well
  as plain OpenMM state files.
- `MassiveGGMTIntegrator` propagates the fourth-moment thermostat sub-dynamics with a
  singularity-free Coulomb splitting instead of a reflective bound on `v2`, so its
  thermostat step no longer fails with a negative square-root argument at large step
  sizes; the per-DOF variable `v2lb` is gone.
- `MassiveGGMTIntegrator` applies the exponential velocity scaling on both sides of
  the Coulomb sub-step, so it solves a single Kepler equation per step.
- With `stochastic=True`, `MassiveGGMTIntegrator` no longer adds Ornstein-Uhlenbeck
  noise to both thermostat variables. Instead, it rescales each velocity and its
  fourth-moment thermostat variable by a common random factor, in the manner of the
  stochastic velocity rescaling of Bussi et al., at the center of the step. The
  friction coefficient is the rate of this rescaling, with the same default, and the
  global variables `b1` and `b2` are gone.
- `ForceMatchingRegressor` takes a `devices` keyword, defaulting to 1, so it no longer
  launches distributed training on multi-GPU hosts; `get_learning_curve()` returns a
  copy, with one row per epoch, and raises before the first fit; positions and forces
  must be given in MD units.

- `ExtendedSpaceSystem` no longer takes `tether_period`, and
  `getTetheringForce()` is gone. Only the x components of extension particles enter
  forces and reports, so the y and z components need no tether.
- The dynamical variables of a sum of couplings keep the order in which they first
  appear in the summed couplings, instead of being sorted by name. This is the order
  expected by `setDynamicalVariableValues` and `setDynamicalVariableVelocities` and
  used by `ExtensionWriter` columns, and copies and serialized sums keep it.
- `ForceMatchingRegressor.fit` no longer shows Lightning's progress bar, model
  summary and informational messages unless the new `verbose` keyword is `True`.
- `DynamicalVariable` raises `TypeError`, like `Bounds`, instead of `ValueError` for
  an invalid unit.

### Added
- `MassiveGGMTIntegrator` accepts `halleyIterations`, the number of Halley iterations
  used to solve the hyperbolic Kepler equation of the Coulomb sub-step; three reach
  double precision.
- `ExtendedSpaceContext.synchronize()` re-synchronizes the physical and extension
  contexts after changes made outside the context's own methods.
- The `state` argument of `ExtendedSpaceSimulation` accepts a file name.
- `InnerProductCoupling.getForceGroups()` returns the force groups of its forces.
- `ExtendedSpaceMetadynamics` accepts periodic and reflective variables together.
- `RegulatedNHLIntegrator.getRegulationParameter()` returns the regulation parameter.

### Fixed
- `pandas` and `typing-extensions` are now declared as runtime dependencies.
- The conda recipe declares every runtime dependency.
- The Linux CI matrix now installs the OpenMM version it names and tests OpenMM 8.4
  through 8.6.
- CI tests Python 3.14 and uses `setup-micromamba` v2.
- `HarmonicCoupling` wraps periodic dynamical variables declared in units other than
  radians (e.g. degrees) with the correct period; previously such couplings produced
  wrong energies across the periodic seam.
- `HarmonicCoupling` accepts non-periodic collective/dynamical variable pairs.
- `ExtendedSpaceMetadynamics` mirrors its kernels across reflective walls. Truncated
  Gaussians halved the bias deposited at a wall, which biased free energies within
  a few `sigma` of it, by up to `biasFactor*kT*ln(2)`, and left the bias with a
  nonzero slope there. Periodic variables use a von Mises kernel, which is smooth
  across the seam and close to a Gaussian for widths small relative to the period.
  The bias table holds each reflective axis as a periodic one of twice its range,
  mirrored about the upper wall, so OpenMM's spline has zero slope at both walls;
  its natural spline gave the bias force an error of about 10% there.
- `DynamicalVariable` is hashable.
- Bounds accept NumPy scalars, reject booleans and non-finite values, and are stored as
  floats.
- An `InnerProductCoupling` with `functions` inside a sum of couplings read the wrong
  dynamical variables.
- Summing couplings no longer alters the couplings being summed.
- `getDynamicalVariableValues` works for dynamical variables used only in `functions`.
- `setParameter` updates parameters shared with the extension context, and
  `setState`, `reinitialize`, `minimizeEnergy` and `loadState` keep both contexts in
  sync.
- Setting dynamical variable values or velocities with the wrong number of entries
  raises an error.
- Lepton definitions after `;` in `functions` are no longer taken as variables.
- `InnerProductCoupling` evaluates parameter derivatives only for its own forces.
- The `setTemperature`, `setFrictionCoefficient` and `setTimeConstant` methods of the
  OpenXPS integrators no longer crash.
- Changing the step size of an integrator, or of an extended-space integrator, updates
  the step-size-dependent thermostat coefficients.
- Copying or pickling an extended-space integrator keeps the integrator classes and the
  `assume_force_first`/`assume_symmetric` flags.
- `ExtendedSpaceSimulation.saveState` stores the extension integrator's variables,
  such as thermostat states, so `loadState` restarts the extension dynamics
  continuously.
- A default extension integrator copied from the physical one gets a different random
  seed, and `CSVRIntegrator` follows OpenMM's seed conventions.
- `CSVRIntegrator` leaves a system with zero kinetic energy at rest instead of producing
  NaN, and raises a clear error when stepped before `registerWithSystem`.
- `MassiveGGMTIntegrator.getTimeConstant` returns a time.
- The kinetic energy and temperature reported by `ExtensionWriter` count one degree of
  freedom per dynamical variable instead of including the unused y and z velocity
  components.
- `ExtensionWriter` header getters work regardless of which columns are enabled and
  raise a clear error before `initialize`.
- `ExtendedSpaceBiasVariable` rejects non-positive `sigma` and `grid_width` values
  smaller than 2.
- `ForceMatchingRegressor.fit` tolerates non-finite validation losses, validates the
  shapes of positions and forces, and no longer reseeds NumPy's global generator;
  `get_parameters()` no longer alters the fitted model; kernel centers start inside
  the variables' ranges; weight decay applies only to kernel weights.
- `ForceMatchingRegressor.get_parameters()` wraps only the centers of periodic
  variables, so the returned parameters reproduce `predict()` for reflective and
  unbounded variables, and it raises before the first fit. `fit` raises when the
  validation split leaves no training data instead of returning an untrained model.
- `ForceMatchingRegressor.fit` draws all its randomness from a local generator and no
  longer reseeds PyTorch's global generator; a failed fit keeps the previous model and
  learning curve; `predict` validates the shape of its input like `fit`.
- `Function` recognizes the Lepton built-ins `square`, `cube` and `recip` instead of
  treating them as variables.
- `ExtensionWriter` reports coupling-function columns in sorted order, so the column
  order no longer changes between processes.
- `CircularBounds` honors the unit it is converted to, so `convert()` and
  `DynamicalVariable` with degree units keep degrees; it rejects bounds that are not a
  full circle from -π to π.
- `PeriodicBounds.wrap` returns the lower bound instead of the upper bound when
  floating-point rounding yields a full period.
- `CollectiveVariableCoupling` accepts one-shot iterables, such as generators, for its
  collective and dynamical variables.
- `MassiveGGMTIntegrator.setFrictionCoefficient` raises for a nonzero value when the
  integrator is deterministic, instead of silently ignoring it.
- `ExtendedSpaceContext.setState`, and therefore `loadState` with a plain OpenMM state
  file, copies restored parameters shared with the extension context (e.g. coupling
  constants) to it, and `reinitialize()` resets them in both contexts.
- `ExtendedSpaceContext.setParameter` and `getParameter` accept parameters that exist
  only in the extension context, such as parameters of coupling functions, and
  `setParameter` synchronizes both contexts afterwards.
- A coupling force placed in a nonzero force group by the user keeps that group when
  the coupling is added to a system; only forces left in group 0 are moved to a
  dedicated group.
- A function variable named like a Lepton coordinate (`x1`, `x2`, ...) no longer
  redirects other variables of the same function to the wrong dynamical variable.
- `MassiveGGMTIntegrator` and `RegulatedNHLIntegrator` (with zero friction) no longer
  produce NaN in the extension context after long runs. As extension integrators, they
  thermostat only the x component of each particle, leaving the inert y and z
  components untouched.
- `InnerProductCoupling` functions receive dynamical variables wrapped into their
  bounds; periodic and reflective variables previously entered functions unwrapped.
- Inner product couplings combined in a sum are merged into one, so couplings that
  share a global parameter get the correct extension force; conflicting function
  definitions or parameter values raise an error.
- `ExtendedSpaceSystem` raises when a force outside the coupling depends on a
  parameter controlled by it, and when the coupling is not a `Coupling`.
- `InnerProductCoupling` raises a clear error for forces without global parameters,
  and for functions named like dynamical variables, while a dynamical variable may
  now be both a force parameter and a function input.
- Arguments of decorated constructors and setters may be named tuples (such as
  `openmm.Vec3`) or other sequences such as `range`, which previously raised errors.
- Summing inner product couplings that define the same dynamical variable
  differently raises an error instead of keeping the first definition.
- `ExtensionWriter` reports the regulated kinetic energy when the extension
  integrator is a `RegulatedNHLIntegrator`, so its temperature no longer exceeds the
  thermostat's.
- The `Coupling.addToExtensionSystem` example builds its periodic expression with an
  f-string, so it no longer gives wrong energies across the periodic seam.
- Passing a coupling that is already part of a system to `ExtendedSpaceSystem` raises
  a clear error instead of crashing in OpenMM. A deep copy of the coupling can be
  used instead.
- `ExtendedSpaceSimulation.saveState` and `loadState` accept file objects open in
  binary mode and `os.PathLike` paths, as does the `state` argument.
- `ForceMatchingRegressor.get_learning_curve()` returns its columns in the documented
  order (`epoch`, `train_loss`, `val_loss`), and the documented loss now matches the
  one minimized, which averages over dynamical variables too.
- `DynamicalVariable` rejects names that are not identifiers, as in Python (letters
  may be non-ASCII, such as `λ`), which previously failed later with an obscure
  Lepton error.
- `Function` takes non-ASCII identifiers such as `λ` as names, and no longer takes
  parts of other Lepton tokens as names, such as `vdw` in `lambda.vdw`.
- Errors about incompatible periodicity explain that periodic variables must have
  the same bounds, and errors of `InnerProductCoupling` list names in sorted order.
