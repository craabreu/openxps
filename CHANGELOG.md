# Changelog

## Unreleased

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
- `ForceMatchingRegressor` takes a `devices` keyword, defaulting to 1, so it no longer
  launches distributed training on multi-GPU hosts; `get_learning_curve()` returns a
  copy, with one row per epoch, and raises before the first fit; positions and forces
  must be given in MD units.

- `ExtendedSpaceSystem` no longer takes `tether_period`, and
  `getTetheringForce()` is gone. The tether never engaged, because the y and z
  components of extension particles rest at the origin with zero velocity.

### Added
- `MassiveGGMTIntegrator` accepts `newtonIterations`.
- `ExtendedSpaceContext.synchronize()` re-synchronizes the physical and extension
  contexts after changes made outside the context's own methods.
- The `state` argument of `ExtendedSpaceSimulation` accepts a file name.
- `InnerProductCoupling.getForceGroups()` returns the force groups of its forces.
- `ExtendedSpaceMetadynamics` accepts periodic and reflective variables together.

### Fixed
- `pandas` and `typing-extensions` are now declared as runtime dependencies.
- The conda recipe declares every runtime dependency.
- The Linux CI matrix now installs the OpenMM version it names and tests OpenMM 8.4
  through 8.6.
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
