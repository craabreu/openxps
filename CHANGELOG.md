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

### Added
- `ExtendedSpaceContext.synchronize()` re-synchronizes the physical and extension
  contexts after changes made outside the context's own methods.
- The `state` argument of `ExtendedSpaceSimulation` accepts a file name.
- `InnerProductCoupling.getForceGroups()` returns the force groups of its forces.

### Fixed
- `pandas` and `typing-extensions` are now declared as runtime dependencies.
- The conda recipe declares every runtime dependency.
- The Linux CI matrix now installs the OpenMM version it names and tests OpenMM 8.4
  through 8.6.
- `HarmonicCoupling` wraps periodic dynamical variables declared in units other than
  radians (e.g. degrees) with the correct period; previously such couplings produced
  wrong energies across the periodic seam.
- `HarmonicCoupling` accepts non-periodic collective/dynamical variable pairs.
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
- A default extension integrator copied from the physical one gets a different random
  seed, and `CSVRIntegrator` follows OpenMM's seed conventions.
- `CSVRIntegrator` leaves a system with zero kinetic energy at rest instead of producing
  NaN, and raises a clear error when stepped before `registerWithSystem`.
- `MassiveGGMTIntegrator.getTimeConstant` returns a time.
