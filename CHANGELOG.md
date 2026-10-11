# Changelog

## Unreleased

### Changed
- Python 3.9 is no longer supported; the minimum version is 3.10.
- `NoBounds` now requires `lower`, `upper` and `unit`, which are typical values of the
  variable used to initialize regression kernels; `DynamicalVariable` requires `bounds`.
- `NoBounds` units are validated and converted like other bounds.
- `Bounds` equality distinguishes periodic, reflective and unbounded kinds.
- `DynamicalVariable.distanceTo` raises for mismatched periods or incompatible units.

### Fixed
- `pandas` and `typing-extensions` are now declared as runtime dependencies.
- The conda recipe declares every runtime dependency.
- The Linux CI matrix now installs the OpenMM version it names.
