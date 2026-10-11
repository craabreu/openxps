# Changelog

## Unreleased

### Changed
- Python 3.9 is no longer supported; the minimum version is 3.10.
- OpenMM 8.4 or newer is required.

### Fixed
- `pandas` and `typing-extensions` are now declared as runtime dependencies.
- The conda recipe declares every runtime dependency.
- The Linux CI matrix now installs the OpenMM version it names and tests OpenMM 8.4
  through 8.6.
