Extended Phase-Space Simulations with OpenMM
============================================

[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/craabreu/openxps/workflows/Linux/badge.svg)](https://github.com/craabreu/openxps/actions?query=workflow%3ALinux)
[![GitHub Actions Build Status](https://github.com/craabreu/openxps/workflows/MacOS/badge.svg)](https://github.com/craabreu/openxps/actions?query=workflow%3AMacOS)
[![GitHub Actions Build Status](https://github.com/craabreu/openxps/workflows/Windows/badge.svg)](https://github.com/craabreu/openxps/actions?query=workflow%3AWindows)
[![GitHub Actions Build Status](https://github.com/craabreu/openxps/workflows/Linter/badge.svg)](https://github.com/craabreu/openxps/actions?query=workflow%3ALinter)
[![Documentation Status](https://github.com/craabreu/openxps/workflows/Docs/badge.svg)](https://github.com/craabreu/openxps/actions?query=workflow%3ADocs)
[![Coverage Report](https://craabreu.github.io/openxps/development/coverage/coverage.svg)](https://craabreu.github.io/openxps/development/coverage)

[![Conda version](https://img.shields.io/conda/v/mdtools/openxps.svg)](https://anaconda.org/mdtools/openxps)
[![Conda platforms](https://img.shields.io/conda/pn/mdtools/openxps.svg)](https://anaconda.org/mdtools/openxps)
[![Conda downloads](https://img.shields.io/conda/dn/mdtools/openxps.svg)](https://anaconda.org/mdtools/openxps)

[![PyPI version](https://img.shields.io/pypi/v/openxps.svg)](https://pypi.org/project/openxps)
[![PyPI version](https://img.shields.io/pypi/pyversions/openxps.svg)](https://pypi.org/project/openxps)
[![PyPI version](https://img.shields.io/pypi/dm/openxps.svg)](https://pypi.org/project/openxps)

[![License](https://img.shields.io/badge/License-MIT-yellowgreen.svg?style=flat)](https://github.com/craabreu/openxps/blob/main/LICENSE.md)

### Overview

OpenXPS runs extended phase-space (XPS) simulations with [OpenMM]. It adds
dynamical variables to a physical system and couples them to collective variables
or to force parameters, as in driven adiabatic free-energy dynamics (d-AFED) and
temperature-accelerated molecular dynamics (TAMD).

Main features:

* Dynamical variables with periodic, reflective or no bounds, in any units.
* Harmonic, custom collective-variable, and inner-product couplings between
  dynamical variables and the physical system. Couplings can be summed.
* `ExtendedSpaceSimulation` and `ExtendedSpaceContext`, which keep the physical and
  extension contexts in sync and save or load both together.
* Lockstep and multiple-time-step (split) integration of the two subsystems.
* Thermostatted integrators: CSVR, massive GGMT, regulated Nosé-Hoover-Langevin and
  symmetric Langevin.
* Extended phase-space metadynamics.
* `ForceMatchingRegressor`, which fits free energy surfaces to mean forces with
  radial basis functions.
* `ExtensionWriter`, which reports extension-system data through
  [CVPack] reporters.

### Installation and Usage

OpenXPS is available as a conda package on the [mdtools] channel. To install it, run:

```bash
    conda install -c conda-forge -c mdtools openxps
```

Or:

```bash
    mamba install -c mdtools openxps
```

To use OpenXPS in your own Python script or Jupyter notebook, simply import it as follows:

```python
    import openxps
```

### Documentation

Documentation for the latest OpenXPS version is available at [Github Pages].

### Copyright

Copyright (c) 2024-2026 [C. Abreu](https://github.com/craabreu)


[Github Pages]: https://craabreu.github.io/openxps/latest
[mdtools]: https://anaconda.org/mdtools/openxps
[OpenMM]: https://openmm.org
[CVPack]: https://github.com/RedesignScience/cvpack
