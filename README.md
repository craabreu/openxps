OpenXPS: Extended Phase-Space Simulations with OpenMM
=====================================================

[![Tests](https://github.com/craabreu/openxps/actions/workflows/Linux.yaml/badge.svg?branch=main)](https://github.com/craabreu/openxps/actions/workflows/Linux.yaml)
[![Coverage](https://craabreu.github.io/openxps/development/coverage/coverage.svg)](https://craabreu.github.io/openxps/development/coverage)
[![Docs](https://github.com/craabreu/openxps/actions/workflows/Docs.yaml/badge.svg?branch=main)](https://github.com/craabreu/openxps/actions/workflows/Docs.yaml)
[![PyPI](https://img.shields.io/pypi/v/openxps.svg)](https://pypi.org/project/openxps)
[![Conda](https://img.shields.io/conda/v/mdtools/openxps.svg)](https://anaconda.org/mdtools/openxps)
[![License](https://img.shields.io/badge/License-MIT-yellowgreen.svg)](https://github.com/craabreu/openxps/blob/main/LICENSE.md)

OpenXPS runs extended phase-space (XPS) simulations with [OpenMM]. It adds
dynamical variables to a physical system and couples them to collective variables
or to force parameters, as in temperature-accelerated molecular dynamics (TAMD),
driven adiabatic free-energy dynamics (d-AFED) and unified free-energy dynamics
(UFED).

### Background

In an extended phase-space simulation, the physical coordinates **r** are augmented
with dynamical variables **s**, which have masses and velocities of their own. Each
dynamical variable sᵢ is coupled to a collective variable θᵢ(**r**), usually through
a harmonic potential ½κᵢ[θᵢ(**r**) − sᵢ]². If the dynamical variables are much
heavier than the collective variables, they move adiabatically: the physical system
relaxes around each value of **s**. If, in addition, the dynamical variables are kept
at a temperature Tₛ higher than the physical temperature T, they cross free-energy
barriers faster, while their distribution remains proportional to
exp[−A(**s**)/k<sub>B</sub>Tₛ], where A is the free energy at the physical temperature,
smoothed by the harmonic coupling. This is the idea behind TAMD
([Maragliano and Vanden-Eijnden, 2006][Maragliano2006]) and d-AFED
([Abrams and Tuckerman, 2008][Abrams2008]).

A dynamical variable can also drive a global parameter of OpenMM forces, such as an
alchemical coupling parameter λ, so that the parameter itself is sampled, as in
λ-dynamics. [Correa et al. (2026)][Correa2026] combined λ-dynamics with d-AFED to
compute solvation free energies.

UFED ([Chen et al., 2012][Chen2012]) combines d-AFED with a metadynamics bias
([Laio and Parrinello, 2002][Laio2002]; [Barducci et al., 2008][Barducci2008]) applied
to the dynamical variables. Free-energy surfaces can then be rebuilt from the mean
forces that the physical system exerts on the dynamical variables, which
[Cuendet and Tuckerman (2014)][Cuendet2014] found to be more accurate than histograms
or the accumulated bias.

Keeping the dynamical variables at their own temperature calls for thermostats that
hold up under these conditions. OpenXPS provides canonical sampling through velocity
rescaling ([Bussi et al., 2007][Bussi2007]), massive generalized Gaussian moment
thermostatting ([Liu and Tuckerman, 2000][Liu2000]), regulated Nosé–Hoover–Langevin
dynamics ([Abreu and Tuckerman, 2021][Abreu2021]) and symmetric Langevin dynamics
([Leimkuhler and Matthews, 2013][Leimkuhler2013]), as well as multiple time-step
integration of the physical and extension systems.

### Features

* [Dynamical variables][DynamicalVariable] with periodic, reflective or no bounds,
  in any units.
* [Harmonic][HarmonicCoupling], [collective-variable][CollectiveVariableCoupling]
  and [inner-product][InnerProductCoupling] couplings between dynamical variables and
  the physical system. Couplings can be summed.
* [`ExtendedSpaceSimulation`][ExtendedSpaceSimulation] and
  [`ExtendedSpaceContext`][ExtendedSpaceContext], which keep the physical and
  extension contexts in sync and save or load both together.
* [Lockstep][LockstepIntegrator] and [multiple time-step][SplitIntegrator]
  integration of the two systems, with the thermostats listed above.
* [Extended phase-space metadynamics][ExtendedSpaceMetadynamics].
* [`ForceMatchingRegressor`][ForceMatchingRegressor], which fits free-energy surfaces
  to mean forces with radial basis functions.
* [`ExtensionWriter`][ExtensionWriter], which reports extension-system data through
  [CVPack] reporters.

### Installation

OpenXPS requires Python 3.10 or newer and OpenMM 8.4 or newer. Install it from the
[mdtools] conda channel:

```bash
conda install -c conda-forge -c mdtools openxps
```

or from PyPI:

```bash
pip install openxps
```

PyTorch and Lightning, which `ForceMatchingRegressor` uses, are installed as
dependencies.

### Quickstart

The script below runs a d-AFED simulation of alanine dipeptide in vacuum, with
dynamical variables coupled to the backbone dihedral angles φ and ψ. The physical
system is kept at 300 K and the dynamical variables at 1500 K. The molecule comes
from [OpenMMTools], which you can install with `pip install openmmtools` or
`conda install -c conda-forge openmmtools`.

```python
import sys

import cvpack
from openmm import unit
from openmmtools import testsystems

import openxps as xps

# Physical system: alanine dipeptide in vacuum
model = testsystems.AlanineDipeptideVacuum()

# Collective variables: the backbone dihedral angles phi and psi
backbone = [
    atom.index for atom in model.topology.atoms() if atom.name in ("C", "CA", "N")
]
phi = cvpack.Torsion(*backbone[0:4], name="phi")
psi = cvpack.Torsion(*backbone[1:5], name="psi")

# Dynamical variables, harmonically coupled to the collective variables
mass = 12 * unit.dalton * (unit.nanometer / unit.radian) ** 2
kappa = 1000 * unit.kilojoule_per_mole / unit.radian**2
phi_s = xps.DynamicalVariable("phi_s", unit.radian, mass, xps.CircularBounds())
psi_s = xps.DynamicalVariable("psi_s", unit.radian, mass, xps.CircularBounds())
coupling = xps.HarmonicCoupling(phi, phi_s, kappa) + xps.HarmonicCoupling(
    psi, psi_s, kappa
)

# Physical system at 300 K (2 fs steps), dynamical variables at 1500 K (8 fs steps)
integrator = xps.SplitIntegrator(
    xps.integrators.SymmetricLangevinIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds
    ),
    xps.integrators.SymmetricLangevinIntegrator(
        1500 * unit.kelvin, 10 / unit.picosecond, 8 * unit.femtoseconds
    ),
)

simulation = xps.ExtendedSpaceSimulation(
    model.topology, xps.ExtendedSpaceSystem(model.system, coupling), integrator
)
simulation.context.setPositions(model.positions)
simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
values = coupling.getCollectiveVariableValues(simulation.context)
simulation.context.setDynamicalVariableValues([values["phi"], values["psi"]])
simulation.context.setDynamicalVariableVelocitiesToTemperature(1500 * unit.kelvin)

simulation.reporters.append(
    cvpack.reporting.StateDataReporter(
        sys.stdout,
        250,
        step=True,
        temperature=True,
        writers=[xps.ExtensionWriter(temperature=True, dynamical_variables=True)],
    )
)
simulation.step(2500)  # 20 ps
```

Each line of the output reports the step, the physical and extension temperatures,
and the values of the dynamical variables (your numbers will differ):

```
#"Step","Temperature (K)","Extension Temperature (K)","phi_s (rad)","psi_s (rad)"
250,250.67811271332351,1666.1861605801755,-2.154638148796597,2.617085205102951
500,272.0123447639219,1187.2009312011787,-2.831159123784501,-2.9213596609714596
...
```

With only two degrees of freedom, the extension temperature fluctuates widely, but
its average approaches 1500 K in longer runs.

### Examples

The [alanine dipeptide examples] include Jupyter notebooks on
[choosing masses and force constants][parameter-selection] for adiabatic
simulations and on [energy conservation][nve-simulation] in constant-energy
simulations.

### Documentation

The documentation of the latest release is available at [GitHub Pages].

### Citing OpenXPS

If you use OpenXPS in published work, please cite the software, using the metadata in
[`CITATION.cff`][CITATION] (GitHub shows it under *Cite this repository*), and the
papers that introduced the methods you used, listed in [Background](#background).

### License

OpenXPS is distributed under the [MIT License][LICENSE].

Copyright (c) 2024-2026 [C. Abreu](https://github.com/craabreu)


[OpenMM]: https://openmm.org
[CVPack]: https://github.com/RedesignScience/cvpack
[OpenMMTools]: https://github.com/choderalab/openmmtools
[mdtools]: https://anaconda.org/mdtools/openxps
[GitHub Pages]: https://craabreu.github.io/openxps/latest
[CITATION]: https://github.com/craabreu/openxps/blob/main/CITATION.cff
[LICENSE]: https://github.com/craabreu/openxps/blob/main/LICENSE.md
[alanine dipeptide examples]: https://github.com/craabreu/openxps/tree/main/examples/alanine_dipeptide
[parameter-selection]: https://github.com/craabreu/openxps/blob/main/examples/alanine_dipeptide/parameter-selection.ipynb
[nve-simulation]: https://github.com/craabreu/openxps/blob/main/examples/alanine_dipeptide/nve-simulation.ipynb

[DynamicalVariable]: https://craabreu.github.io/openxps/latest/api/DynamicalVariable.html
[HarmonicCoupling]: https://craabreu.github.io/openxps/latest/api/HarmonicCoupling.html
[CollectiveVariableCoupling]: https://craabreu.github.io/openxps/latest/api/CollectiveVariableCoupling.html
[InnerProductCoupling]: https://craabreu.github.io/openxps/latest/api/InnerProductCoupling.html
[ExtendedSpaceSimulation]: https://craabreu.github.io/openxps/latest/api/ExtendedSpaceSimulation.html
[ExtendedSpaceContext]: https://craabreu.github.io/openxps/latest/api/ExtendedSpaceContext.html
[LockstepIntegrator]: https://craabreu.github.io/openxps/latest/api/LockstepIntegrator.html
[SplitIntegrator]: https://craabreu.github.io/openxps/latest/api/SplitIntegrator.html
[ExtendedSpaceMetadynamics]: https://craabreu.github.io/openxps/latest/api/ExtendedSpaceMetadynamics.html
[ForceMatchingRegressor]: https://craabreu.github.io/openxps/latest/api/ForceMatchingRegressor.html
[ExtensionWriter]: https://craabreu.github.io/openxps/latest/api/ExtensionWriter.html

[Maragliano2006]: https://doi.org/10.1016/j.cplett.2006.05.062
[Abrams2008]: https://doi.org/10.1021/jp805039u
[Correa2026]: https://arxiv.org/abs/2608.12691
[Chen2012]: https://doi.org/10.1063/1.4733389
[Laio2002]: https://doi.org/10.1073/pnas.202427399
[Barducci2008]: https://doi.org/10.1103/physrevlett.100.020603
[Cuendet2014]: https://doi.org/10.1021/ct500012b
[Bussi2007]: https://doi.org/10.1063/1.2408420
[Liu2000]: https://doi.org/10.1063/1.480769
[Abreu2021]: https://doi.org/10.1080/00268976.2021.1923848
[Leimkuhler2013]: https://doi.org/10.1063/1.4802990
