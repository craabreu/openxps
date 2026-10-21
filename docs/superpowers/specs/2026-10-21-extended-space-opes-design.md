# Extended-space OPES on top of openmm-opes

Date: 2026-10-21. Status: approved in conversation, awaiting review of this document.

## Goal

Give openxps OPES samplers that bias dynamical variables, built on the openmm-opes
package (>= 0.2.0), the way `ExtendedSpaceMetadynamics` is built on
`openmm.app.Metadynamics`.

Two uses, in this order:

1. Gabriela Correa's λ-OPES-d-AFED quinone-in-water runs. Her scripts use an
   unmerged branch of her fork (`gabriela-correa/openxps`, branch `update-openxps`)
   that vendors an older OPES copy and runs a conventional OPES and an OPES-explore
   on θ side by side, reaching into `._opes` for a reporter and a multi-sampler.
   openmm-opes's `CombinedOPES` (OPES_COM, Ray and Rizzi 2025) is the principled form
   of that setup and replaces it.
2. A general openxps feature with the scope of `ExtendedSpaceMetadynamics`: any
   bounded dynamical variables, OPES, OPES-explore and OPES_COM, several samplers
   stepped together, and multiple walkers.

## Decisions

- **Subclasses, mirroring metadynamics.** `ExtendedSpaceOPES` subclasses
  `openmm_opes.OPES` and `ExtendedSpaceCombinedOPES` subclasses
  `openmm_opes.CombinedOPES`. Everything not tied to a simulation or context is
  inherited.
- **Required dependency.** openmm-opes is lightweight (numpy, scipy, openmm), so
  `openmm-opes>=0.2.0` is a required dependency, not an optional extra.
- **Kernels.** The classes use openmm-opes's kernels as they are: Gaussian by
  default, minimum-image on periodic variables (as in PLUMED and in OpenMM's
  metadynamics). No periodic (wrapped or von Mises) kernels: OPES kernels shrink as
  depositions accumulate, so the cut at half a period only matters for the first,
  widest kernels. The `kernelShape` argument is passed through with its default
  `"gaussian"`; `"compact"` stays available because openmm-opes supports it.
- **Metadynamics unchanged.** `ExtendedSpaceMetadynamics` keeps its von Mises and
  mirrored kernels. The two classes therefore use different kernels on periodic
  variables; the OPES docstrings say so.

## API

New module `openxps/opes.py`, exported from `openxps`:

```python
class ExtendedSpaceOPES(openmm_opes.OPES):
    def __init__(self, system: ExtendedSpaceSystem,
                 variables: Sequence[ExtendedSpaceBiasVariable],
                 **kwargs): ...

class ExtendedSpaceCombinedOPES(openmm_opes.CombinedOPES):
    def __init__(self, system: ExtendedSpaceSystem,
                 variables: Sequence[ExtendedSpaceBiasVariable],
                 **kwargs): ...
```

`system` and `variables` replace the parents' arguments of the same names:

- `system` is an `ExtendedSpaceSystem`; the bias force goes into its extension system.
- `variables` are the existing `ExtendedSpaceBiasVariable`s (`dynamical_variable`,
  `sigma`, `grid_width`).

Every other argument is passed to the parent as a keyword argument, with the parent's
names and defaults. The docstrings document `system` and `variables` and link to the
parent's documentation for the rest:
<https://craabreu.github.io/openmm-opes/latest/api/opes/> and
<https://craabreu.github.io/openmm-opes/latest/api/combined-opes/>. `bounded` is
derived from the variables (below), so passing it raises `TypeError`. New openmm-opes
arguments then work without changes to openxps.

### Building the bias variables

As in `ExtendedSpaceMetadynamics.__init__`, for each `ExtendedSpaceBiasVariable`:

- the dynamical variable, in MD units, must be one of `system.getDynamicalVariables()`,
  otherwise `ValueError` naming it;
- `NoBounds` is rejected with `ValueError` (the bias lives on a finite grid), with the
  same message as metadynamics;
- the `openmm.app.BiasVariable` gets `dv.createCollectiveVariable(index)`, the bounds
  as `minValue`/`maxValue`, `sigma` as `biasWidth`, `dv.isPeriodic()` as `periodic`
  and `grid_width` as `gridWidth`.

`bounded` is `True` if any variable has `ReflectiveBounds` and `False` otherwise.
openmm-opes applies it to every non-periodic variable, and with `NoBounds` rejected the
non-periodic variables are exactly the reflective ones. `PeriodicBounds` and
`CircularBounds` give periodic variables.

`sigma` keeps openmm-opes's meaning: with `varianceFrequency=None` it is the standard
deviation of the unbiased distribution, from which the fixed bandwidth is derived.

### Simulations and contexts

openmm-opes reads `simulation.context`, `simulation.currentStep` and
`simulation.step()`, and calls some public methods internally with the simulation or
context it was given. The overrides accept the openxps objects and convert them; the
conversion passes objects that are already converted through unchanged, so the
internal calls keep working.

- `_extensionSimulation(simulation)`: returns `simulation` if it is already the
  wrapper, otherwise `_SimulationWrapper(simulation)` from `openxps/metadynamics.py`
  (moved to a shared private helper if needed).
- `_extensionContext(context)`: returns `context.getExtensionContext()` for an
  `ExtendedSpaceContext`, otherwise `context`.

Overridden methods, each delegating to the parent after converting:

| Method | Classes | Converts |
|---|---|---|
| `step(simulation, steps)` | both | simulation |
| `getCollectiveVariables(simulation)` | both | simulation |
| `getBiasEnergy(context)` | both | context |
| `updateContext(context)` | both | context |
| `getComponentBiasEnergies(context)` | combined | context |
| `stepTogether(simulation, samplers, steps)` (static) | both | simulation, once |

`stepTogether` calls `openmm_opes.OPES.stepTogether` with the wrapped simulation, so
any mix of `ExtendedSpaceOPES` and `ExtendedSpaceCombinedOPES` acting on the same
extension system can run together. Running OPES together with
`ExtendedSpaceMetadynamics` is out of scope.

Inherited without change, where the parent class has them: `getFreeEnergy`,
`getBias`, `getCt`, `getNumKernels`, `getState`/`setState`; `getVariance` and
`getAverageDensity` (OPES); `getComponentBiases` (combined); and the multiple-walker
behavior of `biasDir`, `walkerId` and `saveFrequency`.


### Documented limitations

- The bias force is added to the extension system when the object is created, so it
  must be created before the `ExtendedSpaceSimulation` or `ExtendedSpaceContext`, as
  with metadynamics.
- `extraBias` entries are evaluated on the extension context. A bias in the extension
  system (for instance another sampler) works; a bias in the physical system does not.
- Periodic variables use minimum-image kernels, unlike `ExtendedSpaceMetadynamics`'s
  von Mises kernels.
- openmm-opes rejects a mix of periodic and non-periodic variables ("OPES cannot
  handle mixed periodic/non-periodic variables"), so unlike `ExtendedSpaceMetadynamics`
  the OPES classes cannot bias a periodic and a reflective variable together. The
  error propagates unchanged.
- At most three variables, as in openmm-opes.

## Packaging and docs

- `openmm-opes>=0.2.0` in `pyproject.toml` `dependencies`, in the conda recipe's `run`
  requirements and in the test, docs and deployment environments (mdtools channel).
- API pages for both classes next to `ExtendedSpaceMetadynamics`.
- CHANGELOG entry under *New*.
- A doctest per class, adapted from the alanine dipeptide example in
  `ExtendedSpaceMetadynamics`'s docstring.

## Tests

`openxps/tests/test_opes.py`, Reference platform:

1. **Construction:** the extension system gains exactly one force; the bias variables
   match the dynamical variables (bounds, periodicity, width, grid); `bounded` follows
   `ReflectiveBounds`; `NoBounds` and a foreign dynamical variable raise `ValueError`;
   passing `bounded` raises `TypeError`; other keyword arguments reach the parent.
2. **Equivalence with plain openmm-opes:** drive a one-variable harmonic
   `CollectiveVariableCoupling` with `ExtendedSpaceOPES`; drive a plain
   `openmm_opes.OPES` built on a stand-alone copy of the extension system through the
   same sequence of dynamical-variable values; the deposited kernels, the bias and the
   free energy agree. This tests the adapter, not OPES.
3. **Overrides:** each overridden method accepts both the openxps object and the
   already-converted one and gives the same result; for the combined class, the two
   component bias energies sum to the total.
4. **Stepping together:** `stepTogether` with one `ExtendedSpaceOPES` and one
   `ExtendedSpaceCombinedOPES` deposits on both schedules.
5. **State and walkers:** `getState`/`setState` round-trips; two walkers in one
   `biasDir` share their bias.
6. **Doctests** for both classes.

## First use case

Outside openxps, in `/data3/craabreu/github_repos/lambda-dynamics/`: a port of one of
Gabriela's quinone λ-OPES runs to a single `ExtendedSpaceCombinedOPES` on θ (barrier
126 kJ/mol, explore barrier 21 kJ/mol, as in her script), using the existing
`lambda_setup.py` system, run briefly on the GPU to confirm the workflow. Her
repository is not modified.

## Out of scope

- Periodic kernels in openmm-opes.
- Any change to `ExtendedSpaceMetadynamics`.
- A bias reporter; openmm-opes has none, and `cvpack` reporters can report the
  samplers' quantities if needed later.
- Gabriela's multibath NHL integrator and `ExtensionWriter` changes, and her
  state-ID functions.
