import typing as t

import openmm as mm
from openmm import unit as mmunit
from openmm.app.metadynamics import BiasVariable

from .bounds import NoBounds
from .system import ExtendedSpaceSystem
from .metadynamics import ExtendedSpaceBiasVariable  
from .opes_folder import OPES         

class ExtendedSpaceOPES:

    def __init__(
        self,
        system: ExtendedSpaceSystem,
        variables: t.Sequence[ExtendedSpaceBiasVariable],
        temperature: mmunit.Quantity,
        barrier: mmunit.Quantity,
        frequency: int,
        varianceFrequency: t.Optional[int] = None,
        biasFactor: t.Optional[float] = None,
        exploreMode: bool = False,
        bounded: bool = False,
        **kwargs,
    ) -> None:
        
        system_dvs = system.getDynamicalVariables()
        bias_variables = []

        for var in variables:
            dv = var.dynamical_variable.in_md_units()
            if dv not in system_dvs:
                raise ValueError(f"Dynamical variable {dv.name} not found in ExtendedSpaceSystem")
            if isinstance(dv.bounds, NoBounds):
                raise ValueError(f"Dynamical variable {dv.name} has no bounds")
            index = system_dvs.index(dv)
            bias_variables.append(
                BiasVariable(
                    dv.createCollectiveVariable(index),
                    dv.bounds.lower * dv.bounds.unit,
                    dv.bounds.upper * dv.bounds.unit,
                    var.sigma,
                    dv.isPeriodic(),
                    var.grid_width ,
                )
            )

        self._opes = OPES(
            system.getExtensionSystem(),
            bias_variables,
            temperature=temperature,
            barrier=barrier,
            frequency=frequency,
            varianceFrequency=varianceFrequency,
            biasFactor=biasFactor,
            exploreMode=exploreMode,
            bounded=bounded,
            **kwargs,
        )


    def step(self, simulation, steps: int) -> None:
        class _SimulationWrapper:
            def __init__(self, sim):
                self._sim = sim
            @property
            def context(self):
                return self._sim.context.getExtensionContext()
            @property
            def currentStep(self):
                return self._sim.currentStep
            def step(self, n):
                self._sim.step(n)
        self._opes.step(_SimulationWrapper(simulation), steps)



class MultiOPESSampler:

    def __init__(self, opes_list):
        self.opes_list = list(opes_list)
        self._global_interval = min(o._interval for o in self.opes_list)

    def step(self, simulation, steps: int):

        class _SimulationWrapper:
            def __init__(self, sim):
                self._sim = sim

            @property
            def context(self):
                return self._sim.context.getExtensionContext()

            @property
            def currentStep(self):
                return self._sim.currentStep

            def step(self, n):
                self._sim.step(n)

        simw = _SimulationWrapper(simulation)

        stepsToGo = steps
        groups_per_opes = [{o._force.getForceGroup()} for o in self.opes_list]

        while stepsToGo > 0:
            nextSteps = min(
                stepsToGo,
                self._global_interval - simulation.currentStep % self._global_interval
            )

            simulation.step(nextSteps)
            cur = simulation.currentStep

            for o, groups in zip(self.opes_list, groups_per_opes):
                if cur % o._interval != 0:
                    continue

                position = o.getCollectiveVariables(simw)

                for index, stateIDFunc in enumerate(o.stateIDFuncs):
                    if stateIDFunc(position):
                        o._lastVisitedState = index
                        break

                if o._adaptiveVariance:
                    o._updateSampleStats(position)

                if cur % o.frequency == 0:
                    state = simw.context.getState(getEnergy=True, groups=groups)
                    energy = state.getPotentialEnergy()

                    o.addKernel(position, energy)
                    o.updateContext(simw.context)

                    if o.saveFrequency is not None and cur % o.saveFrequency == 0:
                        o._syncWithDisk()

            stepsToGo -= nextSteps















