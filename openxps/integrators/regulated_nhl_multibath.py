"""
.. module:: openxps.integrators.regulated_nhl
   :platform: Linux, MacOS, Windows
   :synopsis: Regulated Nosé-Hoover-Langevin (NHL) integrator.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import numpy as np
import openmm as mm
from openmm import unit as mmunit

from openxps.utils import preprocess_args

from .utils import IntegratorMixin, add_property

@add_property("temperature", mmunit.kelvin)
@add_property("time constant", mmunit.picosecond)
@add_property("friction coefficient", 1 / mmunit.picosecond)
class MultiBathRegulatedNHLIntegrator(IntegratorMixin, mm.CustomIntegrator):

    @preprocess_args
    def __init__(
        self,
        temperature: mmunit.Quantity | None,
        timeConstant: mmunit.Quantity | None,
        stepSize: mmunit.Quantity,
        *,
        bathTemperatures=None,
        bathTimeConstants=None,
        regulationParameter: float = 1,
        forceFirst: bool = False
    ) -> None:
        super().__init__(stepSize)
        self._forceFirst = forceFirst
        self._regulation_parameter = regulationParameter
          
        self._bath_temperatures = bathTemperatures
        self._bath_time_constants = bathTimeConstants
        self._multi_bath = (bathTemperatures is not None) or (bathTimeConstants is not None)
            
        if self._multi_bath:
            if bathTemperatures is None or bathTimeConstants is None:
                raise ValueError("you must provide bathTemperatures and bathTimeConstants (one per DV)")

        if not self._multi_bath:
            if temperature is None or timeConstant is None:
                raise ValueError("temperature and timeConstant are required")
            self._init_temperature(temperature)
            self._init_time_constant(timeConstant)
            self._init_friction_coefficient(1 / timeConstant)
            
        self._add_variables()
        self.addUpdateContextState()
        self.setKineticEnergyExpression(
            f"0.5*m*v*c*tanh(v/c); c=sqrt({self._regulation_parameter}*kT/m)"
        )
        self._add_boost(1 if forceFirst else 0.5)
        self._add_translation(0.5)
        self._add_thermostat(1)
        self._add_translation(0.5)
        if not forceFirst:
            self._add_boost(0.5)

    def _add_variables(self) -> None:
        if self._multi_bath:
            self.addPerDofVariable("kT", 0)
            self.addPerDofVariable("invQ", 0)
            self.addPerDofVariable("a", 0)
            self.addPerDofVariable("b", 0)
        else:
            self.addGlobalVariable("kT", 0)
            self.addGlobalVariable("invQ", 0)
            self.addGlobalVariable("a", 0)
            self.addGlobalVariable("b", 0)

        self.addPerDofVariable("v1", 0)

        if not self._multi_bath:
            self._update_global_variables()

    def _update_global_variables(self) -> None:
        tau = self.getTimeConstant()
        dt = self.getStepSize()
        friction = self.getFrictionCoefficient()
        kt = mmunit.MOLAR_GAS_CONSTANT_R * self.getTemperature()
        self.setGlobalVariableByName("kT", kt)
        self.setGlobalVariableByName("invQ", 1 / (kt * tau**2))
        self.setGlobalVariableByName("a", np.exp(-friction * dt))
        self.setGlobalVariableByName("b", np.sqrt(1 - np.exp(-2 * friction * dt)) / tau)

    def _add_translation(self, fraction: float) -> None:
        self.addComputePerDof(
            "x",
            f"x + {fraction}*dt*c*tanh(v/c); c=sqrt({self._regulation_parameter}*kT/m)",
        )

    def _add_boost(self, fraction: float) -> None:
        self.addComputePerDof("v", f"v + {fraction}*dt*f/m")

    def _add_v1_boost(self, fraction: float) -> str:
        self.addComputePerDof(
            "v1",
            f"v1 + {fraction}*dt*(m*v*c*tanh(v/c) - kT)*invQ"
            f"; c=sqrt({self._regulation_parameter}*kT/m)",
        )

    def _add_v_scaling(self, fraction: float) -> str:
        return self.addComputePerDof("v", f"v*exp(-{fraction}*dt*v1)")

    def _add_thermostat(self, fraction: float) -> None:
        self._add_v1_boost(0.5 * fraction)
        self._add_v_scaling(0.5 * fraction)
        self.addComputePerDof("v1", "a*v1 + b*gaussian")
        self._add_v_scaling(0.5 * fraction)
        self._add_v1_boost(0.5 * fraction)

    def _set_multi_bath_per_dof(self, system: mm.System) -> None:

        n = system.getNumParticles()
        dt = self.getStepSize()

        if len(self._bath_temperatures) != n:
            raise ValueError(f"bathTemperatures must have length {n} (one per dynamical variable)")
        if len(self._bath_time_constants) != n:
            raise ValueError(f"bathTimeConstants must have length {n} (one per dynamical variable)")

        dt_ps = dt.value_in_unit(mmunit.picosecond)
        kT   = np.zeros((n, 3), dtype=float)
        invQ = np.zeros((n, 3), dtype=float)
        a = np.ones((n, 3), dtype=float)   
        b = np.zeros((n, 3), dtype=float)  

        for i in range(n):
            T = self._bath_temperatures[i]
            tau = self._bath_time_constants[i]

            kt = (mmunit.MOLAR_GAS_CONSTANT_R * T).value_in_unit(mmunit.kilojoule_per_mole)
            tau_ps = tau.value_in_unit(mmunit.picosecond)
            fric_ps = 1 / tau_ps
            
            kT[i,:] = kt
            invQ[i, :] = 1.0 / (kt * tau_ps**2)
            a[i, :] = np.exp(-fric_ps * dt_ps)
            b[i, :] = np.sqrt(1.0 - np.exp(-2.0 * fric_ps * dt_ps)) / tau_ps

        kT_vec   = [mm.Vec3(*row) for row in kT]
        invQ_vec = [mm.Vec3(*row) for row in invQ]
        a_vec    = [mm.Vec3(*row) for row in a]
        b_vec    = [mm.Vec3(*row) for row in b]

        self.setPerDofVariableByName("kT", kT_vec)
        self.setPerDofVariableByName("invQ", invQ_vec)
        self.setPerDofVariableByName("a", a_vec)
        self.setPerDofVariableByName("b", b_vec)

    def registerWithSystem(self, system: mm.System, isExtension: bool) -> None:
        if system.getNumConstraints() > 0:
            raise ValueError("Regulated NHL integrators do not support constraints.")
        if isExtension and self._multi_bath:
            self._set_multi_bath_per_dof(system)
        if self._multi_bath and not isExtension:
            raise ValueError("Multi-bath RegulatedNHLIntegrator is intended for extension systems only.")


