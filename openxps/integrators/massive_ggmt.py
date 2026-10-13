"""
.. module:: openxps.integrators.massive_ggmt
   :platform: Linux, MacOS, Windows
   :synopsis: Massive Generalized Gaussian Moment Thermostat (GGMT) integrator.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import numpy as np
import openmm as mm
from openmm import unit as mmunit

from openxps.integrators.utils import IntegratorMixin, add_property
from openxps.utils import preprocess_args

_COULOMB_STRENGTH = "(2/sqrt(3))*kT^3*invQ2"
_V_FLOOR = "max(abs(v), sqrt(1e-30*kT/m))"


@add_property("temperature", mmunit.kelvin)
@add_property("time constant", mmunit.picosecond)
@add_property("friction coefficient", 1 / mmunit.picosecond)
class MassiveGGMTIntegrator(IntegratorMixin, mm.CustomIntegrator):
    r"""A massive, second- and fourth-moment GGMT integrator :cite:`Liu2000`.

    The massive GGMT integrator is effective at maintaining temperature control under
    steady-state conditions, such as in d-AFED/TAMD :cite:`Maragliano2006,Abrams2008`
    or UFED :cite:`Chen2012` simulations.

    .. note::
        If stochasticity is enabled, the friction coefficient is initially set to the
        inverse of the time constant, but this value can be customized via
        :meth:`setFrictionCoefficient`. Otherwise, the friction coefficient is zero
        and cannot be changed.

    .. warning::
        This integrator is incompatible with systems subject to constraints.

    Parameters
    ----------
    temperature
        The temperature of the heat bath.
    timeConstant
        The time constant (a.k.a. coupling/damping/relaxation time) of the thermostat.
    stepSize
        The integration step size.

    Keyword Arguments
    -----------------
    stochastic
        If True, the integrator will apply a stochastic rescaling to each velocity and
        its fourth-moment thermostat variable at the center of the step.
    forceFirst
        If True, the integrator will apply a force-first scheme. Otherwise, it will
        apply a symmetric operator splitting scheme.
    halleyIterations
        The number of Halley iterations used to solve the hyperbolic Kepler equation
        of the Coulomb sub-step (see Notes). The default reaches double precision.

    Notes
    -----
    Propagated alone, the cubic friction of the GGMT equations has a finite-time
    singularity, which makes Trotter-factorized integrators fail at large step sizes.
    Here it is propagated together with the fourth-moment thermostat variable, as an
    exactly solvable Coulomb problem that requires one hyperbolic Kepler equation per
    step. The stochastic version rescales each velocity and its fourth-moment
    thermostat variable by a common random factor :cite:`Bussi2007` at the center of
    the step, where the friction coefficient sets the rate of the rescaling.

    Example
    -------
    >>> import openxps as xps
    >>> from openmm import unit
    >>> from openmmtools import testsystems

    Symmetric scheme (default)

    >>> integrator = xps.integrators.MassiveGGMTIntegrator(
    ...     temperature=300 * unit.kelvin,
    ...     timeConstant=100 * unit.femtoseconds,
    ...     stepSize=2 * unit.femtoseconds,
    ... )
    >>> integrator
    Per-dof variables:
      v1, v2, active, meanAnomaly, angle, expAngle
    Global variables:
      kT = 2.494338...
      invQ = 40.090...
      invQ2 = 2.416...
    Computation steps:
       0: allow forces to update the context state
       1: v <- v + 0.5*dt*f/m
       2: x <- x + 0.5*dt*v
       3: v1 <- active*(v1 + 0.5*dt*(m*v^2 - kT)*invQ)
       4: v <- v*exp(-0.5*dt*(v1 + kT*v2))
       5: meanAnomaly <- (2*step(sinh0) - 1)*log(abs(sinh0) + cosh0) + ...
       6: angle <- start - residual*coshA/(2*coshA^3 - residual*sinhA); ...
       7: angle <- angle - residual*coshA/(2*coshA^3 - residual*sinhA); ...
       8: expAngle <- expA*(1 - delta + delta^2/2 - delta^3/6); ...
       9: v <- v*cosh0*2*expAngle/(expAngle^2 + 1); ...
      10: v2 <- active*(0.5*(expAngle - 1/expAngle)*...
      11: v <- v*exp(-0.5*dt*(v1 + kT*v2))
      12: v1 <- active*(v1 + 0.5*dt*(m*v^2 - kT)*invQ)
      13: x <- x + 0.5*dt*v
      14: v <- v + 0.5*dt*f/m

    Force-first scheme

    >>> integrator_ff = xps.integrators.MassiveGGMTIntegrator(
    ...     temperature=300 * unit.kelvin,
    ...     timeConstant=40 * unit.femtoseconds,
    ...     stepSize=2 * unit.femtoseconds,
    ...     forceFirst=True
    ... )
    >>> model = testsystems.AlanineDipeptideVacuum()
    >>> try:
    ...     integrator_ff.registerWithSystem(model.system, False)
    ... except ValueError as e:
    ...     print(e)
    Massive GGMT integrators do not support constraints.
    """

    @preprocess_args
    def __init__(  # noqa: PLR0913
        self,
        temperature: mmunit.Quantity,
        timeConstant: mmunit.Quantity,
        stepSize: mmunit.Quantity,
        *,
        stochastic: bool = False,
        forceFirst: bool = False,
        halleyIterations: int = 3,
    ) -> None:
        if halleyIterations < 1:
            raise ValueError("The number of Halley iterations must be at least 1.")
        super().__init__(stepSize)
        self._forceFirst = forceFirst
        self._halley_iterations = halleyIterations
        self._init_temperature(temperature)
        self._init_time_constant(timeConstant)
        self._init_friction_coefficient(1 / timeConstant if stochastic else 0.0)
        self._stochastic = stochastic
        self._add_variables()
        self.addUpdateContextState()
        self._add_boost(1 if forceFirst else 0.5)
        self._add_translation(0.5)
        self._add_thermostat()
        self._add_translation(0.5)
        if not forceFirst:
            self._add_boost(0.5)

    @preprocess_args
    def setFrictionCoefficient(self, value: mmunit.Quantity | float) -> None:
        """
        Set the friction coefficient.

        Parameters
        ----------
        value
            The friction coefficient.

        Raises
        ------
        ValueError
            If the integrator is deterministic and the value is not zero.
        """
        self._init_friction_coefficient(value)
        friction = self.getFrictionCoefficient().value_in_unit(mmunit.picosecond**-1)
        if not self._stochastic and friction != 0:
            self._init_friction_coefficient(0.0)
            raise ValueError(
                "The friction coefficient of a deterministic integrator must be zero."
            )
        self._update_global_variables()

    def _add_variables(self) -> None:
        self.addGlobalVariable("kT", 0)
        self.addGlobalVariable("invQ", 0)
        self.addGlobalVariable("invQ2", 0)
        self.addPerDofVariable("v1", 0)
        self.addPerDofVariable("v2", 0)
        self._add_dof_mask()
        for name in ("meanAnomaly", "angle", "expAngle"):
            self.addPerDofVariable(name, 0)
        if self._stochastic:
            self.addGlobalVariable("a", 0)
            self.addPerDofVariable("normal", 0)
            self.addPerDofVariable("scaleSq", 1)
        self._update_global_variables()

    def _update_global_variables(self) -> None:
        tau = self.getTimeConstant()
        kt = mmunit.MOLAR_GAS_CONSTANT_R * self.getTemperature()
        self.setGlobalVariableByName("kT", kt)
        self.setGlobalVariableByName("invQ", 1 / (kt * tau**2))
        self.setGlobalVariableByName("invQ2", 3 / (8 * kt**3 * tau**2))
        if self._stochastic:
            friction = self.getFrictionCoefficient()
            dt = self.getStepSize()
            self.setGlobalVariableByName("a", np.exp(-friction * dt))

    def _add_translation(self, fraction: float) -> None:
        self.addComputePerDof("x", f"x + {fraction}*dt*v")

    def _add_boost(self, fraction: float) -> None:
        self.addComputePerDof("v", f"v + {fraction}*dt*f/m")

    def _add_v1_boost(self, fraction: float) -> None:
        self.addComputePerDof(
            "v1", self._masked(f"v1 + {fraction}*dt*(m*v^2 - kT)*invQ")
        )

    def _add_coulomb_step(self) -> None:
        """Propagate the fourth-moment sub-dynamics over a full step.

        A constant kick on ``v2`` over half a step precedes and follows the exact
        Coulomb flow, whose final hyperbolic angle solves a Kepler equation by Halley
        iterations. In the stochastic version, the rescaling at the center of the flow
        only changes the orbit energy by ``scaleSq``, so it enters the same equation.
        """
        add = self.addComputePerDof
        kick = "0.5*dt*kT^2*invQ2"
        orbit = (
            "sinh0 = wCoulomb*sqrt(zCoulomb/(2*strength));"
            f" wCoulomb = (2/sqrt(3))*kT*(v2 - {kick});"
            " zCoulomb = sqrt(3)*kT/(m*vFloor^2);"
            f" vFloor = {_V_FLOOR}; strength = {_COULOMB_STRENGTH}"
        )
        if self._stochastic:
            # Both normal variates are drawn in adjacent computations to avoid an
            # OpenMM 8.6.1 bug in merged per-DOF kernels.
            add("normal", "gaussian")
            add(
                "scaleSq",
                "((sqrt(a*normSq) + sqrt((1 - a)*kT)*normal)^2"
                " + (1 - a)*kT*gaussian^2)/normSq;"
                " normSq = m*vFloor^2 + v2Kicked^2/invQ2;"
                f" v2Kicked = v2 - {kick}; vFloor = {_V_FLOOR}",
            )
            time_term = (
                "0.5*dt*energy*sqrt(2*energy)*(1 + scaleSq*sqrt(scaleSq))/strength"
            )
            scale = "sqrt(scaleSq)*"
        else:
            time_term = "dt*energy*sqrt(2*energy)/strength"
            scale = ""
        add(
            "meanAnomaly",
            f"(2*step(sinh0) - 1)*log(abs(sinh0) + cosh0) + sinh0*cosh0 + {time_term};"
            " cosh0 = sqrt(sinh0^2 + 1);"
            f" energy = 0.5*wCoulomb^2 + strength/zCoulomb; {orbit}",
        )
        guess = (
            "(2*step(meanAnomaly) - 1)"
            "*min(0.5*target, 0.5*log(2*target + sqrt(4*target^2 + 1)));"
            " target = abs(meanAnomaly)"
        )

        def correction(angle: str) -> str:
            return (
                "residual*coshA/(2*coshA^3 - residual*sinhA);"
                f" residual = {angle} + sinhA*coshA - meanAnomaly;"
                " sinhA = 0.5*(expA - 1/expA); coshA = 0.5*(expA + 1/expA);"
                f" expA = exp({angle})"
            )

        add("angle", f"start - {correction('start')}; start = {guess}")
        if self._halley_iterations == 1:
            add("expAngle", "exp(angle)")
        else:
            for _ in range(self._halley_iterations - 2):
                add("angle", f"angle - {correction('angle')}")
            # The last correction is applied to exp(angle) by a Taylor factor.
            add(
                "expAngle",
                "expA*(1 - delta + delta^2/2 - delta^3/6);"
                f" delta = {correction('angle')}",
            )
        add(
            "v",
            f"v*{scale}cosh0*2*expAngle/(expAngle^2 + 1);"
            f" cosh0 = sqrt(sinh0^2 + 1); {orbit}",
        )
        add(
            "v2",
            self._masked(
                f"0.5*(expAngle - 1/expAngle)*{_V_FLOOR}*sqrt(m*invQ2) - {kick}"
            ),
        )

    def _add_v_scaling(self, fraction: float) -> None:
        self.addComputePerDof("v", f"v*exp(-{fraction}*dt*(v1 + kT*v2))")

    def _add_thermostat(self) -> None:
        self._add_v1_boost(0.5)
        self._add_v_scaling(0.5)
        self._add_coulomb_step()
        self._add_v_scaling(0.5)
        self._add_v1_boost(0.5)

    def registerWithSystem(self, system: mm.System, isExtension: bool) -> None:
        if system.getNumConstraints() > 0:
            raise ValueError("Massive GGMT integrators do not support constraints.")
        self._set_dof_mask(system, isExtension)
