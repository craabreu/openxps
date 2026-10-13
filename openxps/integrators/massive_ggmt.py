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

    This implementation optionally introduces stochasticity by rescaling each velocity
    and its fourth-moment thermostat variable by a common random factor, in the manner
    of the stochastic velocity rescaling of :cite:`Bussi2007` (see Notes).

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
    The cubic friction term of the GGMT equations has a finite-time singularity when
    propagated in isolation, which appears as a negative argument under a square root
    in Trotter-factorized integrators. This implementation propagates the cubic
    friction together with the fourth-moment thermostat momentum, whose joint
    dynamics is Hamiltonian in suitable variables, by splitting that Hamiltonian into
    an exactly solvable repulsive Coulomb problem and a constant kick. The Coulomb
    flow is obtained from a hyperbolic Kepler equation, solved by a fixed number of
    Halley iterations started from an upper bound of its root, with the hyperbolic
    functions computed from a single exponential. The exponential velocity scaling is
    applied on both sides of the Coulomb sub-step, so that a single Kepler equation is
    solved per step. The resulting step is free of the singularity for any step size.

    In the stochastic version, the vector :math:`(p/\sqrt{m}, p_{\eta_2}/\sqrt{Q_2})`
    is rescaled along its own direction, with its squared norm :math:`\phi` following
    :math:`d\phi = \gamma(2kT - \phi)dt + 2\sqrt{\gamma kT \phi}\,dW`, where
    :math:`\gamma` is the friction coefficient. This rescaling preserves the canonical
    distribution and leaves the phase of the Coulomb orbit unchanged, so it is applied
    exactly at the center of the step without a second Kepler equation.

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
      v1, v2, active, T1, th, eth
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
       5: T1 <- ...
       6: th <- t0 - g*ch/(2*ch*ch*ch - g*sh); g = t0 + sh*ch - T1; ...
       7: th <- th - g*ch/(2*ch*ch*ch - g*sh); g = th + sh*ch - T1; ...
       8: eth <- e*(1 - d + d*d/2 - d*d*d/6); ...
       9: v <- v*sqrt(q*q + 1)*2*eth/(eth*eth + 1); q = w*sqrt(z/(2*c)); ...
      10: v2 <- active*(0.5*(eth - 1/eth)*max(abs(v), sqrt(1e-30*kT/m))*...
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
        for name in ("T1", "th", "eth"):
            self.addPerDofVariable(name, 0)
        if self._stochastic:
            self.addGlobalVariable("a", 0)
            self.addPerDofVariable("R1", 0)
            self.addPerDofVariable("lam2", 1)
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
        """Propagate the (z, w) sub-dynamics over a full step (Coulomb splitting).

        The constant kick on ``v2`` is applied over half a step before and after the
        exact flow of the repulsive Coulomb Hamiltonian ``w^2/2 + c/z``. With
        ``q = sinh(th0) = w*sqrt(z/(2c))`` and ``th`` the final hyperbolic angle, the
        flow satisfies ``z0/z = (q^2 + 1)/cosh(th)^2`` and ``w = sinh(th)*sqrt(2c/z)``,
        so ``v`` and ``v2`` follow from ``th`` alone. The Kepler equation
        ``th + sinh(2 th)/2 = T1`` is solved by Halley iterations started from the
        smaller of its upper bounds ``T1/2`` and ``asinh(2 T1)/2``, with the hyperbolic
        functions written in terms of a single exponential. With two or more
        iterations, the last correction is applied to ``exp(th)`` by a cubic Taylor
        factor, so that no further exponential is needed. The floor on ``|v|`` avoids
        an indeterminate product at the fixed point ``v = 0``.

        In the stochastic version, the rescaling at the center of the Coulomb flow
        multiplies the orbit energy by ``lam2`` at a fixed ``th``, so the Kepler
        equation gains the time term of the second half on the new orbit and ``v``
        gains a factor ``sqrt(lam2)``.
        """
        add = self.addComputePerDof
        kick = "0.5*dt*kT^2*invQ2"
        state = (
            f"q = w*sqrt(z/(2*c)); w = (2/sqrt(3))*kT*(v2 - {kick});"
            f" z = sqrt(3)*kT/(m*vf*vf); vf = {_V_FLOOR}; c = {_COULOMB_STRENGTH}"
        )
        if self._stochastic:
            # Both normal variates are drawn in adjacent computations: in OpenMM
            # 8.6.1, a gaussian draw that follows a merged per-DOF computation
            # without one, after an earlier gaussian draw, reads past the prepared
            # random numbers. The squared norm is that of the state after the first
            # kick, which the Coulomb flow conserves.
            add("R1", "gaussian")
            add(
                "lam2",
                "((sqrt(a*phi) + sqrt((1 - a)*kT)*R1)^2 + (1 - a)*kT*gaussian^2)/phi;"
                f" phi = m*vf*vf + u*u/invQ2; u = v2 - {kick}; vf = {_V_FLOOR}",
            )
            time_term = "0.5*dt*E*sqrt(2*E)*(1 + lam2*sqrt(lam2))/c"
            scale = "sqrt(lam2)*"
        else:
            time_term = "dt*E*sqrt(2*E)/c"
            scale = ""
        add(
            "T1",
            f"(2*step(q) - 1)*log(abs(q) + r) + q*r + {time_term};"
            f" r = sqrt(q*q + 1); E = 0.5*w*w + c/z; {state}",
        )
        guess = (
            "(2*step(T1) - 1)*min(0.5*T, 0.5*log(2*T + sqrt(4*T*T + 1))); T = abs(T1)"
        )
        correction = (
            "g*ch/(2*ch*ch*ch - g*sh); g = {angle} + sh*ch - T1;"
            " sh = 0.5*(e - 1/e); ch = 0.5*(e + 1/e); e = exp({angle})"
        )
        n = self._halley_iterations
        if n == 1:
            add("th", f"t0 - {correction.format(angle='t0')}; t0 = {guess}")
            add("eth", "exp(th)")
        else:
            add("th", f"t0 - {correction.format(angle='t0')}; t0 = {guess}")
            for _ in range(n - 2):
                add("th", f"th - {correction.format(angle='th')}")
            add(
                "eth",
                f"e*(1 - d + d*d/2 - d*d*d/6); d = {correction.format(angle='th')}",
            )
        add("v", f"v*{scale}sqrt(q*q + 1)*2*eth/(eth*eth + 1); {state}")
        add(
            "v2",
            self._masked(f"0.5*(eth - 1/eth)*{_V_FLOOR}*sqrt(m*invQ2) - {kick}"),
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
