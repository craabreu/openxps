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


@add_property("temperature", mmunit.kelvin)
@add_property("time constant", mmunit.picosecond)
@add_property("friction coefficient", 1 / mmunit.picosecond)
class MassiveGGMTIntegrator(IntegratorMixin, mm.CustomIntegrator):
    r"""A massive, second- and fourth-moment GGMT integrator :cite:`Liu2000`.

    The massive GGMT integrator is effective at maintaining temperature control under
    steady-state conditions, such as in d-AFED/TAMD :cite:`Maragliano2006,Abrams2008`
    or UFED :cite:`Chen2012` simulations.

    This implementation optionally introduces stochasticity by applying Langevin-type
    noise to the variables controlling the second- and fourth-moment of the velocity
    distribution.

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
        If True, the integrator will apply stochasticity to the variables controlling
        the second- and fourth-moment of the velocity distribution.
    forceFirst
        If True, the integrator will apply a force-first scheme. Otherwise, it will
        apply a symmetric operator splitting scheme.
    newtonIterations
        The number of Newton iterations used to solve the time equation of the
        Coulomb sub-step (see Notes). The default suffices for double precision.

    Notes
    -----
    The cubic friction term of the GGMT equations has a finite-time singularity when
    propagated in isolation, which appears as a negative argument under a square root
    in Trotter-factorized integrators. This implementation propagates the cubic
    friction together with the fourth-moment thermostat momentum, whose joint
    dynamics is Hamiltonian in suitable variables, by splitting that Hamiltonian into
    an exactly solvable repulsive Coulomb problem and a constant kick. The resulting
    step is free of the singularity for any step size.

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
      v1, v2, z, w, E, s, zmin, th, T1
    Global variables:
      kT = 2.494338...
      invQ = 40.090...
      invQ2 = 2.416...
    Computation steps:
       0: allow forces to update the context state
       1: v <- v + 0.5*dt*f/m
       2: x <- x + 0.5*dt*v
       3: v1 <- v1 + 0.5*dt*(m*v^2 - kT)*invQ
       4: v2 <- v2 - 0.25*dt*kT^2*invQ2
       5: z <- sqrt(3)*kT/max(m*v*v, 1e-30*kT)
    ...
      23: v <- v*exp(-1*dt*(v1 + kT*v2))
    ...
      43: v1 <- v1 + 0.5*dt*(m*v^2 - kT)*invQ
      44: x <- x + 0.5*dt*v
      45: v <- v + 0.5*dt*f/m

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
        newtonIterations: int = 5,
    ) -> None:
        if newtonIterations < 1:
            raise ValueError("The number of Newton iterations must be at least 1.")
        super().__init__(stepSize)
        self._forceFirst = forceFirst
        self._newton_iterations = newtonIterations
        self._init_temperature(temperature)
        self._init_time_constant(timeConstant)
        self._init_friction_coefficient(1 / timeConstant if stochastic else 0.0)
        self._stochastic = stochastic
        self._add_variables()
        self.addUpdateContextState()
        self._add_boost(1 if forceFirst else 0.5)
        self._add_translation(0.5)
        self._add_thermostat(1)
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
        for name in ("z", "w", "E", "s", "zmin", "th", "T1"):
            self.addPerDofVariable(name, 0)
        if self._stochastic:
            self.addGlobalVariable("a", 0)
            self.addGlobalVariable("b1", 0)
            self.addGlobalVariable("b2", 0)
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
            a = np.exp(-friction * dt)
            b1 = np.sqrt(1 - a**2) / tau
            b2 = b1 * np.sqrt(3 / 8) / kt
            self.setGlobalVariableByName("a", a)
            self.setGlobalVariableByName("b1", b1)
            self.setGlobalVariableByName("b2", b2)

    def _add_translation(self, fraction: float) -> None:
        self.addComputePerDof("x", f"x + {fraction}*dt*v")

    def _add_boost(self, fraction: float) -> None:
        self.addComputePerDof("v", f"v + {fraction}*dt*f/m")

    def _add_v1_boost(self, fraction: float) -> None:
        self.addComputePerDof("v1", f"v1 + {fraction}*dt*(m*v^2 - kT)*invQ")

    def _add_v2_boost(self, fraction: float) -> None:
        c = _COULOMB_STRENGTH
        constant_kick = f"v2 - {fraction / 2}*dt*kT^2*invQ2"
        add = self.addComputePerDof
        add("v2", constant_kick)
        add("z", "sqrt(3)*kT/max(m*v*v, 1e-30*kT)")
        add("w", "(2/sqrt(3))*kT*v2")
        add("E", f"0.5*w*w + c/z; c={c}")
        add("zmin", f"c/E; c={c}")
        add("s", "sqrt(2*E)")
        add(
            "th",
            f"(2*step(q) - 1)*log(abs(q) + sqrt(q*q + 1)); q=w*sqrt(z/(2*c)); c={c}",
        )
        add("T1", f"th + 0.5*sinh(2*th) + s*{fraction}*dt/zmin")
        add(
            "th",
            "0.5*select(step(q - 1e6), log(2*q), log(q + sqrt(q*q + 1))); q=2*abs(T1)",
        )
        for _ in range(self._newton_iterations):
            add("th", "th - (th + 0.5*sinh(2*th) - abs(T1))/(1 + cosh(2*th))")
        add("th", "(2*step(T1) - 1)*th")
        add("v", "v*sqrt(z/znew); znew=zmin*cosh(th)^2")
        add("w", "s*tanh(th)")
        add("v2", "(sqrt(3)/2)*w/kT")
        add("v2", constant_kick)

    def _add_v_scaling(self, fraction: float) -> None:
        self.addComputePerDof("v", f"v*exp(-{fraction}*dt*(v1 + kT*v2))")

    def _add_thermostat(self, fraction: float) -> None:
        self._add_v1_boost(0.5 * fraction)
        self._add_v2_boost(0.5 * fraction)
        if self._stochastic:
            self._add_v_scaling(0.5 * fraction)
            self.addComputePerDof("v1", "a*v1 + b1*gaussian")
            self.addComputePerDof("v2", "a*v2 + b2*gaussian")
            self._add_v_scaling(0.5 * fraction)
        else:
            self._add_v_scaling(fraction)
        self._add_v2_boost(0.5 * fraction)
        self._add_v1_boost(0.5 * fraction)

    def registerWithSystem(self, system: mm.System, isExtension: bool) -> None:
        if system.getNumConstraints() > 0:
            raise ValueError("Massive GGMT integrators do not support constraints.")
