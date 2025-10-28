"""
.. module:: openxps.utils
   :platform: Linux, MacOS, Windows
   :synopsis: Utility functions for OpenXPS.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import functools
import inspect
import re
import typing as t

import cvpack
from cvpack.serialization import Serializable
from cvpack.units import Quantity, Unit
from openmm import unit as mmunit

#: The separator used to split checkpoint and XML strings into physical and extension
#: parts.
STRING_SEPARATOR = "\f\f"
BINARY_SEPARATOR = b"::SdXN3dO::"

LEPTON_FUNCTIONS = frozenset(
    {
        "sqrt",
        "exp",
        "log",
        "sin",
        "cos",
        "sec",
        "csc",
        "tan",
        "cot",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "tanh",
        "erf",
        "erfc",
        "min",
        "max",
        "abs",
        "floor",
        "ceil",
        "step",
        "delta",
        "select",
    }
)


def preprocess_args(func: t.Callable) -> t.Callable:
    """
    A decorator that converts instances of unserializable classes to their
    serializable counterparts.

    Parameters
    ----------
        func
            The function to be decorated.

    Returns
    -------
        The decorated function.

    Example
    -------
    >>> from openxps.utils import preprocess_args
    >>> from cvpack import units
    >>> from openmm import unit as mmunit
    >>> @preprocess_args
    ... def function(data):
    ...     return data
    >>> assert isinstance(function(mmunit.angstrom), units.Unit)
    >>> assert isinstance(function(5 * mmunit.angstrom), units.Quantity)
    >>> seq = [mmunit.angstrom, mmunit.nanometer]
    >>> assert isinstance(function(seq), list)
    >>> assert all(isinstance(item, units.Unit) for item in function(seq))
    >>> dct = {"length": 3 * mmunit.angstrom, "time": 2 * mmunit.picosecond}
    >>> assert isinstance(function(dct), dict)
    >>> assert all(isinstance(item, units.Quantity) for item in function(dct).values())
    """
    signature = inspect.signature(func)

    def convert(data: t.Any) -> t.Any:
        if isinstance(data, mmunit.Quantity):
            return Quantity(data)
        if isinstance(data, mmunit.Unit):
            return Unit(data)
        if isinstance(data, t.Sequence) and not isinstance(data, str):
            return type(data)(map(convert, data))
        if isinstance(data, dict):
            return type(data)((key, convert(value)) for key, value in data.items())
        return data

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        for name, data in bound.arguments.items():
            bound.arguments[name] = convert(data)
        return func(*bound.args, **bound.kwargs)

    return wrapper


class Function(Serializable):
    """A function of dynamical variables and global parameters.

    Parameters
    ----------
    name
        The name of the function.
    expression
        The expression of the function.
    **given_parameters
        The given parameters of the function.

    Examples
    --------
    >>> from copy import copy
    >>> from openxps.utils import Function
    >>> f1 = Function("f", "a*x^2", a=1.0)
    >>> f1
    Function("f(x)=a*x^2", a=1.0)
    >>> f2 = Function("g", "exp(-x)*cos(2*pi*y)", pi=3.14159)
    >>> f2
    Function("g(x, y)=exp(-x)*cos(2*pi*y)", pi=3.14159)
    >>> copy(f1)
    Function("f(x)=a*x^2", a=1.0)
    """

    @preprocess_args
    def __init__(
        self, name: str, expression: str, **given_parameters: mmunit.Quantity
    ) -> None:
        self._name = name
        self._expression = expression
        variables, parameters = self._parseDependencies(given_parameters)
        self._variables = variables
        self._parameters = {name: given_parameters[name] for name in parameters}

    def __repr__(self) -> str:
        variables = ", ".join(sorted(self._variables))
        parameters = ", ".join(f"{k}={v}" for k, v in sorted(self._parameters.items()))
        return f'Function("{self._name}({variables})={self._expression}", {parameters})'

    def __copy__(self) -> "Function":
        new = Function.__new__(Function)
        new.__setstate__(self.__getstate__())
        return new

    def __getstate__(self) -> dict[str, t.Any]:
        return {
            "name": self._name,
            "expression": self._expression,
            "variables": self._variables,
            "parameters": self._parameters,
        }

    def __setstate__(self, state: dict[str, t.Any]) -> None:
        self._name = state["name"]
        self._expression = state["expression"]
        self._variables = state["variables"]
        self._parameters = state["parameters"]

    def _parseDependencies(
        self,
        given_parameters: dict[str, mmunit.Quantity],
    ) -> tuple[set[str], set[str]]:
        given_parameters = set(given_parameters.keys())
        dependencies = (
            set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", self._expression))
            - LEPTON_FUNCTIONS
        )
        variables = dependencies - given_parameters
        parameters = dependencies & given_parameters
        if missing := dependencies - (variables | parameters):
            raise ValueError(
                f"Function {self._name} has unknown dependencies: {missing}"
            )
        return variables, parameters

    def getName(self) -> str:
        """Return the name of the function."""
        return self._name

    def getExpression(self) -> str:
        """Return the expression string of the function."""
        return self._expression

    def getVariables(self) -> set[str]:
        """Return the set of variable names in the function."""
        return self._variables

    def getParameters(self) -> dict[str, mmunit.Quantity]:
        """Return the dictionary of parameter names and their values."""
        return self._parameters

    def createCollectiveVariable(
        self,
        all_variables: list[str],
    ) -> cvpack.AtomicFunction:
        """
        Create a collective variable from the function.

        Parameters
        ----------
        all_variables
            The list of all variables in the system.

        Returns
        -------
        cvpack.AtomicFunction
            The collective variable object.

        Examples
        --------
        >>> from openxps.utils import Function
        >>> import openmm as mm
        >>> from math import exp, cos, pi
        >>> fn = Function("g", "exp(-x)*cos(2*pi*y)", pi=3.14159)
        >>> cv = fn.createCollectiveVariable(["x", "y"])
        >>> system = mm.System()
        >>> for _ in range(2):
        ...     _ = system.addParticle(1.0)
        >>> cv.addToSystem(system)
        >>> context = mm.Context(system, mm.VerletIntegrator(0.0))
        >>> x, y = 2, 1
        >>> exp(-x)*cos(2*pi*y)
        0.13533528...
        >>> context.setPositions([mm.Vec3(x, 0.0, 0.0), mm.Vec3(y, 0.0, 0.0)])
        >>> context.getState(getEnergy=True).getPotentialEnergy()
        0.13533528... kJ/mol
        """
        return cvpack.AtomicFunction(
            function=";".join(
                [self._expression]
                + [f"{variable}=x{i + 1}" for i, variable in enumerate(self._variables)]
            ),
            unit=mmunit.kilojoule_per_mole,
            groups=[[all_variables.index(variable) for variable in self._variables]],
            name=self._name,
            **self._parameters,
        )


Function.registerTag("!openxps.utils.Function")
