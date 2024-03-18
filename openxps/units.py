"""
.. module:: openxps.units
   :platform: Linux, MacOS, Windows
   :synopsis: Units of measurement for OpenXPS.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import ast
import typing as t
import functools
import inspect

from openmm import unit as mmunit

from .serializable import Serializable



class Unit(mmunit.Unit, Serializable, ast.NodeTransformer):
    """
    Extension of the OpenMM Unit class to allow serialization and deserialization.

    Parameters
    ----------
    data
        The data to be used to create the unit.
    """
    def __init__(self, data: t.Union[str, mmunit.Unit, dict]) -> None:
        if isinstance(data, str):
            expression = self.visit(ast.parse(data, mode="eval"))
            code = compile(ast.fix_missing_locations(expression), "", mode="eval")
            data = eval(code)  # pylint: disable=eval-used
        if isinstance(data, mmunit.Unit):
            data = dict(data.iter_base_or_scaled_units())
        super().__init__(data)

    def __repr__(self) -> str:
        return self.get_symbol()

    def __getstate__(self) -> str:
        return str(self)

    def __setstate__(self, state: str) -> None:
        self.__init__(state)

    def visit_Name(  # pylint: disable=invalid-name
        self, node: ast.Name
    ) -> ast.Attribute:
        """
        Visit a Name node and transform it into an Attribute node.

        Parameters
        ----------
        node
            The node to be visited and transformed.
        """
        return ast.Attribute(
            value=ast.Name(id="mmunit", ctx=ast.Load()), attr=node.id, ctx=ast.Load()
        )


Unit.register_tag("!openxps.units.Unit")


class Quantity(mmunit.Quantity, Serializable):
    """
    Extension of the OpenMM Quantity class to allow serialization and deserialization.
    """
    def __init__(self, *args: t.Any) -> None:
        if len(args) == 1 and mmunit.is_quantity(args[0]):
            super().__init__(args[0].value_in_unit(args[0].unit), Unit(args[0].unit))
        else:
            super().__init__(*args)

    def __repr__(self):
        return str(self)

    def __getstate__(self) -> t.Dict[str, t.Any]:
        return {
            "value": self.value,
            "unit": str(self.unit),
        }

    def __setstate__(self, keywords: t.Dict[str, t.Any]) -> None:
        self.__init__(keywords["value"], Unit(keywords["unit"]))

    @property
    def value(self) -> t.Any:
        return self._value


Quantity.register_tag("!openxps.units.Quantity")


def preprocess_units(func: t.Callable) -> t.Callable:
    """
    A decorator that converts instances of openmm.unit.Unit and openmm.unit.Quantity
    into openxps.units.Unit and openxps.units.Quantity, respectively.

    Parameters
    ----------
        func
            The function to be decorated.

    Returns
    -------
        The decorated function.

    Example
    -------
    >>> from openxps import units
    >>> from openmm import unit as mmunit
    >>> @units.preprocess_units
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
        if isinstance(data, t.Sequence):
            return type(data)(map(convert, data))
        if isinstance(data, t.Dict):
            return type(data)((key, convert(value)) for key, value in data.items())
        return data

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        for name, data in bound.arguments.items():
            bound.arguments[name] = convert(data)
        return func(*bound.args, **bound.kwargs)

    return wrapper
