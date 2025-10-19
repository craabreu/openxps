"""
.. module:: openxps.serializable
   :platform: Linux, MacOS, Windows
   :synopsis:

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import functools
import inspect
import typing as t

from cvpack.units import Quantity, Unit
from openmm import unit as mmunit


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
