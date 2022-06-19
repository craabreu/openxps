"""
.. module:: utils
   :platform: Unix, Windows, macOS
   :synopsis: Extended Phase-Space Methods with OpenMM

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

"""

from __future__ import annotations

import ast
from typing import Union

from openmm import unit

UnitOrStr = Union[unit.Unit, str]
QuantityOrFloat = Union[unit.Quantity, float]


class _add_unit_module(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> Union[ast.Attribute, ast.Name]:
        if node.id == 'unit':
            return node
        else:
            mod = ast.Name(id='unit', ctx=ast.Load())
            return ast.Attribute(value=mod, attr=node.id, ctx=ast.Load())


def stdval(quantity: QuantityOrFloat) -> QuantityOrFloat:
    """
    Returns the numerical value of a quantity in a unit of measurement compatible with OpenMM's
    standard unit system (mass in Da, distance in nm, time in ps, temperature in K, energy in
    kJ/mol, angle in rad).

    """

    if unit.is_quantity(quantity):
        return quantity.value_in_unit_system(unit.md_unit_system)
    else:
        return quantity


def str2unit(string: str) -> unit.Unit:
    """
    Converts a string into a proper OpenMM unit of measurement.

    """

    tree = _add_unit_module().visit(ast.parse(string, mode='eval'))
    result = eval(compile(ast.fix_missing_locations(tree), '', mode='eval'))
    return result
