"""
.. module:: utils
   :platform: Unix, Windows, macOS
   :synopsis: Extended Phase-Space Methods with OpenMM

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

"""
import ast
from openmm import unit
from typing import Union


Quantity = Union[unit.Quantity, float]


class _add_unit_module(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> Union[ast.Attribute, ast.Name]:
        if node.id == 'unit':
            return node
        else:
            mod = ast.Name(id='unit', ctx=ast.Load())
            return ast.Attribute(value=mod, attr=node.id, ctx=ast.Load())


def in_md_units(quantity: Union[unit.Quantity, float]) -> Union[unit.Quantity, float]:
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
    Converts a string into a proper unit of measurement.

    """
    tree = _add_unit_module().visit(ast.parse(string, mode='eval'))
    result = eval(compile(ast.fix_missing_locations(tree), '', mode='eval'))
    return result
