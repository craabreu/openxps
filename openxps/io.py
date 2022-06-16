"""
.. module:: io
   :platform: Unix, Windows, macOS
   :synopsis: Extended Phase-Space Methods with OpenMM

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

"""

from typing import Any, Union, IO
import yaml


def serialize(object: Any, file: Union[IO, str]):
    """
    Serializes a openxps object.

    """
    dump = yaml.dump(object, Dumper=yaml.CDumper)
    if isinstance(file, str):
        with open(file, 'w') as f:
            f.write(dump)
    else:
        file.write(dump)


def deserialize(file: Union[IO, str]) -> Any:
    """
    Deserializes a openxps object.

    """
    if isinstance(file, str):
        with open(file, 'r') as f:
            object = yaml.load(f.read(), Loader=yaml.CLoader)
    else:
        object = yaml.load(file.read(), Loader=yaml.CLoader)
    return object
