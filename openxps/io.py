"""
.. module:: io
   :platform: Unix, Windows, macOS
   :synopsis: Extended Phase-Space Methods with OpenMM

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

"""

import yaml


def serialize(object, file):
    """
    Serializes a openxps object.

    """
    dump = yaml.dump(object, Dumper=yaml.CDumper)
    if isinstance(file, str):
        with open(file, 'w') as f:
            f.write(dump)
    else:
        file.write(dump)


def deserialize(file):
    """
    Deserializes a openxps object.

    """
    if isinstance(file, str):
        with open(file, 'r') as f:
            object = yaml.load(f.read(), Loader=yaml.CLoader)
    else:
        object = yaml.load(file.read(), Loader=yaml.CLoader)
    return object
