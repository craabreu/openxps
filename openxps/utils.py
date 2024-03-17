"""
.. module:: openxps.utils
   :platform: Linux, Windows, macOS
   :synopsis: Utilities for OpenXPS

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import yaml


def register_serializer(cls: type, tag: str) -> None:
    """
    Register a class for serialization and deserialization with PyYAML.

    Parameters
    ----------
    cls
        The class to be registered for serialization and deserialization.
    tag
        The YAML tag to be used for this class.
    """
    cls.yaml_tag = tag
    yaml.SafeDumper.add_representer(cls, cls.to_yaml)
    yaml.SafeLoader.add_constructor(tag, cls.from_yaml)
