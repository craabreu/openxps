"""
.. module:: openxps.serializable
   :platform: Linux, MacOS, Windows
   :synopsis:

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import yaml


class Serializable(yaml.YAMLObject):
    """
    A mixin class that allows serialization and deserialization of objects with PyYAML.
    """

    @classmethod
    def register_tag(cls, tag: str) -> None:
        """
        Register a class for serialization and deserialization with PyYAML.

        Parameters
        ----------
        tag
            The YAML tag to be used for this class.
        """
        cls.yaml_tag = tag
        yaml.SafeDumper.add_representer(cls, cls.to_yaml)
        yaml.SafeLoader.add_constructor(tag, cls.from_yaml)
