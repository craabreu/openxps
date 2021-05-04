"""
Extended Phase-Space Methods with OpenMM
Extended Phase-Space Methods for Free Energy Calculation with OpenMM
"""

# Add imports here
from .openxps import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
