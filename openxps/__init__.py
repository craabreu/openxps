"""
Extended Phase-Space Methods with OpenMM
Extended Phase-Space Methods for Free Energy Calculation with OpenMM
"""

# Handle versioneer
from ._version import get_versions
from .io import *  # noqa: F401, F403
# Add imports here
from .openxps import *  # noqa: F401, F403
from .testmodels import *  # noqa: F401, F403

versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
