"""
Extended Phase-Space Methods with OpenMM
"""

from . import bounds  # noqa: F401
from ._version import __version__  # noqa: F401
from .bounds import CircularBounds, NoBounds, PeriodicBounds, ReflectiveBounds
from .context import ExtendedSpaceContext  # noqa: F401
from .couplings import (
    CollectiveVariableCoupling,  # noqa: F401
    HarmonicCoupling,  # noqa: F401
    InnerProductCoupling,  # noqa: F401
)
from .dynamical_variable import DynamicalVariable  # noqa: F401
from .extension_writer import ExtensionWriter  # noqa: F401
from .integrator import LockstepIntegrator, SplitIntegrator  # noqa: F401
from .integrators import (
    CSVRIntegrator,
    MassiveGGMTIntegrator,
    RegulatedNHLIntegrator,
    SymmetricLangevinIntegrator,
    SymmetricVerletIntegrator,
)
from .metadynamics import (  # noqa: F401
    ExtendedSpaceBiasVariable,
    ExtendedSpaceMetadynamics,
)
from .simulation import ExtendedSpaceSimulation  # noqa: F401
from .system import ExtendedSpaceSystem  # noqa: F401

__all__ = [
    "bounds",
    "CircularBounds",
    "NoBounds",
    "PeriodicBounds",
    "ReflectiveBounds",
    "CollectiveVariableCoupling",
    "HarmonicCoupling",
    "InnerProductCoupling",
    "ExtendedSpaceContext",
    "ExtendedSpaceSystem",
    "ExtensionWriter",
    "DynamicalVariable",
    "ExtendedSpaceMetadynamics",
    "ExtendedSpaceBiasVariable",
    "ExtendedSpaceSimulation",
    "LockstepIntegrator",
    "SplitIntegrator",
    "SymmetricLangevinIntegrator",
    "CSVRIntegrator",
    "MassiveGGMTIntegrator",
    "RegulatedNHLIntegrator",
    "SymmetricVerletIntegrator",
    "ForceMatchingRegressor",
]


def __getattr__(name: str):
    # ForceMatchingRegressor needs the optional PyTorch, Lightning and pandas, so its
    # module is imported only when the class is first used.
    if name == "ForceMatchingRegressor":
        from .regression import ForceMatchingRegressor  # noqa: PLC0415

        return ForceMatchingRegressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
