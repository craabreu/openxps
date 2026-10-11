"""
Unit and regression test for the openxps package.

"""

# Import package, test suite, and other packages as needed
import sys
from importlib import metadata


def test_openxps_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "openxps" in sys.modules


def test_declared_runtime_dependencies():
    """Every third-party package imported at import time is declared."""
    requires = [
        r.split(";")[0].split(" ")[0].lower() for r in metadata.requires("openxps")
    ]
    for package in (
        "cvpack",
        "lightning",
        "numpy",
        "openmm",
        "pandas",
        "pyyaml",
        "torch",
        "typing-extensions",
    ):
        assert package in requires, f"{package} is not declared in pyproject.toml"


def test_declared_python_floor():
    """PEP 604 annotations at import time require Python >= 3.10."""
    requires_python = metadata.metadata("openxps")["Requires-Python"]
    assert requires_python == ">=3.10"
