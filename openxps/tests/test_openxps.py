"""
Unit and regression test for the openxps package.

"""

# Import package, test suite, and other packages as needed
import re
import subprocess
import sys
from importlib import metadata

import pytest

import openxps


def test_openxps_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "openxps" in sys.modules


REGRESSION_PACKAGES = ("lightning", "pandas", "torch")


def _requirements(extra=None):
    """The requirement names declared for the core install or for an extra."""
    names = []
    for requirement in metadata.requires("openxps"):
        name = re.match(r"[A-Za-z0-9_.-]+", requirement).group(0).lower()
        marker = re.search(r"extra\s*==\s*[\"']([^\"']+)[\"']", requirement)
        if (marker.group(1) if marker else None) == extra:
            names.append(name)
    return names


def test_declared_runtime_dependencies():
    """Every third-party package imported at import time is declared."""
    requires = _requirements()
    for package in (
        "cvpack",
        "numpy",
        "openmm",
        "pyyaml",
        "typing-extensions",
    ):
        assert package in requires, f"{package} is not declared in pyproject.toml"
    for package in REGRESSION_PACKAGES:
        assert package not in requires, f"{package} should be optional"


def test_declared_regression_extra():
    """ForceMatchingRegressor's dependencies are the regression extra."""
    assert sorted(_requirements("regression")) == list(REGRESSION_PACKAGES)


def _run(code):
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


def test_import_does_not_load_regression_dependencies():
    result = _run(
        "import sys, openxps; "
        f"print([m for m in {REGRESSION_PACKAGES!r} if m in sys.modules])"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_regressor_is_available_from_the_package():
    regression = pytest.importorskip("openxps.regression")
    assert openxps.ForceMatchingRegressor is regression.ForceMatchingRegressor


@pytest.mark.parametrize("missing", REGRESSION_PACKAGES)
def test_regressor_without_its_dependencies(missing):
    """Without one of them, openxps imports and the regressor names the extra."""
    result = _run(
        f"import sys; sys.modules[{missing!r}] = None; import openxps; "
        "openxps.SplitIntegrator; openxps.ForceMatchingRegressor"
    )
    assert result.returncode != 0
    assert "ImportError" in result.stderr
    assert "openxps[regression]" in result.stderr
    assert "AttributeError" not in result.stderr


def test_unknown_attribute():
    with pytest.raises(AttributeError, match="no_such_name"):
        getattr(openxps, "no_such_name")  # noqa: B009


def test_declared_python_floor():
    """openmm-opes requires Python >= 3.11."""
    requires_python = metadata.metadata("openxps")["Requires-Python"]
    assert requires_python == ">=3.11"
