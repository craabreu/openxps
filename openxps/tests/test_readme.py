"""
Test that the quickstart example in README.md runs.
"""

import re
from pathlib import Path

import pytest

README = Path(__file__).parents[2] / "README.md"


@pytest.mark.skipif(not README.exists(), reason="README.md is not available")
def test_readme_quickstart_runs(capsys):
    """The Python block of the README runs, here with fewer steps."""
    pytest.importorskip("openmmtools")
    (code,) = re.findall(
        r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S
    )
    code, count = re.subn(r"simulation\.step\(\d+\)", "simulation.step(250)", code)
    assert count == 1
    exec(compile(code, str(README), "exec"), {"__name__": "__readme__"})
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith('#"Step","Temperature (K)","Extension Temperature (K)"')
    assert len(lines) == 2
