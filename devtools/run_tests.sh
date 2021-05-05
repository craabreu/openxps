#!/usr/bin/env bash

# exit when any command fails
set -e

flake8 openxps/
isort openxps/openxps.py
sphinx-build docs/ docs/_build
pytest -v --cov=openxps --doctest-modules openxps/
