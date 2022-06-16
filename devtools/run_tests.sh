#!/usr/bin/env bash

# exit when any command fails
set -e

flake8 openxps/
isort --check-only openxps/
sphinx-build docs/ docs/_build
pytest -v --cov=openxps --doctest-modules openxps/
