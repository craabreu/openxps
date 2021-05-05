#!/usr/bin/env bash
pytest -v --cov=openxps --doctest-modules $@ openxps/
