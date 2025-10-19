#!/usr/bin/env bash
set -e -v
ruff format openxps
ruff check --fix openxps