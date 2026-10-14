Getting Started
===============

Introduction
------------

OpenXPS runs extended phase-space (XPS) simulations with OpenMM. It adds dynamical
variables to a physical system and couples them to collective variables or to force
parameters, as in driven adiabatic free-energy dynamics (d-AFED) and
temperature-accelerated molecular dynamics (TAMD).


Installation
------------

OpenXPS requires Python 3.10 or newer and OpenMM 8.4 or newer. Install it from the
mdtools conda channel::

    conda install -c conda-forge -c mdtools openxps

or from PyPI::

    pip install openxps

PyTorch and Lightning, which :class:`~openxps.ForceMatchingRegressor` uses, are
installed as dependencies.

Usage
-----

To use OpenXPS, import the package in your Python script or Jupyter notebook::

    import openxps

