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

To install OpenXPS in a conda environment, run the following command::

    conda install -c conda-forge -c mdtools openxps

Or use mamba instead::

    mamba install -c mdtools openxps

Usage
-----

To use OpenXPS, import the package in your Python script or Jupyter notebook::

    import openxps

