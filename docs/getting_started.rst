Getting Started
===============

Introduction
------------

OpenXPS is a Python package with pre-defined Collective variable (CV) classes for `OpenMM`_.

Possible use cases include:

*   Adding CVs to `CustomCVForce`_ instances in `OpenMM`_
*   Defining `BiasVariable`_ instances for `Metadynamics`_ simulations
*   Defining `CollectiveVariable`_ instances for `UFED`_ simulations
*   Harnessing the power of `OpenMM`_ for trajectory analysis

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

Example
-------

The following example shows how to use OpenXPS to define CVs for quantifying the helix content of
a sequence of residues in a protein. The example uses the `testsystems` module from the
`OpenMMTools`_ package to create a system with a T4 lysozyme L99A molecule. The longest helix in
this protein occurs between residues LYS60 and ARG80. Four different CVs are defined and added
to the system, which is then used to create an OpenMM `Context`_. The CVs are finally evaluated,
rounded to seven decimal places, and printed to the screen::

    import openxps
    import openmm
    from openmm import app, unit
    from openmmtools import testsystems
    model = testsystems.LysozymeImplicit()
    num_atoms = model.system.getNumParticles()
    residues = [*model.topology.residues()][59:80]
    helix_content = [
        openxps.HelixAngleContent(residues, normalize=True),
        openxps.HelixHBondContent(residues, normalize=True),
        openxps.HelixRMSDContent(residues, num_atoms, normalize=True),
        openxps.HelixTorsionContent(residues, normalize=True),
    ]
    for cv in helix_content:
        model.system.addForce(cv)
    platform = openmm.Platform.getPlatformByName('Reference')
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    for cv in helix_content:
        print(f"{cv.getName()}: {cv.getValue(context, digits=7)}")

The output should be::

    HelixAngleContent: 0.9873989 dimensionless
    HelixHBondContent: 0.93414 dimensionless
    HelixRMSDContent: 0.9946999 dimensionless
    HelixTorsionContent: 0.918571 dimensionless

The output shows that the normalized helix content of residues LYS60-ARG80 of the T4 lysozyme L99A
molecule is close to one, as expected, no matter which measure is used.

.. _BiasVariable:       https://docs.openmm.org/latest/api-python/generated/openmm.app.metadynamics.BiasVariable.html
.. _CollectiveVariable: https://ufedmm.readthedocs.io/en/latest/pythonapi/ufedmm.html#ufedmm.ufedmm.CollectiveVariable
.. _Context:            https://docs.openmm.org/latest/api-python/generated/openmm.openmm.Context.html
.. _CustomCVForce:      https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html
.. _Force:              https://docs.openmm.org/latest/api-python/generated/openmm.openmm.Force.html
.. _Metadynamics:       https://docs.openmm.org/latest/api-python/generated/openmm.app.metadynamics.Metadynamics.html
.. _OpenMM:             https://openmm.org
.. _OpenMMTools:        https://openmmtools.readthedocs.io/en/stable
.. _UFED:               https://ufedmm.readthedocs.io/en/latest/index.html
