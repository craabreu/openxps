# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

from __future__ import annotations

import inspect
import os
import sys

# Incase the project was not installed
import openxps

sys.path.insert(0, os.path.abspath(".."))


def create_class_rst_file(cls, module_name="openxps"):
    name = cls.__name__
    methods = list(cls.__dict__.keys())
    excluded = ["yaml_tag"]
    with open(f"api/{name}.rst", "w") as f:
        included_methods = [
            f"    .. automethod:: {method}\n"
            for method in sorted(methods)
            if not (method.startswith("_") or method in excluded)
        ]
        f.writelines(
            [
                f"{name}\n",
                "=" * len(name) + "\n\n",
                f".. currentmodule:: {module_name}\n",
                f".. autoclass:: {name}\n",
                "    :member-order: alphabetical\n\n",
            ]
            + ["    .. rubric:: Methods\n\n"] * bool(included_methods)
            + included_methods
        )


def create_function_rst_file(func, module_name="openxps"):
    name = func.__name__
    with open(f"api/{name}.rst", "w") as f:
        f.writelines(
            [
                f"{name}\n",
                "=" * len(name) + "\n\n",
                f".. currentmodule:: {module_name}\n",
                f".. autofunction:: {name}\n",
            ]
        )


def create_constant_rst_file(const_name, const_value, module_name="openxps"):
    """Create a .rst file for a constant/variable."""
    with open(f"api/{const_name}.rst", "w") as f:
        f.writelines(
            [
                f"{const_name}\n",
                "=" * len(const_name) + "\n\n",
                f".. currentmodule:: {module_name}\n",
                f".. py:data:: {const_name}\n",
                "    :type: " + type(const_value).__name__ + "\n\n",
            ]
        )


def create_module_docs(module, module_name, output_dir="api", exclude=None):
    """
    Create documentation files for a module with one class/function/constant per file.

    Parameters
    ----------
    module
        The module object to document
    module_name
        The full module name (e.g., 'openxps', 'openxps.bounds')
    output_dir
        The output directory for .rst files (default: 'api')
    exclude
        A list of already documented modules to exclude from the documentation
        (e.g., [openxps.bounds, openxps.integrators, openxps.couplings])
    Returns
    -------
    str or None
        The toctree filename, or None if module is empty
    """
    # Filter by __all__ if it exists, otherwise include all classes/functions
    module_all = getattr(module, "__all__", [])
    if module_all:
        module_all_set = set(module_all)
        # Collect names from excluded modules
        excluded_names = set()
        if exclude:
            for m in exclude:
                excluded_names.update(set(getattr(m, "__all__", [])))
        # Remove excluded names from module_all_set
        module_all_set -= excluded_names
        classes = [
            item
            for item in module.__dict__.values()
            if inspect.isclass(item) and item.__name__ in module_all_set
        ]
        functions = [
            item
            for item in module.__dict__.values()
            if inspect.isfunction(item) and item.__name__ in module_all_set
        ]
        # Get constants/variables (items in __all__ that are not classes or functions)
        constants = []
        for name in module_all:
            # Skip if this name is in excluded modules
            if name in excluded_names:
                continue
            if name not in {item.__name__ for item in classes} and name not in {
                item.__name__ for item in functions
            }:
                if name in module.__dict__:
                    const_value = module.__dict__[name]
                    # Skip modules and other special items
                    if not inspect.ismodule(const_value) and not name.startswith("_"):
                        constants.append((name, const_value))
    else:
        classes = [
            item for item in module.__dict__.values() if inspect.isclass(item)
        ]
        functions = [
            item for item in module.__dict__.values() if inspect.isfunction(item)
        ]
        constants = []

    # If module has no items, return None
    if not classes and not functions and not constants:
        return None

    # Determine toctree filename based on module
    if module_name == "openxps":
        toctree = "main.rst"
        title = "Core"
    else:
        # For submodules, use the module name
        module_short_name = module_name.split('.')[-1]
        toctree = f"{module_short_name}.rst"
        # Capitalize first letter and add "Module"
        title = f"{module_short_name.capitalize()}"

    # Create single toctree with all items
    with open(f"{output_dir}/{toctree}", "w") as f:
        f.write(
            f"{title}\n"
            f"{'=' * len(title)}\n"
            "\n"
            ".. toctree::\n"
            "    :titlesonly:\n"
            "\n"
        )

        # Add classes
        for item in sorted(classes, key=lambda x: x.__name__):
            f.write(f"    {item.__name__}\n")
            create_class_rst_file(item, module_name)

        # Add functions
        for item in sorted(functions, key=lambda x: x.__name__):
            f.write(f"    {item.__name__}\n")
            create_function_rst_file(item, module_name)

        # Add constants
        for const_name, const_value in sorted(constants, key=lambda x: x[0]):
            f.write(f"    {const_name}\n")
            create_constant_rst_file(const_name, const_value, module_name)

        f.write("\n.. testsetup::\n\n    from openxps import *")

    return toctree


# Documentation entries for submodules
bounds_toctree = create_module_docs(openxps.bounds, "openxps.bounds")
couplings_toctree = create_module_docs(openxps.couplings, "openxps.couplings")
integrators_toctree = create_module_docs(openxps.integrators, "openxps.integrators")

# Documentation entries for main module
main_toctree = create_module_docs(
    openxps, "openxps", exclude=(openxps.bounds, openxps.integrators, openxps.couplings)
)


with open("api/index.rst", "w") as f:
    entries = []
    if main_toctree:
        entries.append(f"    {main_toctree}\n")
    if bounds_toctree:
        entries.append(f"    {bounds_toctree}\n")
    if couplings_toctree:
        entries.append(f"    {couplings_toctree}\n")
    if integrators_toctree:
        entries.append(f"    {integrators_toctree}\n")

    f.write(
        "API Reference\n"
        "=============\n"
        "\n"
        ".. toctree::\n"
        "    :maxdepth: 2\n"
        "    :titlesonly:\n"
        "\n"
        + "".join(entries)
    )

# -- Project information -----------------------------------------------------

version = os.getenv("OPENXPS_VERSION", openxps.__version__)
project = f"OpenXPS {version}"
copyright = r"2024 C. Abreu"
author = "Charlles Abreu"
release = ""

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = "4.4"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
]

autosummary_generate = False
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "default"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"
html_static_path = ["_static"]
html_theme_options = {
    # 'logo': 'logo_small.png',
    # 'logo_name': True,
    "github_button": False,
    "github_user": "craabreu",
    "github_repo": "openxps",
}
html_sidebars = {
    "**": ["about.html", "globaltoc.html", "searchbox.html"],
}
html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
# html_short_title = "%s-%s" % (project, version)


def setup(app):
    app.add_css_file("css/custom.css")


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "openxpsdoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "openxps.tex",
        "OpenXPS Documentation",
        "openxps",
        "manual",
    ),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "openxps", "OpenXPS Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "openxps",
        "OpenXPS Documentation",
        author,
        "openxps",
        "Useful Collective Variables for OpenMM",
        "Miscellaneous",
    ),
]

# -- Extension configuration -------------------------------------------------
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_typehints_format = "short"

# Bibliography file
bibtex_bibfiles = ["refs.bib"]

# External links
extlinks = {
    "OpenMM": (
        "http://docs.openmm.org/latest/api-python/generated/openmm.openmm.%s.html",
        "openmm.%s",
    ),
    "CVPack": (
        "https://redesignscience.github.io/cvpack/latest/api/%s.html",
        "cvpack.%s",
    ),
}

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
