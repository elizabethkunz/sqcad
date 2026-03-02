# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# docs/source/conf.py
import os
import sys
from datetime import date

import sphinx_nameko_theme

sys.path.insert(0, os.path.abspath("../../src"))

project = "sqCAD"
copyright = '2026, Elizabeth Kunz, Eli Levenson Falk'
author = "Elizabeth Kunz, Eli Levenson Falk"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = [
    "tutorials.rst",
    "tutorials/**",
]

autosummary_generate = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme_path = [sphinx_nameko_theme.get_html_theme_path()]
html_theme = "nameko"
html_title = "sqCAD"

html_theme_options = {
    "github_url": "https://github.com/elizabethkunz/sqcad",
}

copyright = f"{date.today().year}, Elizabeth Kunz, Eli Levenson Falk"
