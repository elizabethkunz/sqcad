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

sys.path.insert(0, os.path.abspath("../../src"))

project = 'sqCAD'
copyright = '2026, Elizabeth Kunz, Eli Levenson Falk'
author = 'Elizabeth Kunz, Eli Levenson Falk'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinx_design",
]
autosummary_generate = True

html_theme = "pydata_sphinx_theme"
html_title = "sqCAD"

html_theme_options = {
    "github_url": "https://github.com/elizabethkunz/sqcad",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

html_logo = "_static/sqcad_logo.png"   # optional
html_favicon = "_static/favicon.ico"   # optional


html_theme_options = {
    "github_url": "https://github.com/elizabethkunz/sqcad",
}

extensions += ["sphinx.ext.autosummary", "sphinx.ext.autodoc"]
autosummary_generate = True


copyright = f"{date.today().year}, Elizabeth Kunz, Eli Levenson Falk"
