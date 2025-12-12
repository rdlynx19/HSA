# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Emergent Locomotion in HSA Robot'
copyright = '2025, Pushkar Dave'
author = 'Pushkar Dave'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.napoleon', 
    'sphinx.ext.viewcode', 
    'sphinx.ext.intersphinx', 
    'sphinx_rtd_theme'
]

templates_path = ['_templates']
exclude_patterns = []

import os
import sys
# FIX 2: Correct path to project root
sys.path.insert(0, os.path.abspath('..')) 


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# FIX 1: Set the theme to the Read the Docs theme
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

pygments_style = 'sphinx'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'gymnasium': ('https://gymnasium.farama.org/', None),
    'stable_baselines3': ('https://stable-baselines3.readthedocs.io/en/master/', None),
}