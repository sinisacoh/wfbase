# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
import sys
sys.path.append("..")
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'WfBase'
copyright = '2024'
author = 'Sinisa Coh'

# The full version, including alpha/beta/rc tags
release = '0.0.2'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.imgmath',
              'sphinx.ext.viewcode',
              'sphinx_gallery.gen_gallery',
#              'sphinxcontrib.spelling',
]

# https://sphinx-gallery.github.io/stable/getting_started.html
sphinx_gallery_conf = {
     'examples_dirs': '../../examples',   # path to your example scripts
     'gallery_dirs': 'all_examples',  # path to where to save gallery generated output
     'filename_pattern': '/example_',
     'image_srcset': ["2x"],
     'show_signature': False,
     'capture_repr': ('_repr_html_', '__str__', '__repr__'),  # this is suppressed in create_single_examples_page.py
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["__*.rst", "*_template.rst", "sg_execution_times.rst", "all_examples/*.rst"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# preamble for latex formulas
pngmath_latex_preamble=r"\usepackage{cmbright}"
pngmath_dvipng_args=['-gamma 1.5', '-D 110']
pngmath_use_preview=True

# https://sphinxcontrib-spelling.readthedocs.io/en/latest/customize.html
spelling_lang='en_US'
tokenizer_lang='en_US'
spelling_word_list_filename='spelling_wordlist.txt'
spelling_show_whole_line = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'classic'

#html_theme_options = {"rightsidebar":False,
#                      "nosidebar":False}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = ['css/my_custom.css']

html_sidebars = {
          '**':    ['globaltoc.html', 'searchbox.html'],
       }

# remove "show source" from website
html_copy_source=False
html_show_sourcelink=False

# logo
html_logo = "logo.svg"


# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'WfBase.tex', u'WfBase Documentation',
   u'Sinisa Coh', 'manual'),
]

# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'wfbase', u'WfBase Documentation',
     [u'Sinisa Coh'], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output ------------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', 'WfBase', u'WfBase Documentation',
   u'Sinisa Coh', 'WfBase', 'Python software package for electronic structure calculations using Einstein notation',
   'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'


# for autodoc so that things are ordered as in source
autodoc_member_order = 'bysource'

# In order to skip some functions in documentation
def maybe_skip_member(app, what, name, obj, skip, options):
    if name in []:#["tbmodel","add_hop","set_sites","no_2pi"]:
        return True
    else:
        return skip
def setup(app):
    app.connect('autodoc-skip-member', maybe_skip_member)
