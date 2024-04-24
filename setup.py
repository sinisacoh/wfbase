#!/usr/bin/env python

from distutils.core import setup

setup(name = 'wfbase',
      version = '0.0.2',
      author = 'Sinisa Coh',
      author_email = 'sinisa.coh@ucr.edu',
      url = 'https://coh.ucr.edu/wfbase',
      download_url = 'https://coh.ucr.edu/wfbase',
      keywords = 'wannier functions, parsing mathematical expressions, einstein sum, solid state physics, condensed matter physics, materials science, high-throughput calculations',
      py_modules = ['wfbase'],
      license = "gpl-3.0",
      description = "Easy way to compute from first-principles various properties depending on the electronic structure of periodic solids.",
      long_description = "Easy way to compute from first-principles various properties depending on the electronic structure of periodic solids.  Can parse user-provided mathematical expressions, in a human-readable format.  Includes a database of some simple materials.",
      platforms = ['UNIX', 'MAC OS X', 'Windows'],
      install_requires = [\
          "wannierberri", "pyfftw", "spglib", "irrep", "untangle",\
          "imgcat", "numpy", "numba", "pyparsing", "matplotlib", "opt_einsum", "textwrap", "fnmatch", "PIL"
      ],
      )
