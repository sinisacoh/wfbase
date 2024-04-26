#!/usr/bin/env python

# https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/

from setuptools import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

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
      long_description = long_description,
      long_description_content_type = "text/markdown",
      platforms = ['UNIX', 'MAC OS X', 'Windows'],
      install_requires = [\
      "wannierberri", "pyfftw", "spglib", "irrep", "untangle",\
      "imgcat", "opt_einsum",
      "sympy", "numpy", "numba", "pyparsing", "matplotlib",  "pillow", "requests", 
      ],
      python_requires = ">=3.7",
      )
