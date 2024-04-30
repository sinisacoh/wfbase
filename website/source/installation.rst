.. _installation:

Installation
============

If you don't have python installed on your machine, you can download it from `Anaconda's website <https://www.anaconda.com/download/success>`_.

To install WfBase run the following command in your terminal::

  pip install wfbase --upgrade

If the *pip* command from above is raising an error about missing pyfftw, or when running WfBase you get a warning that pyfftw is not installed, try running the following command::

  conda install -c conda-forge pyfftw

and then re-run the pip installation of WfBase::

  pip install wfbase --upgrade
  
The issue with *pyfftw* seems to occur on Apple's M-series processors (*osx-arm64*), as *pip* at the moment seems to be missing a pre-compiled version of *pyfftw* for *osx-arm64*.  See `here for more information <https://stackoverflow.com/questions/68922959/cant-install-pyfftw-with-python-3-9-in-macos-m1>`_.

As of April 2024, some of the packages that WfBase depends on do not
work with the latest version of python (3.12). If this is still the
case, we suggest you try using an earlier version of python, such as 3.11.

Release notes and version list
------------------------------

We recommend you always use latest available version of WfBase. However, if you need to look up an old version of the code, you can find it below.

.. include:: ../local/release/release.rst
  
