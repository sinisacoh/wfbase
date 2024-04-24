.. _installation:

Installation
============

If you don't have python installed on your machine, you can download it from `Anaconda's website <https://www.anaconda.com/download/success>`_.

To install WfBase run the following command in your terminal::

  pip install wfbase --upgrade

If the command above is raising errors about pyfftw or spglib (at the time of writing (April 2024) this seems to happen with Anaconda's python for Apple's M-series processors), please run the following in your terminal, and then repeat the command above to install WfBase::

  conda install -c conda-forge pyfftw
  conda install -c conda-forge spglib

Release notes and version list
------------------------------

We recommend you always use latest available version of WfBase. However, if you need to look up an old version of the code, you can find it below.

.. include:: ../local/release/release.rst
  
