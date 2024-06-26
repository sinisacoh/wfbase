.. _about:

About
=====

People
------

The early version of the WfBase code was developed by Justin Burzachiello, under the guidance of Sinisa Coh, when Justin was an undergraduate student at the University of California, Riverside.  Ming Lei, at that time a graduate student of Sinisa Coh at the University of California, Riverside, developed a database of wannierizations of transition metals.  Stepan Tsirkin (the main developer of the WannierBerri package that WfBase uses under the hood) contributed numerous suggestions to the structure of the code, particularly for the interaction between the WannierBerri and WfBase codes.

Intended use of WfBase and some limitations
-------------------------------------------

WfBase software package is well suited for,

  * quick prototyping, tinkering, and testing of electronic structure calculations,

  * detailed analysis of electronic structure calculations,
  
  * educational use in an introductory solid-state physics
    course or similar.

WfBase software package is likely *not* well suited for,

  * out-of-the-box calculation of physical quantities (such
    calculations are better handled by `Wannier Berri
    <https://wannier-berri.org>`_ and  `Wannier90
    <https://wannier.org>`_),

  * large, computationally intensive, calculations.

Dependence on other computer packages
-------------------------------------
    
The WfBase package relies heavily on several pieces of code developed by many authors.

  * First of all, WfBase heavily relies on using a software package maintained by Stepan Tsirkin called `Wannier Berri <https://wannier-berri.org>`_  This package we use to get Wannier interpolation.

  * Second, the Wannier interpolation, used by Wannier Berri is constructed from the smooth gauge computed by the `Wannier90 <https://wannier.org>`_ software package.

  * Third, Wannier functions in Wannier90 were computed from the Bloch states computed by the first-principles density-functional theory software package `Quantum-ESPRESSO <https://www.quantum-espresso.org>`_.

Development
-----------

Github repository of WfBase is available here

  `<https://github.com/sinisacoh/wfbase>`_

Citation
--------

Please cite the following DOI if you use WfBase in your research papers,

::

  https://doi.org/10.5281/zenodo.12528020

Here is the BibTeX entry you can use,
 
::
  
  @misc{wfbase,
    title = {Wf{B}ase},
    note  = {wavefunction database and computation},
    doi   = {10.5281/zenodo.12528020},
    url   = {https://doi.org/10.5281/zenodo.12528020}
  }
   
Feedback
--------

Please send comments or suggestions for improvement to `this email
address <mailto:sinisa.coh@ucr.edu>`_.

Acknowledgments and Disclaimer
------------------------------

This Web page is based in part upon work supported by the US National
Science Foundation under Grant NSF DMR-1848074. Any opinions, findings, conclusions, or recommendations expressed in this material are those of the author and do not necessarily reflect the views of the National Science Foundation.

