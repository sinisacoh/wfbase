.. _database:

Database
========

WfBase computer package includes a database of the density functional theory (DFT) calculations on some simple solids.

Here are three approaches to accessing the database. In each of these three approaches, you should get in the end a folder with the name *data* containing several files with the *.wf* extension.  Each of these *.wf* files corresponds to one density functional theory (DFT) calculation.

1. Following this link you can download the entire database from
   your browser

     `<https://coh.ucr.edu/wfbase/latest/data.zip>`_

   or you can get the same file using *wget* in the terminal (if
   you have *wget* installed on your machine)::

     wget https://coh.ucr.edu/wfbase/latest/data.zip

   After downloading this file you should unzip it using a command
   such as::

     unzip data.zip

2. Alternatively, you can download the same data directly from
   your python script using the function
   :func:`download_data_if_needed <wfbase.download_data_if_needed>`,

   .. code-block:: Python
   
     import wfbase as wf
     wf.download_data_if_needed()
   
3. The third option to get the database is to first :ref:`install
   <installation>` WfBase and then run the following command from
   your terminal::

     python -c "import wfbase as wf; wf.download_data_if_needed()"

Database entries, which are files with extension *.wf* in folder *data*, contain a nearly complete description of the first-principles electronic band structure.  Due to the exponential localization of the `Wannier functions <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.56.12847>`_, and symmetry capabilities implemented in the `Wannier Berri <https://wannier-berri.org>`_ package, these files are relatively small in size.

Other materials
---------------

At this point, WfBase has a rather small set of materials in its database. If you wish to study material that is currently not in the database, you will have to run the density-functional theory (DFT) calculations on your own.  The example :ref:`here <sphx_glr_all_examples_example_standalone_prepare.py>` provides input files for one of the entries in the database.  Your material will likely need a different set of parameters. Once you run your DFT calculation and obtain the needed output files, you can see the :ref:`following example <sphx_glr_all_examples_example_standalone_recalculate.py>` of how to load that calculation into WfBase.

Information
-----------

Database file with extension *.wf* contains results of the DFT calculations.  Information about these calculations can be accessed using the :func:`info <wfbase.DatabaseWf.info>` function.  Below is an example code that uses the :func:`info <wfbase.DatabaseWf.info>` function to access information about the database.

INSERT example_database_all.py
