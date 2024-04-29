.. meta::
   :keywords: WfBase, electronic structure,
	      density functional theory, wannier function,
	      wannier90, DFT, first-principles, quantum espresso,
	      wannierberri, pyparsing, parse, Einstein notation,
	      nonlinear optics, optical conductivity, anomalous
	      hall effect, wannier interpolation, band structure,
              bloch, periodic insulator

.. toctree::
   :maxdepth: 1
   :hidden:

   about
   installation
   usage
   examples
   database
   quantities
	      
Wavefunction database and computation (WfBase)
==============================================

WfBase is a software package providing an easy way to compute from first-principles various properties depending on the electronic structure of periodic solids.  This package can parse user-provided mathematical expressions, in a human-readable format, using the Einstein notation for indices.  It is well suited, for example, to calculate terms that arise in the perturbation theory context, where one needs to sum over electronic states over arbitrarily dense sampling of the Brillouin zone.

This package also comes with a :ref:`built-in database <database>` of some simple materials.  The accuracy of the electronic structure in the database is on the level of common generalized gradient approximation (GGA) approximations to the density functional theory.  All calculations in the database include fully-relativistic effects, such as the spin-orbit interaction.

To install WfBase run the following command in your terminal (:ref:`more details <installation>`)::

  pip install wfbase --upgrade

Start guide
-----------

Below is a quick introduction to WfBase package.  If you need more information about WfBase, take a look at the :ref:`additional examples <examples>` of using WfBase.  See also the :ref:`detailed technical description <usage>` of all WfBase functionalities.

Accessing database
~~~~~~~~~~~~~~~~~~

After :ref:`downloading the database <database>` you should use the :func:`load <wfbase.load>` function to open one of the database entries,

.. code-block:: Python
		
  db = wf.load("data/au_fcc.wf")

Next, we can use the created object *db* to compute from first-principles various basic electronic-structure quantities on an arbitrarily dense sampling of the Brillouin zone.

.. code-block:: Python
		
  comp = db.do_mesh([16, 16, 16])

A newly created object, *comp*, now stores information about the electronic band structure (*E*) and the needed matrix elements (*A*), along with other :ref:`quantities <quantities>` as well.

Computing user-defined quantities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can now use basic quantities stored in *comp*, such as *E* and *A*, to compute user-defined quantities.  All you need to do here is to call function  :func:`evaluate <wfbase._ComputatorWf.evaluate>` with a string representation of the mathematical expression you wish to calculate. For example, you can provide the following strings to compute the interband optical conductivity,

.. code-block:: Python

  comp.evaluate(
  "sigma_oij <= (j/(numk*volume)) * A_knmi*A_kmnj * (E_kn - E_km)/(E_km - E_kn - hbaromega_o - j*eta)",
  "E_km > ef, E_kn < ef"
  )

Given this information, WfBase will now

* parse the provided strings,

* perform the sum over the repeated indices while enforcing the provided constraints on the empty and occupied states,

* check that the physical units are consistent in the provided expression, and find the physical unit of the resulting quantity *sigma*,

* provide LaTeX'ed equation of the provided expression, as shown here,

.. rst-class:: sphx-glr-horizontal

    *

      .. image-sg:: /all_examples/images/sphx_glr_example_quick_001.png
         :alt: example quick
         :srcset: /all_examples/images/sphx_glr_example_quick_001.png, /all_examples/images/sphx_glr_example_quick_001_2_00x.png 2.00x
         :class: sphx-glr-single-img-larger

* give the final result (optical conductivity), converted to SI units if needed,
		 
.. rst-class:: sphx-glr-horizontal

    *

      .. image-sg:: /all_examples/images/sphx_glr_example_quick_002.png
         :alt: example quick
         :srcset: /all_examples/images/sphx_glr_example_quick_002.png, /all_examples/images/sphx_glr_example_quick_002_2_00x.png 2.00x
         :class: sphx-glr-single-img

Computational provenance
~~~~~~~~~~~~~~~~~~~~~~~~
		 
User can access from within their script detailed information about the computed quantity *sigma*,

.. code-block:: Python

  comp.info("sigma")

The example output of function :func:`info <wfbase._ComputatorWf.info>` can be found :ref:`here <quantities>`.

Using an additional flag to function :func:`info
<wfbase._ComputatorWf.info>` one can also get the automatically
generated python code that WfBase used under the hood to evaluate
the provided expression,

.. code-block:: Python

  comp.info("sigma", show_code = True)

For reproducibility purposes, one can :ref:`run this code directly <sphx_glr_all_examples_example_hybrid_3.py>` if needed.

Furthermore, one can also access information about any other quantity stored in object *comp*, such as the electron band energy (*E*) or Berry connection (*A*),
  
.. code-block:: Python

  comp.info("E")
  comp.info("A")

The :ref:`output <quantities>` of these functions will give you information about the order and meaning of indices of these quantities, as well as the definition of each quantity, sign convention, physical units, and so on.
  
Finally, for a more complete `computational provenance <https://www.computer.org/csdl/magazine/cs/2008/03/mcs2008030009/13rRUwj7cnv>`_, one can also obtain information about the density functional theory (DFT) calculation used to generate the database entry itself,
  
.. code-block:: Python

  db.info()

The example output of this function can be found :ref:`here
<database>`.

INSERT example_quick.py
