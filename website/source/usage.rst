.. _usage:

Usage
=====

Here you can find the `source code <_modules/wfbase.html>`_ of the main WfBase module.

The main WfBase module consists of these two main parts:

* :class:`DatabaseWf <wfbase.DatabaseWf>` deals with loading
  information from the .wf file.  Created by function
  :func:`load <wfbase.load>`.

* :class:`_ComputatorWf <wfbase._ComputatorWf>` main class,
  responsible for doing all calculations. Created by call to
  :func:`do_mesh <wfbase.DatabaseWf.do_mesh>`, or
  :func:`do_path <wfbase.DatabaseWf.do_path>`, or
  :func:`do_list <wfbase.DatabaseWf.do_list>`.

.. automodule:: wfbase
  :members: load, load_from_wannierberri, DatabaseWf, _ComputatorWf, Units, render_latex, display_in_terminal, display_in_separate_window, download_data_if_needed
