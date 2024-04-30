# Wavefunction database and computation (WfBase)
# April 26th, 2024
__version__='0.0.2'

# Copyright 2024 by Sinisa Coh
#
# This file is part of WfBase.  WfBase is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# WfBase is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# A copy of the GNU General Public License should be available
# alongside this source in a file named gpl-3.0.txt.  If not,
# see <http://www.gnu.org/licenses/>.
#
# WfBase is available at https://coh.ucr.edu/wfbase

import wannierberri as wberri
import numpy as np
from numba import njit
from opt_einsum import contract as opteinsum
from copy import deepcopy
import textwrap
import traceback
import fnmatch
from fractions import Fraction
from PIL import Image
import re
import sys
import os
import io
import time
import imgcat
import matplotlib
import matplotlib.pyplot as plt
import requests
from zipfile import ZipFile

import pyparsing as pp
sys.setrecursionlimit(10000)
pp.ParserElement.enablePackrat()

hbar_SI = 1.0545718176461565e-34
electron_charge_SI = 1.602176487e-19
epsilon_zero_SI = 8.854187817620389e-12
speed_of_light_SI = 299792458.0
electron_mass_SI = 9.1093837015e-31
angstrom_SI = 1.0e-10
hartree_SI = 4.35974394E-18
rydberg_SI = hartree_SI / 2.0
bohr_radius_SI = 0.52917720859E-10

QE_MAGN_SIGN = -1.0

def download_data_if_needed(silent = False):
    r"""
    Downloads the latest version of the WfBase database (file *data.zip*)
    and unpacks it in the *data/* folder.  Will not erase previous *data/*
    folder if it already exists.

    You can manually download the same database by following
    :ref:`these instructions <database>`.

    The database can be in any folder, it doesn't need to be in a folder
    called *data* (which is the default used in this documentation).  You
    might also want to place this folder at a fixed place on your machine
    and then simply load it by providing an absolute path to the file
    when you use the *load* function.  This way you don't need to have multiple
    copies of the database on your machine.

    :param silent: If set to *True* will not print an error message if
      unable to download, or unzip, the database, or the database was
      already downloaded. The default is *False*.

    Example usage::

        import wfbase as wf

        # download the database
        wf.download_data_if_needed()

        # open a database file on bcc phase of iron
        db = wf.load("data/fe_bcc.wf")

        # you can also open the database file by providing
        # absolute path.
        db = wf.load("~/work/calculations/data/fe_bcc.wf")
    """

    folder = "data"
    fname = "data.zip"
    if os.path.exists(folder) == True:
        if silent == False:
            _print_without_stopping(""" You called function *download_data_if_needed* in your script.
            This function is supposed to download the WfBase database from a website, and unpack
            it in folder named """ + folder + """.  However, this folder seems to exist already
            in the current path.  Therefore, this script will not download anything.  You probably should
            remove the call to function .download_data_if_needed() from your script.  If you insist
            on forcing a new download of the database, you could rename, or move, folder """ + folder + """
            and then run the script again.""")
        return
    elif os.path.exists(fname) == True:
        if silent == False:
            _print_without_stopping(""" You called function *download_data_if_needed* in your script.
            This function is supposed to download the WfBase database from a website as a
            single zip file called """ + fname + """.  However, this zip file seems to exist already
            in the current path.  Therefore, this script will not download anything.  You probably should
            remove the call to function .download_data_if_needed() from your script.  If you insist
            on forcing a new download of the database, you could rename, or move, zip file """ + fname + """
            and then run the script again.""")
        return
    url = "https://coh.engr.ucr.edu/wfbase/latest/" + fname
    if silent == False:
        print("Trying to connect to " + url)
    r = requests.get(url)
    open(fname, 'wb').write(r.content)
    if silent == False:
        print("Download successful!")
    with ZipFile(fname, "r") as f:
        f.extractall()
    if silent == False:
        print("Unzipped file " + fname)

def load(*args, **kwargs):
    r"""
    This is the function used to open a database file containing information
    about one of the DFT calculations.

    :param data_path: Path to the database .wf file containing information about
      a calculation.  The user should download database .wf file by following
      :ref:`these instructions <database>`.

    :param perform_consistency_check: A flag specifying whether the code
      should check that the computation of various physical quantities is
      done correctly.  The default is True. Set to False only if you are really
      sure what is going on.  It is strongly advised to keep this parameter as is.

    :returns:
      * **db** -- database object of type :class:`DatabaseWf <wfbase.DatabaseWf>`.
        This object can be used next to create computators of various physical
        quantities.

    Example usage::

        import wfbase as wf

        # open a database file on bcc phase of iron
        db = wf.load("data/fe_bcc.wf")

        # create now a computator from the database
        comp = db.do_mesh()

    """
    ret = DatabaseWf()
    ret._load_from_wfbase_database_file(*args, **kwargs)
    return ret

def load_from_wannierberri(*args, **kwargs):
    r"""

    Loads calculation directly using Wannier Berri package without using
    a .wf database file from WfBase.

    See documentation of `Wannier Berri <https://wannier-berri.org>`_ for more
    details on creation of object *System_w90*.

    :param system: This is *System_w90* object from Wannier Berri.

    :param global_fermi_level_ev: This is the Fermi level in eV.  You
      can get this number at the end of your self-consistent DFT calculation.
      WfBase will later shift all band energies by this number, so that
      new Fermi level is zero.

    :returns:
      * **db** -- database object of type :class:`DatabaseWf <wfbase.DatabaseWf>`.
        This object can next be used to create computators of various physical
        quantities.

    Example usage::

        import wfbase as wf
        import wannierberri as wberri

        def main():
            system = wberri.System_w90("run_dft_output/x", berry = True, spin = False)
            db = wf.load_from_wannierberri(system, global_fermi_level_ev = 18.3776)

        if __name__ == "__main__":
            main()

    """
    ret = DatabaseWf()
    ret._load_from_wannierberri_system(*args, **kwargs)
    return ret

class DatabaseWf():
    r"""
    Object of this class contains information about the DFT calculations.  Use
    function :func:`load <wfbase.load>` create object of this class.

    Example usage::

        import wfbase as wf

        # open a database file on bcc phase of iron
        db = wf.load("data/fe_bcc.wf")

        # create now a computator from the database
        comp = db.do_mesh()

    """

    def __init__(self):
        self._only_essentials = False
        self._rng = np.random.default_rng(8318)
        self._loaded_from_wannierberri = False

    def _load_from_wfbase_database_file(self, data_path, perform_consistency_check = True):
        self.__data_path = data_path
        self._system, self._add_info = _read_interface_to_wberri_from_file(self.__data_path)

        if perform_consistency_check == True:
            hashes_current = self._compute_own_hashes()
            for k in hashes_current.keys():
                hash_stored = self._add_info["hash_" + k]
                if k in ["nonhermA", "nonhermS"]:
                    comp = _are_hashes_similar_absolute(hashes_current[k], hash_stored)
                else:
                    comp = _are_hashes_similar_relative(hashes_current[k], hash_stored)
                if comp == False:
                    _raise_value_error("""Something is wrong with computation of
                    quantity """ + k + """.  Value obtained from your combination
                    of database file/installed software/used hardware is different
                    from what is expected.
                    You likely should update your Wannier Berri installation,
                    or you should download newer version of the database.
                    If you know what you are doing and you want to make this
                    message go away, set perform_consistency_check to False,
                    but this is strongly discouraged.""")

    def _load_from_wannierberri_system(self, system, global_fermi_level_ev):
        self._system = system

        self._loaded_from_wannierberri = True

        self._add_info = {}
        self._add_info["fermi_scf_ev"] = global_fermi_level_ev
        self._add_info["num_wann"] = system.num_wann
        self._add_info["cell"] = np.array(system.real_lattice, dtype = float)

    def info(self, print_to_screen = True, full = False):
        r"""
        Returns information about the computation stored in the database .wf file.  This
        information here could be used by the user to redo the DFT calculation
        from scratch, as shown in
        :ref:`this example <sphx_glr_all_examples_example_standalone_prepare.py>`.

        Note that there is a function with the same name that provides information
        about the computator, not about the database .wf file.  See here
        for more information on how to use this other function :func:`info <wfbase._ComputatorWf.info>`.

        :param print_to_screen: Whether the code should print the information
          to the screen.  The default is True.  If set to False, nothing is printed
          but instead this function returns a string.

        :param full: Whether output should be cut to 50 lines per entry or the
          entire information should be shown.

        :returns:
          * **txt** -- string of the text with the information.  This is returned
            only if *print_to_screen* is set to False.

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # print information about the database
            db.info()

        """

        if full == False:
            max_line = 50
        else:
            max_line = None

        out = ""

        if self._loaded_from_wannierberri == False:

            out += "\n"
            out += _make_rst_title("Database file *" + self.__data_path.strip() + "*")
            out += "\n"

            out += _make_rst_field("The version of this database file")
            out += "\n"
            out += _format_one_block_simple_indent(str(self._add_info["data_version"]), indent = 4,
                                                   start_and_end = False, max_line = max_line)
            out += "\n\n"

            out += _make_rst_field("Created using WfBase version")
            out += "\n"
            out += _format_one_block_simple_indent(str(self._add_info["wfbase_version"]), indent = 4,
                                                   start_and_end = False, max_line = max_line)
            out += "\n\n"

        out += _make_rst_field("Currently loaded WfBase version by this script")
        out += "\n"
        out += _format_one_block_simple_indent(str(__version__), indent = 4,
                                               start_and_end = False, max_line = max_line)
        out += "\n\n"

        if self._loaded_from_wannierberri == False:
            out += _make_rst_field("Created using Wannier Berri version")
            out += "\n"
            out += _format_one_block_simple_indent(str(self._add_info["wberri_version"]), indent = 4,
                                                   start_and_end = False, max_line = max_line)
            out += "\n\n"

        out += _make_rst_field("Currently loaded Wannier Berri version")
        out += "\n"
        out += _format_one_block_simple_indent(str(wberri.__version__), indent = 4,
                                               start_and_end = False, max_line = max_line)
        out += "\n\n"

        if self._loaded_from_wannierberri == False:
            out += _make_rst_field("Input file for the SCF computation using pw.x from Quantum ESPRESSO")
            out += "\n"
            out += _format_one_block_simple_indent(_adjust_input_file("scf", str(self._add_info["input_scf"])), indent = 4,
                                                   start_and_end = True, max_line = max_line)
            out += "\n\n"

            out += _make_rst_field("Input file for the NSCF computation using pw.x from Quantum ESPRESSO")
            out += "\n"
            out += _format_one_block_simple_indent(_adjust_input_file("nscf", str(self._add_info["input_nscf"])), indent = 4,
                                                   start_and_end = True, max_line = max_line)
            out += "\n\n"

            out += _make_rst_field("Pseudopotentials used in the calculation")
            out += "\n"
            out += _format_one_block_simple_indent(" , ".join(self._get_psp()), indent = 4,
                                                   start_and_end = False, max_line = max_line)
            out += "\n\n"

            out += _make_rst_field("Input file for pw2wannier90.x from Quantum ESPRESSO")
            out += "\n"
            out += _format_one_block_simple_indent(_adjust_input_file("pw2wan", str(self._add_info["input_pw2wan"])), indent = 4,
                                                   start_and_end = True, max_line = max_line)
            out += "\n\n"

            out += _make_rst_field("Input file for Wannier90")
            out += "\n"
            out += _format_one_block_simple_indent(str(self._add_info["input_w90"]), indent = 4,
                                                   start_and_end = True, max_line = max_line)
            out += "\n\n"

        if print_to_screen:
            print(out)
        else:
            return out

    def _get_psp(self):
        if self._loaded_from_wannierberri == True:
            _stop_because_loaded_from_wannierberri()

        pref = "https://coh.ucr.edu/wfbase/" + str(self._add_info["data_version"]) + "/psp/"
        ret = list(map(lambda s: pref + str(s).strip(), self._add_info["pseudopotentials"]))
        return ret

    def create_dft_input_files(self, folder, ncpu = 16):
        r"""

        Recreates input files for the DFT calculation used in the construction of the
        database.  Stores all of these input files in folder a named *folder*.
        Stops if the *folder* already exists.

        :param folder: the name of the folder in which to store input files.

        :param ncpu: integer specifying how many processors to use in the calculation.
          The default is 16.

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # will store all input files in a folder called run_dft
            db.create_dft_input_files("run_dft")

        """

        if self._loaded_from_wannierberri == True:
            _stop_because_loaded_from_wannierberri()

        if os.path.exists(folder) == True:
            _raise_value_error("Folder " + folder + " already exists.  Stopping. " + 
                               " (To continue, either use a different folder name, or erase folder " + folder + " yourself.)")

        os.mkdir(folder)

        with open(os.path.join(folder, "x.scf.in"), "w") as f:
            f.write(_adjust_input_file("scf", str(self._add_info["input_scf"])))
        with open(os.path.join(folder, "x.nscf.in"), "w") as f:
            f.write(_adjust_input_file("nscf", str(self._add_info["input_nscf"])))
        with open(os.path.join(folder, "x.pw2wan"), "w") as f:
            f.write(_adjust_input_file("pw2wan", str(self._add_info["input_pw2wan"])))
        with open(os.path.join(folder, "x.win"), "w") as f:
            f.write(str(self._add_info["input_w90"]))

        with open(os.path.join(folder, "go"), "w") as f:
            txt = "mkdir _work" + "\n"
            psp = self._get_psp()
            for p in psp:
                txt += "wget " + p + "\n"
            txt +=\
"""mpirun -np """ + str(ncpu) + """ pw.x -npool """ + str(ncpu) + """ -in x.scf.in  >& x.scf.out
mpirun -np """ + str(ncpu) + """ pw.x -npool """ + str(ncpu) + """ -in x.nscf.in >& x.nscf.out
mpirun -np 1  wannier90.x -pp x
mpirun -np """ + str(ncpu) + """ pw2wannier90.x -npool 1 < x.pw2wan >& pw2wan.out
mpirun -np 1  wannier90.x x
"""
            f.write(txt)


    def do_mesh(self, k_mesh = None, shift_k = [0.0, 0.0, 0.0],
                to_compute = ["psi", "A", "S", "dEdk"],
                formatted_output_latex = True, doublet_indices = False, reorder_orbitals = False):
        r"""

        Compute various quantities on a regular k-mesh in the Brillouin zone.
        By default, it will compute the following quantities,

        .. _common_quantites:

        +----------+--------------------------------------------+--------------------------------------------------+
        | Quantity | Short description                          | How to get more information?                     |
        +==========+============================================+==================================================+
        | E        | Electron band energy.                      | Call function *comp.info("E")*.                  |
        |          |                                            | Example :ref:`output <quantities>`.              |
        +----------+--------------------------------------------+--------------------------------------------------+
        | psi      | Electron wavefunction.                     | Call function *comp.info("psi")*.                |
        |          |                                            | Example :ref:`output <quantities>`.              |
        +----------+--------------------------------------------+--------------------------------------------------+
        | A        | Berry connection.                          | Call function *comp.info("A")*.                  |
        |          |                                            | Example :ref:`output <quantities>`.              |
        +----------+--------------------------------------------+--------------------------------------------------+
        | S        | Electron spin magnetic moment.             | Call function *comp.info("S")*.                  |
        |          |                                            | Example :ref:`output <quantities>`.              |
        +----------+--------------------------------------------+--------------------------------------------------+
        | dEdk     | Electron Fermi velocity (times hbar).      | Call function *comp.info("dEdk")*.               |
        |          |                                            | Example :ref:`output <quantities>`.              |
        +----------+--------------------------------------------+--------------------------------------------------+
        | ...      |                                            |                                                  |
        +----------+--------------------------------------------+--------------------------------------------------+

        Code will also compute some other quantities, not listed here.  For example,
        it will construct the default range of energies *hbaromega*, choose a
        default smearing parameter *eta*, etc.  These can be changed by the user.
        See the examples below.

        A complete list of all computed quantities can be obtained by calling the
        :func:`info <wfbase._ComputatorWf.info>` function on the object returned by this function.
        Example output of the function :func:`info <wfbase._ComputatorWf.info>` can be found
        :ref:`here <quantities>`.

        .. note::

          For more details on the computator object returned by this function, see
          the description of the computator class :class:`_ComputatorWf <wfbase._ComputatorWf>`.

        :param k_mesh: Size of the uniform k-mesh on which you want to compute
          these quantities. This should be a vector with three components, one
          for the number of k-points in each direction. The default is the coarse mesh used to
          construct the Wannier functions, but you may want to use a denser
          mesh.  Note that the code will precompute all the quantities here on
          the mesh you specify, and this might take up a lot of your RAM. If
          you wish to save RAM, you could instead compute several smaller grids, one at
          a time.  For example, this can be achieved by randomly shifting the
          k-grid.  See :ref:`this example <sphx_glr_all_examples_example_conv.py>`
          for more details.

        :param shift_k: Shift of the uniform k-mesh. The coordinates for the shift
          are given as dimensionless, reduced, coordinates.  Therefore, this
          parameter expects a set of three numbers between 0 and 1.  If you
          specify numbers outside of this range, then the code will automatically
          reduce them to the range from 0 to 1 by removing the integer part.
          If you set *shift_k* to a string "random" then the code will shift the k-mesh
          by a random amount in all three directions.  This might be useful for
          sampling the k-points.

        :param to_compute: Quantity "E" is always computed.  Here you can list additional
          quantities that you want to compute. These are any combination of
          "psi", "A", "S", "dEdk".  The default is to use all of them: ["psi", "A", "S", "dEdk"].
          See here for :ref:`more information <common_quantites>`.
          If you don't need some of these quantities, don't list them here, so they will not
          be computed, and the code will use less resources.

        :param formatted_output_latex: Boolean value of True or False.  The default
          is True.  If set to False then any quantity evaluated from this computator
          will have a less formatted latex output.  For example, the latex output
          will not use bra and ket notation, etc.

        :param doublet_indices: The default is False.  If set to True then the code will check
          if your band-structure is (at least) two-fold degenerate at each k-point.
          (This happens when a product of inversion and time-reversion symmetry is present,
          such as in inversion symmetric non-magnetic systems, for example.)  If the
          band structure is (at least) two-fold degenerate at each k-point, and this
          parameter is set to True, then the band indices of all quantities will be
          changed, as follows.  If you initially had, for example, 18 bands, then
          band index "n" would normally go over those 18 bands.  But if this parameter
          is set to True then this same system will now have two indices, call them "a"
          and "A".  Index "a" now goes over 9 values while index "A" goes over 2 values.
          Therefore, "a" corresponds to the index of a doublet, and "A" indexes states
          in the doublet.  The choice of the two states in the doublet is randomized
          by the diagonalizer in the Wannier Berri, and there is no special meaning to it.
          Also, if your band structure at some point is 4-fold degenerate, this routine
          will still use the doublet notation, and there is again no special meaning in the
          choice of those two doublets out of the 4 degenerate states.  If you want to avoid
          these high-symmetry points then set *shift_k* to "random", as that will guarantee
          that you can't have more than doubly degenerate bands at every uniform mesh.

        :param reorder_orbitals: The default is False.  If set to True it will reorder orbitals
          in quantities *wfc* and *orbitallabels* so that the spin index of the orbital
          is the slow index.  All other quantities are left unchanged.

        :returns:
          * **comp** -- computator object of type :class:`_ComputatorWf <wfbase._ComputatorWf>`.
            This object can next be used to evaluate various physical quantities.

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # compute quantities on a uniform mesh
            comp = db.do_mesh()

            # create a different computator object, from the same database db
            # but now use a different k-mesh.  There is no need to load
            # the database object again using load function!
            comp_alter = db.do_mesh([4, 4, 4])

            # print information about all quantities in the computator
            comp.info()

            # print information about one of the quantities in the computator
            comp.info("A")

            # change the energy range
            comp.compute_photon_energy("hbaromega", 0.0, 5.0, 51)

            # change the value of the smearing parameter
            comp.replace("eta", "0.2 eV")

            # different way to change the value of the smearing parameter
            comp["eta"] = 0.25

            # you can also do more complex operations on quantities
            comp["S"] = comp["S"][:, :, :, 0]

            db = wf.load("data/au_bcc.wf")
            comp = db.do_mesh()
            print("Shape of matrix A without doubling: ", comp.get_shape("A"))
            comp_new = db.do_mesh(doublet_indices = True)
            print("Shape of matrix A with doubling   : ", comp_new.get_shape("A"))

        """
        start_counter_do = time.perf_counter()

        if k_mesh is None:
            k_mesh = self._system.mp_grid

        if isinstance(shift_k, str):
            if shift_k.lower().strip() == "random":
                shift_k = self._rng.random((3))
            else:
                _raise_value_error("Unrecognized string provided to shift_k: " + shift_k + ".")

        props = {}

        text_trap = io.StringIO()
        sys.stdout = text_trap
        __grid = wberri.Grid(system = self._system, NKFFT = k_mesh, NKdiv = [1, 1, 1])
        sys.stdout = sys.__stdout__
        self._data_K = wberri.data_K.Data_K_R(system = self._system, dK = shift_k, grid = __grid)

        if False:
            # The core below throws an error in current version of wannierberri.
            props["kredvec"] =  {"value": self._data_K.kpoints_all(),
                                 "units": Units(eV = 0, Ang = 0, muB = 0),
                                 "origin_story": "...",
                                 "indices_info": {
                                     "definition": None,
                                     "canonical_names": "...",
                                     "explanation": ["...",
                                                     "..."],
                                     "bands": [],
                                 },
                                 }
        del __grid

        computed = {}

        computed["E"] = np.copy(self._data_K.E_K)

        for thing in to_compute:
            if thing == "psi":
                computed[thing] = _potentially_reorder_orbitals(
                    np.copy(np.transpose(self._data_K.UU_K, (0, 2, 1))),
                    2, reorder_orbitals)
            elif thing == "A":
                computed[thing] = np.copy(self._data_K.A_H)
            elif thing == "S":
                computed[thing] = np.copy(self._data_K.Xbar("SS"))
            elif thing == "dEdk":
                computed[thing] = np.copy(
                    np.diagonal(self._data_K.Xbar('Ham', 1), axis1 = 1, axis2 = 2).transpose(0, 2, 1))
            elif thing == "E":
                continue
            else:
                _raise_value_error("Unknown quantity " + thing)
        del self._data_K

        props["numk"] = {
            "value": int(np.prod(k_mesh)),
            "origin_story": "Total number of k-points in the mesh.",
            "units": Units(eV = 0, Ang = 0, muB = 0),
            "format"          : r"N_{\rm k}",
            "format_conjugate": r"N_{\rm k}",
        }

        props = self.__comp_common_essential(computed, props)
        if self._only_essentials == False:
            props = self.__comp_common(computed, props, reorder_orbitals)

        comp = _ComputatorWf(props, formatted_output_latex,
                             doublet_indices, self._loaded_from_wannierberri)

        comp._computated_using = "do_mesh"

        comp.compute_photon_energy("hbaromega", 0.01, 3.0, 31)
        comp.compute_occupation("f", "E", "ef")
        comp.new("eta", {"value": 0.1, "units": Units(eV = 1)})

        time_do = time.perf_counter() - start_counter_do
        comp._total_seconds_initialize = time_do

        return comp

    def do_list(self, k_list, to_compute = [],
                formatted_output_latex = True,
                doublet_indices = False,
                reorder_orbitals = False):
        r"""

        Similar to :func:`do_mesh <wfbase.DatabaseWf.do_mesh>` with the difference that
        now the returned computator contains information on an arbitrary list
        of k-vectors.

        Use :func:`do_path <wfbase.DatabaseWf.do_path>` for a simple way to generate a path
        between the special k-points in the Brillouin zone.

        :param k_list: List of k-vectors on which you want to do a computation.  These
          vectors are specified as dimensionless reduced coordinates of the reciprocal vectors.

        :param to_compute: List of additional quantities that you want to compute.
          You can take any of these: "psi", "A", "S", "dEdk".  The default is to compute none
          of them.  See here for :ref:`more information <common_quantites>` about these quantities.

        :param formatted_output_latex: Boolean value of True or False.  The default
          is True.  If set to False then any quantity evaluated from this computator
          will have a less formatted latex output.  For example, the latex output
          will not use bra and ket notation, etc.

        :param doublet_indices: Same meaning as in :func:`do_mesh <wfbase.DatabaseWf.do_mesh>`.

        :param reorder_orbitals: The default is False.  If set to True it will reorder orbitals
          in quantities *wfc* and *orbitallabels* so that the spin index of the orbital
          is the slow index.  All other quantities are left unchanged.

        :returns:
          * **comp** -- computator object of type :class:`_ComputatorWf <wfbase._ComputatorWf>`.  This object
            can next be used to evaluate various physical quantities.

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # compute quantities on a list of two k-points
            comp = db.do_list([[0.2, 0.4, 0.2], [0.1, 0.9, 0.4]])

        """
        start_counter_do = time.perf_counter()

        props = {}

        computed = {}
        computed["E"] = []
        for thing in to_compute:
            if thing == "E":
                continue
            computed[thing] = []

        for ii in range(len(k_list)):
            k_one = np.array(k_list[ii])

            text_trap = io.StringIO()
            sys.stdout = text_trap
            __grid = wberri.Grid(system = self._system, NKFFT = [1, 1, 1], NKdiv = [1, 1, 1])
            sys.stdout = sys.__stdout__
            self._data_K = wberri.data_K.Data_K_R(system = self._system, dK = k_one, grid = __grid)
            del __grid

            computed["E"].append(np.copy(self._data_K.E_K[0]))
            for thing in to_compute:
                if thing == "psi":
                    computed[thing].append(_potentially_reorder_orbitals(
                        np.copy(np.transpose(self._data_K.UU_K, (0, 2, 1))[0]),
                        2, reorder_orbitals))
                elif thing == "A":
                    computed[thing].append(np.copy(self._data_K.A_H[0]))
                elif thing == "S":
                    computed[thing].append(np.copy(self._data_K.Xbar("SS")[0]))
                elif thing == "dEdk":
                    computed[thing].append(
                        np.copy(np.diagonal(self._data_K.Xbar('Ham', 1), axis1 = 1, axis2 = 2).transpose(0, 2, 1)[0]))
                elif thing == "E":
                    continue
                else:
                    _raise_value_error("Unknown quantity " + thing)
            del self._data_K

        for k in computed.keys():
            computed[k] = np.array(computed[k])

        props["numk"] = {
            "value": len(k_list),
            "origin_story": "Total number of k-points in the list",
            "units": Units(eV = 0, Ang = 0, muB = 0),
            "format"          : r"N_{\rm k}",
            "format_conjugate": r"N_{\rm k}",
        }

        props["kredvec"] = {
            "value": np.array(k_list),
            "units": Units(eV = 0, Ang = 0, muB = 0),
            "origin_story": "These are reduced coordinates of all k-points in the list.",
            "indices_info": {
                "definition": None,
                "canonical_names": "kr",
                "explanation": ["k-points in the list",
                                "reduced coordinate"],
                "bands": [],
            },
        }

        props = self.__comp_common_essential(computed, props)
        if self._only_essentials == False:
            props = self.__comp_common(computed, props, reorder_orbitals)

        comp = _ComputatorWf(props, formatted_output_latex, doublet_indices, self._loaded_from_wannierberri)

        comp._computated_using = "do_list"

        comp.compute_photon_energy("hbaromega", 0.01, 3.0, 31)
        comp.compute_occupation("f", "E", "ef")
        comp.new("eta", {"value": 0.1, "units": Units(eV = 1)})

        time_do = time.perf_counter() - start_counter_do
        comp._total_seconds_initialize = time_do

        return comp

    def __comp_common_essential(self, computed, props):
        props["ef"] = \
            {
                "value": 0.0,
                "origin_story": """This is the Fermi level.  It is set to zero as we subtract the DFT-computed Fermi
                level on a coarse mesh from the Hamiltonian in the Wannier basis. Note: if you compute band structure
                on a very fine mesh, and then recompute the Fermi level from that band-structure, you will likely
                get a Fermi level that is not exactly zero.  This is to be expected, as the Fermi level will be somewhat
                sensitive on the k-mesh you choose and the smearing you use for the occupations.  However, one can expect
                that 0.0 is a good approximation of the Fermi level for any k-mesh you use, but you may need to check
                this on your own for a very fine mesh.""",
                "units": Units(eV = 1, Ang = 0, muB = 0),
                "format": r"{\rm E}_{\rm F}",
                "format_conjugate": r"{\rm E}_{\rm F}",
            }

        props["eV"] = \
            {
                "value": 1.0,
                "origin_story": "Constant equal to 1 eV.",
                "units": Units(eV = 1, Ang = 0, muB = 0),
                "format": r"{\rm \, eV}",
                "format_conjugate": r"{\rm \, eV}",
            }
        props["Ang"] = \
            {
                "value": 1.0,
                "origin_story": "Constant equal to 1 angstrom.",
                "units": Units(eV = 0, Ang = 1, muB = 0),
                "format": r"{\rm \, \AA}",
                "format_conjugate": r"{\rm \, \AA}",
            }
        props["muB"] = \
            {
                "value": 1.0,
                "origin_story": "Constant equal to 1 bohr magneton.",
                "units": Units(eV = 0, Ang = 0, muB = 1),
                "format": r"\, \mu_{\rm B}",
                "format_conjugate": r"\, \mu_{\rm B}",
            }

        for thing in computed.keys():
            if thing == "E":
                props[thing] = {
                    "value": computed[thing] - float(self._add_info["fermi_scf_ev"]),
                    "origin_story": """Band energies E computed from Wannier interpolation.
                    The Fermi level was computed on a DFT coarse mesh and then bands were shifted
                    so that the Fermi level was set
                    to zero.  If you use a very fine k-mesh then the Fermi level will have
                    to be recomputed for your k-mesh, but it will likely still be close to zero.
                    Note: due to the nature of the Wannier interpolation, some of the bands
                    far above the Fermi level do not correspond to the actual bands computed
                    in the DFT.  Typically this is not a problem as one usually cares about states
                    close to the Fermi level.  All electron states around the Fermi level, as well as
                    in the valence bands should
                    be well reproduced.  Quantities "reliableminenergy" and "reliablemaxenergy"
                    give you an energy window in which band energies are reliable.
                    Similarly, due to the nature of pseudopotential calculations, these band
                    structures do not contain deep core states, but again, these are often not needed.
                    """,
                    "indices_info": {
                        "definition": r"H_*0 u_*0*1 = E_*0*1 u_*0*1",
                        "canonical_names": "kn",
                        "explanation": ["index of a k-point",
                                        "electron band index"],
                        "bands": [1],
                    },
                    "units": Units(eV = 1, Ang = 0, muB = 0),
                    "format": r"E_{*0*1}",
                    "format_conjugate": r"E_{*0*1}",
                }
            elif thing == "psi":
                props[thing] = {
                    "value": computed[thing],
                    "origin_story":"""This is the electron wavefunction written in terms
                    of the localized Wannier state.  The approximate atomic-like orbital
                    characters of these localized Wannier states are given by quantity
                    *orbitallabels* (available only if your database was loaded from
                    the WfBase's database).
                    """,
                    "indices_info": {
                        "definition": r"psi_*0*1*2 = < W_*2 | psi_*0*1 >",
                        "canonical_names": "knp",
                        "explanation": ["index of a k-point",
                                        "electron band index of the state",
                                        "localized orbital index",
                                        ],
                        "bands": [1],
                    },
                    "units": Units(eV = 0, Ang = 0, muB = 0),
                    "format": r"\psi_{*0*1*2}",
                    "format_conjugate": r"\overline{\psi_{*0*1*2}}",
                }
            elif thing == "A":
                props[thing] = {
                    "value": computed[thing],
                    "origin_story":"""This is the Berry connection that can be used to
                    compute the optical matrix elements and other related quantities.
                    """,
                    "indices_info": {
                        "definition": r"< u_*0*1 | i delk_*3 | u_*0*2 >",
                        "canonical_names": "knma",
                        "explanation": ["index of a k-point",
                                        "electron band index of the bra state",
                                        "electron band index of the ket state",
                                        "k-derivative in Cartesian axes (0 for x, 1 for y, 2 for z)",
                                        ],
                        "bands": [1, 2],
                    },
                    "units": Units(eV = 0, Ang = 1, muB = 0),
                    "format": r"\langle u_{*0*1} \lvert "+ _process_latex_imag_j("j") + r" \partial_{k_{*3}} \rvert u_{*0*2} \rangle",
                    "format_conjugate": r"\langle u_{*0*2} \lvert "+ _process_latex_imag_j("j") + r" \partial_{k_{*3}} \rvert u_{*0*1} \rangle",
                }
            elif thing == "S":
                props[thing] = {
                    "value": QE_MAGN_SIGN * computed[thing],
                    "origin_story": """
                    Matrix elements of the spin magnetic moment operator.
                    """,
                    "indices_info": {
                        "definition": r"< u_*0*1 | Mspin_*3 | u_*0*2 >",
                        "canonical_names": "knma",
                        "explanation": ["index of a k-point",
                                        "electron band index of the bra state",
                                        "electron band index of the ket state",
                                        "direction of spin in Cartesian axes (0 for x, 1 for y, 2 for z)",
                                        ],
                        "bands": [1, 2],
                    },
                    "units": Units(eV = 0, Ang = 0, muB = 1),
                    "format": r"\langle \psi_{*0*1}  \lvert M^{\rm spin}_{*3} \rvert \psi_{*0*2} \rangle",
                    "format_conjugate": r"\langle \psi_{*0*2} \lvert M^{\rm spin}_{*3} \rvert \psi_{*0*1} \rangle",
                }
            elif thing == "dEdk":
                props[thing] = {
                    "value": computed[thing],
                    "origin_story":"""This is hbar times the Fermi velocity.
                    """,
                    "indices_info": {
                        "definition": r"d E_*0*1 / d k_*2  (= hbar Vfermi_*0*1*2)",
                        "canonical_names": "kna",
                        "explanation": ["index of a k-point",
                                        "electron band index",
                                        "k-derivative in Cartesian axes (0 for x, 1 for y, 2 for z)",
                                        ],
                        "bands": [1],
                    },
                    "units": Units(eV = 1, Ang = 1, muB = 0),
                    "format": r"\frac{\partial E_{*0*1}}{\partial k_{*2}}",
                    "format_conjugate": r"\frac{\partial E_{*0*1}}{\partial k_{*2}}",
                }
            else:
                _raise_value_error("Unknown quantity: " + thing + " !")

        return props

    def __comp_common(self, computed, props, reorder_orbitals):
        _cell = self._add_info["cell"]

        props["numwann"] = \
            {
                "value": int(self._add_info["num_wann"]),
                "origin_story": "The number of Wannier bands. " + 
                " Also, the number of electron states that are computed at each k-point.",
                "units": Units(eV = 0, Ang = 0, muB = 0),
            }
        props["coarsekmesh"] = \
            {
                "value": np.array(self._system.mp_grid),
                "origin_story": "The size of the coarse k-mesh used to create Wannier functions",
                "indices_info": {
                    "canonical_names": "r",
                    "explanation": ["reduced reciprocal axis",
                                    ],
                    "bands": [],
                },
                "units": Units(eV = 0, Ang = 0, muB = 0),
            }
        props["cell"] = \
            {
                "value": np.array(_cell, dtype = float),
                "origin_story": """Computational unit cell vectors.
                """,
                "indices_info": {
                    "canonical_names": "ia",
                    "explanation": ["index of the cell vector",
                                    "Cartesian axis (0 for x, 1 for y, 2 for z)",
                                    ],
                    "bands": [],
                },
                "units": Units(eV = 0, Ang = 1, muB = 0),
            }
        props["recip"] = \
            {
                "value": np.array(2.0*np.pi*_real_to_recip_no2pi([_cell[0], _cell[1], _cell[2]]), dtype = float),
                "origin_story": """Reciprocal unit cell vectors.
                """,
                "indices_info": {
                    "canonical_names": "ia",
                    "explanation": ["index of the reciprocal cell vector",
                                    "Cartesian axis (0 for x, 1 for y, 2 for z)",
                                    ],
                    "bands": [],
                },
                "units": Units(eV = 0, Ang =-1, muB = 0),
            }
        props["volume"] = \
            {
                "value": np.linalg.det(_cell),
                "origin_story": "The volume of the computational unit cell.",
                "units": Units(eV = 0, Ang = 3, muB = 0),
                "format"          : r"V_{\rm c}",
                "format_conjugate": r"V_{\rm c}",
            }
        if self._loaded_from_wannierberri == False:
            props["orbitallabels"] = \
                {
                    "value": _potentially_reorder_orbitals(np.array(self._add_info["orbital_labels"]), 0, reorder_orbitals),
                    "origin_story": "Array of names of localized atomic-like orbitals" + 
                    " used in the decomposition of the wavefunction *psi*. Uparrow and " + 
                    "downarrow refer to spin angular momentum (opposite to the spin magnetic moment)." + 
                    " " + str(self._add_info["orbital_labels_description"]),
                    "indices_info": {
                        "canonical_names": "p",
                        "explanation": ["localized orbital index",
                        ],
                        "bands": [],
                    },
                    "units": Units(eV = 0, Ang = 0, muB = 0),
                }
            props["atomname"] = \
                {
                    "value": np.array(self._add_info["atom_name"]),
                    "origin_story": "Array of names of atoms in the computational unit cell.",
                    "indices_info": {
                        "canonical_names": "j",
                        "explanation": ["index of the atom in the unit cell",
                        ],
                        "bands": [],
                    },
                    "units": Units(eV = 0, Ang = 0, muB = 0),
                }
            props["atomred"] = \
                {
                    "value": np.array(self._add_info["atom_reduced"], dtype = float),
                    "origin_story": """Reduced coordinates of atom positions.
                    """,
                    "indices_info": {
                        "canonical_names": "jr",
                        "explanation": ["index of the atom in the unit cell",
                                        "reduced coordinates axis",
                        ],
                        "bands": [],
                    },
                    "units": Units(eV = 0, Ang = 0, muB = 0),
                }
            props["reliablemaxenergy"] = \
                {
                    "value": np.array(self._add_info["frozen_max"], dtype = float) - float(self._add_info["fermi_scf_ev"]),
                    "origin_story": """Due to the nature of the Wannier interpolation, the electronic
                    properties are well reproduced for the valence band, up to some energy around the Fermi level.
                    This number gives you a maximal energy up to which you should trust the electronic properties.
                    For a lot of calculations, one does not need to worry about the states that are too far from the Fermi level.
                    The energy of this window includes the fact that the Fermi level is set to zero.
                    """,
                    "units": Units(eV = 1, Ang = 0, muB = 0),
                }
            props["reliableminenergy"] = \
                {
                    "value": np.array(self._add_info["frozen_min_adjusted"], dtype = float) - float(self._add_info["fermi_scf_ev"]),
                    "origin_story": """Similarly to reliablemaxenergy, parameter reliableminenergy gives you the minimal energy
                    for the range with reliable band properties.  States below reliableminenergy, such as core states, are
                    given by Wannier interpolation.  The interaction of valence with core and semi-core states is included, of course,
                    but the energies of core states themselves will not show up here.
                    """,
                    "units": Units(eV = 1, Ang = 0, muB = 0),
                }

        for thing in computed.keys():
            if thing == "E":
                if props[thing]["value"].shape != (props["numk"]["value"], props["numwann"]["value"]):
                    _raise_value_error("Object " + thing + " returned from Wannier Berri has a different shape than expected.")
            elif thing == "psi":
                if props[thing]["value"].shape != (props["numk"]["value"], props["numwann"]["value"], props["numwann"]["value"]):
                    _raise_value_error("Object " + thing + " returned from Wannier Berri has a different shape than expected.")
            elif thing == "A":
                if props[thing]["value"].shape != (props["numk"]["value"], props["numwann"]["value"], props["numwann"]["value"], 3):
                    _raise_value_error("Object " + thing + " returned from Wannier Berri has a different shape than expected.")
            elif thing == "S":
                if props[thing]["value"].shape != (props["numk"]["value"], props["numwann"]["value"], props["numwann"]["value"], 3):
                    _raise_value_error("Object " + thing + " returned from Wannier Berri has a different shape than expected.")
            elif thing == "dEdk":
                if props[thing]["value"].shape != (props["numk"]["value"], props["numwann"]["value"], 3):
                    _raise_value_error("Object " + thing + " returned from Wannier Berri has a different shape than expected.")
            else:
                _raise_value_error("Unknown quantity: " + thing + " !")

        if props["volume"]["value"] < 0.0:
            _raise_value_error("Unit cell vectors are not right-handed, as volume is negative.")

        return props


    def do_path(self, k_str,
                to_compute = [],
                num_steps_first_segment = 30,
                latex_tick_labels = True,
                formatted_output_latex = True,
                doublet_indices = False,
                reorder_orbitals = False):
        r"""
        Similar to :func:`do_mesh <wfbase.DatabaseWf.do_mesh>` with the difference that
        now the returned computator contains information on a list
        of k-vectors between special k-points.

        :param k_str: String describing the path between the special k-points for the given
          symmetry of the system. Conventions here follow that from the Bilbao Crystallographic
          server.

        :param to_compute: List of additional quantities that you want to compute.
          You can take any of these: "psi", "A", "S", "dEdk".  The default is to compute
          none of them. See here for :ref:`more information <common_quantites>` about these quantities.

        :param num_steps_first_segment: The number of points between two first special k-points
          in the list.  The number of points between other special k-points is computed so that
          the density of k-points in the Brillouin zone is nearly constant. The default is 30.

        :param latex_tick_labels: True or False.  The default is True.  If False, then labels
          of special points will not use LaTeX.  For example, Gamma point will simply be rendered
          as "GM" instead of using "$\\Gamma$".

        :param formatted_output_latex: Boolean value of True or False.  The default
          is True.  If set to False then any quantity evaluated from this computator
          will have a less formatted latex output.  For example, the latex output
          will not use bra and ket notation, etc.

        :param doublet_indices: Same meaning as in :func:`do_mesh <wfbase.DatabaseWf.do_mesh>`.

        :param reorder_orbitals: The default is False.  If set to True it will reorder orbitals
          in quantities *wfc* and *orbitallabels* so that the spin index of the orbital
          is the slow index.  All other quantities are left unchanged.

        :returns:
          * **comp** -- computator object of type :class:`_ComputatorWf <wfbase._ComputatorWf>`.  This object
            can next be used to evaluate various physical quantities.

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # compute quantities on a path between these special points
            comp = db.do_path("GM--H--N")

            # plot the band structure
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            comp.plot_bs(ax)
            ax.set_title("Band structure of Fe bcc")
            fig.tight_layout()
            fig.savefig("a.pdf")

        """

        if self._loaded_from_wannierberri == True:
            _stop_because_loaded_from_wannierberri()

        cell = self._add_info["cell"]
        recip = _real_to_recip_no2pi([cell[0], cell[1], cell[2]])

        # get information for the labels of k-points
        kpoint_info = _get_kpoint_label_info(cell,
                                             self._add_info["atom_name"],
                                             self._add_info["atom_reduced"])

        # parse the k-path
        fmt = pp.one_of(list("-_,. "))
        par = list(fmt.split(k_str, include_separators = False))
        pnts = []
        for p in par:
            if p.strip() != "":
                pnts.append(p.strip())

        if len(pnts) == 0:
            _raise_value_error("Need to specify at least one special k-point on the path.")

        if len(pnts) > 1 and num_steps_first_segment < 2:
            _raise_value_error("num_steps_first_segment must be 2 or larger.")

        # get coordinates of endpoints in reduced
        kred_end = []
        for p in pnts:
            if p.upper() not in kpoint_info.keys():
                _raise_value_error("User specified incorrect label of a special k-point \"" + p + 
                                   "\".  Allowed values for this system are " + str(list(kpoint_info.keys())))
            kred_end.append(kpoint_info[p.upper()][0])
        kred_end = np.array(kred_end)

        pnts_new = []
        for p in pnts:
            if len(kpoint_info[p.upper()]) > 1 and latex_tick_labels == True:
                pnts_new.append(kpoint_info[p.upper()][1])
            else:
                pnts_new.append(p.upper())
        pnts = pnts_new

        if kred_end.shape[0] > 1:
            def distance_kpts_red(k0_red, k1_red):
                _c0 = _red_to_cart(recip[0], recip[1], recip[2], k0_red)
                _c1 = _red_to_cart(recip[0], recip[1], recip[2], k1_red)
                return 2.0*np.pi*np.sqrt(np.dot(_c0 - _c1, _c0 - _c1))

            num_steps = []
            num_steps.append(int(round(num_steps_first_segment)))
            kdist_zero = distance_kpts_red(kred_end[0], kred_end[1])
            for i in range(2, kred_end.shape[0]):
                tmp = np.round(num_steps_first_segment * distance_kpts_red(kred_end[i - 1],
                               kred_end[i]) / kdist_zero)
                if tmp < 2:
                    tmp = 2
                num_steps.append(int(tmp))

            k_all_red = []
            k_spec_index = []
            k_spec_index.append(0)
            for i in range(1, kred_end.shape[0]):
                for j in range(num_steps[i - 1]):
                    if i > 1 and j == 0:
                        continue
                    kone = kred_end[i - 1] + (float(j)/float(num_steps[i - 1] - 1))*(kred_end[i] - kred_end[i - 1])
                    k_all_red.append(kone)
                    if j == num_steps[i - 1] - 1:
                        k_spec_index.append(len(k_all_red) - 1)
            k_all_red = np.array(k_all_red)
        else:
            k_all_red = np.array([kred_end[0]])
            k_spec_index = [0]

        k_dist = []
        k_dist.append(0.0)
        for i in range(1, k_all_red.shape[0]):
            k_dist.append(2.0*np.pi*distance_kpts_red(k_all_red[i - 1], k_all_red[i]))
        k_dist = np.cumsum(k_dist)

        k_spec_dist = k_dist[k_spec_index]

        # now compute things in wannierberri
        comp = self.do_list(k_all_red, to_compute, formatted_output_latex, doublet_indices, reorder_orbitals)

        comp._computated_using = "do_path"

        comp.new("kdist", {"value": k_dist,
                           "units": Units(eV = 0, Ang = -1, muB = 0),
                           "origin_story": """Distance between k-points along the specified k-path.
                           You might want to use this for the x-axis of your band structure plot.""",
                           "indices_info": {
                               "canonical_names": "l",
                               "explanation": ["index of all k-points along the path",
                                               ],
                               "bands": [],
                           },
                           })

        comp.new("kspecdist", {"value": k_spec_dist,
                               "units": Units(eV = 0, Ang = -1, muB = 0),
                               "origin_story": """Distance of special points along the specified k-path.
                               For example, you can use this for x-location of special
                               points on your band structure plot.""",
                               "indices_info": {
                                   "canonical_names": "s",
                                   "explanation": ["index of the special k-points along the path",
                                                   ],
                                   "bands": [],
                               },
                               })

        comp.new("kspeclabels", {"value": np.array(pnts),
                                 "units": Units(eV = 0, Ang = 0, muB = 0),
                                 "origin_story": "Labels of special k-points along the specified k-path.",
                                 "indices_info": {
                                     "canonical_names": "s",
                                     "explanation": ["index of the special k-points along the path",
                                                     ],
                                     "bands": [],
                                 },
                                 })

        return comp

    def _compute_own_hashes(self):
        ret = {}
        comp = self.do_mesh([4, 2, 3], [0.123, 0.298, 0.784], ["psi", "A", "S", "dEdk"])
        m_e = comp["E"]
        m_psi = comp["psi"]
        m_a = comp["A"]
        m_s = comp["S"]
        m_dEdk = comp["dEdk"]

        # use energies as one of the hashes
        ret["E"] = m_e

        # use non-hermiticity of A as a check.  This quantity should
        # ideally be very small.
        ret["nonhermA"] = np.sum(np.abs(m_a - m_a.transpose((0, 2, 1, 3)).conjugate()), axis = (1, 2))
        ret["nonhermS"] = np.sum(np.abs(m_s - m_s.transpose((0, 2, 1, 3)).conjugate()), axis = (1, 2))

        # force A and S to be hermitean, so that later there are no
        # issues in computing eigenvalues
        m_a = 0.5*(m_a + m_a.transpose((0, 2, 1, 3)).conjugate())
        m_s = 0.5*(m_s + m_s.transpose((0, 2, 1, 3)).conjugate())

        # signs in front of Bloch states will depend on machine
        # you use. So I just make sure that these signs are randomized
        # each time you run this code you get a different sign.
        # This ensures that signs don't match accidentally.
        rndS = np.random.RandomState(seed = None)
        rnd = (rndS.random(m_a.shape[1]) > 0.5)
        for i,r in enumerate(rnd):
            if r == 1:
                m_a[:,:,i,:] *= -1.0
                m_a[:,i,:,:] *= -1.0
                m_s[:,:,i,:] *= -1.0
                m_s[:,i,:,:] *= -1.0

        m_de = 1.0j*np.sin(m_e[:,:,None] - m_e[:,None,:])

        # this hash will keep track of A and S.  Need to do product to get rid of dependence
        # on the trivial gauge (multiplying a single u-state with a minus sign)
        ret["combined_hash"] = \
            opteinsum("AijB,AjiB,     Aij -> AB", m_s, m_s,      m_de) + \
            opteinsum("AijB,AjiB,     Aij -> AB", m_s, m_a,      m_de) + \
            opteinsum("AijB,AjkB,AkiB,Aij -> AB", m_s, m_s, m_a, m_de) + \
            opteinsum("AijB,AjkB,AkiB,Aij -> AB", m_s, m_a, m_a, m_de) + \
            opteinsum("AijB,AjkB,AkiB,Aij -> AB", m_a, m_a, m_a, m_de)

        # this hash will keep track of dEdk
        ret["dEdk_hash"] = np.sum(m_dEdk) + 1.0j*np.sin(m_dEdk)

        # this hash will keep track of psi
        m_psi_abs2 = np.abs(m_psi)**2
        ret["psi_hash_0"] = np.sum(np.sin(m_psi_abs2[:, 0::2, :] + m_psi_abs2[:, 1::2, :]))
        ret["psi_hash_1"] = opteinsum("knp, kmp -> ", np.conjugate(m_psi), m_psi)
        ret["psi_hash_2"] = opteinsum("knp, knr -> ", np.conjugate(m_psi), m_psi)

        return ret

class _ComputatorWf():
    r"""

    This is a class for an object that stores various quantities and then parses
    mathematical expressions to compute various other physical quantities.  It has
    a funny archaic name so that it is not confused with "calculators" from Wannier Berri.

    In most cases, you will not need to create computator object on your own using
    a constructor. Instead, after loading a database you should use
    :func:`do_mesh <wfbase.DatabaseWf.do_mesh>`,  :func:`do_path <wfbase.DatabaseWf.do_path>`,
    or :func:`do_list <wfbase.DatabaseWf.do_list>` to create the object from this class.
    See the example below.

    Each quantity has a "value", physical "unit", and sometimes other data, such as
    information about how the quantity was constructed, etc.

    The "value" of the quantity is simply a numpy array and it can be modified
    by the user with any numpy operation, such as transpose or reshape or by slicing
    using the [] operator.

    Quantities stored in the computator are constructed in one of these three ways.

    * First, the quantity could be precomputed by the code.  For example, if you
      construct this object using :func:`do_mesh <wfbase.DatabaseWf.do_mesh>` the
      code will precompute various quantities such as electron band energies,
      Berry connection, and so on.

    * Second, quantities could be added to the computator by the user by calling
      the :func:`new <wfbase._ComputatorWf.new>` function.  These added quantites are added
      as ordinary numpy arrays, along with some additional information sent
      to the :func:`new <wfbase._ComputatorWf.new>` function.

    * Third, quantities can be evaluated using the :func:`evaluate <wfbase._ComputatorWf.evaluate>`
      function.  This function parses a mathematical expression using currently available
      quantities and it then stores the resulting quantity in the computator. For example,
      one can use precomputed band energy and Berry connection to compute the optical
      conductivity, and so on.

    .. _structure_computator:

    Here is an example of the structure of the computator object *comp*.  Two quantities
    (*E* and *A*) in this computator were created using function :func:`do_mesh <wfbase.DatabaseWf.do_mesh>`.
    The third quantity (*sigma*) was computed using :func:`evaluate <wfbase._ComputatorWf.evaluate>`.
    See :ref:`examples page <examples>` for various examples that use computators with these quantites.
    For example, you could take a look at :ref:`this example <sphx_glr_all_examples_example_ahc.py>`.
    Each quantity below contains several keys, such as *value* or *units*.  Third column below shows
    you how to access these these keys.
    
    +----------+----------+---------------------------------+
    | Quantity | Key      | How to access?                  |
    +==========+==========+=================================+
    | E        | value    | | *comp["E"]*                   |
    |          |          | | *comp.get("E", "value")*      |
    +          +----------+---------------------------------+
    |          | units    | | *comp.get_units("E")*         |
    |          |          | | *comp.get("E", "units")*      |
    +          +----------+---------------------------------+
    |          | ...      |                                 |
    +----------+----------+---------------------------------+
    | A        | value    | | *comp["A"]*                   |
    |          |          | | *comp.get("A", "value")*      |
    +          +----------+---------------------------------+
    |          | units    | | *comp.get_units("A")*         |
    |          |          | | *comp.get("A", "units")*      |
    +          +----------+---------------------------------+
    |          | ...      |                                 |
    +----------+----------+---------------------------------+
    | sigma    | value    | | *comp["sigma"]*               |
    |          |          | | *comp.get("sigma", "value")*  |
    +          +----------+---------------------------------+
    |          | units    | | *comp.get_units("sigma")*     |
    |          |          | | *comp.get("sigma", "units")*  |
    +          +----------+---------------------------------+
    |          | latex    | | *comp.get_latex("sigma")*     |
    |          |          | | *comp.get("sigma", "latex")*  |
    +          +----------+---------------------------------+
    |          | ...      |                                 |
    +----------+----------+---------------------------------+
    | ...      |          |                                 |
    +----------+----------+---------------------------------+


    To get a list of all quantites in the computator, use function
    :func:`all_quantities <wfbase._ComputatorWf.all_quantities>`. To get all keys stored for a
    specific quantity, use function :func:`all_quantity_keys <wfbase._ComputatorWf.all_quantity_keys>`.
    You can get more information about the quantites in the computator using function
    :func:`info <wfbase._ComputatorWf.info>`. :ref:`Here <quantities>` you can find an example
    output of function :func:`info <wfbase._ComputatorWf.info>`.


    Example usage::

        import wfbase as wf

        # open a database file on bcc phase of iron
        db = wf.load("data/fe_bcc.wf")

        # create a computator object called "comp"
        # the type of this object is _ComputatorWf
        comp = db.do_mesh(formatted_output_latex = False)

        # add new quantity "sigma" to the object comp,
        # evaluate this quantity from the expression below
        comp.evaluate("sigma_ij <= (j / (numk * volume)) * (f_km - f_kn) * A_knmi * A_kmnj")

        # add new quantity called "omicron" to the object comp
        comp.new("omicron", "0.7 * eV")

        # change value of one of the previously stored quantity
        comp.replace("eta", "0.05 * eV")

        # print information about all quantities in this computator
        comp.info()
        # print information about quantity "sigma"
        comp.info("sigma")

        # access some of the stored quantities
        print(comp["omicron"])
        print(comp["sigma"])

        # change the value stored for quantity "sigma"
        # Now sigma is no longer a 3x3 matrix, but a
        # vector.  Also, we multiplied it with 10.
        comp["sigma"] = 10.0 * comp["sigma"][0, :]

        # print information about quantity "sigma"
        # Now a warning will be printed that quantity sigma
        # has been modified.
        comp.info("sigma")

        # since [] returns a copy of the stored object,
        # the following will NOT change the value stored for quantity "sigma"
        # but it will only change the copy of hte array.
        cpy = comp["sigma"]
        cpy = 20.0 * cpy
        print(comp["sigma"]) # --> unchanged by the previous line

        # the following also does not change the value of sigma
        comp["sigma"][:] = 1.0

        # the following, on the other hand, does change the value of sigma
        comp["sigma"] = 1.0

        # list all quantities stored in this computator
        print(comp.all_quantities())

    """

    def __init__(self,
                 quantities={},
                 formatted_output_latex = True,
                 doublet_indices = False,
                 loaded_from_wannierberri = False):
        self.__quantities = quantities

        self._computated_using = ""

        self._total_seconds_initialize = None

        self._db_loaded_from_wannierberri = loaded_from_wannierberri

        for k in quantities.keys():
            self.__check_core(k)
            self.__check_quantity_has_required(quantities[k])
            self.__set_to_numpy_array(k)

        self.__allow_early_changes = True
        if doublet_indices == True:
            self.__try_to_convert_all_band_indices_from_singlets_to_doublets()
            for k in quantities.keys():
                self.__check_core(k)
                self.__check_quantity_has_required(quantities[k])
                self.__set_to_numpy_array(k)
        self.__doublet_indices = doublet_indices
        self.__allow_early_changes = False

        self.__did_user_mess_with_values = {}
        for core in self.all_quantities():
            self.__did_user_mess_with_values[core] = False

        self.__added_later_by_user = []

        self.__formatted_output_latex = formatted_output_latex

        self._order_parsed = 0

        self._reorg_parser = ParserReorg()

        # create parsing object from pyparsing
        # This code is heavily based on eval_arith.py and simpleArith.py from pyparsing github repository
        #
        # basic building blocks are either integers or symbols that have letters with additional special characters
        # order of these three things below matters
        operand = _get_operand()
        # This class will be called whenever you wish to evaluate one operand
        operand.set_parse_action(EvalConstVar)
        self._parser = pp.infix_notation(
            operand,
            [
                (pp.oneOf("Real Imag") , 1, pp.opAssoc.RIGHT, EvalFuncOp    ), # various function calls
                (         "#"          , 1, pp.opAssoc.RIGHT, EvalConjugOp  ), # complex conjugation
                (         "&"          , 1, pp.opAssoc.RIGHT, EvalDOneOp    ), # operation of dividing 1 by the object
                (         "^"          , 2, pp.opAssoc.LEFT , EvalPowerOp   ), # power raising (strictly speaking this should be a RIGHT not LEFT to follow conventions.  But we don't allow user to do A^B^C so it doesn"t matter.)
                (pp.oneOf("+ -")       , 1, pp.opAssoc.RIGHT, EvalSignOp    ), # sign in front of an object
                (pp.oneOf("* /")       , 2, pp.opAssoc.LEFT , EvalMultDivOp ), # multiplication and division
                (pp.oneOf("+ -")       , 2, pp.opAssoc.LEFT , EvalAddSubOp  ), # addition and subtraction
                (pp.oneOf("<= <+= <<="), 2, pp.opAssoc.LEFT , EvalArrowOp   ), # perform assignment
            ],
        )

        # this is parser for brute force sums
        operand_bfs = _get_operand()
        operand_bfs.set_parse_action(BfsConstVar)
        self._parser_brute_force_sums = pp.infix_notation(
            operand_bfs,
            [
                (pp.oneOf("Real Imag") , 1, pp.opAssoc.RIGHT, BfsFuncOp    ),
                (         "#"          , 1, pp.opAssoc.RIGHT, BfsConjugOp  ),
                (         "&"          , 1, pp.opAssoc.RIGHT, BfsDOneOp    ),
                (         "^"          , 2, pp.opAssoc.LEFT , BfsPowerOp   ),
                (pp.oneOf("+ -")       , 1, pp.opAssoc.RIGHT, BfsSignOp    ),
                (pp.oneOf("* /")       , 2, pp.opAssoc.LEFT , BfsMultDivOp ),
                (pp.oneOf("+ -")       , 2, pp.opAssoc.LEFT , BfsAddSubOp  ),
                (pp.oneOf("<= <+= <<="), 2, pp.opAssoc.LEFT , BfsArrowOp   ),
            ],
        )

        self._verbose_evaluate = False

    def __try_to_convert_all_band_indices_from_singlets_to_doublets(self):
        if self["numwann"]%2 != 0:
            _raise_value_error("It is not possible to use doublet-index notation if you have odd number of states.")

        ene = self["E"]
        de = ene[:, 1::2] - ene[:, :-1:2]
        if np.max(np.abs(de)) > 1.0E-11:
            _raise_value_error("It is not possible to use doublet-index notation as your bands are not at least twice degenerate.  Your system either has broken P*T symmetry, or something went wrong with symmetrization.")

        for core in self.all_quantities():
            if "indices_info" in self.all_quantity_keys(core):
                old_quant = self.__get_entire_quantity(core)
                new_quant = self.__do_doublets_one_quant(old_quant, core, value_already_doubled = False)
                self.__change_quantity(core, new_quant)

    def __do_doublets_one_quant(self, old_quant, core, value_already_doubled = False):
        quant = deepcopy(old_quant)

        indices_info = quant["indices_info"]

        if value_already_doubled == False:
            for b in indices_info["bands"]:
                if quant["value"].shape[b] != self["numwann"]:
                    _raise_value_error("What is marked as band index in quantity " + core + " does not have the right shape.")

        new_shape = []
        tmp0 = 0
        orig_inds = []
        tmp1 = 0
        new_inds = []
        for i,ii in enumerate(quant["value"].shape):
            if i in indices_info["bands"]:
                new_shape.append(ii//2)
                new_shape.append(2)
                new_inds.append(str(tmp1) + str(tmp1 + 1))
                tmp1 += 2
            else:
                new_shape.append(ii)
                new_inds.append(str(tmp1))
                tmp1 += 1
            orig_inds.append(str(tmp0))
            tmp0 += 1

        if value_already_doubled == False:
            quant["value"] = quant["value"].reshape(list(new_shape))

        if "format" in quant.keys():
            new_format = quant["format"]
            for i in range(len(new_inds) - 1, -1, -1):
                new_format = new_format.replace("*" + orig_inds[i], "*" + "*".join(new_inds[i]))
            quant["format"] = new_format

        if "format_conjugate" in quant.keys():
            new_format_conjugate = quant["format_conjugate"]
            for i in range(len(new_inds) - 1, -1, -1):
                new_format_conjugate = new_format_conjugate.replace("*" + orig_inds[i], "*" + "*".join(new_inds[i]))
            quant["format_conjugate"] = new_format_conjugate

        new_indices_info = {}
        new_canonical_names = ""
        for j, jj in enumerate(indices_info["canonical_names"]):
            if j in indices_info["bands"]:
                new_canonical_names += jj.lower() + jj.upper()
            else:
                new_canonical_names += jj
        if len(new_canonical_names) != len(list(set(new_canonical_names))):
            _raise_value_error("Indices not unique: " + new_canonical_names)
        new_indices_info["canonical_names"] = new_canonical_names

        if "definition" in indices_info.keys():
            new_definition = indices_info["definition"]
            if new_definition is not None:
                for i in range(len(new_inds) - 1, -1, -1):
                    new_definition = new_definition.replace("*" + orig_inds[i], "*" + "*".join(new_inds[i]))
            new_indices_info["definition"] = new_definition

        new_bands = []
        for b in indices_info["bands"]:
            new_bands.append(list(map(int, list(new_inds[b]))))
        new_indices_info["bands"] = new_bands

        new_explanation = []
        for j, exp in enumerate(indices_info["explanation"]):
            if j in indices_info["bands"]:
                new_explanation.append("doublet index (" + exp + ")")
                new_explanation.append("index within the doublet (0 or 1)")
            else:
                new_explanation.append(exp)
        new_indices_info["explanation"] = new_explanation

        quant["indices_info"] = new_indices_info

        return quant

    def get_initialization_time(self):
        r"""
        Returns time, in seconds, it took to initialize this computator.
        Most of this time is spent in calls to Wannier Berri to get
        all required quantities, such as band energy, Berry connection,
        etc.
        """
        return self._total_seconds_initialize

    def __getitem__(self, core):
        self.__does_core_exist(core)
        return deepcopy(self.__quantities[core]["value"])

    def __setitem__(self, core, value):
        if core not in self.all_quantities():
            _raise_value_error("Quantity "+ core + " is not defined.  You must add it using .new(...) method function.")

        self.__quantities[core]["value"] = value
        self.__did_user_mess_with_values[core] = traceback.extract_stack()
        for k in ["origin_story", "latex", "exec"]:
            if k in self.all_quantity_keys(core):
                del self.__quantities[core][k]
        self.__quantities[core]["origin_story"] = ""
        self.__set_to_numpy_array(core)

    def __change_quantity(self, core, quantity):
        self.__does_core_exist(core)
        if self.__allow_early_changes == True:
            self.__quantities[core] = quantity

    def all_quantities(self):
        r"""

        Returns a list of all quantities stored in the computator object.

        :returns:
          * **lst** -- list containing all quantities stored in the object.

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # compute quantities on a mesh
            comp = db.do_mesh()

            # get a list of all quantities stored in comp
            lst = comp.all_quantities()

            # access one of the quantities in comp
            print(comp["E"])

        """
        return list(self.__quantities.keys())

    def all_quantity_keys(self, core):
        r"""

        Returns a list of keys for all data stored about the single quantity *core*.

        :returns:
          * **lst** -- list containing keys for quantity *core*

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # compute quantities on a mesh
            comp = db.do_mesh()

            # get a list of all quantities stored in comp
            keys = comp.all_quantity_keys("E")

            # access one of the keys (units) stored about quantity "E"
            print(str(comp.get("E", "units")))

        """
        self.__does_core_exist(core)
        return sorted(list(self.__quantities[core].keys()))

    def get(self, core, key = "value"):
        r"""

        Returns value of the specified quantity *core*.  This returned
        value (typically an array of numbers) is a copy of the value
        stored in the computator object.  Therefore, if you change the copy
        of this value, the one stored in the computator object will not
        be changed.  If you want to actually change the value of the
        quantity stored in the computator use the [] operator, as shown
        in the example below, or use the :func:`replace <wfbase._ComputatorWf.replace>`
        function.

        :param core: name of the quantity you wish to get

        :param key: Which part of the quantity you want to get.  The default is "value"
          which returns the value of the quantity (typically an array of numbers, for
          example).  This could also be "units" to get the physical unit of the
          quantity.

        :returns:
          * **val** -- returned value of the specified quantity.

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            # now "energy" is a shallow copy of quantity "E"
            energy = comp.get("E")
        
            units = comp.get("E", "units")

            print("Energy of the first band at the first kpoint is", energy[0, 0], "in units of", units)

            # this will change array "energy" but not quantity "E" stored in the comp!
            energy = energy + 10.0

            # this will change the quantity "E" stored in the comp
            comp["energy"] = energy * 3.4

        """
        self.__does_core_exist(core)
        return deepcopy(self.__quantities[core][key])

    def __get_entire_quantity(self, core):
        self.__does_core_exist(core)
        return deepcopy(self.__quantities[core])

    def get_shape(self, core):
        r"""

        Returns the shape of the specified quantity *core*.

        :param core: name of the quantity whose shape you want to get.

        :returns:
          * **shp** -- shape of the quantity

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            print("Band energy is stored in array of this shape: ", comp.get_shape("E"))

        """
        self.__does_core_exist(core)
        return self.__quantities[core]["value"].shape

    def get_ndim(self, core):
        r"""

        Returns the dimensionality of the specified quantity *core*.

        :param core: name of the quantity whose dimensionality you want to get.

        :returns:
          * **ndim** -- dimensionality.  0 for single number.  1 for vector, etc.

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            print("Band energy is stored in array of this dimensionality: ", comp.get_ndim("E"))

        """
        self.__does_core_exist(core)
        return self.__quantities[core]["value"].ndim

    def get_latex(self, core):
        r"""

        This returns an object that stores information about LaTeX'ed
        definition of *core*.  Here *core* is a quantity that was computed
        using the :func:`evaluate <wfbase._ComputatorWf.evaluate>` function.

        :param core: Name of the quantity.

        :returns:
          * **lat** -- object that contains information about LaTeX'ed
            definition of *core*.

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            # evaluate some object
            comp.evaluate("sigma_ij <= (j / (numk * volume)) * (f_km - f_kn) * A_knmi * A_kmnj")

            # now get LaTeX'ed data about this object
            lat = comp.get_latex("sigma")

            wf.render_latex(lat, "test.png")
            wf.display_in_separate_window("test.png")
            wf.display_in_terminal("test.png")

        """

        self.__does_core_exist(core)
        if self.__did_user_mess_with_values[core] == False:
            if self._is_parsed(core):
                if "ind" in self.all_quantity_keys(core):
                    use_ind = self.get(core, "ind")
                else:
                    use_ind = ""
                if "latex" in self.all_quantity_keys(core):
                    use_latex = self.get(core, "latex")
                else:
                    use_latex = ""
                return _LatexExpression(core, use_ind, use_latex)

        return None

    def get_units(self, core):
        r"""

        Returns units of quantity *core*.  This is a product of arbitrary power
        of Angstroms, electron-volts, and Bohr's magneton.

        :param core: Name of the quantity.

        :returns:
          * **unit** -- object that contains information about units.

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            # evaluate some object
            comp.evaluate("sigma_ij <= (j / (numk * volume)) * (f_km - f_kn) * A_knmi * A_kmnj")

            print("Units of sigma are: ", comp.get_units("sigma"))

        """

        self.__does_core_exist(core)
        return self.get(core, "units")

    def compute_in_SI(self, core, prefactor = None):
        r"""
        Internally in WfBase all quantities are specified in units of eV, Angstrom, and
        Bohr magneton.  This function will return the numerical value of the
        physical quantity *core* in SI units.  If *prefactor* is not None (default),
        then the returned numerical value will be multiplied by *prefactor*.

        Here *prefactor* is a string that consists of various constants
        of nature (hbar, electron charge, etc, as listed below).  If prefactors are specified
        then this function will return an additional object that contains LaTeX expression
        for the product of the *prefactor* and *core*.

        This function does not change any property of the quantity *core* itself.  Everything
        in the computator, and all cores, are always specified in eV, Angstrom, and Bohr
        magneton.  The only numbers in SI units are those returned by this function.

        :param core: name of the quantity you wish to get.

        :param prefactor: optional parameter.  If specified then the returned quantity will
          be multiplied by this prefactor.  The prefactor provided to this function is a string
          of the form "e^2 / (hbar * epszero)" or similar.  The allowed constants are
          "e" for electron charge, "epszero" for vacuum permittivity, "c" for speed of light
          "me" for electron mass, and "hbar" for reduced Planck's constant.

        :returns:
          * **val** -- returned value of the specified quantity.

          * **lat**  -- object that contains LaTeX expression for the product of *core* and *prefactor*.
            Returned only if *prefactor* is not *None* (default).

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            # evaluate some object
            comp.evaluate("sigma_ij <= (j / (numk * volume)) * (f_km - f_kn) * A_knmi * A_kmnj")

            # get units of this object
            print("Units of sigma are: ", comp.get_units("sigma"))

            # convert units to SI and multiply with e^2/hbar
            result, result_latex = comp.compute_in_SI("sigma", "e^2 / hbar")
            print("result = ", result)

            wf.render_latex(result_latex, "latex.png")
            wf.display_in_separate_window("latex.png")
            wf.display_in_terminal("latex.png")

        """
        self.__does_core_exist(core)

        ret_value = self.__quantities[core]["units"]._to_SI(self.get(core, "value"))

        if prefactor is not None:
            ret_latex_obj= self.get_latex(core)
            data = _parse_prefactor_SI_units_fundamental_constants(prefactor)
            ret_value = data._numerical_value() * ret_value
            ret_latex_obj = _LatexExpression(core = "",
                                             ind = "",
                                             rhs = ret_latex_obj._rhs,
                                             prefactor = data._to_latex())
            return (ret_value, ret_latex_obj)
        else:
            return ret_value

    def get_as_dictionary(self, want_cores_in, key = "value"):
        r"""

        Returns values of multiple quantities at once in form of a dictionary.

        :param want_cores_in: A string or a list of strings.  Each string is either
          a name of the quantity you want to return, or it contains an asterisk or question mark
          to match possibly multiple quantities at once.  See the example below.

        :param key: Which part of the quantity you want to get.  The default is "value"
          which returns the values of the quantities (typically an array of numbers, for
          example).

        :returns:
          * **dic** -- dictionary containing values of all quantities that match.

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            # this will return all quantities that match
            dic = comp.get_as_dictionary(["atom*", "A", "num?"])

            print(dic["numk"])

        """

        if isinstance(want_cores_in, str):
            want_cores = [want_cores_in]
        else:
            want_cores = want_cores_in
        ret = {}
        for k in self.all_quantities():
            found_match = False
            for c in want_cores:
                if fnmatch.fnmatch(k, c):
                    found_match = True
            if found_match == True:
                ret[k] = self.get(k, key)
        return ret

    def new(self, core, data, units_as = None):
        r"""

        Adds a new quantity to the computator.

        :param core: This is the name of the new quantity.

        :param data: This is the data associated with the new quantity.
          This can be one of three things.

          * First option -- is that *data* can be a dictionary that contains key
            "value" that is a number or a numpy array.  The
            dictionary can also contain key "units" (defaults to dimensionless).
            The key "units" should be of type :class:`Units <wfbase.Units>`
            as shown in the example below.
            User can also specify key "format" and "format_conjugate" which
            give a way to format this quantity in LaTeX.  See the example
            below how to use "format".  In short, one needs to specify \*0 at the place
            in the LaTeX expression where the first index of the quantity goes,
            \*1 for the second, etc.  Another entry in the dictionary
            could be "origin_story" which is a string describing the quantity *core*.

          * Second option -- is that *data* is a string, such
            as "3.0 eV * muB^2 / Ang" or similar.  Allowed units are eV, Ang, and muB,
            for electronvolt, angstrom, and bohr radius.  You must use multiplication
            signs between units, such as "eV * muB".  (You are not allowed to
            use "eV muB" or "3.0 eV".) You can use parentheses, division, and power
            operator (^).

          * Third option -- is to simply make *data* a number or a numpy array. The
            units will be set to dimensionless by default (unless you specified those
            with parameter *units_as*).

        :param units_as: Ignored if set to *None* (default).  Otherwise, units of the
            new quantity will be equal to the units of quantity *units_as*.  Stops
            if units were specified through parameter *data* (either first or second option
            above).

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            # create new quantity with dimensionless value of 3.0
            comp.new("alpha", 3.0)

            # create new quantity with value of 3.0 * eV^2 * muB^2 / Ang
            # (notice that below there must be a multiplication sign after "3.0")
            comp.new("beta", "3.0 * (eV * muB)^2 / Ang")

            # different way to achieve the same thing
            comp.new("gamma", {"value": 3.0, "units": wf.Units(eV = 1)})

            # adding extra information about the quantity
            comp.new("delta", {"value": 3.0,
                               "units": wf.Units(eV = 1),
                               "origin_story": "Give here some information about delta."})

            # information about how to display this quantity
            comp.new("epsilon", {"value": np.array([[ 10.0       ,  9.0 + 3.0j,  1.0 + 2.0j],
                                                    [  9.0 - 3.0j, 12.0       , 22.0 - 8.0j],
                                                    [  1.0 - 2.0j, 22.0 + 8.0j, 45.0       ]]),
                                 "units": wf.Units(eV = 1),
                                 "format": r"\langle v_{*0} \lvert X \rvert v_{*1} \rangle",
                                 "format_conjugate": r"\langle v_{*1} \lvert X \rvert v_{*0} \rangle"})

            # one can also define quantities that are vectors, or tensors in general
            comp.new("zeta", [3.0, 4.0, 5.0, 6.0])

            # units of new quantity eta will be copy-pasted from gamma
            comp.new("eta", 3.0, units_as = "gamma")

            comp.info("alpha")
            comp.info("beta")
            comp.info("gamma")
            comp.info("delta")
            comp.info("epsilon")

            # evaluate new quantity using quantity defined earlier
            comp.evaluate("kappa_ik <= epsilon_ij * #epsilon_kj")

            # expression for kappa should be formatted using the bra-ket notation for epsilon
            comp.info("kappa", display = True)

        """
        if core in self.all_quantities():
            _raise_value_error("Quantity \"" + str(core) + """\" already exists. You can't change its value using function .new.
            Btw, you can change the value of the variable with comp[\"""" + core + """\"] = ... if you want.  You can also
            use .replace instead of .new to rewrite previous quantity with new one.""")
        self.__check_similar_tilde(core)
        if isinstance(data, str):
            data = _parse_value_and_units(data)
        if isinstance(data, int) or \
           isinstance(data, float) or \
           isinstance(data, complex) or \
           isinstance(data, np.ndarray) or \
           isinstance(data, list):
            data = {"value": np.array(data)}
        if "units" not in data.keys():
            if units_as is not None:
                data["units"] = self.get_units(units_as)
            else:
                data["units"] = Units()
        else:
            if units_as is not None:
                _raise_value_error("You specified units twice.  Once using the parameter *data*, second time using parameter *units_as*.  Do one or the other, but not both.")
        self.__check_core(core)
        self.__check_quantity_has_required(data)

        self.__quantities[core] = data
        self.__did_user_mess_with_values[core] = False
        self.__set_to_numpy_array(core)

        self.__added_later_by_user.append(core)

    def replace(self, core, data, units_as = None):
        r"""

        Removes previously existing quantity *core* and replaces it with new
        quantity *core* with data provided in *data*.

        :param core: This is the name of the quantity that you want to replace.

        :param data: This is the data associated with the new quantity.
          Same as *data* parameter used in :func:`new <wfbase._ComputatorWf.new>`.

        :param units_as: Same as *units_as* parameter used in :func:`new <wfbase._ComputatorWf.new>`.

        Example usage::

            import wfbase as wf

            db = wf.load("data/fe_bcc.wf")
            comp = db.do_mesh()
            comp.new("beta", "3.0 * (eV * muB)^2 / Ang")
            comp.replace("beta", "4.0 * (eV * muB)^2 / Ang")

        """

        if core not in self.all_quantities():
            _raise_value_error("Quantity \"" + str(core) + "\" does not exist already, so you can't replace it.  Use the function .new() instead.")
        self.remove(core)
        self.new(core, data, units_as)

    def remove(self, core):
        r"""

        Removes previously existing quantity *core*.

        :param core: This is the name of the quantity that you want to remove.

        Example usage::

            import wfbase as wf

            db = wf.load("data/fe_bcc.wf")
            comp = db.do_mesh()
            comp.new("beta", "3.0 * (eV * muB)^2 / Ang")
            comp.remove("beta")

        """

        if core not in self.all_quantities():
            _raise_value_error("Quantity \"" + str(core) + "\" doesn't exist.   Therefore, it can't be removed.")
        del self.__quantities[core]
        del self.__did_user_mess_with_values[core]

        if core in self.__added_later_by_user:
            self.__added_later_by_user.remove(core)

    def __check_core(self, core):
        if core.count("~") > 1:
            _raise_value_error("Variable name: " + core + " is invalid.  It must contain at most only one ~.")
        if core.startswith("~") == True:
            _raise_value_error("Variable name: " + core + " is invalid.  It can't start with ~.")
        if core.endswith("~") == True:
            _raise_value_error("Variable name: " + core + " is invalid.  It can't end with ~.")
        if not core.replace("~","").isalpha() or core == "":
            _raise_value_error("Variable name: " + core + " is invalid.  It must contain only letters and at most one ~.")

        if core == "j":
            _raise_value_error("Quantity name j is not allowed as it might be confusing " + 
                               "as we use the same symbol to represent square root of negative one.")

    def __does_core_exist(self, core):
        if core not in self.all_quantities():
            _raise_value_error("Specified core \"" + core + "\" does not exist.")

    def __check_similar_tilde(self, core):
        core_use = core.replace("~", "")
        keys = self.all_quantities()
        for k in keys:
            if k.replace("~", "") == core_use:
                _raise_value_error("Quantity \"" + str(core) + """\" does not already exists, but a similarly named quantity
                (ignoring the tilde symbol, ~) does exist!  This is not allowed as these are too similar.  Pick
                a more unique name.""")

    def __check_quantity_has_required(self, data):
        if "value" not in data.keys():
            _raise_value_error("Did not specify value.")
        if "units" not in data.keys():
            _raise_value_error("Did not specify units.")

    def __set_to_numpy_array(self, core):
        self.__quantities[core]["value"] = np.array(self.__quantities[core]["value"])

    def _return_in_latex(self, core, ind, do_latex_conjugate = False):
        if "format" in self.all_quantity_keys(core) and \
           self.__formatted_output_latex == True and \
           self.__did_user_mess_with_values[core] == False:
            performed_latex_conjugate = False
            if do_latex_conjugate == False:
                use_str = self.get(core, "format")
            else:
                if "format_conjugate" in self.all_quantity_keys(core):
                    use_str = self.get(core, "format_conjugate")
                    performed_latex_conjugate = True
                else:
                    use_str = self.get(core, "format")

            ret = _replace_star_with_indices(use_str, ind)

            if do_latex_conjugate == True and performed_latex_conjugate == False:
                ret = r" \overline{ " + ret.strip() + r" } "
        else:
            if ind is None:
                ret = _nicefy_core(core)
            else:
                ret = _nicefy_core(core) + r"_{" + _nicefy_subscript(ind) + r"}"
            if do_latex_conjugate == True:
                ret = r" \overline{ " + ret.strip() + r" } "
        return ret

    def compute_occupation(self, out_core = "f", energy = "E", fermi = "ef", kbtemp = None):
        r"""

        Computes a quantity that has entries close to 1 at all places
        where *energy* is less than *fermi* and 0 otherwise.  Uses a Fermi-Dirac
        distribution if temperature is specified.  Otherwise, temperature is zero.

        .. note::

          This quantity can be used to enforce occupations of states while
          evaluating physical quantities.  The same effect can be achieved
          using the *conditions* tag while calling  :func:`evaluate <wfbase._ComputatorWf.evaluate>`
          function.  The benefit of using *conditions* tag is that it reduces
          the number of operations needed to do the computation.  See examples below

        :param out_core: Name of the occupation factor quantity.  Defaults to "f".
          (This function will remove previously existing quantity with the same name.)

        :param energy: Name of the energy quantity.  Defaults to "E".

        :param fermi: Name of the fermi level quantity (or a float).  Defaults to "ef".

        :param kbtemp: The default is None, which means zero temperature.   You can also specify
          a quantity, or give a floating point number in units of eV.

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            comp.compute_occupation("f", "E", "ef")

            # computes band energy of occupied states
            comp.evaluate("sumA <= E_nk * f_nk")

            # same but without using "f"
            comp.evaluate("sumB <= E_nk", "E_nk < ef")

            # same computation by directly using numpy operations
            import numpy as np
            sumC = np.sum(comp["E"][ comp["E"] < comp["ef"]])

            print(comp["sumA"], comp["sumB"], sumC)

        """
        if out_core in self.all_quantities():
            self.remove(out_core)

        use_energy = np.real(self.get(energy, "value"))

        if isinstance(fermi, str):
            if self.get(fermi, "units")._check_units_the_same(self.get(energy, "units")) == False:
                _raise_value_error("Units of " + energy + " and " + fermi + " are not the same!")
            if self.get_ndim(fermi) != 0:
                _raise_value_error("Fermi level must be a single number.")
            use_fermi = np.real(self.get(fermi, "value"))
        else:
            use_fermi = np.real(float(fermi))

        if kbtemp is None:
            value = np.array(use_energy < use_fermi, dtype = float)
            data = {
                "value": value,
                "origin_story": "Has an entry equal to 1 for all entries where " + 
                str(energy) + " < " + str(fermi) + " otherwise it is 0.",
                "units": Units(eV = 0, Ang = 0, muB = 0),
            }
        else:
            if isinstance(kbtemp, str):
                if self.get(kbtemp, "units")._check_units_the_same(self.get(energy, "units")) == False:
                    _raise_value_error("Units of " + energy + " and " + kbtemp + " are not the same!")
                if self.get_ndim(kbtemp) != 0:
                    _raise_value_error("Temperature must be a single number.")
                use_kbtemp = np.real(self.get(kbtemp, "value"))
            else:
                use_kbtemp = np.real(float(kbtemp))

            value = _fermi_dirac(use_energy, use_fermi, use_kbtemp)
            data = {
                "value": value,
                "origin_story": "Fermi-Dirac occupation factor between 1 and 0. " + 
                "It is close to 1 for all entries where " + 
                str(energy) + " < " + str(fermi) + " otherwise it is close to 0.",
                "units": Units(eV = 0, Ang = 0, muB = 0),
            }

        if "indices_info" in self.all_quantity_keys(energy):
            energy_data = self.get(energy, "indices_info")
            data["indices_info"] = {"canonical_names": energy_data["canonical_names"],
                                    "explanation": energy_data["explanation"],
                                    "bands": energy_data["bands"],
                                    }
        else:
            data["origin_story"] += " The indices of this quantity are the same as those of *" + str(energy) + "*."

        self.new(out_core, data)


    def compute_occupation_derivative(self, out_core = "dfdE", energy = "E", fermi = "ef", kbtemp = 0.05):
        r"""

        Computes a derivative of the Fermi-Dirac distribution.

        :param out_core: Name of the occupation factor quantity.  Defaults to "dfdE".
          (This function will remove previously existing quantity with the same name.)

        :param energy: Name of the energy quantity.  Defaults to "E".

        :param fermi: Name of the fermi level quantity (or a float).  Defaults to "ef".

        :param kbtemp: Quantity (or a floating point number) giving kb*temperature in eV.
          Defaults to 0.05 eV.

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            comp.compute_occupation_derivative("dfdE", "E", "ef", 0.01)

        """
        if out_core in self.all_quantities():
            self.remove(out_core)

        use_energy = np.real(self.get(energy, "value"))

        if isinstance(fermi, str):
            if self.get(fermi, "units")._check_units_the_same(self.get(energy, "units")) == False:
                _raise_value_error("Units of " + energy + " and " + fermi + " are not the same!")
            if self.get_ndim(fermi) != 0:
                _raise_value_error("Fermi level must be a single number.")
            use_fermi = np.real(self.get(fermi, "value"))
        else:
            use_fermi = np.real(float(fermi))

        if isinstance(kbtemp, str):
            if self.get(kbtemp, "units")._check_units_the_same(self.get(energy, "units")) == False:
                _raise_value_error("Units of " + energy + " and " + kbtemp + " are not the same!")
            if self.get_ndim(kbtemp) != 0:
                _raise_value_error("Temperature must be a single number.")
            use_kbtemp = np.real(self.get(kbtemp, "value"))
        else:
            use_kbtemp = np.real(float(kbtemp))

        value = _fermi_dirac_deriv(use_energy, use_fermi, use_kbtemp)
        data = {
            "value": value,
            "origin_story": "Derivative of the Fermi-Dirac occupation factor.",
            "units": Units(eV = -1, Ang = 0, muB = 0),
        }

        if "indices_info" in self.all_quantity_keys(energy):
            energy_data = self.get(energy, "indices_info")
            data["indices_info"] = {"canonical_names": energy_data["canonical_names"],
                                    "explanation": energy_data["explanation"],
                                    "bands": energy_data["bands"],
                                    }
        else:
            data["origin_story"] += " The indices of this quantity are the same as those of *" + str(energy) + "*."

        allinds = ""
        for j in range(self.get_ndim(energy)):
            allinds += "*" + str(j)
        data["format"] = r"\frac{\partial f}{\partial " + energy.strip() + r"_{" + allinds + r"} }"
        data["format_conjugate"] = data["format"]

        self.new(out_core, data)


    def compute_kronecker(self, out_core, core, ind, core2 = None, ind2 = None):
        r"""

        Computes a Kronecker delta symbol that has one on diagonal and zero otherwise.

        :param out_core: Name of the Kronecker delta quantity. No default.
          (This function will remove previously existing quantity with the same name.)

        :param core: The Kronecker delta will have the first index the same shape as
          *ind*-th index of the quantity *core*.

        :param ind: First index to use in the construction of the Kronecker delta.

        :param core2: The Kronecker delta will have the second index the same shape as
          *ind2*-th index of the quantity *core2*.  The default is to use the same as *ind*.

        :param ind2: Second index to use in the construction of the Kronecker delta.  The default
          is to have the same as *ind*.

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            # this will create kronecker with shape numwann * numwann
            comp.compute_kronecker("d", "E", 1)

            # this will include only diagonal parts
            comp.evaluate("B_nm <= d_nm/(E_km - E_kn + j*eta)")

            # this will exclude diagonal parts, where n == m
            comp.evaluate("C_nm <= (1.0 - d_nm)/(E_km - E_kn + j*eta)")

        """

        if out_core in self.all_quantities():
            self.remove(out_core)
        if core2 is None:
            core2 = core
        if ind2 is None:
            ind2 = ind
        self.new(out_core, {"value": np.eye(self.get_shape(core)[ind], self.get_shape(core2)[ind2]),
                            "origin_story": """Kronecker delta.  Equals one when two indices
                                               are the same (on the diagonal) and equals zero otherwise.""",
                            "indices_info": {
                                "canonical_names": "xy",
	                        "explanation": ["same as index #" + str(ind) + " of quantity *" + core,
                                                "same as index #" + str(ind2) + " of quantity *" + core2,
                                                ],
                                "bands": [],
                            },
                            "units": Units(eV = 0, Ang = 0, muB = 0)}
                 )

    def compute_identity(self, out_core, size):
        r"""

        Gives an identity matrix of shape (size, size). One on diagonal, zero off-diagonal.

        :param out_core: Name of this quantity.

        :param size: Size of the matrix.

        :returns:
          * **mat** -- The identity matrix.

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            # this will create 3x3 identity matrix
            comp.compute_identity("one", 3)

        """

        if out_core in self.all_quantities():
            self.remove(out_core)
        self.new(out_core, {"value": np.eye(size, dtype = complex),
                            "origin_story": "Identity matrix.",
                            "indices_info": {
                                "canonical_names": "xy",
	                        "explanation": ["generic index.",
                                                "generic index."
                                                ],
                                "bands": [],
                            },
                            "units": Units(eV = 0, Ang = 0, muB = 0),
                            "format": r"I^{" + str(size) + "}",
                            "format_conjugate": r"I^{" + str(size) + "}"}
                 )

    def compute_photon_energy(self, out_core = "hbaromega", emin = 0.5, emax = 3.0, steps = 31):
        r"""

        Computes an array of photon energy in eV.

        :param out_core: Name of the quantity for photon energy.  Defaults to "hbaromega".
          (This function will remove previously existing quantity with the same name.)

        :param emin: Minimal energy in the range.  Defaults to 0.5.  Units are eV.

        :param emax: Maximal energy in the range.  Defaults to 3.0.  Units are eV.

        :param steps: The number of equidistant steps.  Defaults to 31.

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            # this will create photon energies from 0.5 to 5 eV in 51 steps
            comp.compute_photon_energy("omicron", 0.5, 5.0, 51)

            # this achieves the same thing
            import numpy as np
            comp.new("delta", {"value": np.linspace(0.01, 5.0, 51, endpoint = True),
                               "units": wf.Units(eV = 1)})

        """
        if out_core in self.all_quantities():
            self.remove(out_core)
        self.new(out_core, {"value": np.linspace(emin, emax, steps, endpoint = True),
                            "origin_story": "Photon energies hbar*omega on an equidistant mesh of values.",
                            "indices_info": {
                                "canonical_names": "o",
                                "explanation": ["index of the photon energy",
                                                ],
                                "bands": [],
                            },
                            "units": Units(eV = 1, Ang = 0, muB = 0),
                            "format": r"\hbar \omega_{*0}",
                            "format_conjugate": r"\hbar \omega_{*0}"})

    def __confirm_consistent_definition_of_variables(self, cores, txt):
        for c in cores:
            if c not in self.all_quantities():
                _raise_value_error("Can't compute " + txt.strip() + " without having quantity " + c.strip() + ".")
            if self.__did_user_mess_with_values[c] == True:
                _raise_value_error("Can't compute " + txt.strip() + " because user has changed quantity A, so I'm not sure what happened to it.")
            if c in self.__added_later_by_user:
                _raise_value_error("Can't compute " + txt.strip() + " because user has created their own quantity A.")

    def compute_orbital_character(self, out_core):
        r"""
        Computes approximate orbital character of the wavefunction.

        || < W_p | psi_kn > ||^2

        (If you earlier used doublet_indices = True then the indices
        above on psi are are *knN* instead of *kn*.)

        This is a dimensionless number.  The sum of this number over index *p* is 1.0.

        :param out_core: Name of the quantity for the orbital character.
          (This function will remove previously existing quantity with the same name.)

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            # computes orbital character
            comp.compute_orbital_character("O")

        """
        self.__confirm_consistent_definition_of_variables(["psi"], "orbital character")

        if out_core in self.all_quantities():
            self.remove(out_core)

        if self.__doublet_indices == False:
            tmp = self.evaluate("_knp <= Real(psi_knp * #psi_knp)")
        else:
            tmp = self.evaluate("_knNp <= Real(psi_knNp * #psi_knNp)")

        quant = \
            {"value": tmp,
             "origin_story": """Approximate orbital character of wavefunction
             in terms of localized atomic-like function.  The sum over orbitals
             adds up to 1. The description of localized orbitals are given by
             quantity *orbitallabels* (available only if your database was loaded
             from the WfBase's database).
             """,
             "indices_info": {
                 "definition": r"|| < W_*2 | u_*0*1 > ||^2",
                 "canonical_names": "knmdo",
                 "explanation": ["index of a k-point",
                                 "electron band index",
                                 "localized orbital index",
                                 ],
                 "bands": [1],
             },
             "units": Units(eV = 0, Ang = 0, muB = 0),
             "format" : \
             r"\lvert \langle W_{*2} \vert \psi_{*0 *1} \rangle \rvert^2",
             "format_conjugate" :\
             r"\lvert \langle W_{*2} \vert \psi_{*0 *1} \rangle \rvert^2"}

        if self.__doublet_indices == True:
            quant = self.__do_doublets_one_quant(quant, out_core, value_already_doubled = True)

        self.new(out_core, quant)


    def compute_optical_offdiagonal(self, out_core, hbaromega):
        r"""
        Computes matrix elements for the off-diagonal (interband) interaction
        of electrons with electromagnetic waves.

        < psi_kn | H_offdiagonal | psi_km >

        (If you earlier used doublet_indices = True then the indices above are *knN* and *kmM*
        instead of *kn* and *km*.)

        Units of this quantity are eV.  This effectively assumes that the maximum
        electric field of the applied electromagnetic wave is 1 eV/Ang.

        The exact computed quantity is

        (1 / 2)  (E_kn - E_km) * < u_kn | i del_k_d | u_km > / hbaromega_o

        The diagonal elements (n = m) are set to zero. (When indices are doubled then
        matrix elements within the doublet are set to zero.)

        :param out_core: Name of the quantity for the matrix element.
          (This function will remove previously existing quantity with the same name.)

        :param hbaromega: Name of the quantity containing photon energies.

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            # compute interband optical matrix element for predefined
            # photon energies "hbaromega"
            comp.compute_optical_offdiagonal("L", "hbaromega")

            # this will create photon energies from 0.01 to 5 eV in 51 steps
            comp.compute_photon_energy("omicron", 0.01, 5.0, 51)

            # recompute matrix elements for these new energies
            comp.compute_optical_offdiagonal("L", "omicron")

        """
        if self.get(hbaromega, "units")._check_units_the_same(Units(eV = 1)) == False:
            _raise_value_error("Units of " + hbaromega + " are not eV!")

        self.__confirm_consistent_definition_of_variables(["A", "E"], "optical matrix element")

        if out_core in self.all_quantities():
            self.remove(out_core)

        if self.__doublet_indices == False:
            tmp = self.evaluate("_knmdo <= 0.5 * (E_kn - E_km) * A_knmd / "+hbaromega+"_o")
            for i in range(tmp.shape[1]):
                tmp[:,i,i,:,:] = 0.0
        else:
            tmp = self.evaluate("_knNmMdo <= 0.5 * (E_knN - E_kmM) * A_knNmMd / "+hbaromega+"_o")
            for i in range(tmp.shape[1]):
                tmp[:,i,:,i,:,:,:] = 0.0

        quant = \
            {"value": tmp,
             "origin_story": """Off-diagonal matrix element for electron-light interaction.
             Diagonal elements of this matrix are set to zero by hand.
             The maximal electric field strength of the incoming light is set to 1 V/Ang.
             The unit of the matrix element is therefore energy (eV).
             """,
             "indices_info": {
                 "definition": "0.5 (E_*0*1 - E_*0*2) < u_*0*1 | i del_k_*3 | u_*0*2 > / " + hbaromega + "_*4",
                 "canonical_names": "knmdo",
                 "explanation": ["index of a k-point",
                                 "electron band index of the bra state",
                                 "electron band index of the ket state",
                                 "direction of the E-field of light in Cartesian (0 for x, 1 for y, 2 for z)",
                                 "index for photon energy " + hbaromega,
                                 ],
                 "bands": [1, 2],
             },
             "units": Units(eV = 1, Ang = 0, muB = 0),
             "format" : \
             r"\langle \psi_{*0 *1} \lvert  H^{\rm inter}_{*3 *4} \rvert \psi_{*0 *2} \rangle",
             "format_conjugate" :\
             r"\langle \psi_{*0 *2} \lvert  H^{\rm inter}_{*3 *4} \rvert \psi_{*0 *1} \rangle"}

        if self.__doublet_indices == True:
            quant = self.__do_doublets_one_quant(quant, out_core, value_already_doubled = True)

        self.new(out_core, quant)


    def compute_optical_offdiagonal_polarization(self, out_core, hbaromega, polarization):
        r"""
        Similar to :func:`compute_optical_offdiagonal <wfbase._ComputatorWf.compute_optical_offdiagonal>` but now
        the matrix element is computed for a specified polarization only.

        :param out_core: Name of the quantity for the matrix element.
          (This function will remove previously existing quantity with the same name.)

        :param hbaromega: Name of the quantity containing photon energies.

        :param polarization: String describing the polarization direction. For linearly
          polarized light use "x", or "y", or "z".  For circularly polarized light,
          use one of these "x + i y", "x - i y", "x + i z", "x - i z", "y + i z", "y - i z".

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            # compute interband optical matrix element for predefined
            # photon energies "hbaromega"
            comp.compute_optical_offdiagonal_polarization("L", "hbaromega", "x + i y")

        """

        if self.get(hbaromega, "units")._check_units_the_same(Units(eV = 1)) == False:
            _raise_value_error("Units of " + hbaromega + " are not eV!")

        self.__confirm_consistent_definition_of_variables(["A", "E"], "optical matrix element")

        if out_core in self.all_quantities():
            self.remove(out_core)

        if self.__doublet_indices == False:
            tmp = self.evaluate("_knmdo <= 0.5 * (E_kn - E_km) * A_knmd / "+hbaromega+"_o")
            for i in range(tmp.shape[1]):
                tmp[:,i,i,:,:] = 0.0
        else:
            tmp = self.evaluate("_knNmMdo <= 0.5 * (E_knN - E_kmM) * A_knNmMd / "+hbaromega+"_o")
            for i in range(tmp.shape[1]):
                tmp[:,i,:,i,:,:,:] = 0.0

        use_pol = polarization.replace(" ", "").lower().replace("i", "j")

        if use_pol == "x":
            direction = np.array([1.0, 0.0, 0.0], dtype = complex)
        elif use_pol == "y":
            direction = np.array([0.0, 1.0, 0.0], dtype = complex)
        elif use_pol == "z":
            direction = np.array([0.0, 0.0, 1.0], dtype = complex)
        elif use_pol in ["x+jy", "x+yj"]:
            direction = np.array([1.0, 1.0j, 0.0], dtype = complex)
        elif use_pol in ["x-jy", "x-yj"]:
            direction = np.array([1.0,-1.0j, 0.0], dtype = complex)
        elif use_pol in ["x+jz", "x+zj"]:
            direction = np.array([1.0, 0.0, 1.0j], dtype = complex)
        elif use_pol in ["x-jz", "x-zj"]:
            direction = np.array([1.0, 0.0,-1.0j], dtype = complex)
        elif use_pol in ["y+jz", "y+zj"]:
            direction = np.array([0.0, 1.0, 1.0j], dtype = complex)
        elif use_pol in ["y-jz", "y-zj"]:
            direction = np.array([0.0, 1.0,-1.0j], dtype = complex)
        else:
            _raise_value_error("Unknown polarization. Must be one of these: x, y, z, x+iy, x-iy, x+iz, x-iz, y+iz, y-iz.")

        if use_pol in ["x", "y", "z"]:
            pol_type = "linear"
        else:
            pol_type = "circular"

        latex_pol = use_pol.replace("j", "i").replace("+", " + ").replace("-", " - ")
        latex_pol = latex_pol.replace("x", r"{\rm x}").replace("y", r"{\rm y}").replace("z", r"{\rm z}")
        if "+" in latex_pol:
            latex_pol_conj = latex_pol.replace("+", "-")
        elif "-" in latex_pol:
            latex_pol_conj = latex_pol.replace("-", "+")
        else:
            latex_pol_conj = latex_pol

        if self.__doublet_indices == False:
            tmp = opteinsum("knmdo, d -> knmo", tmp, np.conjugate(direction))
        else:
            tmp = opteinsum("knNmMdo, d -> knNmMo", tmp, np.conjugate(direction))

        quant = {"value": tmp,
                 "origin_story": """Off-diagonal matrix element for electron-light interaction
                 for a """ + pol_type + " " + use_pol + """ polarization of light.
                 Diagonal elements of this matrix are set to zero by hand.
                 The maximal electric field strength of the incoming light is set to 1 V/Ang.
                 The unit of the matrix element is therefore energy (eV).
                 """,
                 "indices_info": {
                     "definition": "0.5 (E_*0*1 - E_*0*2) < u_*0*1 | i delk_" + use_pol.strip() + " | u_*0*2 > / " + hbaromega + "_*3",
                     "canonical_names": "knmo",
                     "explanation": ["index of a k-point",
                                     "electron band index of the bra state",
                                     "electron band index of the ket state",
                                     "index for photon energy " + hbaromega,
                                     ],
                     "bands": [1, 2],
                 },
                 "units": Units(eV = 1, Ang = 0, muB = 0),
                 "format" : \
                 r"\langle \psi_{*0 *1} \lvert  H^{\rm inter}_{ " + latex_pol      + r", *3 } \rvert \psi_{*0 *2} \rangle",
                 "format_conjugate" :\
                 r"\langle \psi_{*0 *2} \lvert  H^{\rm inter}_{ " + latex_pol_conj + r", *3 } \rvert \psi_{*0 *1} \rangle"}

        if self.__doublet_indices == True:
            quant = self.__do_doublets_one_quant(quant, out_core, value_already_doubled = True)

        self.new(out_core, quant)

    def compute_hbar_velocity(self, out_core):
        r"""

        Computes matrix elements for the diagonal (intraband) and
        off-diagonal (interband) velocity operator times hbar,

        < psi_kn | hbar * v_d | psi_km >

        Units of this quantity are eV * Ang

        The exact computed quantity for the diagonal is

        (d E_kn / d k_d)  delta_nm

        for off-diagonal (n != m) it is

        i  (E_kn - E_km) * < u_kn | i del_k_d | u_km >

        (If you earlier used doublet_indices = True then the indices above are *knN* and *kmM*
        instead of *kn* and *km*.)

        :param out_core: Name of the quantity for the matrix element.
          (This function will remove previously existing quantity with the same name.)

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            comp.compute_hbar_velocity("hbarv")

        """

        self.__confirm_consistent_definition_of_variables(["A", "E", "dEdk"], "velocity matrix element")

        if out_core in self.all_quantities():
            self.remove(out_core)

        if self.__doublet_indices == False:
            tmp = self.evaluate("_knmd <= 1.0j * (E_kn - E_km) * A_knmd")
            diagonal = self["dEdk"]
            for i in range(tmp.shape[1]):
                tmp[:,i,i,:] = diagonal[:,i,:]
        else:
            tmp = self.evaluate("_knNmMd <= 1.0j * (E_knN - E_kmM) * A_knNmMd")
            diagonal = self["dEdk"]
            for i in range(tmp.shape[1]):
                for I in range(tmp.shape[2]):
                    for J in range(tmp.shape[4]):
                        if I == J:
                            tmp[:,i,I,i,J,:] = diagonal[:,i,I,:]
                        else:
                            tmp[:,i,I,i,J,:] = 0.0

        quant = {"value": tmp,
                 "origin_story": """Matrix element of hbar*velocity operator.
                 Includes both diagonal elements (computed from the Fermi velocity) and off-diagonal
                 elements (computed from the Berry connection).
                 """,
                 "indices_info": {
                     "definition": "[if *1 == *2]  (d E_*0*1 / d k_*3) delta_*1*2\n[if *1 != *2]  i (E_*0*1 - E_*0*2) < u_*0*1 | i del_k_*3 | u_*0*2 >",
                     "canonical_names": "knmd",
                     "explanation": ["index of a k-point",
                                     "electron band index of the bra state",
                                     "electron band index of the ket state",
                                     "direction of velocity in Cartesian (0 for x, 1 for y, 2 for z)",
                                     ],
                     "bands": [1, 2],
                 },
                 "units": Units(eV = 1, Ang = 1, muB = 0),
                 "format" : \
                 r"\langle \psi_{*0 *1} \lvert  \hbar v_{*3} \rvert \psi_{*0 *2} \rangle",
                 "format_conjugate" :\
                 r"\langle \psi_{*0 *2} \lvert  \hbar v_{*3} \rvert \psi_{*0 *1} \rangle"}

        if self.__doublet_indices == True:
            quant = self.__do_doublets_one_quant(quant, out_core, value_already_doubled = True)

        self.new(out_core, quant)


    def _process_all_greater_lesser(self, conditions, code, brute_force_sums):

        conditions = re.split(",|:|;|\n", conditions)

        self._filters = []

        counter = 0

        if brute_force_sums == False:
            code._add_raw("_f = []\n", to_preamble = True)

        for cond in conditions:
            cond = cond.strip()
            if cond == "":
                continue
            if "<" in cond:
                sp = cond.split("<")
                op = "<"
            elif ">" in cond:
                sp = cond.split(">")
                op = ">"
            elif "!=" in cond:
                continue
            else:
                _raise_value_error("Condition must contain either < or > or !=.")
            if len(sp) != 2:
                _raise_value_error("Condition must contain ony one of < or >.")
            left = sp[0].strip()
            right = sp[1].strip()

            sp = left.split("_")
            if len(sp) != 2:
                _raise_value_error("Left of " + op + " there has to be a tensor written as A_ijk or similar.")
            left_core = sp[0].strip()
            left_ind = sp[1].strip()
            if "_" in right:
                _raise_value_error("Left of " + op + " there has to be a constant not a tensor.")
            if left_core not in self.all_quantities():
                _raise_value_error("Tensor " + left_core + " in the greater/lesser condition on the left, is not defined.")
            if len(left_ind) == 0:
                _raise_value_error("Tensor " + left_core + " in the greater/lesser condition on the left should have indices specified.")
            if len(left_ind) != len(self.get_shape(left_core)):
                _raise_value_error("Tensor " + left_core + " in the greater/lesser condition on the left was not specified with the correct number of indices.")
            if len(left_ind) != len(set(left_ind)):
                _raise_value_error("Tensor " + left_core + " in the greater/lesser condition on the left should not have repeating indices.")

            parsing_right =  pp.Combine(      pp.Word(pp.nums, min = 1) + "." + pp.Word(pp.nums, min = 1)).set_results_name("float") | \
                             pp.Combine("-" + pp.Word(pp.nums, min = 1) + "." + pp.Word(pp.nums, min = 1)).set_results_name("float") | \
                             pp.Combine(      pp.Word(pp.nums, min = 1) + ".").set_results_name("float") | \
                             pp.Combine("-" + pp.Word(pp.nums, min = 1) + ".").set_results_name("float") | \
                             pp.Word(      pp.nums, min = 1).set_results_name("integer") | \
                             pp.Word("-" + pp.nums, min = 1).set_results_name("integer") | \
                             pp.Combine(pp.Word(pp.alphas, min = 1) + "~" + pp.Word(pp.alphas, min = 1)).set_results_name("constant") | \
                             pp.Word(pp.alphas, min = 1).set_results_name("constant")

            par = _my_parse_string(parsing_right, right, parse_all = True)

            if par.get_name() in ["integer", "float"]:
                right = str(right)
                right_bfs = right
                right_latex = str(right)
                right_type = "int or float"
            elif par.get_name() in ["constant"]:
                if right not in self.all_quantities():
                    _raise_value_error("Constant: " + right + " in the greater/lesser condition on the right, is not defined.")
                right_latex = self._return_in_latex(right, None)
                right_bfs = "__object_" + right
                right = "__object[\"" + right + "\"]"
                right_type = "constant"
            else:
                _raise_value_error("Should not happen.")

            if brute_force_sums == False:
                code._add_raw("_f.append(np.real(__object[\"" + left_core + "\"]) " + op + " np.real(" + right + "))")

            # these are filters to be used to enforce which terms appear in the sum
            txt  = "_f[" + str(counter) + "]"
            for i, one_ind in enumerate(left_ind):
                tmp = []
                for j in range(len(left_ind)):
                    if i == j:
                        tmp.append("_s[\"" + one_ind + "\"]")
                    else:
                        tmp.append(":")
                txt += "[" + ",".join(tmp)  + "]"

            self._filters.append({
                "cond_value": txt,
                "cond_inds": left_ind,
                "cond_latex": self._return_in_latex(left_core, left_ind) + " " + op + " " + right_latex,
                "cond_for_brute_force": {"left_core": left_core,
                                         "left_index": left_ind,
                                         "op": op,
                                         "right": right_bfs,
                                         "right_type": right_type}
            })

            counter += 1

        # now compute slices which simplify computation by removing terms that will be removed eventually anyways.
        # I call this partially filtered as slicing operations always produce rectangular arrays.
        # you can't slice indices one by one and get anything that is not rectangular
        tt = ""
        for f in self._filters:
            tt += f["cond_inds"]
        tt = "".join(sorted(set(tt)))
        self._partially_filtered_indices = tt

        if brute_force_sums == False:
            code._add_raw("_s = {}\n", to_preamble = True)

        for ind in self._partially_filtered_indices:
            txt = ""
            txt += "_s[\"" + str(ind) + "\"] = ("
            # now go over all filters and find in which is ind appearing
            tmp = []
            for f, ff in enumerate(self._filters):
                if ind in ff["cond_inds"]:
                    to_sum_over = list(range(len(ff["cond_inds"])))
                    to_sum_over.remove(ff["cond_inds"].index(ind))
                    tmp.append("np.sum(_f[" + str(f) + "], axis = " + str(tuple(to_sum_over)) + ")")
            if len(tmp) == 0:
                _raise_value_error("This should not happen.")
            txt += " + ".join(tmp)
            txt += ") > 0\n"
            if brute_force_sums == False:
                code._add_raw(txt)
                #code._add_raw("print(\"Value of _s[" + str(ind)  +"] is : \", _s[\"" + str(ind) + "\"])" + "\n")

        to_loop_over = ", ".join(list(map(lambda x: "\"" + x + "\"", list(self._partially_filtered_indices))))
        if len(to_loop_over) > 0:
            txt = ""
            txt += "for ind in [" + to_loop_over + "]:" + "\n"
            # make sure you don't slice too much
            txt += "    if True not in _s[ind]:\n"
            txt += "        raise ValueError(\"Condition on index \" + ind + \" is so restrictive that it removes all elements.\")\n"
            # no point in slicing things if there is nothing to slice
            txt += "    if False not in _s[ind]:\n"
            txt += "        _s[ind] = slice(None)\n"
            if brute_force_sums == False:
                code._add_raw(txt)


    def _process_all_diagonals(self, conditions, code, brute_force_sums):

        conditions = re.split(",|:|;|\n", conditions)

        self._diagonals = []

        if brute_force_sums == False:
            code._add_raw("_orig_shp = {}\n", to_preamble = True)

        for cond in conditions:
            cond = cond.strip()
            if cond == "":
                continue
            if "<" in cond:
                continue
            elif ">" in cond:
                continue
            elif "!=" in cond:
                sp = cond.split("!=")
            else:
                _raise_value_error("Condition must contain either < or > or !=.")
            if len(sp) != 2:
                _raise_value_error("Condition on diagonals must contain ony one of !=")
            left = sp[0].strip()
            right = sp[1].strip()

            parsing_indices = pp.Word(pp.alphas, min = 1, max = 1).set_results_name("single_index")

            left = _my_parse_string(parsing_indices, left, parse_all = True)
            left = left[0]
            right = _my_parse_string(parsing_indices, right, parse_all = True)
            right = right[0]

            if left == right:
                _raise_value_error("Condition on diagonals can't be between same indices!")

            if [left, right] not in self._diagonals and [right, left] not in self._diagonals:
                self._diagonals.append(sorted([left, right]))
        tt = ""
        for dia in self._diagonals:
            tt += "".join(dia)
        tt = "".join(sorted(set(tt)))
        self._all_diagonal_indices = tt
        self._all_diagonal_indices_stored_shape = ""

        for dia in self._diagonals:
            # if indices in the diagonal don't have the same shape that could lead to ambiguities
            if brute_force_sums == False:
                code._add_raw("if \"" + dia[0] +"\" in _orig_shp.keys() and \"" + 
                              dia[1] +"\" in _orig_shp.keys():\n    if _orig_shp[\"" + 
                              dia[0] + "\"] != _orig_shp[\"" + dia[1] + "\"]:\n        raise ValueError(\"Indices " +  
                              dia[0] + " and " + dia[1] + " must be of the same length, as they are used in a condition.\")", \
                              to_preamble = False, to_the_top = True)

    def evaluate(self,
                 in_eqns,
                 conditions = "",
                 brute_force_sums = False,
                 optimize_divisions = True,
                 optimize_recomputation = True,
                 show_latex_with_div_opts = False):
        r"""
        This function can be used to evaluate a wide range of mathematical expressions.
        Einstein summation over repeating indices is assumed.

        Quick example::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            # calculate something
            comp.evaluate("sigma_ij <= (j / (numk * volume)) * (f_km - f_kn) * A_knmi * A_kmnj")

        :param in_eqns: String containing mathematical expression(s) you wish to evaluate.
          This string can contain the following.

          * "Numbers," -- either integer, real, or complex.  For example, you can
            use numbers in these formats 2, 3.14, 3j, 4.3 + 2.0j, j.

          * "Quantities" -- defined in this computator.  For example, "E_kn" corresponds
            to the band energy associated with k-point k and band n.  This quantity
            can be anyone that is currently defined for the computator.  You can use the
            function :func:`all_quantity_keys <wfbase._ComputatorWf.all_quantity_keys>` to
            get a list of all quantities currently in the computator.  You can
            also use :func:`info <wfbase._ComputatorWf.info>` to get
            extended information about all quantities in the computator.
            Note that you can use any set of letters as indices for quantities.
            For example, if your expression contains multiple
            sums over bands, you can use "E_kn" and "E_km", or any other combination
            of indices such as "E_qw". Quantities that are constants (not tensors with
            indices) you can simply refer to as "Q" or "QwR" or similar.  For constants
            don't use the underscore symbol, as these quantities don't have indices.

            In the name of the quantity, you can use ~ symbol (once at most).  When doing
            LaTeX rendering everything after ~ will be rendered as a superscript.  For example
            "Abc~intra_knm" will be rendered in LaTeX as "Abc^{intra}_{knm}".

          * "Basic operators" -- such as addition (+), subtraction (-), multiplication (*),
            division (/), exponentiation (^).  Note that the multiplication symbol *
            is required.  For example, you must use "A_ij * B_jk".  If you simply use
            "A_ij B_jk" you will get an error message.  Certain potentially ambiguous expressions
            are not allowed, such as "A / B * C".  You should either use "A / (B * C)"
            or "(A / B) * C" depending on your intention.  Similarly, you
            are not allowed to write "A^B^C" as that could be ambiguous.  Instead,
            write "(A^B)^C" or "A^(B^C)".

          * "Plus and minus symbol" -- can be used either in between two quantities,
            such as "A_ij - B_ij", but you can also use it in front of a symbol, such
            as "-A_ij - B_ij".  This expression will change the sign of the
            quantity "A" relative to the former expression, as expected.

          * "Parentheses." --  Only regular parentheses "(" and ")" are allowed.

          * "Assignment operators" -- are one of these three: <=, <+=, or <<=.
            The assignment operator assigns to the new quantity on the
            left the expression given on the right.  For example, "A_ik <= B_ij * C_jk"
            would do a regular matrix multiplication of B and C and assign
            the result to a new quantity A.  Related assignment operator "<+=" assumes
            that the quantity "A" on the left already exists. This operator will compute
            the value of the right hand side and add it to the value of the
            preexisting quantity "A".  This is similar to the behavior of "+="
            operator in python.  Finally, "<<=" operator will erase the previously existing
            quantity "A" on the left, and it will create a new quantity with the
            value of whatever is on the right of "<<=".

            On the left of "<=" you can either have a tensor quantity "A_ik", or a constant
            quantity "D", or you can simply have "_ik", or simply "_".  If you have
            something like "_ik" on the left of "<=" then this function
            will simply return a numerical value of the tensor and will not create
            any new quantity.  Finally, if you simply write "A_ij * B_jk" without even
            using "<=" operator, you will again get a numerical value of the matrix product
            without creating a new quantity.  The indices of the returned quantity in this case
            will be simply sorted in alphabetical order ("ik") in the present case.  To reduce
            ambiguity, it is therefore recommended to always use one of the assignment operators
            in your computations. (This option is not allowed when *brute_force_sums* is *True*.
            One must use one of the arrow operators (<=, <+=, <<=) if *brute_force_sums* is *True*.)

            Assignment operators can be used only once per expression.

          * "Complex conjugation" -- is done using the # operator.  Note that this operator
            does not transpose any of the indices of the matrix, it only does complex conjugation.
            The # operator must appear before the quantity whose complex conjugation you
            are taking.  For example, in this case, quantity B would be conjugated:  "A_ij * #B_jl".
            If you also wish to transpose B then you need to do so explicitly "A_ij * #B_lj" by
            swapping indices on B.  (If quantity B_jl is a matrix element of operator O,
            "< j | O | l >" then #B_jl will refer simply to "conjugate(< j | O | l >)".  Now, if operator O
            happens to be a Hermitian operator, then this quantity is by definition equal to
            <l|O|j>.  LaTeX parser will display in this case <l|O|j> instead of
            conjugate(<j|O|l>).  In other words, #B_jl will be parsed into <l|O|j> instead
            of conjugate(<j|O|l>) which doesn't look as pretty.)

          * "Functions" -- Real and Imag can also be used in the expression.   These take
            real and imaginary parts of the complex numbers.  For example, "Real(A_ij + B_ji)"

        :param conditions: This string contains all restrictions you wish to perform on the
          sums in your computation.  The string can contain more than one condition.  Conditions
          must be separated from each other by a comma.  For example, "E_kn < ef , E_km > ef, n != m"
          would limit any sum containing indices k and n to only those for which E_kn is less than ef.
          An additional limitation would occur for the sum over k and m.  The third condition would ensure that no
          sum over n and m includes the n==m term.   Also, if n and m appear in the output indices, the diagonal
          terms are set to zero. (The code will also avoid dividing by n==m term of the tensors in the denominator.
          This is useful if you divide 1/(A_n - A_m) and then use the condition n!=m.)
          The lesser/greater conditions must be formatted so that
          on the left of < or > there is a single tensor quantity.  The indices should not repeat.

          If one of the indices is summed over, the other index in the condition must appear in the same sum.
          Otherwise, the meaning of the condition is imprecise.  (For example, this would occur if one has
          the condition "E_kn < ef" but in the expression we have a sum only over n, but not over k.  For example,
          if the expression "B_k <= E_kn * E_kn" we have a sum over repeated index n but not over k.)
        
          On the right of < or > you must use either a constant (quantity without indices) or a number.
          No other formatting is allowed for conditions with lesser/greater.
          Only real parts of the left and right hand side are used in determining conditions.

          The formatting of conditions involving != is much simpler.  On the left and right of != you
          can only have a single index.

        :param brute_force_sums: Boolean. The default is False.  The code will use different algorithms to evaluate
          the quantity, depending on the value of this parameter.  If False (default) then the code will
          evaluate quantity using numpy vectorization.  If True, then the code will use brute-force for loops
          compiled via Numba.  The two approaches should give numerically the same values.  Depending
          on your machine, it is probably more optimal to use default (False) when you have moderately
          dense k-grids.  If you need to sample denser k-grids, you should probably use random sampling
          and perform several calculations with a moderately dense k-grid unless you reach convergence.
          See :ref:`this example <sphx_glr_all_examples_example_conv.py>` for more details on how to do this.  If you really
          have a need to do a single-shot calculation with very many k-points, you may want to set this
          flag to *True* and test if that makes your calculation faster.  (Whether brute-force sums are faster or not
          will depend not only on the number of k-points but also the type of calculation you are evaluating.)

          (There is a slight difference in the two approaches (numpy vectorization vs Numba)
          since in certain edge cases numpy vectorization approach will stop, while Numba approach will still do the
          calculation.  For example, this will happen in certain edge cases with *conditions*, if
          some condition is too restrictive on the sum, numpy vectorization will stop, but with this parameter set to *True* it
          will do the calculation.  Another difference is that when *brute_force_sums* is *True* you must
          use one of the arrow operators (<=, <+=, or <<=) in all of your evaluation expressions.)

        :param optimize_divisions: Boolean.  The default is True.  If True then the code will internally
          preoptimize the expression by replacing expressions such as A*B*C/(D*E) with something like A*B*C*(1/D)*(1/E).
          In many cases, this is faster to evaluate (less overhead) as this is now a single product
          of five quantities (A, B, C, 1/D, and 1/E).  For debugging purposes, the user could set this flag
          to False, but True should give the same result faster in most cases.  If in doubt, inspect the output of
          :func:`info <wfbase._ComputatorWf.info>`  (with the *show_code* flag set to True) to see what exact operation the code
          uses to evaluate your expression.  (This parameter is ignored if *brute_force_sums* is *True*.)

        :param optimize_recomputation:  Boolean.  The default is True.  If True then the code will optimize certain
          computations.  For example, if the expression (E_nk - E_mk) appears twice in the single call
          to this function (it could be either in the same line or in two different lines, as long
          as it is in the same call to this function), then in the second appearance of this expression
          code will reuse previously computed value.  For debugging purposes, the user could set this
          flag to False, but True should give the same result faster. If in doubt, inspect the output of
          :func:`info <wfbase._ComputatorWf.info>`  (with *show_code* flag set to True) to see what exact operation the code
          uses to evaluate your expression. (This parameter is ignored if *brute_force_sums* is *True*.)

        :param show_latex_with_div_opts: Boolean.  The default is False.  If True then the code will return LaTeX'd
          code that includes optimizations that were potentially used if the *optimize_divisions* flag is True.

        Example usage::

            import wfbase as wf
            import numpy as np

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            comp.verbose_evaluate()

            # This creates new quantity, called "test".
            # This quantity depends on one index (k).
            # The value of this quantity is the sum of squares
            # of all band energies with that k.  The sum over *n* is implied as
            # index *n* does not appear on the left of <=
            comp.evaluate("test_k <= E_kn * E_kn")
            print(comp["test"])

            # direct way to compute the same thing using numpy
            direct = np.sum(comp["E"]*comp["E"], axis = 1)
            print(direct)

            # this removes quantity "test"
            comp.remove("test")

            # same calculation as above, but using brute-forced sums and Numba
            # instead of numpy vectorization
            comp.evaluate("test_k <= E_kn * E_kn", brute_force_sums = True)
            comp.remove("test")

            # since now index *n* appears next to the tensor "test"
            # on the left of <= we don't sum over n
            comp.evaluate("test_kn <= E_kn * E_kn")

            comp.remove("test")
            # now we sum over both *k* and *n* as neither index
            # appears next to the tensor "test" on the left of <=
            comp.evaluate("test <= E_kn * E_kn")

            comp.remove("test")
            # this basically makes "test" be a copy of "E", as nothing
            # is summed over
            comp.evaluate("test_kn <= E_kn")

            comp.remove("test")
            # this makes "test" be a transpose of "E"
            comp.evaluate("test_nk <= E_kn")

            comp.remove("test")
            # now the sum is performed only over occupied states
            comp.evaluate("test <= E_kn * E_kn", "E_kn < ef")

            comp.remove("test")
            # now the sum is performed only over occupied states with energy above -2 eV.
            comp.evaluate("test <= E_kn * E_kn", "E_kn < ef, E_kn > -2.0")

            comp.remove("test")
            # now the sum is performed only over occupied states with energy above -2 eV,
            # the diagonal terms (n == m) are not included in the sum
            comp.evaluate("test <= E_kn * E_kn", "E_kn < ef, E_kn > -2.0, n != m")

            comp.remove("test")
            # this will create a tensor with four indices.  Tensor "E" will be
            # repeated over the missing indices.
            comp.evaluate("test_knmi <= E_kn * A_kmni")

            comp.remove("test")
            # same thing, but now we perform sum over three of the indices
            # so we are left with one index (k)
            comp.evaluate("test_k <= E_kn * A_kmni")

            comp.remove("test")
            # this will create a tensor with three indices.  E_kn does
            # not depend on m, so it will simply copy E_kn along m'th index.
            comp.evaluate("test_knm <= E_kn - E_km")

            comp.remove("test")
            # this can be made more complicated as well...
            comp.evaluate("test_ki <= (E_kn - E_km) * A_knmi * (E_kn + E_km)")

            comp.remove("test")
            # One is allowed to use numbers as well (but there must be a
            # multiplication sign between any two quantities if you want them
            # multiplied.  So, you must use ... 2 * E_kn ...  not simply ... 2 E_kn ...
            comp.evaluate("test_ki <= (E_kn - E_km) * A_knmi * (E_kn + 2 * E_km)")

            comp.remove("test")
            # One can use "eV", "Ang", or "muB" to introduce quantities in the expression with units
            comp.evaluate("test_ki <= (E_kn - E_km) * A_knmi * (E_kn + 2 * E_km + 0.1 * eV)")

            comp.remove("test")
            # ... division is also allowed, as well as complex numbers
            comp.evaluate("test_ki <= (E_kn - E_km) * A_knmi / (E_kn + 2j * E_km)")

            comp.remove("test")
            # ... and multiple levels of parentheses
            comp.evaluate("test_ki <= (E_kn - E_km) * A_knmi / ((E_kn + 2j * E_km) * A_knni)")

            comp.remove("test")
            # ... you can also raise to a power
            comp.evaluate("test_ki <= (E_kn - E_km)^2 * A_knmi")

            comp.remove("test")
            # ... and take complex conjugate, or real, or imaginary part
            # (note that complex conjugation does not transpose any indices!  It only
            # complex conjugates each element of the tensor)
            comp.evaluate("test_ki <= (E_kn - E_km) * #A_knmi * Real(1 / (E_kn - 2j* E_km)) * Imag(A_knni)")

            # you can redefine quantity "test" that already exists, without the need to call "remove"
            # notice how we use here <<= instead of <=
            comp.evaluate("test_knm <<= E_kn - E_km")

            # you can also add to the previously existing quantity using <+=
            comp.evaluate("test_knm <+= E_kn + 3.0 * E_km")

            comp.remove("test")
            # notice how the regular use of <= creates quantity "test"
            comp.evaluate("test_knm <= E_kn - E_km")
            # if you simply want to get the numerical array, and you
            # don't want to create a new quantity, you can simply do:
            value = comp.evaluate("_knm <= E_kn - E_km")
            # if you don't use <= at all, then there will be no sums performed
            # and the order of indices is alphabetical (kab in the case below)
            value = comp.evaluate("E_kb - E_ka")

        """

        split_eqns = []
        for ln in in_eqns.split("\n"):
            l = ln.strip()
            if l != "":
                split_eqns.append(l)

        if len(split_eqns) < 1:
            _raise_value_error("Need to specify at least one valid equation when calling evaluate function.")

        # put code for each line in input equations in here
        code = _StorePythonCode(optimize_recomputation = optimize_recomputation)
        code_only_for_latex = _StorePythonCode(optimize_recomputation = optimize_recomputation)

        self._process_all_diagonals(conditions, code, brute_force_sums)
        self._process_all_greater_lesser(conditions, code, brute_force_sums)

        if brute_force_sums == False:
            code_dic = {"opteinsum": opteinsum, "np": np}
        else:
            code_dic = {"njit": njit, "np": np}

        num_eqns = len(split_eqns)

        # go over each equation
        for eqn_i in range(num_eqns):

            raw_eqn = split_eqns[eqn_i]

            # this removes all divisions and returns inverse operator & instead.
            # also, this removes some parentheses that are not needed.
            # That should generate a more optimized code.
            use_eqn = raw_eqn
            if brute_force_sums == False:
                if optimize_divisions == True:
                    use_eqn = self._reorg_parser.reorganize(raw_eqn)
                ret, store_to, mode = self._work_on_one_equation_using_vectorizations(use_eqn, code, code_dic, eqn_i, num_eqns)
            else:
                ret, store_to, mode = self._work_on_one_equation_using_brute_force_sums(use_eqn, code, code_dic, eqn_i, num_eqns)

            # return numpy array immediately if you are doing things like _ik <= A_ij * B_jk or simply  A_ij * B_jk
            if store_to is None:
                return ret["value"]
            if store_to == "no use of left arrow":
                _check_that_return_indices_compatible_with_filter(ret["ind"], self._filters)
                return ret["value"]

            # get latex by parsing everything again, but now with raw eqn
            if mode in ["<=", "<<="]:
                if show_latex_with_div_opts == True:
                    use_for_latex_eqn = use_eqn
                else:
                    use_for_latex_eqn = raw_eqn

                par_only_for_latex = _my_parse_string(self._parser, use_for_latex_eqn, parse_all = True)
                par_only_for_latex = par_only_for_latex[0]
                if isinstance(par_only_for_latex, EvalArrowOp):
                    ret_only_for_latex, _ignore, _ignore = par_only_for_latex.eval(self,
                            code_only_for_latex, allow_storing_data = True, call_from_main_evaluate = True)
                else:
                    ret_only_for_latex = par_only_for_latex.eval(self, code_only_for_latex)
                ret["latex"] = ret_only_for_latex["latex"]

            # otherwise, store everything that is needed
            ret["parsed"] = True
            if mode == "<=": # definition
                ret["parsed_string"] = raw_eqn
                ret["parsed_condition"] = conditions
                ret["order_parsed"] = self._order_parsed
                self._order_parsed += 1
                ret["brute_force_sums"] = brute_force_sums
                if store_to in self.all_quantities():
                    _raise_value_error("Quantity " + store_to + \
                    " already defined! You can't use <= if quantity on the left of it was defined already.  Use <<= to overwrite previously defined quantity.")
                self.new(store_to, ret)
            elif mode == "<<=": # replacement
                ret["parsed_string"] = raw_eqn
                ret["parsed_condition"] = conditions
                ret["brute_force_sums"] = brute_force_sums
                if store_to not in self.all_quantities():
                    _raise_value_error("Quantity " + store_to + \
                    " was not defined earlier!  Operator <<= is used to replace the tensor on the left of it with new value.")
                self.remove(store_to)
                self.new(store_to, ret)
                if brute_force_sums == False:
                    code.changed_value_of_this_core("__object[\"" + store_to + "\"]")
            elif mode == "<+=": # addition
                if store_to not in self.all_quantities():
                    _raise_value_error("Quantity " + store_to + " not defined!  Operator <+= can only be used when quantity on the left of it was defined already")
                if self.get_units(store_to)._check_units_the_same(ret["units"]) == False:
                    _raise_value_error("Units don't match when using the <+= operator.  On the left of <+= units are: " + str(self.get_units(store_to)) + \
                                       " while on the right of <+= units are: " + str(ret["units"]) + ".")
                self[store_to] += ret["value"]
                if brute_force_sums == False:
                    code.changed_value_of_this_core("__object[\"" + store_to + "\"]")
            else:
                _raise_value_error("Unknown mode " + mode + ".")

            if mode in ["<=", "<<="]:
                if self._verbose_evaluate == True:
                    self.info(store_to)

        # how many times did code do recomputation optimization
        self._from_last_evaluation_num_used_stored = code.how_many_times_used_stored()


    def _work_on_one_equation_using_vectorizations(self, use_eqn, code, code_dic, eqn_i, num_eqns):
        code_txt_import  = ""
        code_txt_import += "import numpy as np\n"
        code_txt_import += "from opt_einsum import contract as opteinsum\n"

        # This will parse the string.  Code is not yet generated.  That happens later when you call par.eval(...)
        par = _my_parse_string(self._parser, use_eqn, parse_all = True)
        if len(par) != 1:
            _raise_value_error("PyParsing returned something not expected?!")
        par = par[0]

        # check whether this topmost thing in the parsing tree is using one of "<=" operators
        if isinstance(par, EvalArrowOp):
            # this generates python code equivalent for one equation
            # if user specified _ij <= A_ij * ...  then store_to will be None
            # otherwise store_to will be X if you did X_ij <= A_ij * ...
            ret, store_to, mode = par.eval(self, code, allow_storing_data = True, call_from_main_evaluate = True)
        else:
            ret = par.eval(self, code)
            store_to = "no use of left arrow"
            mode = None

        if store_to is None and mode != "<=":
            _raise_value_error("If equation is of the form _ij <= ... or _ <= ... then you must use <= operator, not <+= or <<=.")

        if store_to is None or store_to == "no use of left arrow":
            if num_eqns != 1:
                _raise_value_error("If equation is of the form _ij <= ... or you don't use <= at the top of the parsing tree, then you can't have more than one equation per evaluation.")

        code._add_raw("__value = " + ret["value"])

        if eqn_i < num_eqns - 1:
            code._add_raw("#")

        code.start_new_chunk()

        code_txt = code.get_code_chunk(eqn_i, prefix = "")

        # run this part of the code
        code_dic["__object"] = self
        time_exec = _nice_exec(code_txt, code_dic)

        # store what came out of exec
        ret["value"] = code_dic["__value"]

        # store code used to compute this quantity
        ret["exec"] = {"code": _decorate_code_into_a_function(code_txt_import + "\n" + code_txt),
                       "eqn_order_from_1": eqn_i + 1,
                       "num_eqns": num_eqns}

        # store time spent in exec
        ret["total_seconds_exec"] = time_exec

        # release memory for variables inside exec's
        if eqn_i == num_eqns - 1:
            def_vars = code.get_all_defined_variables()
            def_vars.sort()
            del_txt = ""
            for dd in def_vars:
                del_txt += "del " + dd + "\n"
            del_txt +="del _f\n"
            del_txt +="del _s\n"
            exec(del_txt, code_dic)

        return ret, store_to, mode

    def _work_on_one_equation_using_brute_force_sums(self, use_eqn, code, code_dic, eqn_i, num_eqns):
        # There are certain checks on the syntax that we do with regular parsing (one
        # without brute forcing sums).  Since we don't do these checks on the brute
        # force sums, I here have a dummy parsing, just to make sure that the
        # same checks are applied in both cases.  In the future one would probably
        # want to find a better way to deal with this.  Either one can separate out
        # checks from the (non-bfs) parser, or duplicate code.  Both options
        # don't look clean to me.  Third option is to simply remove the dummy parsing below.
        # Another problem is that self._parser_brute_force_sums does not do units,
        # so I need to steal units from the dummy parser below.  Finally, there are
        # some checks that appear in the opteinsum which are skipped if we don't use
        # it.
        if True:
            __par = _my_parse_string(self._parser, use_eqn, parse_all = True)
            __par = __par[0]
            if isinstance(__par, EvalArrowOp):
                __ret, __store_to, __mode = __par.eval(self, deepcopy(code), allow_storing_data = True, call_from_main_evaluate = True)
            else:                __ret = __par.eval(self, deepcopy(code))

        code_txt_import  = ""
        code_txt_import += "from numba import njit\n"
        code_txt_import += "import numpy as np\n"

        data_about_used_tensors = []

        par = _my_parse_string(self._parser_brute_force_sums, use_eqn, parse_all = True)
        if len(par) != 1:
            _raise_value_error("PyParsing returned something not expected?!")
        par = par[0]
        if isinstance(par, BfsArrowOp):
            ret, store_to, mode = par.eval(data_about_used_tensors, self, allow_storing_data = True, call_from_main_evaluate = True)
            ret["units"] = __ret["units"]
        else:
            # this condition is here as otherwise I would need to figure
            # out which indices were duplicated (therefore summed over) and which were not
            # but this choice would have to be consistent with what is done in case with brute_force_sums = False
            _raise_value_error("If parameter brute_force_sums is set to True, then you must use one of the arrow operators " + 
                               "in your expressions: <=, <+= or <<=.")

        if store_to is None and mode != "<=":
            _raise_value_error("If equation is of the form _ij <= ... or _ <= ... then you must use <= operator, not <+= or <<=.")

        if store_to is None or store_to == "no use of left arrow":
            if num_eqns != 1:
                _raise_value_error("If equation is of the form _ij <= ... or you don't use <= at the top of the parsing tree, then you can't have more than one equation per evaluation.")

        filters_bfs = []
        for ii in range(len(self._filters)):
            filters_bfs.append(self._filters[ii]["cond_for_brute_force"])

        # now create code that involves all brute force sums
        initialization_code = _create_brute_force_sums(self, code, code_dic, eqn_i, expression = ret["value"], indices_want = ret["ind"], data_about_used_tensors = data_about_used_tensors,
                                                      filters_bfs = filters_bfs, diagonals = self._diagonals)

        if eqn_i < num_eqns - 1:
            code._add_raw("#")

        code.start_new_chunk()

        code_txt = code.get_code_chunk(eqn_i, prefix = "")

        # run this part of the code
        time_exec = _nice_exec(code_txt, code_dic)

        # store what came out of exec
        ret["value"] = code_dic["__value"]

        # store code used to compute this quantity
        ret["exec"] = {"code": _decorate_code_into_a_function(code_txt_import + "\n" + initialization_code + "\n" + code_txt),
                       "eqn_order_from_1": eqn_i + 1,
                       "num_eqns": num_eqns}

        # store time spent in exec
        ret["total_seconds_exec"] = time_exec

        if eqn_i == num_eqns - 1:
            del code_dic

        return ret, store_to, mode


    def _is_parsed(self, core):
        ret = False
        if "parsed" in self.all_quantity_keys(core):
            if self.get(core, "parsed") == True:
                ret = True
        return ret

    def _what_to_say_if_user_messed(self, core):
        if self.__did_user_mess_with_values[core] == False:
            _raise_value_error("User did not mess with "+core)

        out = ""
        out += _format_one_block("""The quantity """ + core + """ has been accessed at some point and the user
        might have changed its numerical value.""")
        out += "\n\n"
        out += _format_one_block( """**Therefore it is possible that the stored description of this
        object is no longer valid. If you want more information about this object, try printing documentation about it before you modify it.** """)
        out += "\n"
        out += _format_one_block("Here is the location in the code where this quantity was accessed by the user and it might have been modified by the code.")
        out += "\n"
        for ln in self.__did_user_mess_with_values[core]:
            out += _format_one_block(str(ln), indent = 8, width = 200)
            out += "\n"
        out += "\n"
        return out

    def info(self, core = None, print_to_screen = True, display = False, full = False, show_code = False, allow_multiple_expressions = False):
        r"""

        This function provides information about various quantities in the computator.

        Note that there is a function with the same name that provides information
        about the database .wf file, not the computator.  See here
        for more information on how to use this other function :func:`info <wfbase.DatabaseWf.info>`.

        :param core: If set to None then the function will return information
          about all quantities in the computator.  Otherwise, it will give
          information only about the quantity *core*.

        :param print_to_screen: Boolean.  If set to True (default) then the information
          about the quantity will be printed on the screen by this function.  If set to False
          nothing is printed on the screen.  Instead, in this case function returns a string
          with the same information.

        :param display: Boolean.  If set to True it will display inside the terminal LaTeX-ed
          equations if they are present in the description of the quantity.  This feature works only
          in terminals that support imgcat.  One such terminal is iTerm2 on OS X.  You can find information about
          `installing imgcat in iTerm2 here <https://iterm2.com/documentation-images.html>`_.

          If set to False (default) nothing is displayed in the terminal.
        
        :param full: Boolean.  If set to False (default) only some of the numerical values of the quantity are
          shown.  If set to True, all values are shown.

        :param show_code:  Boolean.  If set to True the function will show the python code
          used to compute the quantity.  The default is False.

        :param allow_multiple_expressions: Boolean.  If set to True, and if *show_code* set to True, will
          display python code used to compute the quantity, even if multiple expressions were processed
          in a single call to :func:`evaluate <wfbase._ComputatorWf.evaluate>` and you are trying to access information
          for a quantity that is not the first one that was evaluated.  The default is False.

        :returns:
          * **txt** -- String with information about the quantities.  This string is returned only
            if *print_to_screen* is set to False.

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            # shows information about quantity "E"
            comp.info("E")

            # compute some quantity
            comp.evaluate("sigma_oij <= (j / (numk * volume)) * (f_km - f_kn) * Real((E_km - E_kn) / (E_km - E_kn - hbaromega_o - j*eta)) * A_knmi * A_kmnj")

            # shows information about newly computed quantity
            comp.info("sigma")

            # ... also displays LaTeX-ed equation defining this quantity
            # This works only in terminals that support imgcat, such as iTerm2 on OS X.
            # You can find information about installing imgcat in iTerm2 here:
            #     https://iterm2.com/documentation-images.html>
            comp.info("sigma", display = True)

        """
        if core is not None:
            ret = self.__info_one_core(core, print_to_screen = print_to_screen, display = display,
                                       full = full, show_code = show_code, allow_multiple_expressions = allow_multiple_expressions)
            if print_to_screen == False:
                return ret
        else:
            keys = self.all_quantities()
            keys.sort()
            ret = []
            for k in keys:
                ret.append(self.__info_one_core(k, print_to_screen = print_to_screen, display = display,
                                                full = full, show_code = show_code, allow_multiple_expressions = allow_multiple_expressions))
            if print_to_screen == False:
                return "\n".join(ret)

    def __info_one_core(self, core, print_to_screen, display, full, show_code, allow_multiple_expressions):
        self.__does_core_exist(core)

        out = ""
        out += "\n"
        out += _make_rst_title("Quantity *" + core.strip() + "*")
        out += "\n"

        out += _make_rst_field("Shape")
        out += _format_one_block(str(self.get_shape(core)))
        out += "\n\n"

        out += _make_rst_field("Value")
        if full == False:
            with np.printoptions(precision = 3, linewidth = 80):
                out += _format_one_block_simple_indent(str(np.array(self[core])), indent = 4, start_and_end = False, max_line = 8)
        else:
            with np.printoptions(precision = 6, linewidth = 80, threshold = sys.maxsize):
                out += _format_one_block_simple_indent(str(np.array(self[core])), indent = 4, start_and_end = False, max_line = None)
        out += "\n"

        out += _make_rst_field("Units")
        out += _format_one_block(str(self.get(core, "units")))
        out += "\n\n"

        show_latex = False

        if "origin_story" in self.all_quantity_keys(core) or self._is_parsed(core):
            if self.__did_user_mess_with_values[core] != False:
                out += _make_rst_field("Origin story")
                out += self._what_to_say_if_user_messed(core)
            else:
                if self._is_parsed(core):
                    out += _make_rst_field("Origin story")
                    out += _format_one_block("This quantity was computed by parsing the following string")
                    out += "\n\n"
                    out += _format_one_block(self.get(core, "parsed_string"), indent = 8)
                    out += "\n\n"
                    if self.get(core, "parsed_condition") != "":
                        out += _format_one_block("... under the following conditions")
                        out += "\n\n"
                        out += _format_one_block(self.get(core, "parsed_condition"), indent = 8)
                        out += "\n\n"
                    #
                    if show_code == True:
                        out += _make_rst_field("Python code")
                        out += "\n"
                        #
                        data_exec = self.get(core, "exec")
                        code = data_exec["code"]
                        which_eq = data_exec["eqn_order_from_1"]
                        number_eq = data_exec["num_eqns"]
                        #
                        if number_eq > 1 and which_eq > 1:
                            if allow_multiple_expressions == True:
                                out += _format_one_block(\
                                  "*Note!* You evaluated " + str(number_eq) + " equation(s) in the same string at the same time. " +
                                  "Your quantity " + str(core) + " was evaluated as the equation number " + str(which_eq) + \
                                  " .  Therefore, there were " + str(which_eq - 1) + " expression(s) computed before this one. " + 
                                  "The output below includes computations of only one quantity.  However, some of the terms here " + 
                                  "might have been computed in earlier quantities.  Therefore, the code below potentially can't be executed " + 
                                  "on its own, in isolation from the earlier " + str(which_eq - 1) + " expression(s).  Additionally, " + 
                                  "it is possible that some of the quantities used in this expression were modified by the previous expression(s).")
                                out += "\n\n"
                                out += _format_one_block_simple_indent(code, indent = 0, start_and_end = True, dont_indent_first = False)
                            else:
                                out += _format_one_block(\
                                    "Will not display code because this quantity was evaluated as equation number " + str(which_eq) + \
                                    " out of total " + str(number_eq) + " equations.  This can lead to ambiguities. If you still " + 
                                    "insist on getting partial code for this quantity, " + 
                                    "please set parameter allow_multiple_expressions to True when you call .info() function.")
                                out += "\n\n"
                        else:
                            out += _format_one_block_simple_indent(code, indent = 0, start_and_end = True, dont_indent_first = False)
                        out += "\n"
                    #
                    out += _make_rst_field("LaTeX")
                    latex_obj = self.get_latex(core)
                    latex_source = "$" + latex_obj.get_string() + "$"
                    out += _format_one_block(latex_source, indent = 4)
                    out += "\n\n"
                    show_latex = True
                else:
                    out += _make_rst_field("Origin story")
                    out += _format_one_block(self.get(core, "origin_story"), indent = 4)
                    out += "\n\n"
                    if "indices_info" in self.all_quantity_keys(core):
                        txt_ind, txt_def = _process_index_information(self.get(core, "indices_info"))
                        out += _make_rst_field("Indices")
                        out += _format_one_block_simple_indent(txt_ind, indent = 4, start_and_end = False)
                        out += "\n"
                        out += _format_one_block_simple_indent(txt_def, indent = 4, start_and_end = False)
                        out += "\n\n"
                    if "latex" in self.all_quantity_keys(core):
                        out += _make_rst_field("LaTeX")
                        out += _format_one_block(self.get(core, "latex"), indent = 4)
                        out += "\n\n"
        else:
            if "latex" in self.all_quantity_keys(core):
                out += _make_rst_field("LaTeX")
                out += _format_one_block(self.get(core, "latex"), indent = 4)
                out += "\n\n"

        if print_to_screen:
            print(out)

        if show_latex == True:
            if display == True:
                latex_obj = self.get_latex(core)
                render_latex(latex_obj, "__tmp.png")
                display_in_terminal("__tmp.png")
                print("\n\n")

        return out

    def verbose_evaluate(self, verbose = True):
        r"""

        If *verbose* is set to True then the code will print information about
        each expression it evaluates.

        :param verbose: whether to print information or not.  If not specified,
          it defaults to True.

        Example usage::

            import wfbase as wf
            import numpy as np

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            comp.verbose_evaluate()

        """
        self._verbose_evaluate = verbose

    def report(self):
        r"""

        This function returns a LaTeX equation for all quantities that were parsed
        by the user, in the order in which they were parsed. (The function skips
        quantities that were later modified by the user.)

        :returns:
          * **txt** -- String with LaTeX equation.

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # create now a computator from the database
            comp = db.do_mesh()

            # compute some quantity
            comp.evaluate("sigma_oij <= (j / (numk * volume)) * (f_km - f_kn) * Real((E_km - E_kn) / (E_km - E_kn - hbaromega_o - j*eta)) * A_knmi * A_kmnj")

            # you can print the equation to the screen
            print(comp.report())
            # ... or, you can also render the equation
            wf.render_latex(comp.report(), "latex.pdf")

        """
        out = r"\begin{align}" + "\n"

        keys = self.all_quantities()
        use_cores = []
        order_cores = []
        for core in keys:
            if self._is_parsed(core):
                if self.__did_user_mess_with_values[core] == False:
                    use_cores.append(core)
                    order_cores.append(self.get(core, "order_parsed"))
        srt = np.argsort(order_cores)
        use_cores = np.array(use_cores)[srt].tolist()

        for core in use_cores:
            latex_obj = self.get_latex(core)
            latex_source = latex_obj.get_string(inside_align = True)
            out += latex_source + "\n" + r"\notag" + "\n" + r"\\" + "\n"
        if out.endswith(r"\\" + "\n"):
            out = out[:-3]

        out += r"\end{align}"

        return out

    def plot_bs(self, ax, plot_bands = True, plot_spec = True, plot_fermi = True, plot_xticks = True):
        """

        Plots the band structure.

        :param ax: Matplotlib's axes onto which you wish to plot the band structure.

        :param plot_bands: If *True*, will plot the electron bands. (Default).

        :param plot_spec: If *True*, will plot special k-points on the path. (Default).

        :param plot_fermi: If *True*, will plot the Fermi level. (Default).

        :param plot_xticks: If *True*, will plot the x-ticks of special points. (Default).

        Example usage::

            import wfbase as wf

            # open a database file on bcc phase of iron
            db = wf.load("data/fe_bcc.wf")

            # compute quantities on a path between these special points
            comp = db.do_path("GM--H--N")

            # plot the band structure
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            comp.plot_bs(ax)
            ax.set_title("Band structure of Fe bcc")
            fig.tight_layout()
            fig.savefig("a.pdf")

        """

        if self._computated_using not in ["do_path", "do_list"]:
            _raise_value_error("Function .plot_bs you can use only on computators that were constructed using .do_path or .do_list")
        if self._computated_using == "do_path":
            if plot_bands == True:
                for i in range(self["E"].shape[1]):
                    ax.plot(self["kdist"], self["E"][:, i], "k-", lw = 1.0, zorder = -200)
            if plot_spec == True:
                for i in range(1, self["kspecdist"].shape[0] - 1):
                    ax.axvline(self["kspecdist"][i], c = "b", ls = "--", lw = 0.7, zorder = -300)
        if self._computated_using == "do_list":
            if plot_bands == True:
                for i in range(self["E"].shape[1]):
                    ax.plot(range(self["E"].shape[0]), self["E"][:, i], "k-")
        if plot_fermi == True:
            ax.axhline(self["ef"], c = "r", zorder = -100, lw = 0.7)
        if self._computated_using == "do_path":
            if plot_xticks == True:
                ax.set_xticks(self["kspecdist"])
                ax.set_xticklabels(self["kspeclabels"])
            ax.set_xlim(self["kdist"][0], self["kdist"][-1])
        if self._db_loaded_from_wannierberri == False:
            ax.set_ylim(self["reliableminenergy"], self["reliablemaxenergy"])

def _decorate_code_into_a_function(code):
    use = ""
    use += "def evaluate_directly(__object):\n"
    use += code
    use += "return __value\n"
    use = _format_one_block_simple_indent(use, indent = 4, start_and_end = False, dont_indent_first = True)
    return use

def _replace_star_with_indices(use_str, ind):
    fmt = pp.White() |\
        pp.Combine("*" + pp.Word(pp.nums, min = 1)).set_results_name("index") |\
        pp.Word(pp.printables, exclude_chars = "* ")
    par = _my_parse_string(fmt[...], use_str, parse_all = True)
    ret = ""
    for p in par:
        if p[0] == "*":
            num_index = int(p[1:])
            if num_index >= len(ind):
                _raise_value_error("Index #" + str(num_index) + " does not exist in the formatting or definition of the quantity. " + use_str)
            ret += ind[num_index]
        else:
            ret += p
    return ret

def _process_index_information(indices_info):
    ret_ind = ""
    ret_ind += "This quantity has these indices\n[" + ", ".join(indices_info["canonical_names"]) + "]"
    ret_ind += "\n"
    if "definition" in indices_info.keys():
        ret_ind += "\nDefinition of the quantity in terms of the same indices as above\n" + \
            _replace_star_with_indices(indices_info["definition"], list(indices_info["canonical_names"]))
        ret_ind += "\n"

    ret_def = ""
    for i, exp in enumerate(indices_info["explanation"]):
        ret_def += "#" + str(i) + " index \"" + indices_info["canonical_names"][i] + "\" corresponds to the " + exp + "\n"

    return ret_ind, ret_def


def _create_brute_force_sums(comp, code, code_dic, eqn_i, expression, indices_want, data_about_used_tensors, filters_bfs, diagonals):
    # go over all indices that appear on the right of the arrow
    all_indices_on_right = ""
    for d in data_about_used_tensors:
        all_indices_on_right += d["indices"]
    all_indices_on_right = sorted(list(set(all_indices_on_right)))

    for j in indices_want:
        if j not in all_indices_on_right:
            _raise_value_error("Index " + j + " appears on the left of arrow operator, but not on the right.")

    # flush all old shared data from code_dic
    keys = list(code_dic.keys())
    for k in keys:
        if str(k).startswith("__object_") or str(k).startswith("__size_"):
            del code_dic[k]

    # also need to include quantities that appear in filters_bfs
    data_filters_bfs = []
    for f in filters_bfs:
        data_filters_bfs.append({"key": f["left_core"].replace("__object_", ""), "indices": f["left_index"]})
        if f["right_type"] == "constant":
            data_filters_bfs.append({"key": f["right"].replace("__object_", ""), "indices": ""})

    initialization_code = ""
    # add all matrices needed in code_dic
    for d in data_about_used_tensors + data_filters_bfs:
        key = d["key"]
        key_reduced = key.replace("~", "")
        # this will make a deepcopy
        if "__object_" + key_reduced not in code_dic.keys():
            code_dic["__object_" + key_reduced] = comp[key]
            initialization_code += "__object_" + key_reduced + " = __object[\"" + key + "\"]" + "\n"

        shape = code_dic["__object_" + key_reduced].shape
        indices = d["indices"]
        for i, j in enumerate(indices):
            if "__size_" + j not in code_dic.keys():
                code_dic["__size_" + j] = shape[i]
                initialization_code += "__size_" + j + " = __object.get_shape(\"" + key + "\")[" + str(i) + "]" + "\n"
            else:
                if code_dic["__size_" + j] != shape[i]:
                    _raise_value_error("Index " + j + " of quantity " + key + " has shape " + shape[i] + " but earlier in the expression this index appeared with shape " + code_dic["__size_" + j] + " which is different!")

    func_name = "_tmp_func__" + str(eqn_i).zfill(3)
    code._add_raw(r"@njit" + "\n", indent4 = 0)
    code._add_raw(r"def " + func_name + "(__value):" + "\n", indent4 = 0)
    num_indents = 0

    # order indices in the order in which Numba will be the fastest (this involves some guessing...)
    optimal_order = _find_optimal_order_indices(all_indices_on_right, data_about_used_tensors + data_filters_bfs, indices_want)

    for j in optimal_order:
        code._add_raw("for " + j + " in range(__size_" + j + "):" + "\n", indent4 = num_indents + 1)
        num_indents += 1

    things_in_if_statement = []
    #
    for f in filters_bfs:
        num_appear_on_right = 0
        for j in f["left_index"]:
            if j in all_indices_on_right:
                num_appear_on_right += 1
        num_appear_on_left = 0
        for j in f["left_index"]:
            if j in indices_want:
                num_appear_on_left += 1
        if num_appear_on_left > 0:
            _raise_value_error("Some, of the indices \"" + f["left_index"] + "\" used in the condition appear on the left hand side of the arrow operator.")
        if num_appear_on_right == len(f["left_index"]):
            if f["right_type"] == "constant":
                use_right = f["right"].replace("~", "").strip() + ".real"
            else:
                use_right = f["right"].strip()
            things_in_if_statement.append("__object_" + f["left_core"].replace("~", "") + "[" + ",".join(f["left_index"]) + "].real " + f["op"] + " " + use_right)
        elif num_appear_on_right != 0:
            _raise_value_error("Some, but not all indices \"" + f["left_index"] + "\" used in the condition appear on the right hand side of the arrow operator.")
    #
    for d in diagonals:
        num_appear_on_right = 0
        for j in d:
            if j in all_indices_on_right:
                num_appear_on_right += 1
        num_appear_on_left = 0
        for j in d:
            if j in indices_want:
                num_appear_on_left += 1
        if num_appear_on_right == 2 and num_appear_on_left in [0, 2]:
            things_in_if_statement.append(d[0] + " != " + d[1])
            if code_dic["__size_" + d[0]] != code_dic["__size_" + d[1]]:
                _raise_value_error("Condition " + d[0] + " != " + d[1] + " is applied onto indices that are not of the same shape.")
        elif num_appear_on_right == 2 and num_appear_on_left == 1:
            _raise_value_error("In condition " + d[0] + " != " + d[1] + " you have both indices appear on the right of arrow and only one appear on the left of it.  This is ambiguous.")

    if len(things_in_if_statement) > 0:
        code._add_raw("if " + " and ".join(things_in_if_statement) + ":" + "\n", indent4 = num_indents + 1)
        num_indents += 1

    if indices_want == "":
        value_indices = ""
    else:
        value_indices = "[" + ",".join(indices_want) + "]"

    _expr_multi = _pack_code_multiline("__value" + value_indices + " += ", expression)

    for em in _expr_multi:
        code._add_raw(em + "\n", indent4 = num_indents + 1)

    if indices_want == "":
        code._add_raw("__value = np.array([0.0], dtype = complex)" + "\n")
    else:
        size_indices = []
        for j in indices_want:
            size_indices.append("__size_" + j)
        code._add_raw("__value = np.zeros((" + ",".join(size_indices) + "), dtype = complex)" + "\n")

    code._add_raw(func_name + "(__value)" + "\n")

    if indices_want == "":
        code._add_raw("__value = __value[0]" + "\n")

    return initialization_code

def _pack_code_multiline(first, second):
    second_parts = textwrap.wrap(second, width = 80, break_long_words = False)
    ret = []
    for i in range(len(second_parts)):
        if i == 0:
            tmp = first + second_parts[i]
        else:
            tmp = " " * len(first) + second_parts[i]
        if i != len(second_parts) - 1:
            tmp = tmp + " \\"
        ret.append(tmp)
    return ret


def _find_optimal_order_indices(all_indices_on_right, data_tensors, indices_want):
    # take indices that appear on the right of the arrow, but not on the left
    # as we want to have the ones on the left at the end anyways
    only_right = sorted(list(set(all_indices_on_right).difference(indices_want)))

    # keep score for each index
    score_for_only_right = np.zeros(len(only_right), dtype = int)

    longest_tensor = 0
    for d in data_tensors:
        ind = deepcopy(d["indices"])
        if len(ind) > longest_tensor:
            longest_tensor = len(ind)


    for d in data_tensors:
        ind = deepcopy(d["indices"])
        if ind != "":
            for i, l in enumerate(only_right):
                if l in ind:
                    where = ind.index(l)
                    score_for_only_right[i] += longest_tensor - where

    srt = np.argsort(score_for_only_right)[::-1]

    ret = "".join(np.array(list(only_right))[srt]) + indices_want

    return list(ret)

class Units():
    """
    Class that stores information about a physical quantity's
    units of electron volts (eV), angstroms (Ang), and Bohr magnetons (muB).

    :param eV: Power associated with electron-volts.  For example, if this
      parameter has value 2 then that represents units eV^2. The default is zero.

    :param A: Same as *eV* but for angstrom.

    :param muB: Same as *eV* but for Bohr magneton.

    Example usage::

        import wfbase as wf
        import numpy as np

        # open a database file on bcc phase of iron
        db = wf.load("data/fe_bcc.wf")

        # create now a computator from the database
        comp = db.do_mesh()

        # defines a new quantity, gamma, with value 3.0 in units of eV
        comp.new("gamma", {"value": 3.0, "units": wf.Units(eV = 1)})

        # defines a new quantity, delta, with value 2.5 in units of eV * A^2 / muB
        comp.new("delta", {"value": 2.5, "units": wf.Units(eV = 1, Ang = 2, muB = -1)})

    """
    def __init__(self, eV = 0.0, Ang = 0.0, muB = 0.0):
        self._eV = float(eV)
        self._Ang = float(Ang)
        self._muB = float(muB)
        self.__tolerance = 1.0E-8

    def _to_SI(self, value):
        factor = 1.0
        if np.abs(self._eV) > self.__tolerance:
            factor *= electron_charge_SI**self._eV
        if np.abs(self._Ang) > self.__tolerance:
            factor *= angstrom_SI**self._Ang
        if np.abs(self._muB) > self.__tolerance:
            factor *= (electron_charge_SI * hbar_SI / (2.0 * electron_mass_SI))**self._muB
        return value * factor

    def _check_units_the_same(self, another):
        unit_diffs = np.abs(np.array([
            self._eV  - another._eV,
            self._Ang - another._Ang,
            self._muB - another._muB]))

        return all(unit_diffs < self.__tolerance)

    def _multiply(self, another):
        return Units(eV  = self._eV  + float(another._eV ),
                     Ang = self._Ang + float(another._Ang),
                     muB = self._muB + float(another._muB))

    def _inverse(self):
        return Units(eV  = (-1.0)*self._eV ,
                     Ang = (-1.0)*self._Ang,
                     muB = (-1.0)*self._muB)

    def _divide(self, another):
        return Units(eV  = self._eV  - float(another._eV ),
                     Ang = self._Ang - float(another._Ang),
                     muB = self._muB - float(another._muB))

    def _exponent(self, ex):
        if self._is_trivial():
            if ex["units"]._is_trivial() == False:
                _raise_value_error("You can't have exponent with units!")
            return Units(eV = 0.0, Ang = 0.0 , muB = 0.0)
        else:
            # we need to get the numerical value of the exponent before it is evaluated
            # this is a chicken and an egg problem.  Therefore, I will allow here
            # only simple exponents that consist only of numerical quantities,
            # and parentheses or operations, as these might appear in the parsing.
            _simple = ex["value"]
            for s in _simple:
                if s not in "()*+-/^j.0123456789 ":
                    _raise_value_error("""If you use (...)^(,,,) and if "..." has units
                    then ",,," can only be a numerical value.
                    You are not allowed to use constants or tensors, such as A_ij^(B + 3),
                    as long as A has units.  You are allowed to do things
                    like A_ij^(-3.0) or similar. """ + _simple)
            try:
                numerical_value = eval(_simple)
            except Exception:
                traceback.print_stack()
                print()
                _raise_value_error("""Currently exponents can only be numerical values.
                You are not allowed to use anything too complex in the exponent.
                You are allowed to do things like A_ij^(-3.0) or similar. """ + _simple)

            if abs(np.imag(numerical_value)/np.abs(numerical_value)) > self.__tolerance:
                _raise_value_error("You can't raise something with units to exponent that is not real!")
            return Units(eV  = self._eV  * float(np.real(numerical_value)),
                         Ang = self._Ang * float(np.real(numerical_value)),
                         muB = self._muB * float(np.real(numerical_value)))

    def _exponent_float(self, vv):
        return Units(eV  = self._eV  * vv,
                     Ang = self._Ang * vv,
                     muB = self._muB * vv)

    def _is_trivial(self):
        return (np.abs(self._eV ) < self.__tolerance) and \
               (np.abs(self._Ang) < self.__tolerance) and \
               (np.abs(self._muB) < self.__tolerance)

    def __str__(self):
        out = ""
        if np.abs(self._eV - 1.0) < self.__tolerance:
            out += r" eV "
        elif self._eV > self.__tolerance:
            out += r" eV^"+str(self._eV).strip() + " "
        elif self._eV < (-1.0)*self.__tolerance:
            out += r" eV^("+str(self._eV).strip() + ") "

        if np.abs(self._Ang - 1.0) < self.__tolerance:
            out += r" Ang "
        elif self._Ang > self.__tolerance:
            out += r" Ang^"+str(self._Ang).strip() + " "
        elif self._Ang < (-1.0)*self.__tolerance:
            out += r" Ang^("+str(self._Ang).strip() + ") "

        if np.abs(self._muB - 1.0) < self.__tolerance:
            out += r" muB "
        elif self._muB > self.__tolerance:
            out += r" muB^"+str(self._muB).strip() + " "
        elif self._muB < (-1.0)*self.__tolerance:
            out += r" muB^("+str(self._muB).strip() + ") "

        if out == "":
            out = "1"

        return out.strip()

def render_latex(latex_str, fname, dpi = 300):
    """
    Renders mathematical LaTeX expression *latex_str* and saves it into file *fname*.

    :param latex_str: mathematical LaTeX expression to be rendered.

    :param fname: name of the output file.  Either png or pdf format.

    :param dpi: dots per inch for png file (ignored for pdf).

    Example usage::

        import wfbase as wf

        # open a database file on bcc phase of iron
        db = wf.load("data/fe_bcc.wf")

        # create now a computator from the database
        comp = db.do_mesh()

        # evaluate some object
        comp.evaluate("sigma_ij <= (j / (numk * volume)) * (f_km - f_kn) * A_knmi * A_kmnj")

        # now get LaTeX'ed data about this object
        lat = comp.get_latex("sigma")

        wf.render_latex(lat, "test.png")

    """

    if len(fname.strip()) < 5:
        _raise_value_error("Must provide a valid filename of a png or a pdf file.")

    if fname.strip()[-4:].lower() not in [".png", ".pdf"]:
        _raise_value_error("Must provide a valid filename of a png or a pdf file.")
    else:
        file_format = fname.strip()[-3:].lower()

    expression = str(latex_str).strip()
    if expression.startswith(r"\begin{align}") and expression.endswith(r"\end{align}"):
        expression = expression
    elif expression.startswith(r"$") and expression.endswith(r"$"):
        expression = expression
    else:
        expression = "$" + expression + "$"

    expression = expression.replace("\n", " ")

    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}"
    fig = plt.figure()
    text = fig.text(0.0, 0.0, expression, ha = "center", va = "center")

    if file_format == "png":
        fig.savefig(fname, dpi = dpi)
        bbox = text.get_window_extent()
        width, height = (bbox.size / float(dpi)) + 0.1
        fig.set_size_inches((width, height))
        text.set_position((0.5, 0.5))
        fig.savefig(fname.strip(), dpi = dpi)
    elif file_format == "pdf":
        fig.savefig(fname)
        bbox = text.get_window_extent()
        width, height = bbox.size
        fig.set_size_inches(((width /72.0) + 0.05,
                             (height/72.0) + 0.05))
        text.set_position((0.5, 0.5))
        fig.savefig(fname.strip())
    else:
        print("Should not happen.")
        exit()

def display_in_terminal(fname):
    """
    Displays image file *fname* in the terminal.  Only supported on terminals
    that allow imgcat, such as iTerm2 on OS X.  You can find information about
    `installing imgcat in iTerm2 here <https://iterm2.com/documentation-images.html>`_.

    :param fname: name of the file.

    Example usage::

        import wfbase as wf

        # open a database file on bcc phase of iron
        db = wf.load("data/fe_bcc.wf")

        # create now a computator from the database
        comp = db.do_mesh()

        # evaluate some object
        comp.evaluate("sigma_ij <= (j / (numk * volume)) * (f_km - f_kn) * A_knmi * A_kmnj")

        # now get LaTeX'ed data about this object
        lat = comp.get_latex("sigma")

        wf.render_latex(lat, "test.png")
        wf.display_in_terminal("test.png")

    """

    f = open(fname)
    imgcat.imgcat(f)
    f.close()

def display_in_separate_window(fname):
    """
    Displays image file *fname* in a separate window.  The result will
    depend on your operating system and how it interacts with python's
    PIL module.

    :param fname: name of the file.

    Example usage::

        import wfbase as wf

        # open a database file on bcc phase of iron
        db = wf.load("data/fe_bcc.wf")

        # create now a computator from the database
        comp = db.do_mesh()

        # evaluate some object
        comp.evaluate("sigma_ij <= (j / (numk * volume)) * (f_km - f_kn) * A_knmi * A_kmnj")

        # now get LaTeX'ed data about this object
        lat = comp.get_latex("sigma")

        wf.render_latex(lat, "test.png")
        wf.display_in_separate_window("test.png")

    """
    im = Image.open(fname)
    im.show()

####################################################################
# Below this point we have private classes and functions that user #
# is not meant to interact with directly.                          #
####################################################################

class _StorePythonCode():
    # It is essential here that this code is simply defining, and modifying, variables
    # one by one.  So that if you work on variable A first then you later work on B, you
    # will then never go back to working on A.

    def __init__(self, optimize_recomputation = True):
        self.__code_chunks = [""]
        self.__code_preamble = [""]
        self.__variable_names = []

        # used to retrieve previously computed values
        self.__broadcast_info = {}
        self.__einsum_info = {}

        # remember which variables might have been updated with <+= or <<= operators.  You can't then use stored values involving those variables.
        self.__which_cores_got_updated = []

        self.__optimize_recomputation = optimize_recomputation

        self.__counter_used_stored = {
            "einsum_simple": 0,
            "einsum_jumbled_indices":0,
            "broadcast_simple": 0,
            "broadcast_jumbled_indices":0
        }

        self._added_soft_div = False

    def get_num_chunks(self):
        return len(self.__code_chunks)

    def get_code_chunk(self, ii, prefix = " "*4):
        ret = ""

        for l in self.__code_preamble[ii].split("\n"):
            if l == "":
                continue
            ret += prefix + l.rstrip() + "\n"

        for l in self.__code_chunks[ii].split("\n"):
            if l == "":
                continue
            ret += prefix + l.rstrip() + "\n"

        return ret

    def _add_raw(self, txt, to_preamble = False, to_the_top = False, indent4 = 0):
        inde = "    " * indent4
        if to_preamble == False:
            if to_the_top == False:
                self.__code_chunks[-1] += inde + txt.rstrip() + "\n"
            else:
                self.__code_chunks[-1] = inde + txt.rstrip() + "\n" + self.__code_chunks[-1]
        else:
            if to_the_top == False:
                self.__code_preamble[-1] += inde + txt.rstrip() + "\n"
            else:
                self.__code_preamble[-1] = inde + txt.rstrip() + "\n" + self.__code_preamble[-1]

    def give_me_unique_variable_name(self, base):
        i = 0
        while(True):
            ret = base + str(i).zfill(2)
            if ret not in self.__variable_names:
                return ret
            i = i + 1

    def start_new_chunk(self):
        self.__code_chunks.append("")
        self.__code_preamble.append("")

    def get_all_defined_variables(self):
        return deepcopy(self.__variable_names)

    def _add_definition(self, variable, right_code, adjust_newlines = False, to_preamble = False):
        if variable in self.__variable_names:
            _raise_value_error("Duplicate variable name!")
        self.__variable_names.append(variable.strip())
        if adjust_newlines == True:
            if len(right_code) >  50:
                right_code = right_code.replace("\\\n", "\\\n" + " "*(len(variable.strip()) + 13))
            else:
                right_code = right_code.replace("\\\n", " ")
        if to_preamble == False:
            self.__code_chunks[-1] += variable.strip() + " = " + right_code + "\n"
        else:
            self.__code_preamble[-1] += variable.strip() + " = " + right_code + "\n"
        return variable

    def _add_definition_from_einsum(self, base, einsum_1, einsum_2_with_newlines, do_copy):
        einsum_2 = einsum_2_with_newlines.replace("\\\n", " ")

        sp = einsum_1.split("->")
        left_sp = sp[0].replace(",", "").strip()
        right_sp = sp[1].strip()
        for i in right_sp:
            if i not in left_sp:
                _raise_value_error("Index \"" + i + "\" appears on the left of assignment operator (<=, <<=, <+=) but this index does not appear on the right." )

        if self.__optimize_recomputation:
            # check if we ever did the same einsum before
            for var in self.__einsum_info.keys():
                # check if there is something with the same tensors
                if einsum_2 == self.__einsum_info[var]["einsum_2"]:
                    # check if any of the variables got changed
                    values_got_changed = False
                    for core in self.__which_cores_got_updated:
                        if core in einsum_2:
                            values_got_changed = True
                    if values_got_changed == False:
                        if einsum_1 == self.__einsum_info[var]["einsum_1"]:
                            self.__counter_used_stored["einsum_simple"] += 1
                            return var
                        else:
                            # this checks if we earlier computed something with equivalent indices, but not exactly the same
                            left = self.__einsum_info[var]["einsum_1"]
                            right = einsum_1
                            mm = _find_1_to_1_map_from_left_to_right(left, right)
                            if mm is not None:
                                self.__counter_used_stored["einsum_jumbled_indices"] += 1
                                return var

        # compute it, as we didn't find this before
        variable = self.give_me_unique_variable_name(base)
        if do_copy == False:
            ret = self._add_definition(variable,         "opteinsum(\"" + einsum_1 + "\",\\\n" + einsum_2_with_newlines + ")" , adjust_newlines = True)
        else:
            # probably could go without np.copy here, and at other places,
            # but better be safe than sorry, as opteinum in many cases
            # returns a shallow copy (even if indices are contracted)
            ret = self._add_definition(variable, "np.copy(opteinsum(\"" + einsum_1 + "\",\\\n" + einsum_2_with_newlines + "))", adjust_newlines = True)

        if self.__optimize_recomputation:
            # store in case we need it later
            if variable in self.__einsum_info.keys():
                _raise_value_error("Duplicate variable name, einsum!")
            self.__einsum_info[variable] = {"einsum_1": einsum_1, "einsum_2": einsum_2}

        return ret

    def changed_value_of_this_core(self, core):
        self.__which_cores_got_updated.append(core)

    def store_broadcast_info_for_lookup(self, variable, info):
        if self.__optimize_recomputation:
            if variable in self.__broadcast_info.keys():
                _raise_value_error("Duplicate variable name, broadcast!")
            self.__broadcast_info[variable] = info

    def check_if_did_this_broadcast_before(self, vals, inds, operations):
        if self.__optimize_recomputation:
            for var in self.__broadcast_info.keys():
                # check if there is an equivalent computation that was done already
                if operations == self.__broadcast_info[var]["input_operations"]:
                    if vals == self.__broadcast_info[var]["input_vals"]:
                        # check if any of the variables got changed in the meantime
                        values_got_changed = False
                        for core in self.__which_cores_got_updated:
                            for v in vals:
                                if core in v:
                                    values_got_changed = True
                        if values_got_changed == False:
                            # this means that earlier we computed the exact same thing, with the same set of indices
                            if inds == self.__broadcast_info[var]["input_inds"]:
                                self.__counter_used_stored["broadcast_simple"] += 1
                                return {"value": var,
                                        "ind": self.__broadcast_info[var]["ret_ind"],
                                        "units": self.__broadcast_info[var]["ret_units"]}
                            else:
                                # this checks if we earlier computed something with equivalent indices, but jumbled around
                                # This applies only to the case when the jumbled indices don't appear in conditions.
                                # If they appear in conditions (even if equivalent) the code will ignore that and
                                # will recompute everything from scratch.
                                left = ",".join(self.__broadcast_info[var]["input_inds"])
                                right = ",".join(inds)
                                mm = _find_1_to_1_map_from_left_to_right(left, right)
                                if mm is not None:
                                    self.__counter_used_stored["broadcast_jumbled_indices"] += 1
                                    return {"value": var,
                                            "ind": _convert_left_to_right(self.__broadcast_info[var]["ret_ind"], mm),
                                            "units": self.__broadcast_info[var]["ret_units"]}
        return None

    def how_many_times_used_stored(self):
        return deepcopy(self.__counter_used_stored)

def _check_const(tokens):
    if len(tokens) != 1:
        _raise_value_error("Unexpected behavior of pyparsing.")

def _check_func(tokens):
    if len(tokens) != 1:
        _raise_value_error("Unexpected behavior of pyparsing.")
    if (len(tokens[0]) != 2) or (str(tokens[0][0]) not in ["Real", "Imag"]):
        _raise_value_error("Incorrect input.  Problem with function parsing.")

def _check_conjug(tokens):
    if len(tokens) != 1:
        _raise_value_error("Unexpected behavior of pyparsing.")
    if (len(tokens[0]) != 2) or (str(tokens[0][0]) != "#"):
        _raise_value_error("Incorrect input.  Problem with complex conjugation.")

def _check_d_one(tokens):
    if len(tokens) != 1:
        _raise_value_error("Unexpected behavior of pyparsing.")
    if (len(tokens[0]) != 2) or (str(tokens[0][0]) != "&"):
        _raise_value_error("Incorrect input.  Problem with & operator.")

def _check_sign(tokens):
    if len(tokens) != 1:
        _raise_value_error("Unexpected behavior of pyparsing.")
    if len(tokens[0]) != 2 or str(tokens[0][0]) not in ["+", "-"]:
        _raise_value_error("Incorrect input.  Problem with sign in front of a symbol.")

def _check_power(tokens):
    if len(tokens) != 1 or tokens[0][1] != "^":
        _raise_value_error("Unexpected behavior of pyparsing.")
    if len(tokens[0]) != 3:
        _raise_value_error("Don't allow things like A^B^C.  Use parentheses instead.  For example (A^B)^C.")

def _check_mult_div(tokens):
    if len(tokens) != 1:
        _raise_value_error("Unexpected behavior of pyparsing.")
    for j in range(1, len(tokens[0]), 2):
        op  = tokens[0][j]
        if op != "*" and op != "/":
            _raise_value_error("This should never happen.  Expected * or /")

    count_divisions = 0
    for j in range(1, len(tokens[0]), 2):
        op  = tokens[0][j]
        if op == "/":
            count_divisions +=1
    if count_divisions > 1:
        _raise_value_error("Not allowing terms like A / B / C or similar, as it might be ambiguous.  Use parantheses to clarify what you mean.")

    if count_divisions == 1:
        if tokens[0][-2] != "/":
            _raise_value_error("Not allowing terms like A / B * C or similar, as it might be ambiguous.  Use parantheses to clarify what you mean.")

def _check_add_sub(tokens):
    if len(tokens) != 1:
        _raise_value_error("Unexpected behavior of pyparsing.")
    for j in range(1, len(tokens[0]), 2):
        op  = tokens[0][j]
        if op != "+" and op != "-":
            _raise_value_error("This should never happen.  Expected + or -")

def _check_ein(tokens):
    if len(tokens) != 1:
        _raise_value_error("Unexpected behavior of pyparsing.")
    if len(tokens[0]) != 3 or str(tokens[0][1]) not in ["<=", "<+=", "<<="]:
        _raise_value_error("Arrow must appear as ... <= ...    If you want something fancier, use parantheses to clear up what you mean.")

class ReorgConstVar():
    def __init__(self, s, loc, tokens):
        _check_const(tokens)
        self.value = tokens[0]
    def eval(self, parent):
        return self.value.strip()

class ReorgFuncOp():
    def __init__(self, s, loc, tokens):
        _check_func(tokens)
        self.func = tokens[0][0]
        self.value = tokens[0][1]
    def eval(self, parent):
        return self.func + "(" + self.value.eval(parent = self) + ")"

class ReorgConjugOp():
    def __init__(self, s, loc, tokens):
        _check_conjug(tokens)
        self.value = tokens[0][1]
    def eval(self, parent):
        return "#" + "(" + self.value.eval(parent = self) + ")"

class ReorgSignOp():
    def __init__(self, s, loc, tokens):
        _check_sign(tokens)
        self.sign, self.value = tokens[0]
    def eval(self, parent):
        return self.sign + "(" + self.value.eval(parent = self) + ")"

class ReorgPowerOp():
    def __init__(self, s, loc, tokens):
        _check_power(tokens)
        self.value = tokens[0]
    def eval(self, parent):
        return "(" + self.value[0].eval(parent = self) + ")^(" + self.value[2].eval(parent = self) + ")"

def _take_inverse_of(obj, parent):
    if isinstance(obj, ReorgConstVar):
        return "&"  + obj.eval(parent = parent)
    elif isinstance(obj, ReorgFuncOp):
        return "&(" + obj.eval(parent = parent) + ")"
    elif isinstance(obj, ReorgConjugOp):
        return "&(" + obj.eval(parent = parent) + ")"
    elif isinstance(obj, ReorgSignOp):
        return "&(" + obj.eval(parent = parent) + ")"
    elif isinstance(obj, ReorgPowerOp):
        return "&(" + obj.eval(parent = parent) + ")"
    elif isinstance(obj, ReorgMultDivOp):
        return        obj.eval(parent = parent, add_inverse = True)
    elif isinstance(obj, ReorgAddSubOp):
        return "&"  + obj.eval(parent = parent)
    elif isinstance(obj, ReorgArrowOp):
        return "&"  + obj.eval(parent = parent)
    else:
        _raise_value_error("Unknown object in taking inverse.")

class ReorgMultDivOp():
    def __init__(self, s, loc, tokens):
        _check_mult_div(tokens)
        self.value = tokens[0]

    def eval(self, parent, add_inverse = False):
        if add_inverse == False:
            ret = self.value[0].eval(parent = self)
            for j in range(1, len(self.value), 2):
                op  = self.value[j]
                if op == "*":
                    nxt = self.value[j + 1].eval(parent = self)
                elif op == "/":
                    nxt = _take_inverse_of(self.value[j + 1], self)
                else:
                    _raise_value_error("Unexpected operator!")
                ret = ret + " * " + nxt
        else:
            ret = _take_inverse_of(self.value[0], self)
            for j in range(1, len(self.value), 2):
                op  = self.value[j]
                if op == "*":
                    nxt = _take_inverse_of(self.value[j + 1], self)
                elif op == "/":
                    nxt = self.value[j + 1].eval(parent = self)
                else:
                    _raise_value_error("Unexpected operator!")
                ret = ret + " * " + nxt
        # it is more optimal NOT to have extra parentheses here as then
        # something like A*B*C/(D*E) will be one product A*B*C*(1/D)*(1/E)
        # instead of A*B*C*((1/D)*(1/E)) which has more overhead
#        if len(self.value) > 1:
#            ret = "( " + ret + " )"
        return ret

class ReorgAddSubOp():
    def __init__(self, s, loc, tokens):
        _check_add_sub(tokens)
        self.value = tokens[0]
    def eval(self, parent):
        ret =  self.value[0].eval(parent = self)
        for j in range(1, len(self.value), 2):
            op  = self.value[j]
            nxt = self.value[j+1].eval(parent = self)
            ret = ret + " " + op + " " + nxt
        if len(self.value) > 1:
            ret = "(" + ret + ")"
        return ret

class ReorgArrowOp():
    def __init__(self, s, loc, tokens):
        _check_ein(tokens)
        self.value = tokens[0]
    def eval(self, parent):
        if parent is None:
            return self.value[0].eval(parent = self) + " " + self.value[1] + " " + self.value[2].eval(parent = self)
        else:
            return "(" + self.value[0].eval(parent = self) + " " + self.value[1] + " " + self.value[2].eval(parent = self) + ")"

class ParserReorg():
    def __init__(self):
        operand = _get_operand()
        operand.set_parse_action(ReorgConstVar)
        self._rparser = pp.infix_notation(
            operand,
            [
                (pp.oneOf("Real Imag") , 1, pp.opAssoc.RIGHT, ReorgFuncOp    ), # various function calls
                (         "#"          , 1, pp.opAssoc.RIGHT, ReorgConjugOp  ), # complex conjugation
                (         "^"          , 2, pp.opAssoc.LEFT , ReorgPowerOp   ), # power raising (strictly speaking this should be a RIGHT not LEFT to follow conventions.  But we don't allow user to do A^B^C so it doesn't matter.)
                (pp.oneOf("+ -")       , 1, pp.opAssoc.RIGHT, ReorgSignOp    ), # sign in front of an object
                (pp.oneOf("* /")       , 2, pp.opAssoc.LEFT , ReorgMultDivOp ), # multiplication or division
                (pp.oneOf("+ -")       , 2, pp.opAssoc.LEFT , ReorgAddSubOp  ), # addition and subtraction
                (pp.oneOf("<= <+= <<="), 2, pp.opAssoc.LEFT , ReorgArrowOp   ), # perform assignment
            ]
        )

    def reorganize(self, txt):
        return _my_parse_string(self._rparser, txt, parse_all = True)[0].eval(parent = None)

def _perform_partial_filtering(obj, ind, partially_filtered_indices):
    ret = deepcopy(obj)
    for i, ii in enumerate(ind):
        if ii in partially_filtered_indices:
            tmp = []
            for j in range(len(ind)):
                if j == i:
                    tmp.append("_s[\"" + ii + "\"]")
                else:
                    tmp.append(":")
            ret += "[" + ",".join(tmp) + "]"
    return ret

def _process_latex_imag_j(s):
    return s.replace("j", r" i")

def _save_info_about_original_shapes(orig_val, ind, comp, code):
    for i, ii in enumerate(ind):
        if ii in comp._all_diagonal_indices:
            if ii not in comp._all_diagonal_indices_stored_shape:
                code._add_definition("_orig_shp[\"" + ii + "\"]", orig_val + ".shape[" + str(i) + "]", to_preamble = True)
                comp._all_diagonal_indices_stored_shape += ii

class EvalConstVar():
    def __init__(self, s, loc, tokens):
        _check_const(tokens)
        self.text = tokens[0]
        self.name = tokens.get_name()

    def eval(self, comp, code, calling_from_left_arrow = False, try_latex_conjugate = False):
        did_latex_conjugate = False
        if self.name == "ind_for_left_arrow":
            _raise_value_error("You can't evaluate thing that starts with underscore.  This thing needs to be to the left of <= operator or similar.")
        elif self.name in ["float", "integer"]:
            ret = {"value": str(self.text),
                   "ind": "",
                   "units": Units(eV = 0, Ang = 0 , muB = 0),
                   "latex": self.text.strip()}
        elif self.name in ["imaginary float", "imaginary integer"]:
            ret = {"value": str(complex(self.text)),
                   "ind": "",
                   "units": Units(eV = 0, Ang = 0 , muB = 0),
                   "latex": _process_latex_imag_j(self.text.strip())}
        elif self.name in ["imaginary one"]:
            ret = {"value": "complex(1.0j)",
                   "ind": "",
                   "units": Units(eV = 0, Ang = 0 , muB = 0),
                   "latex": _process_latex_imag_j("j")}
        elif self.name in ["tensor"]:
            core, ind = self.text.split("_")
            if comp[core].shape == ():
                _raise_value_error("Object " + core + " does not have a shape of a tensor.")
            orig_val = "__object[\"" + core + "\"]"
            ret = {"value": _perform_partial_filtering(orig_val, ind, comp._partially_filtered_indices),
                   "ind": ind,
                   "units": comp.get(core, "units"),
                   "latex": comp._return_in_latex(core.strip(), ind.strip(), try_latex_conjugate)}
            did_latex_conjugate = True
            _save_info_about_original_shapes(orig_val, ret["ind"], comp, code)
        elif self.name in ["constant"]:
            if comp[self.text].shape != ():
                _raise_value_error("Object " + self.text + " is not a constant.")
            ret = {"value": "__object[\""+ self.text +"\"]",
                   "ind": "",
                   "units": comp.get(self.text, "units"),
                   "latex": comp._return_in_latex(self.text.strip(), None, try_latex_conjugate)}
            did_latex_conjugate = True
        else:
            _raise_value_error("Unknown name!")
        ret = _resolve_indices_if_called_from_left_arrow(calling_from_left_arrow, ret, code, comp)
        if calling_from_left_arrow != False:
            ret = _apply_diagonals_on_output_if_indices_remain(code, ret, comp)
        if try_latex_conjugate == True:
            return ret, did_latex_conjugate
        else:
            return ret

class EvalFuncOp():
    def __init__(self, s, loc, tokens):
        _check_func(tokens)
        self.func  = tokens[0][0]
        self.value = tokens[0][1]

    def eval(self, comp, code, calling_from_left_arrow = False):
        ret = self.value.eval(comp = comp, code = code)
        if self.func == "Real":
            ret["value"] = "np.real(" + ret["value"] + ")"
            ret["latex"] = r" {\rm Re} \left( " + ret["latex"].strip() + r" \right) "
        elif self.func == "Imag":
            ret["value"] = "np.imag(" + ret["value"] + ")"
            ret["latex"] = r" {\rm Im} \left( " + ret["latex"].strip() + r" \right) "
        else:
            _raise_value_error("Unknown function.")
        ret = _resolve_indices_if_called_from_left_arrow(calling_from_left_arrow, ret, code, comp)
        if calling_from_left_arrow != False:
            ret = _apply_diagonals_on_output_if_indices_remain(code, ret, comp)
        return ret

class EvalConjugOp():
    def __init__(self, s, loc, tokens):
        _check_conjug(tokens)
        self.value = tokens[0][1]

    def eval(self, comp, code, calling_from_left_arrow = False):
        if isinstance(self.value, EvalConstVar):
            ret, did_latex_conjugate = self.value.eval(comp = comp, code = code, try_latex_conjugate = True)
        else:
            ret = self.value.eval(comp = comp, code = code)
            did_latex_conjugate = False
        ret["value"] = "np.conjugate(" + ret["value"] + ")"
        if did_latex_conjugate == False: # revert to basic conjugation.
            ret["latex"] = r" \overline{ " + ret["latex"].strip() + r" } "
        ret = _resolve_indices_if_called_from_left_arrow(calling_from_left_arrow, ret, code, comp)
        if calling_from_left_arrow != False:
            ret = _apply_diagonals_on_output_if_indices_remain(code, ret, comp)
        return ret

def _prepare_soft_divide(numer, denom_value, denom_ind, code, comp, parantheses_around):
    all_whch = []
    for dia in comp._diagonals:
        howmany = 0
        for i in dia:
            if i in denom_ind:
                howmany += 1
        if howmany == 0 or howmany == 1:
            continue

        whch = {"ind": [],
                "osh": None,
                "sli": []}
        for j, jj in enumerate(denom_ind):
            if jj is None:
                continue
            if jj in dia:
                whch["ind"].append(j)
                if whch["osh"] is None:
                    whch["osh"] = "_orig_shp[[[\"" + jj + "\"]]]"
                if jj in comp._partially_filtered_indices:
                    whch["sli"].append("_s[[[\"" + jj + "\"]]]")
                else:
                    whch["sli"].append(None)
        all_whch.append(whch)

    if len(all_whch) == 0:
        return None

    if code._added_soft_div == False:
        txt = """
def _soft_divide(_x, _y, _whc_all):
    if len(_whc_all) == 0: raise ValueError("Should not occur.")
    _yc = np.copy(_y)
    _all_dd = False
    for _whc in _whc_all:
        _dd = np.zeros(tuple([_whc["osh"]] * len(_whc["ind"])), dtype = bool)
        np.fill_diagonal(_dd, True)
        for _i in range(len(_whc["ind"])):
            if _whc["sli"][_i] is not None:
                _dd = _dd[tuple([slice(None) if _i != _j else _whc["sli"][_i] for _j in range(len(_whc["ind"]))])]
        _slic = tuple([slice(None) if _i in _whc["ind"] else None for _i in range(_y.ndim)])
        _dd = _dd[_slic] + np.zeros(_y.shape, dtype = bool)
        _all_dd = np.logical_or(_dd, _all_dd)
    _yc[_all_dd] = np.inf
    _ret = np.nan_to_num(_x / _yc)
    return _ret
        """.strip() + "\n"
        code._add_raw(txt, to_preamble = True, to_the_top = True)
        code._added_soft_div = True

    str_all_whch = str(all_whch)
    str_all_whch = str_all_whch.replace(r"'_s[[[", "_s[").replace("]]]'", "]").replace(r"'_orig_shp[[[", "_orig_shp[")
    str_all_whch = str_all_whch.replace("{", "\n    {")

    ret = "_soft_divide(" + numer + ", " + denom_value + ", " + str_all_whch + ")"

    if parantheses_around == True:
        ret = "(" + ret + ")"
    return ret

class EvalDOneOp():
    def __init__(self, s, loc, tokens):
        _check_d_one(tokens)
        self.value = tokens[0][1]

    def eval(self, comp, code, calling_from_left_arrow = False):
        ret = self.value.eval(comp = comp, code = code)
        tsd = _prepare_soft_divide("1.0", ret["value"], list(ret["ind"]), code, comp, parantheses_around = True)
        if tsd is None:
            ret["value"] = "(1.0/(" + ret["value"] + "))"
        else:
            ret["value"] = tsd
        ret["latex"] = r" \left( " + ret["latex"].strip() + r" \right)^{-1} "
        ret["units"] = ret["units"]._inverse()
        ret = _resolve_indices_if_called_from_left_arrow(calling_from_left_arrow, ret, code, comp)
        if calling_from_left_arrow != False:
            ret = _apply_diagonals_on_output_if_indices_remain(code, ret, comp)
        return ret

class EvalSignOp():
    def __init__(self, s, loc, tokens):
        _check_sign(tokens)
        self.sign, self.value = tokens[0]

    def eval(self, comp, code, calling_from_left_arrow = False):
        mult = {"+": 1.0, "-": -1.0}[self.sign]
        ret = self.value.eval(comp = comp, code = code)
        ret["value"] = "(" + str(mult)  + ") * (" + ret["value"] + ")"
        ret["latex"] = r"\left( " + self.sign.strip() + " " + ret["latex"] + r" \right)"
        ret = _resolve_indices_if_called_from_left_arrow(calling_from_left_arrow, ret, code, comp)
        if calling_from_left_arrow != False:
            ret = _apply_diagonals_on_output_if_indices_remain(code, ret, comp)
        return ret

class EvalPowerOp():
    def __init__(self, s, loc, tokens):
        _check_power(tokens)
        self.value = tokens[0]

    def eval(self, comp, code, calling_from_left_arrow = False):
        trmL = self.value[0].eval(comp = comp, code = code)
        trmR = self.value[2].eval(comp = comp, code = code)

        if trmR["ind"] != "":
            _raise_value_error("Exponents must be simply numbers.")
        if trmR["units"]._is_trivial() == False:
            _raise_value_error("Exponents can't have units.")

        ret = {"value": "np.power(" + trmL["value"] + ", " + trmR["value"] + ")",
               "ind": trmL["ind"],
               "units": trmL["units"]._exponent(trmR),
               "latex": r"\left(" + trmL["latex"].strip() + r"\right)"+ r"^{ " + trmR["latex"].strip() + r" } "} # maybe parantheses not needed if trmL is a single object?
        ret = _resolve_indices_if_called_from_left_arrow(calling_from_left_arrow, ret, code, comp)
        if calling_from_left_arrow != False:
            ret = _apply_diagonals_on_output_if_indices_remain(code, ret, comp)
        return ret

def _check_that_return_indices_compatible_with_filter(indices_to, filters):
    for filt in filters:
        found = ""
        for i in filt["cond_inds"]:
            if i in indices_to:
                found += i
        if found != "":
            _raise_value_error("You specified greater/lesser condition involving indices " + filt["cond_inds"] + " but now you are not summing over index " + found + ".")

def _add_filters_to_einsum(indices_from, indices_to, filters):
    filter_vals = []
    filter_from = []
    filter_latex = []

    _check_that_return_indices_compatible_with_filter(indices_to, filters)

    for filt in filters:
        howmany = 0
        for i in filt["cond_inds"]:
            if i in indices_from:
                howmany += 1
        if howmany == 0:
            continue
        elif howmany != len(filt["cond_inds"]):
            _raise_value_error("You specified greater/lesser condition involving indices " + filt["cond_inds"] + ". " + 
                             "Therefore, for any sum in your expression, if any of these indices appear " + 
                             "in the sum, all other should appear as well.  Otherwise, greater/lesser condition you " + 
                             "specified makes no sense.")
        filter_from.append(filt["cond_inds"])
        filter_vals.append(filt["cond_value"])
        filter_latex.append(filt["cond_latex"])

    return filter_vals, filter_from, filter_latex

def _add_diagonals_to_einsum(indices_from, indices_to, comp):
    diagonal_vals = []
    diagonal_from = []
    diagonal_latex = []

    for idia, dia in enumerate(comp._diagonals):
        howmany_from = 0
        for i in dia:
            if i in indices_from:
                howmany_from += 1
        howmany_to = 0
        for i in dia:
            if i in indices_to:
                howmany_to += 1
        if howmany_from == 2 and howmany_to == 1:
            _raise_value_error("You have specified this condition on indices ( " + dia[0] + " != " + dia[1] + " ) but now you have one index " + 
                               "on the left of arrow operator and both on the right of it.  It is ambiguous what you want to do with this condition.")
        if howmany_from == 0 or howmany_from == 1:
            continue
        if howmany_to == 1 or howmany_to == 2:
            continue

        diagonal_from.append("".join(dia))
        val = "(1.0 - np.eye(_orig_shp[\"" + dia[0] + "\"], _orig_shp[\"" + dia[1] + "\"]))"
        val = _perform_partial_filtering(val, dia, comp._partially_filtered_indices)
        diagonal_vals.append(val)
        diagonal_latex.append(dia[0] + r" \neq " + dia[1])

    return diagonal_vals, diagonal_from, diagonal_latex

def _remove_outer_para(s):
    #Removes parentheses such as ( 4 + 5 * (3 - 1) ).  But leaves alone stuff like (3 + 43) (2 - 3)

    if len(s) < 13:
        return s

    if s[:6] != r"\left(" or s[-7:] != r"\right)":
        return s

    counter = 0
    for i in range(6, len(s) - 7):
        if s[i:].startswith(r"\left("):
            counter += 1
        elif s[i:].startswith(r"\right)"):
            counter -= 1
        if counter < 0:
            return s

    if counter != 0:
        _raise_value_error("Missing parentheses!")

    return s[6:-7]

class EvalMultDivOp():
    def __init__(self, s, loc, tokens):
        _check_mult_div(tokens)
        self.value = tokens[0]

    def eval(self, comp, code, calling_from_left_arrow = False):
        vals = []
        inds = []
        operations = []
        latex = ""
        tmpL = self.value[0].eval(comp = comp, code = code)
        vals.append(tmpL["value"])
        inds.append(tmpL["ind"])
        units = tmpL["units"]
        latex += tmpL["latex"].strip()

        for j in range(1, len(self.value), 2):
            op  = self.value[j]
            nxt = self.value[j + 1]

            if op == "/" and isinstance(nxt, EvalAddSubOp):
                tmpR = nxt.eval(comp = comp, code = code, no_outer_para = True)
            else:
                tmpR = nxt.eval(comp = comp, code = code)

            vals.append(tmpR["value"])
            inds.append(tmpR["ind"])
            operations.append(op)

            if op == "*":
                units = units._multiply(tmpR["units"])
                latex += r" \, " + tmpR["latex"].strip()
            elif op == "/":
                units = units._divide(tmpR["units"])
                latex = r" \frac{ " + _remove_outer_para(latex.strip()) + r" }{ " + tmpR["latex"].strip() + r" } "
            else:
                _raise_value_error("Wrong operator!?")

#        if operations[-1] != "/":
#            latex = r" \left( " + latex + r" \right) "

        if calling_from_left_arrow == False:
            ret = _broadcast_indices(vals, inds, operations, units, code, comp)
            ret["latex"] = latex
        else:
            tmp_indices_from = ",".join(inds)
            indices_to =  calling_from_left_arrow
            filter_vals, filter_from, filter_latex = _add_filters_to_einsum(tmp_indices_from, indices_to, comp._filters)
            diagonal_vals, diagonal_from, diagonal_latex = _add_diagonals_to_einsum(tmp_indices_from, indices_to, comp)
            indices_from = ",".join(inds + filter_from + diagonal_from)
            indices = indices_from + "->" + indices_to
            summed_over_indices = sorted(set("".join(inds)).difference(indices_to))
            if operations[-1] == "/":
                tsd = _prepare_soft_divide("1.0", vals[-1], list(inds[-1]), code, comp, parantheses_around = True)
                if tsd is None:
                    vals[-1] = "(1.0/(" + vals[-1] + "))"
                else:
                    vals[-1] = tsd
            einsum_1 = indices
            einsum_2 = ",\\\n".join(vals + filter_vals + diagonal_vals)
            ret = {"value": code._add_definition_from_einsum("__mult",  einsum_1, einsum_2, do_copy = False),
                   "ind": calling_from_left_arrow,
                   "units": units}

            summed_over_index = "".join(sorted(summed_over_indices)).strip()
            if len(summed_over_index) > 0:
                ret["latex"] = r" \displaystyle\sum_{ " + _nicefy_subscript(summed_over_index) + r" }"
            else:
                ret["latex"] = " "
            if len(filter_latex) > 0 or len(diagonal_latex) > 0:
                if len(summed_over_index) == 0:
                    _raise_value_error("This should not happen.  If you have nothing summing over, then there should be no conditions.")
                ret["latex"] += r"^{\substack{"
                ret["latex"] += r" \\ ".join(filter_latex + diagonal_latex)
                ret["latex"] += r"}}"
            ret["latex"] += " " + latex.strip() + r" "

        if calling_from_left_arrow != False:
            ret = _apply_diagonals_on_output_if_indices_remain(code, ret, comp)

        return ret

class EvalAddSubOp():
    def __init__(self, s, loc, tokens):
        _check_add_sub(tokens)
        self.value = tokens[0]

    def eval(self, comp, code, calling_from_left_arrow = False, no_outer_para = False):
        vals = []
        inds = []
        latex = ""
        operations = []
        tmpL = self.value[0].eval(comp = comp, code = code)
        vals.append(tmpL["value"])
        inds.append(tmpL["ind"])
        units = tmpL["units"]
        latex += tmpL["latex"].strip()
        prev_latex = tmpL["latex"].strip()

        for j in range(1, len(self.value), 2):
            op  = self.value[j]
            nxt = self.value[j+1]

            tmpR = nxt.eval(comp = comp, code = code)
            vals.append(tmpR["value"])
            inds.append(tmpR["ind"])
            operations.append(op)

            latex += " " + str(op).strip() + " " + tmpR["latex"].strip()

            if units._check_units_the_same(tmpR["units"]) == False:
                _raise_value_error("Units in your expression do not match!  You are trying to " + 
                                   {"+": "add", "-": "subtract"}[str(op).strip()] + " term with units of [" + 
                                   str(units) + "] to term with units of [" + str(tmpR["units"]) + "]." +
                                   "BREAK The first term is:  " + prev_latex.strip()  + "\n" + 
                                   "BREAK The second term is: " + tmpR["latex"].strip() + "\n")

            prev_latex = tmpR["latex"].strip()

        # THIS MIGHT BE DONE BETTER.  MAYBE NO NEED TO BROADCAST
        # HERE TO INDICES THAT YOU WILL LATER COLLAPSE IN THE EINSUM?

        ret = _broadcast_indices(vals, inds, operations, units, code, comp)

        latex = latex.strip()

        if no_outer_para == False:
            latex = r" \left( " + latex + r" \right) "

        ret["latex"] = latex

        ret = _resolve_indices_if_called_from_left_arrow(calling_from_left_arrow, ret, code, comp)
        if calling_from_left_arrow != False:
            ret = _apply_diagonals_on_output_if_indices_remain(code, ret, comp)

        return ret

class EvalArrowOp():
    def __init__(self, s, loc, tokens):
        _check_ein(tokens)
        self.value = tokens[0]

    def eval(self, comp, code, allow_storing_data = False, call_from_main_evaluate = False):
        if self.value[0].name not in ["ind_for_left_arrow", "tensor", "constant"]:
            _raise_value_error("Incorrect usage of <= operator.  On the left of <= there should be something of the form X_Y where X can be a new variable, or X can be empty string.  Y should be set of indices or empty.")

        if call_from_main_evaluate == False:
            # It is not clear how to deal with conditions.  For example, if one says m!=n but then
            # mn appear both in nested arrow and main arrow, these m and n indices could refer to different
            # things.  For example, they might correspond to matrices with different shapes, and
            # our _orig_shp thing saves a global shape for a fixed index.
            _raise_value_error("Not allowing nested assignment operators, as that might lead to ambiguity.")

        if allow_storing_data == False:
            if self.value[0].name != "ind_for_left_arrow":
                _raise_value_error("Incorrect usage of <= operator.  On the left of <= there should be _ or _ijk or similar.")

        if "_" in self.value[0].text:
            store_to = self.value[0].text.split("_")[0]
            if store_to == "":
                store_to = None
            indices_want = self.value[0].text.split("_")[1]
        else:
            store_to = self.value[0].text
            indices_want = ""

        if len(indices_want) != len(list(set(indices_want))):
            _raise_value_error("You have duplicate indices on the left of the assignment operator <=, <<=, or <+= .")

        ret = self.value[2].eval(comp = comp, code = code, calling_from_left_arrow = indices_want)

        if allow_storing_data == False:
            return ret
        else:
            return ret, store_to, self.value[1]

def _resolve_indices_if_called_from_left_arrow(calling_from_left_arrow, ret, code, comp):
    if calling_from_left_arrow == False:
        return ret
    elif ret["ind"] == calling_from_left_arrow:
        _check_that_return_indices_compatible_with_filter(calling_from_left_arrow, comp._filters)
        return ret
    else:
        summed_over_indices = sorted(set("".join(ret["ind"])).difference(calling_from_left_arrow))
        how_many_indices_summing_over = len(summed_over_indices)

        latex_start = ret["latex"]
        tmp_indices_from = ret["ind"]
        indices_to = calling_from_left_arrow
        filter_vals, filter_from, filter_latex = _add_filters_to_einsum(tmp_indices_from, indices_to, comp._filters)
        diagonal_vals, diagonal_from, diagonal_latex = _add_diagonals_to_einsum(tmp_indices_from, indices_to, comp)
        if how_many_indices_summing_over == 0 and (len(filter_vals) > 0 or len(diagonal_vals) > 0):
            _raise_value_error("Should not happen.  Can't apply conditions if there is no summation.")
        if how_many_indices_summing_over > 0:
            summed_over_index = "".join(sorted(summed_over_indices)).strip()
            if len(summed_over_index) > 0:
                latex = r" \displaystyle\sum_{ " + _nicefy_subscript(summed_over_index) + r" }"
            else:
                latex = r" "
            if len(filter_latex)>0 or len(diagonal_vals) > 0:
                if len(summed_over_index) == 0:
                    _raise_value_error("This should not happen.  If you have nothing summing over, then there should be no conditions.")
                latex += r"^{\substack{"
                latex += r" \\ ".join(filter_latex + diagonal_latex)
                latex += r"}}"
            latex += r" \left( " + latex_start + r" \right) "
        else:
            latex = latex_start
        indices_from = ",".join([ret["ind"]] + filter_from + diagonal_from)
        indices = indices_from + "->" + indices_to
        einsum_1 = indices
        einsum_2 = ",\\\n".join([ret["value"]] + filter_vals + diagonal_vals)
        if len([ret["value"]] + filter_vals + diagonal_vals) > 1:
            do_copy = False
        else:
            do_copy = True
        ret = {"value": code._add_definition_from_einsum("__reso",  einsum_1 , einsum_2, do_copy = do_copy),
               "ind": calling_from_left_arrow,
               "units": ret["units"],
               "latex": latex}

        return ret

def _apply_diagonals_on_output_if_indices_remain(code, ret, comp):
    all_val = []
    for idia, dia in enumerate(comp._diagonals):
        howmany = 0
        for i in dia:
            if i in ret["ind"]:
                howmany += 1
        if howmany == 0 or howmany == 1:
            continue

        where_dia_0 = ret["ind"].index(dia[0])
        where_dia_1 = ret["ind"].index(dia[1])

        if where_dia_0 < where_dia_1:
            val = "(1.0 - np.eye(_orig_shp[\"" + dia[0] + "\"], _orig_shp[\"" + dia[1] + "\"]))"
        else:
            val = "(1.0 - np.eye(_orig_shp[\"" + dia[1] + "\"], _orig_shp[\"" + dia[0] + "\"]))"
        # no need here for partial filtering as there is no < or > condition applied to the output indices!
#        val = _perform_partial_filtering(val, dia, comp.partially_filtered_indices)
#        val = "(" + val + ")"

        tmp = []
        for j in ret["ind"]:
            if j in dia:
                tmp.append(":")
            else:
                tmp.append("None")
        val = val + "[" + ", ".join(tmp) + "]"

        all_val.append(val)

    if len(all_val) == 0:
        return ret

    all_val = " * ".join(all_val)

    variable = code.give_me_unique_variable_name("__removediag")
    code._add_definition(variable, "(" + ret["value"] + ") * (" + all_val + ")")

    ret["value"] = variable
    return ret

def _broadcast_indices(vals, inds, operations, units, code, comp):
    if (len(vals) != len(inds)) or (len(vals) != len(operations) + 1):
        _raise_value_error("Inconsistent input!")

    # check if we ever called broadcast with these same parameters
    bef = code.check_if_did_this_broadcast_before(vals, inds, operations)
    if bef is not None:
        return bef

    # broadcast function will create new variable in the python code for exec
    # this will be the name of that variable
    result_val_name = code.give_me_unique_variable_name("__brod")

    l_val, l_ind = vals[0], inds[0]

    #obtain a sorted set representing the indices that the end-result should have
    ret_ind = "".join(sorted(set("".join(inds))))

    # construct variable where we will store result
    #
    # place None at missing indices
    result_axes = ["None" for i in ret_ind]
    # leave other indices as they are
    for i in l_ind:
        result_axes[ret_ind.find(i)] = ":"

    # now make sure that indices are in the right order
    if l_ind != "":
        jss_l_ind = "".join(sorted(set(l_ind)))
        if l_ind != jss_l_ind:
            einsum_1 = l_ind + "->" + jss_l_ind
            einsum_2 = l_val
            code._add_definition(result_val_name, "np.copy(opteinsum(\"" + einsum_1 + "\", " + einsum_2 + "))")
        else:
            code._add_definition(result_val_name, "np.copy(" + l_val + ")")
    else:
        code._add_definition(result_val_name, "np.array(" + l_val + ")")

#    if len(result_axes) > 0:
#        code._add_raw(result_val_name + " = " + result_val_name + "[" + ",".join(result_axes) + "]")

    for j in range(1, len(vals)):
        r_val, r_ind = vals[j], inds[j]

        if r_ind == "":
            if operations[j - 1] == "+":
                code._add_raw(result_val_name + " = " + result_val_name + " + (" + r_val + ")")
            elif operations[j - 1] == "-":
                code._add_raw(result_val_name + " = " + result_val_name + " - (" + r_val + ")")
            elif operations[j - 1] == "*":
                code._add_raw(result_val_name + " = " + result_val_name + " * (" + r_val + ")")
            elif operations[j - 1] == "/":
                code._add_raw(result_val_name + " = " + result_val_name + " / (" + r_val + ")")
            else:
                _raise_value_error("Unknown operation!")
        else:
            r_axes = ["None" for i in ret_ind]
            for i in r_ind:
                r_axes[ret_ind.find(i)] = ":"
            if len(r_axes) > 0:
                r_axes_str = "[" + ",".join(r_axes) + "]"
            else:
                r_axes_str = ""
            r_ind_ssj = "".join(sorted(set(r_ind)))
            if r_ind == r_ind_ssj:
                rot_r_val_code = "(" + r_val + ")"
            else:
                rot_r_val_code = "np.copy(opteinsum(\"" + r_ind + "->" + r_ind_ssj + "\", " + r_val + "))"
            if operations[j - 1] in ["+", "-", "*", "/"]:
                if len(result_axes) > 0:
                    result_axes_str = "[" + ",".join(result_axes) + "]"
                else:
                    result_axes_str = ""
                if operations[j - 1] != "/":
                    code._add_raw(result_val_name + " = " + 
                                  result_val_name + result_axes_str + " " + 
                                  operations[j - 1] + " " + 
                                  rot_r_val_code + r_axes_str)
                else:
                    r_ind_ssj_expanded = []
                    pos = 0
                    for ra in r_axes:
                        if ra == ":":
                            r_ind_ssj_expanded.append(r_ind_ssj[pos])
                            pos += 1
                        elif ra == "None":
                            r_ind_ssj_expanded.append(None)
                        else:
                            _raise_value_error()
                    if pos != len(r_ind_ssj):
                        _raise_value_error("Missing something.  Should not happen.")

                    tsd = _prepare_soft_divide(result_val_name + result_axes_str, rot_r_val_code + r_axes_str, list(r_ind_ssj_expanded), \
                                               code, comp, parantheses_around = False)
                    if tsd is None:
                        code._add_raw(result_val_name + " = " + 
                                      result_val_name + result_axes_str + " " + 
                                      operations[j - 1] + " " + 
                                      rot_r_val_code + r_axes_str)
                    else:
                        code._add_raw(result_val_name + " = " + tsd)
                # here the problem is that it might happen that result_axes_str and r_axes_str have None at the same place
                # in that case you should squeeze out those indices
                to_squeeze = []
                for k in range(len(result_axes)):
                    if result_axes[k] == "None" and r_axes[k] == "None":
                        to_squeeze.append(k)
                if len(to_squeeze) > 0:
                    code._add_raw(result_val_name + " = " +  result_val_name + ".squeeze(axis = " + str(tuple(to_squeeze)) + ")")
            else:
                _raise_value_error("Unknown operation!")
            for i in r_ind:
                result_axes[ret_ind.find(i)] = ":"

    if "None" in result_axes:
        _raise_value_error("Hm, this shouldn't happen.  All indices must be sliced eventually...")

    ret = {"value": result_val_name, "ind": ret_ind, "units": units}

    # store parameters you sent to this function
    # in case we ever call this function again with the same parameters
    code.store_broadcast_info_for_lookup(result_val_name, {"input_vals": vals, "input_inds": inds, "input_operations": operations, "ret_ind": ret_ind, "ret_units": units})

    return ret

def _nicefy_core(txt):
    ll = txt.strip()

    if ll.count("~") > 1:
        _raise_value_error("Wrong format of tensor/constant: " + ll)
    if ll.startswith("~") == True:
        _raise_value_error("Name: " + ll + " is invalid.  It can't start with ~.")
    if ll.endswith("~") == True:
        _raise_value_error("Name: " + ll + " is invalid.  It can't end with ~.")

    if ll.count("~") == 0:
        core = ll
        superscript = ""
    else:
        sp = ll.split("~")
        core = sp[0].strip()
        superscript = sp[1].strip()

    if core in "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi pi rho sigma tau upsilon phi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega".split(" "):
        core = "\\" + core
    else:
        if len(core) > 1:
            core = r"\mathrm{" + core + r"}"

    if superscript == "":
        return core
    else:
        return core + r"^{\mathrm{ " + superscript +" }}"

def _nicefy_subscript(txt):
    ll = txt.strip()
    return "".join(ll)


class BfsConstVar():
    def __init__(self, s, loc, tokens):
        _check_const(tokens)
        self.value = tokens[0]
        self.name = tokens.get_name()
    def eval(self, data, parent):
        if self.name in ["tensor"]:
            core, ind = self.value.split("_")
            core = core.strip()
            ind = ind.strip()
            data.append({"key": core, "indices": ind})
            return "__object_" + core.replace("~", "") + "[" + ",".join(ind) + "]"
        elif self.name in ["constant"]:
            core = self.value.strip()
            data.append({"key": core, "indices": ""})
            return "__object_" + core.replace("~", "")
        elif self.name in ["imaginary one"]:
            return "(1.0j)"
        return self.value.strip()

class BfsFuncOp():
    def __init__(self, s, loc, tokens):
        _check_func(tokens)
        self.func = tokens[0][0]
        self.value = tokens[0][1]
    def eval(self, data, parent):
        if self.func == "Real":
            return "(" + self.value.eval(data, parent = self) + ").real"
        elif self.func == "Imag":
            return "(" + self.value.eval(data, parent = self) + ").imag"
        else:
            _raise_value_error("Unknown function.")

class BfsConjugOp():
    def __init__(self, s, loc, tokens):
        _check_conjug(tokens)
        self.value = tokens[0][1]
    def eval(self, data, parent):
        return "(" + self.value.eval(data, parent = self) + ").conjugate()"

class BfsSignOp():
    def __init__(self, s, loc, tokens):
        _check_sign(tokens)
        self.sign, self.value = tokens[0]
    def eval(self, data, parent):
        return self.sign + "(" + self.value.eval(data, parent = self) + ")"

class BfsDOneOp():
    def __init__(self, s, loc, tokens):
        _check_d_one(tokens)
        self.sign, self.value = tokens[0]
    def eval(self, data, parent):
        return "(1.0/(" + self.value.eval(data, parent = self) + "))"

class BfsPowerOp():
    def __init__(self, s, loc, tokens):
        _check_power(tokens)
        self.value = tokens[0]
    def eval(self, data, parent):
        term = self.value[0].eval(data, parent = self)
        exponent = self.value[2].eval(data, parent = self)
        if exponent in ["1", "1.0"]:
            return "(" + term + ")"
        elif exponent in ["2", "2.0"]:
            return "((" + term + ")*(" + term + "))"
        elif exponent in ["-1", "-1.0"]:
            return "(1.0/(" + term + "))"
        else:
            return "(" + term + ")**(" + exponent + ")"

class BfsMultDivOp():
    def __init__(self, s, loc, tokens):
        _check_mult_div(tokens)
        self.value = tokens[0]
    def eval(self, data, parent):
        ret = self.value[0].eval(data, parent = self)
        for j in range(1, len(self.value), 2):
            op  = self.value[j]
            nxt = self.value[j + 1].eval(data, parent = self)
            ret = ret + " " + op + " " + nxt
        if len(self.value) > 1:
            ret = "(" + ret + ")"
        return ret

class BfsAddSubOp():
    def __init__(self, s, loc, tokens):
        _check_add_sub(tokens)
        self.value = tokens[0]
    def eval(self, data, parent):
        ret = self.value[0].eval(data, parent = self)
        for j in range(1, len(self.value), 2):
            op  = self.value[j]
            nxt = self.value[j+1].eval(data, parent = self)
            ret = ret + " " + op + " " + nxt
        if len(self.value) > 1:
            ret = "(" + ret + ")"
        return ret

class BfsArrowOp():
    def __init__(self, s, loc, tokens):
        _check_ein(tokens)
        self.value = tokens[0]

    def eval(self, data, parent, allow_storing_data = False, call_from_main_evaluate = False):
        if self.value[0].name not in ["ind_for_left_arrow", "tensor", "constant"]:
            _raise_value_error("Incorrect usage of <= operator.  On the left of <= there should be something of the form X_Y where X can be a new variable, or X can be empty string.  Y should be set of indices or empty.")

        if call_from_main_evaluate == False:
            _raise_value_error("Not allowing nested assignment operators, as that might lead to ambiguity.")

        if allow_storing_data == False:
            if self.value[0].name != "ind_for_left_arrow":
                _raise_value_error("Incorrect usage of <= operator.  On the left of <= there should be _ or _ijk or similar.")

        if "_" in self.value[0].value:
            store_to = self.value[0].value.split("_")[0]
            if store_to == "":
                store_to = None
            indices_want = self.value[0].value.split("_")[1]
        else:
            store_to = self.value[0].value
            indices_want = ""

        if len(indices_want) != len(list(set(indices_want))):
            _raise_value_error("You have duplicate indices on the left of the assignment operator <=, <<=, or <+= .")

        ret = {}
        ret["value"] = self.value[2].eval(data, parent = self)
        ret["ind"] = indices_want

        return ret, store_to, self.value[1]


class UnitConstVar():
    def __init__(self, s, loc, tokens):
        _check_const(tokens)
        self.text = tokens[0]
        self.name = tokens.get_name()

    def eval(self):
        if self.name in ["float", "integer"]:
            ret = float(self.text)
        elif self.name == "basic_unit":
            if self.text.lower() == "ev":
                ret = Units(eV = 1)
            elif self.text.lower() == "ang":
                ret = Units(Ang = 1)
            elif self.text.lower() == "mub":
                ret = Units(muB = 1)
            else:
                _raise_value_error("Unknown unit: " + self.text + ".")
        else:
            _raise_value_error("Unknown name!")
        return ret

class UnitPowerOp():
    def __init__(self, s, loc, tokens):
        _check_power(tokens)
        self.value = tokens[0]

    def eval(self):
        trmL = self.value[0].eval()
        trmR = self.value[2].eval()

        if isinstance(trmL, Units) == False:
            _raise_value_error("Must raise one of basic units to a power.")
        if isinstance(trmR, float) == False:
            _raise_value_error("Exponents must be simply a number.")

        return trmL._exponent_float(trmR)

class UnitMultDivOp():
    def __init__(self, s, loc, tokens):
        _check_mult_div(tokens)
        self.value = tokens[0]

    def eval(self):
        ret = self.value[0].eval()
        if isinstance(ret, Units) == False:
            _raise_value_error("Needs to be one of basic units.")

        for j in range(1, len(self.value), 2):
            op  = self.value[j]
            nxt = self.value[j + 1]
            tmpR = nxt.eval()
            if isinstance(tmpR, Units) == False:
                _raise_value_error("Needs to be one of basic units.")

            if op == "*":
                ret = ret._multiply(tmpR)
            elif op == "/":
                ret = ret._divide(tmpR)
            else:
                _raise_value_error("Wrong operator!?")
        return ret

def _parse_value_and_units(raw_string):
    ret = {}

    string = raw_string.strip()

    if " " not in string:
        ret["value"] = float(string)
        ret["units"] = Units(eV = 0, Ang = 0, muB = 0)
        return ret

    if string.count("*") == 0:
        _raise_value_error("""If you are specifying this value as string, then the string must
        be of the form \"3.0 * eV/Ang\" or similar.  In other words, you _must_ include
        multiplication sign between number and units.""")
    sp = string.index("*")
    ret["value"] = float(string[:sp])
    units_str = string[sp + 1:]

    # now parse units out of the string
    operand = \
        pp.Combine("-" + pp.Word(pp.nums, min = 1) + "." + pp.Word(pp.nums, min = 1)).set_results_name("float") | \
        pp.Combine(pp.Word(pp.nums, min = 1) + "." + pp.Word(pp.nums, min = 1)).set_results_name("float") | \
        pp.Combine("-" + pp.Word(pp.nums, min = 1) + ".").set_results_name("float") | \
        pp.Combine(pp.Word(pp.nums, min = 1) + ".").set_results_name("float") | \
	pp.Word("-" + pp.nums, min = 1).set_results_name("integer") | \
	pp.Word(pp.nums, min = 1).set_results_name("integer") | \
        pp.Word(pp.alphas, min = 1).set_results_name("basic_unit")
    operand.set_parse_action(UnitConstVar)
    parser = pp.infix_notation(
	operand,
        [
            (         "^"          , 2, pp.opAssoc.LEFT , UnitPowerOp   ), # power raising
            (pp.oneOf("* /")       , 2, pp.opAssoc.LEFT , UnitMultDivOp ), # multiplication and division
        ],
        )
    par = _my_parse_string(parser, units_str, parse_all = True)
    if len(par) != 1:
        _raise_value_error("PyParsing returned something not expected?!")
    par = par[0]
    ret["units"] = par.eval()

    return ret

class FundConst():
    def __init__(self, cnst):
        self._data = {"hbar": 0.0, "e": 0.0, "epszero": 0.0, "c": 0.0, "me": 0.0}
        self._data[cnst] = 1.0
    def _exponent_float(self, ex):
        for k in self._data.keys():
            self._data[k] *= ex
        return self
    def _multiply_with(self, obj):
        for k in self._data.keys():
            self._data[k] = self._data[k] + obj._data[k]
        return self
    def _divide_by(self, obj):
        for k in self._data.keys():
            self._data[k] = self._data[k] - obj._data[k]
        return self
    def _to_latex(self):
        keys = list(self._data.keys())
        keys.sort()
        numerator = []
        denominator = []
        for k in keys:
            if np.abs(self._data[k]) < 1.0E-8:
                continue
            tmp = _fund_const_in_latex(k)
            if self._data[k] > 0.0:
                sign =  1
            else:
                sign = -1
            expon = str(Fraction(sign * self._data[k]).limit_denominator(1000))
            if expon == "1":
                expon = ""
            else:
                expon = r"^{" + expon.strip() + r"}"

            if sign == 1:
                numerator.append(tmp + expon)
            else:
                denominator.append(tmp + expon)
        if len(numerator) == 0 and len(denominator) == 0:
            return ""
        if len(numerator) != 0 and len(denominator) == 0:
            return " ".join(numerator)
        if len(numerator) == 0 and len(denominator) != 0:
            return r"\frac{1}{" + " ".join(denominator) + "}"
        if len(numerator) != 0 and len(denominator) != 0:
            return r"\frac{" + " ".join(numerator) + "}{" + " ".join(denominator) + "}"

    def _numerical_value(self):
        ret = 1.0
        for k in self._data.keys():
            ret *= np.power(_fund_const_to_numerical(k), self._data[k])
        return ret

def _fund_const_in_latex(key):
    if key == "hbar"   : return r"\hbar"
    if key == "e"      : return r"e"
    if key == "epszero": return r"\epsilon_0"
    if key == "c"      : return r"c"
    if key == "me"     : return r"m_{\rm e}"
    _raise_value_error("Unknown fundamental constant.")

def _fund_const_to_numerical(key):
    if key == "hbar"   : return hbar_SI
    if key == "e"      : return electron_charge_SI
    if key == "epszero": return epsilon_zero_SI
    if key == "c"      : return speed_of_light_SI
    if key == "me"     : return electron_mass_SI
    _raise_value_error("Unknown fundamental constant.")

class PrefactorSIConstVar():
    def __init__(self, s, loc, tokens):
        _check_const(tokens)
        self.text = tokens[0]
        self.name = tokens.get_name()

    def eval(self):
        if self.name in ["float", "integer"]:
            ret = float(self.text)
        elif self.name == "fundamental_constant":
            if self.text.lower() in ["hbar", "e", "epszero", "c", "me"]:
                ret = FundConst(self.text.lower())
            else:
                _raise_value_error("Unknown fundamental constant: " + self.text + ".")
        else:
            _raise_value_error("Unknown name!")
        return ret

class PrefactorSIPowerOp():
    def __init__(self, s, loc, tokens):
        _check_power(tokens)
        self.value = tokens[0]

    def eval(self):
        trmL = self.value[0].eval()
        trmR = self.value[2].eval()
        if isinstance(trmL, FundConst) == False:
            _raise_value_error("Only allowed to raise one of the fundamental constants to a power. " + str(trmL))
        if isinstance(trmR, float) == False:
            _raise_value_error("Exponents must be simply a number. " + str(trmR))
        return trmL._exponent_float(trmR)

class PrefactorSIMultDivOp():
    def __init__(self, s, loc, tokens):
        _check_mult_div(tokens)
        self.value = tokens[0]

    def eval(self):
        ret = self.value[0].eval()
        if isinstance(ret, FundConst) == False:
            _raise_value_error("Only allowed to multiply/divide fundamental constants. " + str(ret))
        for j in range(1, len(self.value), 2):
            op  = self.value[j]
            nxt = self.value[j + 1]
            tmpR = nxt.eval()
            if isinstance(tmpR, FundConst) == False:
                _raise_value_error("Only allowed to multiply/divide fundamental constants. " + str(tmpR))
            if op == "*":
                ret = ret._multiply_with(tmpR)
            elif op == "/":
                ret = ret._divide_by(tmpR)
            else:
                _raise_value_error("Wrong operator!?")
        return ret

def _parse_prefactor_SI_units_fundamental_constants(raw_string):
    string = raw_string.strip()
    operand = \
        pp.Combine("-" + pp.Word(pp.nums, min = 1) + "." + pp.Word(pp.nums, min = 1)).set_results_name("float") | \
        pp.Combine(pp.Word(pp.nums, min = 1) + "." + pp.Word(pp.nums, min = 1)).set_results_name("float") | \
        pp.Combine("-" + pp.Word(pp.nums, min = 1) + ".").set_results_name("float") | \
        pp.Combine(pp.Word(pp.nums, min = 1) + ".").set_results_name("float") | \
	pp.Word("-" + pp.nums, min = 1).set_results_name("integer") | \
	pp.Word(pp.nums, min = 1).set_results_name("integer") | \
        pp.Word(pp.alphas, min = 1).set_results_name("fundamental_constant")
    operand.set_parse_action(PrefactorSIConstVar)
    parser = pp.infix_notation(
	operand,
        [
            (         "^"          , 2, pp.opAssoc.LEFT , PrefactorSIPowerOp   ), # power raising
            (pp.oneOf("* /")       , 2, pp.opAssoc.LEFT , PrefactorSIMultDivOp ), # multiplication and division
        ],
        )
    par = _my_parse_string(parser, string, parse_all = True)
    if len(par) != 1:
        _raise_value_error("PyParsing returned something not expected?!")
    par = par[0]
    return par.eval()

def _are_hashes_similar_relative(hash_0, hash_1, tol = 1.0E-5):
    if np.max(np.abs(hash_0 - hash_1))/np.max(np.abs(hash_1)) < tol:
        return True
    else:
        return False

def _are_hashes_similar_absolute(hash_0, hash_1, tol = 1.0E-5):
    if np.max(np.abs(hash_0 - hash_1)) < tol:
        return True
    else:
        return False

def _get_kpoint_label_info(cell, name, red):
    if red.shape[0] != 1:
        _raise_value_error("Crystal structure not supported")

    p = cell[0, 2]
    if np.abs(p) < 1.0E-9:
        _raise_value_error("Crystal structure not supported")

    tmp_bcc = np.array([[ p, p, p], [-p, p, p], [-p,-p, p]])
    tmp_fcc = np.array([[-p, 0, p], [ 0, p, p], [-p, p, 0]])

    if np.max(np.abs(cell - tmp_bcc)) < 1.0E-9:
        struc_kind = "bcc"
    elif np.max(np.abs(cell - tmp_fcc)) < 1.0E-9:
        struc_kind = "fcc"
    else:
        _raise_value_error("Crystal structure not supported")

    # the nomenclature is from bilbao crystallography server
    if struc_kind == "bcc":
        ret = {
            "GM": [[0.   , 0.   , 0.   ], r"$\Gamma$"],
            "H" : [[0.   , 1./2., 0.   ]],
            "N" : [[1./4., 1./4., 0.   ]],
            "P" : [[1./4., 1./4., 1./4.]],
        }
        recip_alt = cell[0, 2] * _real_to_recip_no2pi([cell[0], cell[1], cell[2]])
    elif struc_kind == "fcc":
        ret = {
            "GM": [[0.   , 0.   , 0.   ], r"$\Gamma$"],
            "X" : [[0.   , 1./2., 0.   ]],
            "M" : [[1./2., 1./2., 0.   ]],
            "U" : [[1./8., 1./2., 1./8.]],
            "K" : [[3./8., 3./8., 0.   ]],
            "L" : [[1./4., 1./4., 1./4.]],
            "W" : [[1./4., 1./2., 0.   ]],
        }
        recip_alt = cell[0, 2] * _real_to_recip_no2pi([cell[0], cell[1], cell[2]])
    else:
        _raise_value_error("Crystal structure not supported")

    # convert to conventions used in QE
    for k in ret.keys():
        ret[k][0] = _cart_to_red(recip_alt[0], recip_alt[1], recip_alt[2], ret[k][0])

    return ret

def _cart_to_red(a1, a2, a3, cart):
    cnv = np.array([a1, a2, a3])
    cnv = cnv.T
    cnv = np.linalg.inv(cnv)
    return np.dot(cnv, cart)

def _red_to_cart(a1, a2, a3, red):
    return np.array(red[0]*a1 + red[1]*a2 + red[2]*a3)

def _real_to_recip_no2pi(real):
    ret = []
    ret.append(np.cross(real[1], real[2]))
    ret.append(np.cross(real[2], real[0]))
    ret.append(np.cross(real[0], real[1]))
    ret = np.array(ret) / np.dot(real[0], np.cross(real[1], real[2]))
    return ret

def _get_operand():
    return \
        pp.Combine(pp.Word(pp.nums, min = 1) + "." + pp.Word(pp.nums, min = 1) + "j").set_results_name("imaginary float") | \
        pp.Combine(pp.Word(pp.nums, min = 1) + "." + "j").set_results_name("imaginary float") | \
        pp.Combine(pp.Word(pp.nums, min = 1) + "j").set_results_name("imaginary integer") | \
        pp.Literal("j").set_results_name("imaginary one") | \
        pp.Combine(pp.Word(pp.nums, min = 1) + "." + pp.Word(pp.nums, min = 1)).set_results_name("float") | \
        pp.Combine(pp.Word(pp.nums, min = 1) + ".").set_results_name("float") | \
        pp.Word(pp.nums, min = 1).set_results_name("integer") | \
        pp.Combine("_" + pp.Word(pp.alphas, min = 1)).set_results_name("ind_for_left_arrow") | \
        pp.Literal("_").set_results_name("ind_for_left_arrow") | \
        pp.Combine(pp.Word(pp.alphas, min = 1) + "~" + pp.Word(pp.alphas, min = 1) + "_" + pp.Word(pp.alphas, min = 1)).set_results_name("tensor") | \
        pp.Combine(pp.Word(pp.alphas, min = 1) + "~" + pp.Word(pp.alphas, min = 1)).set_results_name("constant") | \
        pp.Combine(pp.Word(pp.alphas, min = 1) + "_" + pp.Word(pp.alphas, min = 1)).set_results_name("tensor") | \
        pp.Word(pp.alphas, min = 1).set_results_name("constant")

def _my_parse_string(obj, *args, **kwargs):
    try:
        return obj.parse_string(*args, **kwargs)
    except pp.ParseException as pe:
        traceback.print_stack()
        print()
        print("\n" + 
              _format_one_block_simple_indent("Problem with parsing of this string (note, problem might occur after symbol ^ below):\n\n" + pe.explain(depth = 0),
                                             indent = 0,
                                             string_indent = "+++  ",
                                             start_and_end = True,
                                             special_string = "+") + \
              "\n")
        exit()

def _raise_value_error(s):
    raise ValueError("\n\n" + "*"*81 + "\n" + 
                     _format_one_block(s,
                                      indent = None,
                                      width = 80,
                                      initial_indent    = "***  ",
                                      subsequent_indent = "***  ") + \
                     "\n" + "*"*81 + "\n")

def _print_without_stopping(s):
    print("\n\n" + "&"*81 + "\n" + 
          _format_one_block(s,
                            indent = None,
                            width = 80,
                            initial_indent    = "&&&  ",
                            subsequent_indent = "&&&  ") + \
          "\n" + "&"*81 + "\n")

def _nice_exec(code_txt, code_dic):
    try:
        exec_time = time.perf_counter()
        exec(code_txt, code_dic)
        exec_time = time.perf_counter() - exec_time
        return exec_time
    except Exception as err:
        traceback.print_stack()
        print()
        print(repr(err) + "  exception occured while trying to execute this code:")
        #
        if err.__class__.__name__ == "SyntaxError":
            line_number = err.lineno
        else:
            cl, exc, tb = sys.exc_info()
            line_number = traceback.extract_tb(tb)[-1][1]
        #
        tmp = code_txt.split("\n")
        code_use_txt = ""
        for i in range(len(tmp)):
            if i != line_number - 1:
                code_use_txt += "             " + tmp[i] + "\n"
            else:
                code_use_txt += "==PROBLEM==> " + tmp[i] + "\n"
        out = _format_one_block_simple_indent(code_use_txt, indent = 0, start_and_end = True, special_string = "$")
        print(out)
        raise err

class _LatexExpression():
    def __init__(self, core, ind, rhs, prefactor = ""):
        self._core = core.strip()
        self._ind = ind.strip()
        self._prefactor = prefactor.strip()
        self._rhs = rhs.strip()

    def get_string(self, inside_align = False):
        latex_source = ""
        if self._core != "":
            latex_source += _nicefy_core(self._core)
            if self._ind != "":
                latex_source += r"_{" + _nicefy_subscript(self._ind) + r"}"
        if self._core != "" and self._rhs != "":
            if inside_align == False:
                latex_source += r" \Leftarrow "
            else:
                latex_source += r" & \Leftarrow "
        if self._rhs != "":
            if self._prefactor != "":
                latex_source += r"\left[ " + self._prefactor + r" \right] \times \left[ " + self._rhs + r" \right]"
            else:
                latex_source += self._rhs
        return latex_source

    def __str__(self):
        return self.get_string()

def _format_one_block(raw_text, indent = 4, width = 80, initial_indent = None, subsequent_indent = None):
    text = raw_text.strip()

    if "BREAK" in text:
        sp = text.split("BREAK")
        ret = []
        for s in sp:
            ret.append(_format_one_block(s, indent, width, initial_indent, subsequent_indent))
        return ("\n" + subsequent_indent + "\n").join(ret)

    text = text.replace("\n", "")
    while "  " in text:
        text = text.replace("  ", " ")

#    while "  " in text:
#        text = text.replace("  ", " ")
#    while "\n " in text:
#        text = text.replace("\n ", " ")
#    while " \n" in text:
#        text = text.replace(" \n", " ")

    if initial_indent is None:
        initial_indent = " "*indent
    if subsequent_indent is None:
        subsequent_indent = " "*indent

    wrapper = textwrap.TextWrapper(width = width, # this includes indents
                                   expand_tabs = True,
                                   tabsize = 4,
                                   replace_whitespace = True,
                                   drop_whitespace = True,
                                   initial_indent = initial_indent,
                                   subsequent_indent = subsequent_indent,
                                   break_long_words = True,
                                   break_on_hyphens = True)

    return wrapper.fill(text = text)

def _format_one_block_simple_indent(raw_text, indent = 4, string_indent = "", start_and_end = True, max_line = None, dont_indent_first = False, special_string = "#"):
    max_len = 1
    ret = ""
    sp = raw_text.split("\n")
    for i, s in enumerate(sp):
        if max_line is not None:
            if i >= max_line:
                one = " " * indent + string_indent + "... cutting long output ... set parameter \"full\" to True to get complete output." + "\n"
                if len(one) > max_len:
                    max_len = len(one)
                ret += one
                break
        if dont_indent_first == False:
            one = " "*indent + string_indent + s + "\n"
        else:
            if i == 0:
                one = string_indent + s + "\n"
            else:
                one = " "*indent + string_indent + s + "\n"
        if len(one) > max_len:
            max_len = len(one)
        ret += one

    if start_and_end == True:
        tmp = special_string*(max_len + 2) + "\n"
    else:
        tmp = ""

    ret = tmp + ret.rstrip() + "\n" + tmp

    return ret

def _make_rst_title(title):
    out = ""
    out += "="*(len(title.strip()) + 2)
    out += "\n"
    out += " " + title.strip()
    out += "\n"
    out += "="*(len(title.strip()) + 2)
    out += "\n"
    return out

def _make_rst_subtitle(subtitle):
    out = ""
    out += " " + subtitle.strip()
    out += "\n"
    out += "-"*(len(subtitle.strip()) + 2)
    out += "\n"
    return out

def _make_rst_field(title):
    out = ""
    out += ":"
    out += title.strip()
    out += ":"
    out += "\n"
    return out

def _find_1_to_1_map_from_left_to_right(left, right):
    if len(left) != len(right):
        return None
    given_left_return_right = {}
    for i in range(len(left)):
        l = left[i]
        r = right[i]
        if l in given_left_return_right.keys():
            if given_left_return_right[l] != r:
                return None
        else:
            given_left_return_right[l] = r
    return given_left_return_right

def _convert_left_to_right(left, given_left_return_right):
    ret = ""
    for l in left:
        ret += given_left_return_right[l]
    return ret


class _InterfaceToWberri(wberri.System_w90):
    # This is a class that derives from Wannier Berri's System_w90 class.
    # This class is only used as an interface to Wannier Berri, and this
    # class is therefore not supposed to be used directly by the user.
    def __init__(self, syst_raw = None):
        if syst_raw is not None:
            self.mp_grid = deepcopy(syst_raw.mp_grid)
            self.num_wann = deepcopy(syst_raw.num_wann)
            self.periodic = deepcopy(syst_raw.periodic)
            self.real_lattice = deepcopy(syst_raw.real_lattice)
            self.recip_lattice = deepcopy(syst_raw.recip_lattice)
            self.use_wcc_phase = deepcopy(syst_raw.use_wcc_phase)
            self.iRvec = deepcopy(syst_raw.iRvec)
            self._XX_R = deepcopy(syst_raw._XX_R)
            self.__add_empty_symmetry()

    def fill_in_from_dictionary(self, ff):
        self.mp_grid = ff["mp_grid"]
        self.num_wann = ff["num_wann"]
        self.periodic = ff["periodic"]
        self.real_lattice = ff["real_lattice"]
        self.recip_lattice = ff["recip_lattice"]
        self.use_wcc_phase = ff["use_wcc_phase"]
        self.iRvec = ff["iRvec"]
        self.__add_empty_symmetry()

        do_regular = False
        if "use_reduced_sym" in ff.keys():
            if ff["use_reduced_sym"] == True:
                self._XX_R = _from_reduced_dic_to_XX(ff)
            else:
                do_regular = True
        else:
            do_regular = True

        if do_regular == True:
            self._XX_R = {}
            for k in ff.keys():
                if len(k) > 7:
                    if k[:7] == "_XX_R__":
                        self._XX_R[k[7:]] = _few_to_many_bits(ff[k])

    def __add_empty_symmetry(self):
        # Symmetry is not stored in a numpy array
        # Therefore I need to reconstruct symmetry from scratch
        # as I don't want to store this object.
        self.symgroup = wberri.symmetry.Group([],
                                              recip_lattice = self.recip_lattice,
                                              real_lattice = self.real_lattice)

def _replace_with_reduced_data(fn_in, fn_out, reduced_XX, reduced_XX_common):
    g = open(fn_in, "rb")
    gg = np.load(g, allow_pickle = False)
    f = open(fn_out, "wb")

    # get stuff that is already stored, except for full matrices
    data = {}
    for k in gg.keys():
        if len(k) > 7:
            if k[:7] == "_XX_R__":
                continue
        data[k] = gg[k]

    # add stuff for symmetry reduction
    data["use_reduced_sym"] = True
    for X in reduced_XX.keys():
        for k in reduced_XX[X].keys():
            data["reduced_sym__" + X.strip() + "__" + k.strip()] = _many_to_few_bits(reduced_XX[X][k])
    for k in reduced_XX_common.keys():
        data["reduced_sym__common__" + k.strip()] = _many_to_few_bits(reduced_XX_common[k])

    np.savez_compressed(f, **data)
    f.close()
    g.close()

def _from_reduced_dic_to_XX(ff):
    all_X = []
    for k in ff.keys():
        if len(k) > 13:
            if k[:13] == "reduced_sym__":
                sp = k.split("__")
                if len(sp) != 3:
                    print("problem")
                    exit()
                if sp[1] == "common":
                    continue
                all_X.append(sp[1])
    all_X = list(sorted(set(all_X)))

    s_rotc = ff["reduced_sym__common__s_rotc"]
    s_inv = ff["reduced_sym__common__s_inv"]
    s_tr = ff["reduced_sym__common__s_tr"]
    r_irr = ff["reduced_sym__common__r_irr"]
    s_max_denom = ff["reduced_sym__common__s_max_denom"]
    s_orb = _squarerootform_to_complex(ff["reduced_sym__common__s_orb"], s_max_denom)
    r_rel = _oned_to_llist(ff["reduced_sym__common__r_rel"])
    r_rel_oper = _oned_to_llist(ff["reduced_sym__common__r_rel_oper"])
    r_star = _oned_to_llist(ff["reduced_sym__common__r_star"])

    _XX_R = {}
    for X in all_X:

        mat_in_eigen_space = _oned_to_llist(ff["reduced_sym__" + X + "__mat_in_eigen_space"], nump = True)
        #
        e_max_denom = ff["reduced_sym__" + X + "__e_max_denom"]
        e_data = _squarerootform_to_complex(ff["reduced_sym__" + X + "__e_data"], e_max_denom)
        #
        cache_eig_matrices = _oned_to_complicated(ff["reduced_sym__" + X + "__e_info"],
                                                  e_data,
                                                  ff["reduced_sym__" + X + "__e_keys"])
        reconstructed = _from_eigen_to_general(X,
                                               mat_in_eigen_space,
                                               cache_eig_matrices,
                                               s_rotc,
                                               s_inv,
                                               s_tr,
                                               s_orb,
                                               r_irr,
                                               r_rel,
                                               r_rel_oper,
                                               r_star,
                                               ff["reduced_sym__" + X + "__parity_I"],
                                               ff["reduced_sym__" + X + "__parity_TR"])

        _XX_R[X] = _few_to_many_bits(reconstructed)

    return _XX_R

def _many_to_few_bits(x):
    if not isinstance(x, np.ndarray):
        return x
    if x.dtype == np.complex128:
        return np.array(x, dtype = np.complex64)
    elif x.dtype == np.float64:
        return np.array(x, dtype = np.float32)
    else:
        return x

def _few_to_many_bits(x):
    if not isinstance(x, np.ndarray):
        return x
    if x.dtype == np.complex64:
        return np.array(x, dtype = np.complex128)
    elif x.dtype == np.float32:
        return np.array(x, dtype = np.float64)
    else:
        return x

def _write_interface_to_wberri_to_file(fname, obj, add_info = {}):
    f = open(fname, "wb")
    data={
        "mp_grid": obj.mp_grid,
        "num_wann": obj.num_wann,
        "periodic": obj.periodic,
        "real_lattice": obj.real_lattice,
        "recip_lattice": obj.recip_lattice,
        "use_wcc_phase": obj.use_wcc_phase,
        "iRvec": obj.iRvec,
        }
    for k in obj._XX_R.keys():
        data["_XX_R" + "__" + k] = _many_to_few_bits(obj._XX_R[k])
    for keys in add_info.keys():
        data["add_info__" + keys.strip()] = add_info[keys]
    np.savez_compressed(f, **data)
    f.close()

def _read_interface_to_wberri_from_file(fname):
    f = open(fname, "rb")
    ff = np.load(f, allow_pickle = False)
    ret = _InterfaceToWberri(None)
    ret.fill_in_from_dictionary(ff)
    add_info = {}
    for k in ff.keys():
        if "add_info__" in k:
            if ff[k].dtype.type is np.string_:
                add_info[k.replace("add_info__", "")] = str(ff[k])
            else:
                add_info[k.replace("add_info__", "")] = ff[k]
    f.close()

    return ret, add_info

def _append_to_npz(fin, fout, extra_add_info, overwrite_key = False):
    if os.path.exists(fout) == True:
        _raise_value_error("File " + fout + " already exists.  Stopping.")
    f = open(fin, "rb")
    ff = np.load(f, allow_pickle = False)
    data = dict(ff)
    for k in extra_add_info.keys():
        usek = "add_info__" + k.strip()
        if overwrite_key == False and usek in data.keys():
            _raise_value_error("Duplicate key " + usek + ".  Stopping.")
        data[usek] = extra_add_info[k]
    g = open(fout, "wb")
    np.savez_compressed(g, **data)
    g.close()
    f.close()

def _llist_to_oned(ll):
    num_terms = len(ll)
    len_each_term = []
    for l in ll:
        len_each_term.append(len(l))
    ret = [num_terms] + len_each_term
    for l in ll:
        ret = ret + l
    ret = np.array(ret)
    if ret.ndim != 1:
        print("Not a list of lists?!")
        exit()
    return ret

def _oned_to_llist(oned, nump = False):
    num_terms = int(np.round(np.real(oned[0])))
    len_each_term = np.array(np.round(np.real(oned[1 : num_terms + 1])), dtype = int).tolist()
    use = oned[num_terms + 1:]
    ret = []
    for i in range(num_terms):
        if nump == False:
            ret.append(use[:len_each_term[i]].tolist())
        else:
            ret.append(np.array(use[:len_each_term[i]]))
        use = use[len_each_term[i]:]
    if len(use) != 0:
        print("Missing!")
        exit()
    return ret

def _np_to_ls(lmat):
    ret = []
    for l in lmat:
        ret.append(l.tolist())
    return ret

def _complicated_to_oned(dicmat):
    keys = list(dicmat.keys())
    keys.sort()

    num_terms = len(keys)
    first_shape_each_term = []
    for k in keys:
        first_shape_each_term.append(dicmat[k].shape[0])

    remaining_shape = list(dicmat[keys[0]].shape[1:])
    for k in keys:
        if remaining_shape != list(dicmat[k].shape[1:]):
            print(remaining_shape, list(dicmat[k].shape[1:]))
            print("Inconsistent shapes!")
            exit()

    ret_info = [num_terms] + first_shape_each_term + [len(remaining_shape)] + remaining_shape

    ret_data = np.copy(dicmat[keys[0]])
    for k in keys[1:]:
        ret_data = np.vstack((ret_data, dicmat[k]))

    keys = ",".join(keys)

    return np.array(ret_info), ret_data, np.array(keys)

def _oned_to_complicated(e_info, e_data, e_keys):
    num_terms = e_info[0]
    first_shape_each_term = e_info[1 : num_terms + 1]
    #remaining_shape_len = e_info[num_terms + 1]
    #remaining_shape = e_info[num_terms + 2 : num_terms + 2 + remaining_shape_len]

    keys = str(e_keys).split(",")
    dicmat = {}
    start = 0
    for i, k in enumerate(keys):
        dicmat[k] = e_data[start : start + first_shape_each_term[i]]
        start += first_shape_each_term[i]
    if start != e_data.shape[0]:
        print("Missing terms!")
        exit()
    return dicmat

def _complex_to_squarerootform(mat, max_denom):
    tmp_re = float(max_denom)*(np.real(mat)**2)*np.sign(np.real(mat))
    tmp_im = float(max_denom)*(np.imag(mat)**2)*np.sign(np.imag(mat))

    if np.max(np.abs(tmp_re - np.round(tmp_re))) > 1.0E-8:
        print(np.max(np.abs(tmp_re - np.round(tmp_re))))
        print(tmp_re - np.round(tmp_re))
        print("Problem!")
        exit()
    if np.max(np.abs(tmp_im - np.round(tmp_im))) > 1.0E-8:
        print(np.max(np.abs(tmp_im - np.round(tmp_im))))
        print("Problem! im")
        exit()
    if np.max(np.abs(tmp_re)) > 128.0*256.0 - 1.0:
        print("Too large!")
        exit()
    if np.max(np.abs(tmp_im)) > 128.0*256.0 - 1.0:
        print("Too large!")
        exit()

    ret = np.array([np.array(np.round(tmp_re), dtype = np.int16),
                    np.array(np.round(tmp_im), dtype = np.int16)])
    return ret

def _squarerootform_to_complex(mat, max_denom):
    tmp_re = np.array(mat[0], dtype = float)
    tmp_im = np.array(mat[1], dtype = float)
    tmp_re = np.sqrt(np.abs(tmp_re) / float(max_denom)) * np.sign(tmp_re)
    tmp_im = np.sqrt(np.abs(tmp_im) / float(max_denom)) * np.sign(tmp_im)
    ret = tmp_re + 1.0j*tmp_im
    ret[np.abs(ret) < 1.0E-8] = 0.0
    return ret

def _from_eigen_to_general(X, mat_in_eigen_space, cache_eig_matrices, s_rotc, s_inv, s_tr, s_orb, r_irr, r_rel, r_rel_oper, r_star, parity_I, parity_TR):
    maxR = np.max(r_irr)
    for ll in r_rel:
        maxR = np.max([maxR, np.max(ll)])

    shp = list(cache_eig_matrices[list(cache_eig_matrices.keys())[0]].shape)
    shp = shp[1:3] + [maxR + 1] + shp[3:]

    ret = np.zeros(shp, dtype = complex)
    populated = []
    for ii in range(len(r_irr)):
        pick_one_ind = r_irr[ii]
        subgr = " ".join(list(map(str, r_star[ii])))
        mat_reconstructed = opteinsum("b, b... -> ...", mat_in_eigen_space[ii], cache_eig_matrices[subgr])

        ret[:,:,pick_one_ind] = mat_reconstructed
        populated.append(pick_one_ind)

        for i in range(len(r_rel[ii])):
            tmp_trns = _transform_one(s_rotc, s_inv, s_tr, s_orb, r_rel_oper[ii][i], mat_reconstructed, parity_I, parity_TR)
            if r_rel[ii][i] not in populated:
                ret[:,:,r_rel[ii][i]] = tmp_trns
            else:
                if np.max(np.abs(tmp_trns - ret[:,:,r_rel[ii][i]])) > 1.0E-10:
                    print("Hm, this should not happen.  One can arrive at this R vector by applying to different operations to different irreducible R vectors.")
                    print("But, depending on which combination you use, you get a different matrix?!")
                    exit()
            populated.append(r_rel[ii][i])

    return ret

def _rotate_matrix(X, L, R):
    if X.ndim == 2:
        return L.dot(X).dot(R)
    elif X.ndim == 3:
        X_shift = X.transpose(2, 0, 1)
        tmpX = L.dot(X_shift).dot(R)
        return tmpX.transpose(0, 2, 1).reshape(X.shape)
    else:
        raise ValueError()

def _transform_one(s_rotc, s_inv, s_tr, s_orb, i, mat_in, parity_I, parity_TR):
    mat = np.copy(mat_in)
    # on AA and SS you need to deal with vectors
    # here I daggered everything relative to wberri as I'm rotating from irr to rotated
    if mat.ndim == 3:
        mat = np.tensordot(mat, s_rotc[i].T.conj(), axes=1).reshape(mat.shape)
    elif mat.ndim > 3:
        raise ValueError("transformation of tensors is not implemented")
    if s_inv[i] == True:
        mat *= parity_I
    if s_tr[i] == True:
        mat = (mat/parity_TR).conj()
    return _rotate_matrix(mat, s_orb[i], s_orb[i].T.conj())

def _nice_exp(x):
    arg = np.maximum(np.minimum(x, 100.0), -100.0)
    return np.exp(arg)

def _fermi_dirac(en, mu, kbt):
    return 1.0 / (_nice_exp((en - mu)/kbt) + 1.0)

def _fermi_dirac_deriv(en, mu, kbt):
    return (-1.0 / (_nice_exp((en - mu)/kbt) + 1.0)**2) * _nice_exp((en - mu)/kbt) * (1.0 / kbt)

def _potentially_reorder_orbitals(mat, ind, reorder_orbitals):
    if reorder_orbitals == True:
        if ind == 0:
            d = mat.shape[0]
            return np.reshape(np.transpose(np.reshape(mat, (d//2, 2)), (1, 0)), (d))
        elif ind == 2:
            d = mat.shape[2]
            return np.reshape(np.transpose(np.reshape(mat, (mat.shape[0], mat.shape[1], d//2, 2)), (0, 1, 3, 2)), (mat.shape[0], mat.shape[1], d))
        else:
            _raise_value_error("This should not happen.")
    else:
        return mat

def _adjust_input_file(kind, text):
    if kind == "scf" or kind == "nscf":
        ln = text.split("\n")
        ret = ""
        for l in ln:
            if "pseudo_dir = " in l:
                ret += l.split("=")[0] + r"= '.'" + "\n"
            else:
                ret += l + "\n"
    elif kind == "pw2wan":
        ln = text.split("\n")
        ret = ""
        for l in ln:
            if "outdir = " in l:
                ret += l.split("=")[0] + r"= '_work'" + "\n"
            else:
                ret += l + "\n"
    else:
        _raise_value_error("This should not happen.")
    return ret

def _stop_because_loaded_from_wannierberri():
    traceback.print_stack()
    print("\n\n\n")
    _raise_value_error("""You loaded your calculation directly from Wannier Berri using
    function load_from_wannierberri, and not through the WfBase database.  Therefore
    some information you are trying to access is not available""")
