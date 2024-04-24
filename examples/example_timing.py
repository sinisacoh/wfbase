"""
Timing
======

Minimal testing of the speed of computer code is much better than
no testing at all.  In this simple example, we compare the typical behavior
of calculations in wfbase using a computer with 16 GB of RAM.  Behavior
on your computer might be different.

As the number of k-points in the fine interpolation mesh increases,
it becomes less efficient to store all of that information in the
memory.  Instead, it is more efficient to perform many calculations
over smaller grids.  This same strategy is used in example
:ref:`sphx_glr_all_examples_example_conv.py`

On the other hand, if one uses brute forced sums (this uses Numba
under the hood) then scaling with a k-point size is better.

These results will also depend on what kind of expression you are using.
Depending on the number of distinct indices, and the number of contractions,
you might find that brute forced sums (using Numba) might become less
efficient than numpy vectorizations.

"""

# Copyright under GNU General Public License 2024
# by Sinisa Coh (see gpl-wfbase.txt)

import wfbase as wf
import numpy as np
import pylab as plt
import time

def main():

    wf.download_data_if_needed()
    
    db = wf.load("data/fe_bcc.wf")

    figt, axt = plt.subplots()
    fig , axs = plt.subplots(4, 3, figsize = (12.0, 12.0))
    
    plot_times(check_timings(do_method_1, db, range(8, 25, 4), chunk_size = None), \
               axs[0], axt, "k", "Einsum, single shot")
    plot_times(check_timings(do_method_1, db, range(8, 49, 8), chunk_size = 8   ), \
               axs[1], axt, "r", "Einsum, smaller chunks")
    plot_times(check_timings(do_method_2, db, range(8, 49, 4), chunk_size = None), \
               axs[2], axt, "g", "Brute force sum, single shot")
    plot_times(check_timings(do_method_2, db, range(8, 49, 8), chunk_size = 8   ), \
               axs[3], axt, "b", "Brute force sum, smaller chunks")

    common_axes_range(axs)

    fig.tight_layout()
    fig.savefig("fig_timing.pdf")
    
    axt.set_ylim(0.0)
    figt.tight_layout()
    figt.savefig("fig_timing_total.pdf")
    
def do_method_1(db, nk):
    comp = db.do_mesh([nk, nk, nk], shift_k = "random", to_compute = ["A"])
    comp.evaluate("sigma_oij <= (j / (numk * volume)) * (f_km - f_kn) * \
                   Real((E_km - E_kn) / (E_km - E_kn - hbaromega_o - j*eta)) * \
                   A_knmi * A_kmnj")
    return comp

def do_method_2(db, nk):
    comp = db.do_mesh([nk, nk, nk], shift_k = "random", to_compute = ["A"])
    comp.evaluate("sigma_oij <= (j / (numk * volume)) * (f_km - f_kn) * \
                   Real((E_km - E_kn) / (E_km - E_kn - hbaromega_o - j*eta)) * \
                   A_knmi * A_kmnj", brute_force_sums = True)
    return comp

def check_timings(method, db, all_nks, chunk_size = None):

    method(db, 3)
    
    times = {"nk":[], "tot": [], "ini": [], "exe": []}
    for nk in all_nks:
        print(nk)
        times["nk"].append(nk)
        if chunk_size is None:
            tot_time = time.perf_counter()
            comp = method(db, nk)
            tot_time = time.perf_counter() - tot_time
            times["tot"].append(tot_time)
            times["ini"].append(comp.get_initialization_time())
            times["exe"].append(comp.get("sigma", "total_seconds_exec"))
        else:
            if nk % chunk_size != 0:
                print("Wrong chunk_size.")
                exit()
            nchunks = nk // chunk_size
            tmp_tot_time = 0.0
            tmp_ini_time = 0.0
            tmp_exe_time = 0.0
            for i in range(nchunks**3):
                tot_time = time.perf_counter()
                comp = method(db, chunk_size)
                tot_time = time.perf_counter() - tot_time
                tmp_tot_time += tot_time
                tmp_ini_time += comp.get_initialization_time()
                tmp_exe_time += comp.get("sigma", "total_seconds_exec")
            times["tot"].append(tmp_tot_time)
            times["ini"].append(tmp_ini_time)
            times["exe"].append(tmp_exe_time)
    for k in times.keys():
        times[k] = np.array(times[k])
    return times

def plot_times(times, axs, axt, col, label):
    data_nk  = times["nk"]    
    data_tot = 1000.0 * (times["tot"]) / (times["nk"]**3)
    data_ini = 1000.0 * (times["ini"]) / (times["nk"]**3)
    data_exe = 1000.0 * (times["exe"]) / (times["nk"]**3)
    data_ovh = 1000.0 * (times["tot"] - (times["ini"] + times["exe"])) / (times["nk"]**3)
    
    axt.plot(data_nk, data_tot, col + "o-", label = label)
    axt.set_title("Total time (per k-point)")
    axt.legend()
    axt.set_xticks(data_nk)
    axt.set_xlabel(r"Fine k-mesh")
    axt.set_ylabel(r"Time per k-point (ms)")    
    
    axs[0].plot(data_nk, data_ini, col + "o-", label = label)
    axs[0].set_title("Grid initialization time (per k-point)")
    axs[1].plot(data_nk, data_exe, col + "o-", label = label)
    axs[1].set_title("Quantity evaluation time (per k-point)")
    axs[2].plot(data_nk, data_ovh, col + "o-", label = label)
    axs[2].set_title("Overhead time (per k-point)")
    axs[0].set_ylabel(r"Time per k-point (ms)")    
    for ax in axs:
        ax.legend()
        ax.set_xticks(data_nk)
        ax.set_xlabel(r"Fine k-mesh")
    
def common_axes_range(all_axs):
    xmin = []
    xmax = []
    ymax = []
    for axs in all_axs:
        for ax in axs:
            xmin.append(ax.get_xlim()[0])
            xmax.append(ax.get_xlim()[1])
            ymax.append(ax.get_ylim()[1])
    xmin = max(xmin)
    xmax = max(xmax)
    ymax = max(ymax)
    for axs in all_axs:
        for ax in axs:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(0.0 , ymax)
            
main()
