"""
Band structure (all)
====================

Band structure of all transition metals (in a bcc structure).

"""

# Copyright under GNU General Public License 2024
# by Sinisa Coh (see gpl-wfbase.txt)

import wfbase as wf
import pylab as plt

wf.download_data_if_needed()

cases = """sc ti v  cr mn fe co ni cu zn
           y  zr nb mo tc ru rh pd ag cd
           XX hf ta w  re os ir pt au hg"""

fig, axs = plt.subplots(3, 10, figsize = (20, 6))
row = 0
for line in cases.split("\n"):
    column = 0
    for ll in line.split(" "):
        if ll == "":
            continue
        if ll == "XX":
            axs[row][column].remove()
            column += 1
            continue
        db = wf.load("data/" + ll + "_bcc.wf")
        comp = db.do_path("GM--H--N")
        ax = axs[row][column]
        comp.plot_bs(ax)
        if column == 0:
            ax.set_ylabel(r"$E_{nk}$ (eV)")
        ax.set_ylim(-12.0, 12.0)
        ax.set_title(ll.title() + " (bcc)")
        column += 1
    row += 1
fig.tight_layout()
fig.savefig("fig_bs_all.pdf")
