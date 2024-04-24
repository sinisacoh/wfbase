"""
Spin-resolved band structure
============================

Spin resolved band structure of Fe (bcc) near the Fermi level.

"""

# Copyright under GNU General Public License 2024
# by Sinisa Coh (see gpl-wfbase.txt)

import wfbase as wf
import numpy as np
import pylab as plt

def plot_multicolor_line(ax, x, y, col, rng = None, lw = 2.0, cmap = "coolwarm", negative = True):
    from matplotlib.collections import LineCollection
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    if rng is None:
        rng = np.max(np.abs(col))
    if negative == True:
        norm = plt.Normalize(-rng, rng)
    else:
        norm = plt.Normalize(0.0, rng)
    lc = LineCollection(segments, cmap = cmap, norm=norm)
    lc.set_array(col)
    lc.set_linewidth(lw)
    line = ax.add_collection(lc)

wf.download_data_if_needed()
    
db = wf.load("data/fe_bcc.wf")
comp = db.do_path("GM--N--H", num_steps_first_segment = 300, to_compute = ["S"])
fig, ax = plt.subplots()
comp.plot_bs(ax, plot_bands = False, plot_fermi = False, plot_spec = False)
for i in range(comp["numwann"]):
    plot_multicolor_line(ax, comp["kdist"], comp["E"][:, i], comp["S"][:,i,i,2].real)
ax.set_title("Spin-resolved band structure of Fe (bcc)")
ax.set_ylabel(r"$E_{nk}$ (eV)")
ax.set_ylim(-3, 3)
fig.tight_layout()
fig.savefig("fig_bs_spin.pdf")
    



