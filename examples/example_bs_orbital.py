"""
Orbital-resolved band structure
===============================

Orbital resolved band structure of Fe (bcc) near the Fermi level.

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
comp = db.do_path("GM--N--H", num_steps_first_segment = 300, to_compute = ["psi"])
    
fig, axs = plt.subplots(2, 3, figsize = (3*4.0, 2*3.0))
for i in range(2):
    for j in range(3):
        if j == 0:
            # sp3d2_(+-x,+-y)
            orbs = range( 0 + i,  8, 2)
            cmap = "Reds"
        elif j == 1:
            # sp3d2_(+-z)
            orbs = range( 8 + i, 12, 2)
            cmap = "Blues"
        elif j == 2:
            # t2g
            orbs = range(12 + i, 18, 2)
            cmap = "Greens"
        ax = axs[i][j]
        comp.plot_bs(ax, plot_bands = False, plot_fermi = False, plot_spec = False)
        for n in range(comp["numwann"]):
            chars = np.sum((np.abs(comp["psi"][:, n, :])**2)[:, orbs], axis = 1)
            plot_multicolor_line(ax, comp["kdist"], comp["E"][:, n], chars, cmap = cmap, negative = False)
        if j == 0:
            ax.set_ylabel(r"$E_{nk}$ (eV)")
        ax.set_title(" ".join(list(map(lambda s: "$" + s + "$", comp["orbitallabels"][orbs]))))
        ax.set_ylim(-3.0, 3.0)
fig.tight_layout()
fig.savefig("fig_bs_orbital.pdf")
    



