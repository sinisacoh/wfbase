"""
Band structure plot
===================

Electron band-structure plot along the high-symmetry
lines in the Brillouin zone.

"""

# Copyright under GNU General Public License 2024
# by Sinisa Coh (see gpl-wfbase.txt)

import wfbase as wf
import pylab as plt

wf.download_data_if_needed()

db = wf.load("data/au_fcc.wf")
comp = db.do_path("GM--X--W--L--GM")
fig, ax = plt.subplots()
comp.plot_bs(ax)
ax.set_ylabel(r"$E_{nk}$ (eV)")
ax.set_title("Band structure of Au (fcc)")
fig.tight_layout()
fig.savefig("fig_bs_au.pdf")
    
db = wf.load("data/fe_bcc.wf")
comp = db.do_path("GM--H--N--GM--P--N--H")
fig, ax = plt.subplots()
comp.plot_bs(ax)
ax.set_title("Band structure of Fe (bcc)")
ax.set_ylabel(r"$E_{nk}$ (eV)")
fig.tight_layout()
fig.savefig("fig_bs_fe.pdf")
    
