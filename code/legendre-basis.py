# Adapted from https://github.com/arvoelke/phd/blob/master/code/legendre-basis.py

import os
import numpy as np

from dashedlines import HandlerDashedLines, LineCollection
from nengolib.networks.rolling_window import _legendre_readout as delay_readout

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['text.usetex'] = True
plt.rcParams['font.serif'] = 'cm'

figname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "../figures", "legendre-basis.pdf")

q = 12
theta = 1.0

thetas = np.linspace(0, theta, 1000)
data = np.zeros((q, len(thetas)))

for i, thetap in enumerate(thetas):
    data[:, i] = delay_readout(q, thetap / theta)

cmap = sns.color_palette("GnBu_d", len(data))  # sns.cubehelix_palette(len(data), light=0.7, reverse=True)

with sns.axes_style('ticks'):
    with sns.plotting_context('paper', font_scale=4):
        plt.figure(figsize=(14, 6))
        for i in range(len(data)):
            plt.plot(thetas / theta, data[i], c=cmap[i],
                     lw=3, alpha=0.7, zorder=i)

        plt.xlabel(r"$\theta' / \theta$ (Unitless)", labelpad=20)
        plt.ylabel(r"$\mathcal{P}_i$")
        lc = LineCollection(len(cmap) * [[(0, 0)]], lw=10,
                            colors=cmap)
        plt.legend([lc], [r"$i = 0 \ldots d - 1$"], handlelength=2,
                   handler_map={type(lc): HandlerDashedLines()},
                   bbox_to_anchor=(0.2, 1.2), loc='upper left', borderaxespad=0.,
                   fancybox=False, frameon=False)

        sns.despine(offset=15)

        plt.savefig(figname, bbox_inches='tight')
