""" Code to plot intensities along intensity equator
    for all three planets explored in the paper. 
    Varies the number of source points, N, and plots
    Ifull for comparison. Also outputs times
    Last updated by JLCarter on 2024, 01 18
"""

"""stuff to import"""
import myDefs as md

# stuff for starry
import theano
theano.config.gcc__cxxflags += " -fexceptions"

import starry

# stuff for plotting and dispay
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import mplot3d
import sympy
from IPython.display import display
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# regular stuff
import numpy as np
import time

""" initialize planets and stars """
# a is in AU, Rp is in Earth radii, Rs is in solar radii

# parameters for KELT-9b from Gaudi et al. (2017)
# parameters for Kepler-91 b from Esteves et al. (2015)
# parameters for Trappist-1 d from Agol, et al. 2021 doi 10.3847/psj/abd022

planetNAME_all = ["KELT-9 b", "Kepler-91 b", "Trappist-1 d"]
aAll = np.array([0.03462, 0.0731, 0.02227])
RpAll = np.array([21.21, 15.32, 0.77])
RsAll = np.array([2.362, 6.3, 0.1192])

aRsAll, RpRsAll = md.getScaled(aAll, RpAll, RsAll)

# star-planet seperation in units of Rp, need for Starry
sepAll = aRsAll / RpRsAll
# stellar radius in units of Rp, need for Starry
RsAll = 1 / RpRsAll

# coordinates of center of source in units of Rp
xs = 0
ys = 0
zs = sepAll

source_npts = [1, 4, 9, 16, 25, 
               32, 45, 60, 69, 88, 109,
               124, 145, 172, 193,
               216, 249, 276, 305
              ]

# planet coordinates are 0,0 for substellar point if zs = sep
lat = np.array([0])
lon = np.array([0])

""" get the intensities """
# initialize as zeros
Istarry = np.array([np.zeros(len(source_npts)), np.zeros(len(source_npts)), np.zeros(len(source_npts))])
Ifinite = np.array([np.zeros(len(source_npts)), np.zeros(len(source_npts)), np.zeros(len(source_npts))])
Ifull = np.array([np.zeros(len(source_npts)), np.zeros(len(source_npts)), np.zeros(len(source_npts))])

# get the analytically determined values
for p in range(0,len(sepAll)):
    Ifullholder = md.getIfull(lat, lon, xs, ys, zs[p], RsAll[p])
    Ifull[p] = np.full(shape = len(source_npts), fill_value = Ifullholder)

# get numerically determined intensities
starry.config.lazy = False
for p in range(0,len(sepAll)):
    for i in range(0,len(source_npts)):
        starryMAPS = starry.Map(reflected=True, source_npts=source_npts[i])
        # get the intensities at substellar points, lat = 0, lon = 0
        Istarry[p,i] = starryMAPS.intensity(lat = lat, lon=lon, 
                         xs=xs, ys=ys, zs=zs[p],
                         rs = RsAll[p]
                        )
        if source_npts[i] == 1:
            # set first value of Ifinite to Iptxi case, else is nan becasue dr = 0
            Ifinite[p,i] = md.getIptxi(lat,lon, xs, ys, zs[p],1)
        else:
            Ifinite[p,i] = md.getIfinite(lat, lon, xs, ys, zs[p], RsAll[p], source_npts[i])
    
    # END LOOP i
# END LOOP p

# normalize to ppm
fluxFRAC = 1e-6
Istarry /= fluxFRAC
Ifinite /= fluxFRAC
Ifull /= fluxFRAC

# do subtractions
Istardiff = Istarry-Ifull
Ifindiff = Ifinite-Ifull

# now do absolute values
Iabsstardiff = np.abs(Istardiff)
Iabsfindiff = np.abs(Ifindiff)

# get averages to help with visualizing
IabsaveStarry = np.empty_like(Iabsstardiff)
IabsaveFin = np.empty_like(Iabsfindiff)
for p in range(0,len(RsAll)):
    IabsaveStarry[p] = np.array(
        np.full(shape = len(source_npts),
                            fill_value = np.average(Iabsstardiff[p]))
        )
    IabsaveFin[p]= np.array(
        np.full(shape = len(source_npts),
                            fill_value = np.average(Iabsfindiff[p]))
        )

"""time to plot """
# to convert figure sizes
mm = 1/25.4 
rot = 60 # rotation of labels
xTICKS = [1,25, 50, 75, 100, 125, 150,175,200,225,250,275,300]

tickSIZE = 10
axeslabelSIZE = 11
subtitleSIZE = 13
titleSIZE = 15
legSIZE = 10

markerSIZE = 25
lineSIZE = 1.5

figX = 190*mm
figY = 170*mm

xvar = source_npts # every other is even/odd, starting with add N

starcolor = 'b'
starmark = 's'
fincolor = 'r'
finmark = 'o'

colTitles = [r'$|I_{ext,starry}-I_{full}|$',
             r'$|I_{ext,\cos\xi\cos\theta^\prime}-I_{full}|$',
            ]

fig ,((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,
                        figsize=(figX, figY), 
                        sharex = 'all',
                        sharey = 'row',
                        constrained_layout=True
                       )

for axs, planets, p in zip([[ax1, ax2], [ax3, ax4], [ax5, ax6]],
                           planetNAME_all,
                           range(0,len(planetNAME_all))
                        ):
    # plot the differences
    curIabsstardiff = Iabsstardiff[p]
    curIabsfindiff = Iabsfindiff[p]
    for i in range(0,len(xvar)):
        if xvar[i] % 2 == 0: # is even, fill in the marker
            fillstar = starcolor
            fillfin = fincolor
        else:
            fillstar = 'w'
            fillfin = 'w'
        axs[0].scatter(xvar[i], curIabsstardiff[i], label = '$|I_{starry}-I_{full}|$',
                       s = markerSIZE, 
                       marker = starmark, c = fillstar, edgecolors = starcolor, 
                       linewidths = 1,
                       zorder = 1)
        axs[1].scatter(xvar[i], curIabsfindiff[i], label = '$|I_{finite}-I_{full}|$',
                       s = markerSIZE, 
                       marker = finmark, c = fillfin, edgecolors = fincolor, 
                       linewidths = 1,
                       zorder = 2)
    
    # plot the average difference
    axs[0].plot(xvar, IabsaveStarry[p], 
                '--k', 
                label = 'Ave$|I_{starry}-I_{full}|$', 
                linewidth = lineSIZE)
    axs[1].plot(xvar, IabsaveFin[p], 
                '-.k', 
                label = 'Ave$|I_{finite}-I_{full}|$', 
                linewidth = lineSIZE)
    
    axs[0].set_ylabel("Intensity (ppm of $I_{star}$)", fontsize = axeslabelSIZE)
    for subax,titles in zip(axs,
                     colTitles
                    ):
        # add annotation with planet name
        subax.annotate(planets,
                       xy = (0.05, 0.85), xycoords = 'axes fraction',
                       fontsize = subtitleSIZE)
        # set ticks
        subax.tick_params(labelsize=tickSIZE)
        subax.set_ylim([np.min([np.min(Iabsfindiff[p]),np.min(Iabsstardiff[p])])*1.5, 
                        np.max([np.max(Iabsfindiff[p]),np.max(Iabsstardiff[p])])*1.5
                       ]
                      )
        if p == 0:
            # add titles
            subax.set_title(titles, fontsize = titleSIZE)
        if p == len(planetNAME_all)-1:
            # add x-axis stuff
            subax.set_xlabel('$N$', fontsize = axeslabelSIZE)
            subax.set_xticks(ticks = xTICKS)
            subax.set_xticklabels(xTICKS,rotation = rot)


# save time
fileNAME = "AllSubstellarVsourcePoints.jpg"
fileNAME = fileNAME.replace(" ","")
plt.savefig(fileNAME, bbox_inches="tight", dpi = 300)
