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

# array of number of Q to use
# source_npts = [1, 4, 9, 16, 25, 
#                32, 45, 60, 69, 88, 109,
#                124, 145, 172, 193,
#                216, 249, 276, 305
#               ]
# pick 10 for clarity.
source_npts = [1, 4, 25, 60 , 109,
               172, 216, 249, 305
              ]

# planet coordinates
res = 300
lon = np.linspace(-180,180,res)
lat = np.zeros(res)

""" get the intensities """
# initialize as zeros
Istarry = np.array([np.zeros((len(source_npts),res)), 
                    np.zeros((len(source_npts),res)), 
                    np.zeros((len(source_npts),res))
                   ]
                  )
Ifinite = np.array([np.zeros((len(source_npts),res)), 
                    np.zeros((len(source_npts),res)), 
                    np.zeros((len(source_npts),res))
                   ]
                  )
Ifull = np.array([np.zeros(res), np.zeros(res), np.zeros(res)])

# get the analytically determined values
for p in range(0,len(sepAll)):
    Ifull[p] = md.getIfull(lat, lon, xs, ys, zs[p], RsAll[p])

# get numerically determined intensities
starry.config.lazy = False
for p in range(0,len(sepAll)):
    for i in range(0,len(source_npts)):
        starryMAPS = starry.Map(reflected=True, source_npts=source_npts[i])
        # get the intensities at substellar points, lat = 0, lon = 0
        IstarryHOLDER = starryMAPS.intensity(lat = lat, lon=lon, 
                         xs=xs, ys=ys, zs=zs[p],
                         rs = RsAll[p]
                        )
        Istarry[p,i] = np.reshape(IstarryHOLDER, [res,])
        if source_npts[i] == 1:
            # set first value of Ifinite to Iptxi case, else is nan becasue dr = 0
            Ifinite[p,i] = md.getIptxi(lat,lon, xs, ys, zs[p],res)
        else:
            Ifinite[p,i] = md.getIfinite(lat, lon, xs, ys, zs[p], RsAll[p], source_npts[i])
    # END LOOP i
# END LOOP p

# normalize to ppm
fluxFRAC = 1e-6
Istarry /= fluxFRAC
Ifinite /= fluxFRAC
Ifull /= fluxFRAC

"""time to plot """
# to convert figure sizes
mm = 1/25.4 
#sizes of things
tickSIZE = 10
axeslabelSIZE = 11
subtitleSIZE = 13
titleSIZE = 15
legSIZE = 10

markerSIZE = 25
lineSIZE = 1.5

lineSTYLE = 'dashdot'
fullSTYLE = 'solid'

fullcolor = '0'

figX = 190*mm
figY = 170*mm



# first, set Istarry and Ifinite next to each other
fig ,((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,
                        figsize=(figX, figY), 
                        sharex = 'all',
                        sharey = 'row',
                        constrained_layout=True
                       )

colTitles = [r'$I_{ext,starry}$',
             r'$I_{ext,\cos\xi\cos\theta^\prime}$',
            ]

for axs, planets, p in zip([[ax1, ax2], [ax3, ax4], [ax5, ax6]],
                           planetNAME_all,
                           range(0,len(planetNAME_all))
                        ):
    for i in range(0,len(source_npts)):
        mylabel = "N = {:n}".format(source_npts[i])
        axs[0].plot(lon, Istarry[p,i], 
                    linestyle = lineSTYLE, 
                    label = mylabel, 
                    linewidth = lineSIZE,
                      )
        axs[1].plot(lon, Ifinite[p,i], 
                    linestyle = lineSTYLE,
                    linewidth = lineSIZE,
                      )
    # end i
    
    axs[0].set_ylabel("Intensity (ppm of $I_{star}$)", fontsize = axeslabelSIZE)
    for subax,titles in zip(axs,
                     colTitles
                    ):
        subax.plot(lon, Ifull[p], 
                   linestyle = fullSTYLE,
                   color = fullcolor,
                   label = '$I_{full}$', 
                   linewidth = lineSIZE*2, 
                   zorder = 0)
        # add annotation with planet name
        subax.annotate(planets,
                       xy = (0.05, 0.85), xycoords = 'axes fraction',
                       fontsize = subtitleSIZE)
        # set ticks
        subax.tick_params(labelsize=tickSIZE)
        if p == 0:
            # add titles
            subax.set_title(titles, fontsize = titleSIZE)
        if p == len(planetNAME_all)-1:
            subax.set_xlabel('Longitude [deg]', fontsize = axeslabelSIZE)


lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

# Finally, the legend
fig.legend(lines[0:len(source_npts)+1], 
           labels[0:len(source_npts)+1], 
           bbox_to_anchor = (1.15, 0.965),
           ncol=1,
           fontsize = legSIZE
          )

# save time
fileNAME = "AllIntensityEquator.jpg"
fileNAME = fileNAME.replace(" ","")
plt.savefig(fileNAME, bbox_inches="tight", dpi = 300)
