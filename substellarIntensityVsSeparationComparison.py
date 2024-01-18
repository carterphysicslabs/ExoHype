""" Code to plot substellar intensities as function of 
    star-planet separation in units of Rs
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

""" initialize planet and star 
    comment/uncomment desired planet
"""
# a is in AU, Rp is in Earth radii, Rs is in solar radii

# parameters from Gaudi et al. (2017)
planetNAME = 'KELT-9b'
a = 0.03462 
Rp = 21.21
Rs = 2.362

# Esteves et al. (2015)
# planetNAME = 'Kepler-91b'
# a = 0.0731 
# Rp = 15.32 
# Rs = 6.30 

# from Agol, et al. 2021 doi 10.3847/psj/abd022
# planetNAME = "Trappist-1 d"
# a = 0.02227 
# Rp = 0.788 
# Rs = 0.1192 

print('Running substellarIntensityVsSeparationComparison.py for ', planetNAME)

""" code to compute intensities and maps """
# get scaled values in terms of Rs
aRstrue, RpRs = md.getScaled(a, Rp, Rs)

aRs = np.linspace(1.5, 5, 1000)
# star-planet seperation in units of Rp, need for Starry
sep = aRs / RpRs 
sepTrue = aRstrue / RpRs
# stellar radius in units of Rp, need for Starry
Rs = 1 / RpRs

# coordinates of center of source in units of Rp
xs = 0
ys = 0
zs = sep
zsTrue = sepTrue

# number of source points for extended source models
source_npts = 25

# paper coordinates have substellar point at 0, 0 if zs = sep
lat = np.array([0])
lon = np.array([0])


""" Config and parameter settings for STARRY """
starry.config.lazy = False
# initilize the maps from starry
mappt = starry.Map(reflected=True, source_npts=1)
mapext = starry.Map(reflected=True, source_npts=source_npts)

# now get intensities
# point source approximation using equation (27)
Ipt = mappt.intensity(lat = lat, lon=lon, 
                     xs=xs, ys=ys, zs=zs,
                     rs = Rs
                    )

# extended source, uses equation (27) over a loop, see lines 1385-1405 of starry ver 1.2.0 here:
# https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/_core/core.py#L1385
Iext = mapext.intensity(lat = lat, lon=lon, 
                     xs=xs, ys=ys, zs=zs,
                     rs = Rs
                    )

""" get intensities from myDefs """
Ifull = np.zeros(np.shape(zs))
Iplane = np.zeros(np.shape(zs))
Ifinite = np.zeros(np.shape(zs))
Ifinite2 = np.zeros(np.shape(zs))
for i in range(len(zs)):
    Iplane[i] = md.getIplane(lat,lon, xs, ys, zs[i])
    Ifull[i] = md.getIfull(lat, lon, xs, ys, zs[i], Rs)
    Ifinite[i] = md.getIfinite(lat,lon, xs, ys, zs[i], Rs, source_npts)
    Ifinite2[i] = md.getIfinite2(lat, lon, xs, ys, zs[i], Rs, source_npts)

# reshape to be np.shape(aRs), only need to for Iext, and Ipt
Ipt = np.reshape(Ipt, np.shape(aRs))
Iext = np.reshape(Iext, np.shape(aRs))
Ifinite = np.reshape(Ifinite, np.shape(aRs))
Ifinite2 = np.reshape(Ifinite2, np.shape(aRs))

""" convert everything to ppm, get maximum value """
# do all in ppm
fluxFRAC = 1e-6
Ipt /= fluxFRAC
Iext /= fluxFRAC
Iplane /= fluxFRAC
Ifull /= fluxFRAC
Ifinite /= fluxFRAC
Ifinite2 /= fluxFRAC

themax = np.max([np.nanmax(Ipt),
                 np.nanmax(Iext),
                 np.nanmax(Iplane),
                 np.nanmax(Ifull), 
                 np.nanmax(Ifinite),
                 np.nanmax(Ifinite2)
                ]
               )

""" common figure stuff """
# to convert figure sizes
mm = 1/25.4 

#sizes of things
tickSIZE = 8
axeslabelSIZE = 10
subtitleSIZE = 12
titleSIZE = 12
legSIZE = 10

markerSIZE = 20
lineSIZE = 1.5

figX = 190*mm
figY = 115*mm


""" initalize stuff for plotting """
ymin = 0
ymax = themax*1.05

xvar = aRs

theIntensities = [Ifull,
            Ifinite,
            Ipt,
            Iplane,
            Iext,
            Ifinite2
           ]

intensitylabels = [r'$I_{full}$', 
             r'$I_{ext,\cos\xi\cos\theta^\prime}$', 
             r'$I_{pt,starry}$',
             r'$I_{plane}$',
             r'$I_{ext,starry}$',
             r'$I_{ext,\cos\xi}$', 
            ]

linespecs = ['-k',
             '-.b',
             '--m',
             ':g',
             '-c',
             '-.r',
            ]

""" Time to plot """
fig = plt.figure(figsize=(figX, figY))
ax = plt.axes()
for vals, labels, specs in zip(theIntensities, intensitylabels,linespecs):
    ax.plot(xvar, 
            vals, 
            specs, 
            label = labels, 
            linewidth = lineSIZE,
           )


# add markers for "true" values
ax.plot([aRstrue, aRstrue], [ymin,ymax], 
        '--k', 
        linewidth = 0.25 * lineSIZE, 
        zorder = 4,
       )


ax.set_xlabel('Star-planet seperation ($R_s$)', fontsize = axeslabelSIZE)
ax.set_ylabel("Intensity (ppm of $I_{star}$)", fontsize = axeslabelSIZE)

ax.set_ylim([ymin, ymax])
ax.set_xlim([1.5,5])
ax.tick_params(labelsize=tickSIZE)

ax.legend(fontsize = legSIZE)

# save time
fileNAME = planetNAME+"substellarVdistance.jpg"
fileNAME = fileNAME.replace(" ","")
plt.savefig(fileNAME, bbox_inches="tight", dpi=300)

""" print some descriptions """
print('{} accepted value of aRs = {:.02f}'.format(planetNAME,aRstrue))
print('The maximum calculated value was {:.02f} ppm.'.format(themax))