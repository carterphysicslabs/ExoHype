""" Code to create difference plot along intensity equator 
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

print('Running equatorSubtractions.py for ', planetNAME)

""" code to compute intensities and maps """
# get scaled values in terms of Rs
aRs, RpRs = md.getScaled(a, Rp, Rs)
# star-planet seperation in units of Rp, need for Starry
sep = aRs / RpRs 
# stellar radius in units of Rp, need for Starry
Rs = 1 / RpRs

# resolution of lat and lon grid
res = 300

# get points on equator
equatorLON = np.linspace(-180,180,res)
equatorLAT = np.zeros(res)

# number of source points for extended source models
source_npts = 25

# location of center of star. zs = sep cooresponds to out of the page in figures.
xs = 0
ys = 0
zs = sep

""" Config and parameter settings for STARRY """
starry.config.lazy = False
# initilize the maps from starry
mappt = starry.Map(reflected=True, source_npts=1)
mapext = starry.Map(reflected=True, source_npts=source_npts)

# now get intensities
# point source approximation using equation (27)
Iptequa = mappt.intensity(lat = equatorLAT, lon=equatorLON, 
                     xs=xs, ys=ys, zs=zs,
                     rs = Rs
                    )

# extended source, uses equation (27) over a loop, see lines 1385-1405 of starry ver 1.2.0 here:
# https://github.com/rodluger/starry/blob/b72dff08588532f96bd072f2f1005e227d8e4ed8/starry/_core/core.py#L1385
Iextequa = mapext.intensity(lat = equatorLAT, lon=equatorLON, 
                     xs=xs, ys=ys, zs=zs,
                     rs = Rs
                    )

""" get intensities from myDefs """
Ifiniteequa = md.getIfinite(equatorLAT, equatorLON, xs, ys, zs, Rs, source_npts)
Ifinite2equa = md.getIfinite2(equatorLAT, equatorLON, xs, ys, zs, Rs, source_npts)
Ifullequa = md.getIfull(equatorLAT,equatorLON, xs, ys, zs, Rs)
Iplaneequa = md.getIplane(equatorLAT, equatorLON, xs, ys, zs)

# change shape to all be (res, 1)
Ifiniteequa = np.reshape(Ifiniteequa, [res, 1])
Ifinite2equa = np.reshape(Ifinite2equa, [res, 1])
Ifullequa = np.reshape(Ifullequa, [res, 1])
Iplaneequa = np.reshape(Iplaneequa, [res, 1])

""" convert everything to ppm, calculate differences, 
    and get min and max values for scaling """
# do all in ppm
fluxFRAC = 1e-6
Iptequa /= fluxFRAC
Iextequa /= fluxFRAC
Ifiniteequa /= fluxFRAC
Ifinite2equa /= fluxFRAC
Ifullequa /= fluxFRAC
Iplaneequa /= fluxFRAC


finite2tostarry = Ifinite2equa - Iextequa
exttofinite = Iextequa - Ifiniteequa
finitetofull = Ifiniteequa - Ifullequa
fulltoplane = Ifullequa - Iplaneequa
pttoplane = Iptequa - Iplaneequa
finitetoplane = Ifiniteequa-Iplaneequa

# get extent of illumination and fully illuminated zone to mark on plots
xfull, yfull = md.getFullLatLong(RpRs, aRs, res ** 2)
xtermextPOS, xtermextNEG, ytermext = md.getExtLatLong(RpRs, aRs, res ** 2)

""" common figure stuff """
# to convert figure sizes
mm = 1/25.4 

#sizes of things
tickSIZE = 8
axeslabelSIZE = 10
subtitleSIZE = 12
titleSIZE = 12
legSIZE = 10

# markerSIZE = 15
lineSIZE = 1.5

figX = 190*mm
figY = 120*mm


""" initalize stuff for plotting """
thediffs = [finite2tostarry,
            exttofinite,
            finitetofull,
            fulltoplane,
            pttoplane,
            finitetoplane
           ]

difflabels = [r'$I_{ext,\cos\xi}-I_{ext,starry}$', 
             r'$I_{ext,starry}-I_{ext,\cos\xi\cos\theta^\prime}$', 
             r'$I_{ext,\cos\xi\cos\theta^\prime}-I_{full}$',
             r'$I_{full}-I_{plane}$',
             r'$I_{pt,starry}-I_{plane}$',
             r'$I_{ext,\cos\xi\cos\theta^\prime}-I_{plane}$',
            ]

linespecs = ['-k',
             '-.b',
             '--m',
             ':r',
             '-g',
             '-.c'
            ]

fig = plt.figure(figsize=(figX, figY))
ax = plt.axes()
# inset
axin = inset_axes(ax, width = "27%", height ="30%",loc=2, borderpad=2)

""" Time to plot """
# the difference plots
for diff, labels, specs in zip(thediffs, difflabels,linespecs):
    ax.plot(equatorLON, diff, specs, label = labels, linewidth = lineSIZE)
    axin.plot(equatorLON,diff,specs,label = labels, linewidth = lineSIZE)

# axes limits and limits of zones
xfullmin = np.nanmin(xfull)
xfullmax = np.nanmax(xfull)
xunmin = -180 - np.nanmin(ytermext)
xunmax = 180 - np.nanmax(ytermext)

ymin, ymax = ax.get_ylim()

axin.set_xlim([xunmin,xunmax])
axin.set_ylim([np.nanmin(finite2tostarry)*1.05, np.nanmax(finite2tostarry)*1.05])

for axis in [ax,axin]:
    axis.plot([xfullmin, xfullmin],[ymin, ymax], '--c', linewidth = 0.25 *lineSIZE)
    axis.plot([xfullmax, xfullmax],[ymin, ymax], '--c', linewidth = 0.25 *lineSIZE)
    axis.tick_params(labelsize = tickSIZE)

ax.plot([xunmin, xunmin],[ymin, ymax], '--k', linewidth = 0.25 *lineSIZE)
ax.plot([xunmax, xunmax],[ymin, ymax], '--k', linewidth = 0.25 *lineSIZE)

# labels
ax.set_xlabel('Longitude [deg]', fontsize = axeslabelSIZE)
ax.set_ylabel('Differences (ppm)', fontsize = axeslabelSIZE)

ax.set_xlim([-180, 180])
ax.set_ylim([ymin, ymax])

ax.legend(fontsize = legSIZE)

# save time
fileNAME = planetNAME+"IntensityEquator.jpg"
fileNAME = fileNAME.replace(" ","")
plt.savefig(fileNAME, bbox_inches="tight", dpi=300)

""" print some descriptions """
signsDiff = np.sign(fulltoplane)
longs = np.array([])
for i in range(0,len(signsDiff)):
    if signsDiff[i] == -1:
        longs = np.append(longs, equatorLON[i])
minVAL = np.nanmin(np.abs(longs))
maxVAL = np.nanmax(np.abs(longs))

print('Given Ifull-Iplane for {}:'.format(planetNAME))
print('Between 0 and {:.2f} degrees latitude, the plane parallel ray model underestimates the instellation'.format(
    minVAL)
     )
print('Within the fully illuminated zone, the plane parallel ray model overestimates the instellation between')
print('{:.2f} and {:.2f} degrees latitude'.format(minVAL, maxVAL))
print('Up to {:.2f} degrees beyond the pole on the anti-stellar side of the plane, the plane parallel ray model under estimates the instellation'.format(
xunmax-90)
     )