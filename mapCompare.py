""" Code to create intensity maps 
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
# planetNAME = 'KELT-9b'
# a = 0.03462 
# Rp = 21.21
# Rs = 2.362

# Esteves et al. (2015)
# planetNAME = 'Kepler-91b'
# a = 0.0731 
# Rp = 15.32 
# Rs = 6.30 

# from Agol, et al. 2021 doi 10.3847/psj/abd022
planetNAME = "Trappist-1 d"
a = 0.02227 
Rp = 0.788 
Rs = 0.1192 

print('Running for ', planetNAME)

""" code to compute intensities and maps """
# get scaled values in terms of Rs
aRs, RpRs = md.getScaled(a, Rp, Rs)
# star-planet seperation in units of Rp, need for Starry
sep = aRs / RpRs 
# stellar radius in units of Rp, need for Starry
Rs = 1 / RpRs

# resolution of lat and lon grid
res = 300

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

# get latidude and longitude so all are the same
lat1, lon1 = mappt.get_latlon_grid(res=res, projection='rect')
# reshape to 1D
lon = np.reshape(lon1, res ** 2)
lat = np.reshape(lat1, res ** 2)

""" get the intensities from starry """
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
Ifinite = md.getIfinite(lat, lon, xs, ys, zs, Rs, source_npts)
Ifinite2 = md.getIfinite2(lat, lon, xs, ys, zs, Rs, source_npts)
Ifull = md.getIfull(lat,lon, xs, ys, zs, Rs)
Iplane = md.getIplane(lat, lon, xs, ys, zs)

""" convert everything to ppm and get min and max values for scaling """
# do all in ppm
fluxFRAC = 1e-6
Ipt /= fluxFRAC
Iext /= fluxFRAC
Ifinite /= fluxFRAC
Ifinite2 /= fluxFRAC
Ifull /= fluxFRAC
Iplane /= fluxFRAC
# get max and mins
theMAX = np.nanmax([np.nanmax(Ipt), 
                    np.nanmax(Iext),
                   np.nanmax(Ifinite), 
                    np.nanmax(Ifinite2), 
                    np.nanmax(Ifull),
                    np.nanmax(Iplane)
                   ])
theMIN = np.nanmin([np.nanmin(Ipt), 
                    np.nanmin(Iext),
                   np.nanmin(Ifinite), 
                    np.nanmin(Ifinite2), 
                    np.nanmin(Ifull),
                    np.nanmin(Iplane)
                   ])

# reshape for contours and imshow
Ipt = np.reshape(Ipt, [res,res])
Iext = np.reshape(Iext, [res,res])
Ifinite = np.reshape(Ifinite, [res,res])
Ifinite2 = np.reshape(Ifinite2, [res,res])
Ifull = np.reshape(Ifull, [res,res])
Iplane = np.reshape(Iplane, [res,res])

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
figY = 170*mm

# color map
cmap = 'plasma'
norm = mpl.colors.Normalize(vmin=theMIN, vmax=theMAX)

extent = (-180,np.max(lon),-90,np.max(lat))

""" initalize stuff for plotting """
titles = [r'$I_{ext,starry}$ Method', 
          r'$I_{ext,\cos\xi}$ Method',
          r'$I_{ext,\cos\xi\cos\theta^\prime}$ Method',
          r'$I_{full}$ Method',
          r'$I_{pt,starry}$ Method',
          r'$I_{plane}$ Method'
         ]
# for contours
# levels = np.array([0, 1, 5, 10, 30, 50, 75])
levels = (np.array([0, 5, 10, 30, 50, 75]), #extstarry
          np.array([0, 5, 10, 30, 50, 75]), #extcosxi
          np.array([0, 1, 10, 30, 50]), #extcosxicosthetaprime
          np.array([30, 50]), #full
          np.array([0, 10, 30, 50]), #ptstarry
          np.array([0, 10, 30, 50]), #plane
         )
fmt = {0: "0%", 1: "1%", 5: "5%", 10: "10%", 30: "30%", 50: "50%", 75: "75%"}

fig ,((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,
                        figsize=(figX, figY), 
                        sharex = 'all',
                        sharey = 'all',
                        constrained_layout=True
                       )

""" Time to plot """
for axis, img, tt, lvls in zip([ax1, ax2, ax3, ax4, ax5, ax6], 
                         [Iext,Ifinite2,Ifinite,Ifull,Ipt,Iplane], 
                         titles,
                         levels
                        ):
    sc = axis.imshow(img, 
                     origin="lower", 
                     interpolation="none", 
                     zorder=-3,
                     extent = extent,
                     cmap=cmap, 
                     norm = norm,
                    )
    axis.set_title(tt, fontsize = subtitleSIZE)
    # add extents of zones
    axis.plot(xfull, yfull, '--k', linewidth = lineSIZE)
    # set axis limits and ticks
    axis.set_ylim([np.min(lat), np.max(lat)])
    axis.set_xlim([np.min(lon), np.max(lon)])
    axis.tick_params(labelsize=tickSIZE)
    # plot contours
    X = np.linspace(-180, np.max(lon), img.shape[1])
    Y = np.linspace(-90, np.max(lat), img.shape[0])
    cont = axis.contour(
        X,
        Y,
        img / theMAX * 100,
        lvls,
        colors="w",
        antialiased=True,
        linewidths=0.5*lineSIZE,
        linestyles="dotted",
    )

    labels = axis.clabel(cont, 
                         fontsize=tickSIZE, 
                         fmt=fmt,
                        )
    for l in labels:
        l.set_rotation(0)

# add labels to axes
ax5.set_xlabel('Longitude [deg]', fontsize = axeslabelSIZE)
ax6.set_xlabel('Longitude [deg]', fontsize = axeslabelSIZE)
ax1.set_ylabel('Latitude [deg]', fontsize = axeslabelSIZE)
ax3.set_ylabel('Latitude [deg]', fontsize = axeslabelSIZE)
ax5.set_ylabel('Latitude [deg]', fontsize = axeslabelSIZE)

# add the color bar
cbar_ax = fig.add_axes([1.02, 0.05, 0.025, 0.90])
cbar = plt.colorbar(sc, cax = cbar_ax)
cbar.set_label(label="Intensity (ppm of $I_{star}$)", fontsize=legSIZE)
cbar_ax.tick_params(labelsize = tickSIZE)


# save time
fileNAME = planetNAME+"MapCompare.jpg"
fileNAME = fileNAME.replace(" ","")
plt.savefig(fileNAME, bbox_inches="tight", dpi=300)

""" print differences to report in paper """
starrydiff = Iext-Ifull
print('For '+ planetNAME + ', max(Istarry-Ifull) = {:0.04f} ppm of Istar'.format(np.nanmax(starrydiff)))
print('For '+ planetNAME + ', max(Ifinite-Ifull) = {:0.04f} ppm of Istar'.format(np.nanmax(Ifinite-Ifull)))
print('For '+ planetNAME + ', max(Ifull-Iplane) = {:0.04f} ppm of Istar'.format(np.nanmax(Ifull-Iplane)))
