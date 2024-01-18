""" Code to print out subtraction table with 
    computation time in seconds
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

print('Running for ', planetNAME)

""" set up planet and star parameters """
aRs, RpRs = md.getScaled(a, Rp, Rs)
# star-planet seperation in units of Rp, need for Starry
sep = aRs / RpRs 
# stellar radius in units of Rp, need for Starry
Rs = 1 / RpRs

# coordinates of center of source in units of Rp
xs = 0
ys = 0
zs = sep

# if have a single lat, lon to look at do here, else can use
# lat, lon = map.get_latlon_grid(res=res, projection='rect')
# must be np.array!
lat = np.array([0])
lon = np.array([0])

# array of number of Q to use
source_npts = [1, 4, 9, 16, 25, 
               32, 45, 60, 69, 88, 109,
               124, 145, 172, 193,
               216, 249, 276, 305
              ]

""" code to compute intensities and times """
starry.config.lazy = False

# get the values at substellar point
Istarry = np.zeros(len(source_npts))
Ifinite = np.zeros(len(source_npts))
starrytime = np.zeros(len(source_npts))
finitetime = np.zeros(len(source_npts))
for i in range(0,len(source_npts)):
    t1 = time.time()
    starryMAPS = starry.Map(reflected=True, source_npts=source_npts[i])
    # get the intensities at substellar points, lat = 0, lon = 0
    Istarry[i] = starryMAPS.intensity(lat = lat, lon=lon, 
                     xs=xs, ys=ys, zs=zs,
                     rs = Rs
                    )
    starrytime[i] = time.time()-t1
    
    t2 = time.time()
    Ifinite[i] = md.getIfinite(lat, lon, xs, ys, zs, Rs, source_npts[i])
    finitetime[i] = time.time()-t2

# set first value of Ifinite to Iptxi case, else is nan becasue dr = 0
Ifinite[0] = md.getIptxi(lat,lon, xs, ys, zs,1)

# get the analytically determined value
t3 = time.time()
Ifull = md.getIfull(lat, lon, xs, ys, zs, Rs)
fulltime = time.time()-t3

starryMfinite = Istarry-Ifinite
finiteMfull = Ifinite - Ifull
starryMfinite/=1e-6
finiteMfull/=1e-6
for i in range(0, len(source_npts)):
    print(
        'N = {:d}, Istarry - Ifinite = {:0.04f}, Ifinite - Ifull = {:0.04f}, starrytime = {:0.04f}, finitetime = {:1.03E}'.format(
            source_npts[i], starryMfinite[i], finiteMfull[i],starrytime[i],finitetime[i])
         )
print('fulltime = {:1.03E}'.format(fulltime))



