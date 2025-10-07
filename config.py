import numpy as np
from astropy import units as u
from astropy import constants as c
from astropy.coordinates import EarthLocation

from pathlib import Path

WAVELENGTH = np.arange(0, 12001, 1)
WMIN = 3825
WMAX = 9200
ROOTDIR = str(Path(__file__).resolve().parent) + '/'
MDSPEC = ROOTDIR + 'sdsstemplates/m5.active.ha.na.k.fits'

#Flare star params
plx = 10.27
dist = 1000 / plx * u.pc
Rstar = 0.1 * u.Rsun

#Image set range - depends on how many images you stored
start = 0
end = 74


HOUR2SEC = 3_600
DAY2SEC = HOUR2SEC * 24
deg2arcsec = 3_600
deg2rad = np.pi / 180
ha2deg = 15.0

#window size for median smoothing
window = 9

### CTIO weather conditions
RH = 0.0 #relative humidity (percent)
TEMPERATURE = 15 + 273.15 #surface air temp (Kelvin)
P_S = 760 #surface pressure (millibar)

BETA = 4.5908e-6 * TEMPERATURE
CHI = 1 + 5.302e-3 * np.sin(np.deg2rad(EarthLocation.of_site('Cerro Tololo').lat.value))**2 - \
      5.83e-6 * np.sin(np.deg2rad(2 * EarthLocation.of_site('Cerro Tololo').lat.value))**2 - 3.15e-7 * EarthLocation.of_site('Cerro Tololo').height.value

P_W = RH * 1e-4 * np.exp(77.345 + 0.0057 * TEMPERATURE - (7235 / TEMPERATURE)) / TEMPERATURE**8.2
D_W = (1 + P_W * (1 + 3.7e-4 * P_W) * (-2.37321e-3 + (2.23366 / TEMPERATURE) - (710.792 / TEMPERATURE**2) + (7.75141e4 / TEMPERATURE**3))) * (P_W / TEMPERATURE)
D_S = (1 + (P_S - P_W) * (57.90e-8 - (9.3250e-4 / TEMPERATURE) + (0.25844 / TEMPERATURE**2))) * ((P_S - P_W) / TEMPERATURE)



AMS = np.linspace(1.05,1.2,num=6,dtype='float') ###FBB I would getread of this and add the AM as a file produced with the first code that extracts the chips from the N images
AM =  np.array([1.46071187, 1.46383532, 1.46696313, 1.4701134 , 1.47325367,
       1.47644162, 1.47972646, 1.48297042, 1.48629275, 1.48949247,
       1.49289569, 1.49621113, 1.49956752, 1.50292117, 1.50630044,
       1.50972821, 1.51323743, 1.51670154, 1.52018407, 1.52369808,
       1.52722673, 1.53077701, 1.53434693, 1.53785739, 1.54148284,
       1.54508301, 1.54880347, 1.55260932, 1.55638765, 1.56019322,
       1.56392776, 1.56760547, 1.57134818, 1.575136  , 1.57902216,
       1.58298778, 1.58688571, 1.59078537, 1.59485041, 1.59882875,
       1.6028305 , 1.6068912 , 1.61097471, 1.6150698 , 1.61919505,
       1.62346431, 1.62762555, 1.63188249, 1.6361399 , 1.64049296,
       1.64482594, 1.64925459, 1.65361239, 1.6580181 , 1.6624244 ,
       1.66693561, 1.67142261, 1.67604518, 1.68070194, 1.68524247,
       1.68981158, 1.694539  , 1.69922954, 1.70404494, 1.70876083,
       1.71348053, 1.71828197, 1.7231769 , 1.72809362, 1.73293621,
       1.73788664, 1.74284825, 1.74787617, 1.75291492])


def get_img_string(key):
    codes={'flare1ep1':'20200206287824000819zg13o2',
           'flare1ep3':'20200206302627000819zg13o2',
           'flare2ep1':'20190418224780000864zg08o3',
           'flare3ep1':'20180402425405000864zg10o2'
    }
    return codes[key]

def get_flr_coord(key):
    coords = {'flare1':[186.09002468556912, 65.31234685437119],
              'flare2':[239.9337200957272, 75.04305873103452],
              'flare3':[255.0794984723742, 78.31401820936068]}
    return coords[key]

batoid_trace = np.array([[4000, 4200,4400,4600,4800, 5100, 5300, 5500],[-4.7265763558433775, 
                         -2.6427375813610112, 
                         -1.0704452846444354, 
                         0.029975162321676407, 
                         0.9870705151456214, 
                         2.0014959056264345, 
                         2.528961277402921,
                         2.9450944784608026]])

batoid_params = np.array([-5.26219831e-08,  5.85778052e-04, -1.58081754e+00])

linenames = ['Ca II H', 'Ca II K', r'H$\delta$', r'H$\gamma$', r'H$\beta$']

linedict = {'Ca II H': np.array([2.39550000e-13, 3.93184712e+03, 5.51710509e+00]),
 'Ca II K': np.array([4.52232596e-13, 3.96757926e+03, 7.61958415e+00]),
 'H$\\delta$': np.array([3.88426913e-13, 4.10018988e+03, 7.61420211e+00]),
 'H$\\gamma$': np.array([4.10208099e-13, 4.33913340e+03, 8.00815074e+00]),
 'H$\\beta$': np.array([3.88398001e-13, 4.86003526e+03, 8.02575104e+00])}

quiescent_spectra = {"m7": "/sdsstemplates/m7.active.ha.na.k.fits",
			"m5": "/sdsstemplates/m5.active.ha.na.k.fits"}
quiescent_spectranpy = {"m7": "/m7.npy",
			"m5": "/m5.npy"}
