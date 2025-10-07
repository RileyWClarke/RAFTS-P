import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, FK5
from astropy.io import fits

import scipy
from scipy.interpolate import interp1d
from scipy.optimize import minimize

#import rubin_sim.phot_utils.bandpass as Bandpass
#import rubin_sim.phot_utils.Sed as Sed

from config import *
from mdwarf_interp import *

import astropy.constants as const
import astropy.units as u
from astropy.modeling.models import BlackBody

import warnings
#suppress warnings
warnings.filterwarnings('ignore')

def calc_w_eff(fluxc, filt, wavec):
    #Calc effective lambda
    w_eff = np.exp(np.sum( fluxc * filt * np.log(wavec)) / 
                   np.sum( fluxc * filt))

    return w_eff

def gaussian(x, A=1.0, mu=0.0, sigma=1.0):
    """
    Calculate the Gaussian function.

    Parameters:
    x (array-like): Input values where the Gaussian function will be evaluated.
    A (float): Amplitude of the Gaussian (default is 1.0).
    mu (float): Mean of the Gaussian (default is 0.0).
    sigma (float): Standard deviation of the Gaussian (default is 1.0).

    Returns:
    array: Values of the Gaussian function evaluated at x.
    """
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def lorentzian(x, A, x0, gamma):
    """
    Generate a Lorentzian function.

    Parameters:
    - x : array-like
        The input values (independent variable).
    - A : float
        The amplitude (height) of the peak.
    - x0 : float
        The center of the peak.
    - gamma : float
        The full-width at half-maximum (FWHM) of the peak.

    Returns:
    - f : array-like
        The Lorentzian function values.
    """
    return A / (1 + ((x - x0) / (gamma / 2)) ** 2)

def make_bb(wavelengths, temp, normed = 1.0):

    """
    Creates blackbody spectrum

    Parameters
    -----------
    temp: float
        Effective temperature of blackbody in Kelvin
    wavelenths: ndarray
        wavelength array of blackbody in Angstroms
    normed : float
        blackbodies are normalized to 1.0 at this value

    Returns
    -----------
    ndarray:
        flux array of blackbody in ergs / (cm^2 * s * Angstrom)
    """
    h = (const.h).cgs
    c = (const.c).cgs
    k = (const.k_B).cgs
    
    l = u.Quantity(wavelengths, unit=u.angstrom)

    T = u.Quantity(temp, unit=u.K)

    F_lambda = (((2*h*c**2)/l**5) * (1/(np.exp((h*c)/(l*k*T)) - 1)))

    return F_lambda.value * normed 

def sed_integ(w, f):
    """
    Integrate a spectrum f over wavelength w

    Parameters
    -----------
    w: ndarray
        wavelength array in Angstroms
    f: ndarray
        flux array in ergs / (cm^2 * s * Angstrom)
    Returns
    -----------
    float:
        integrated flux in ergs / (cm^2 * s)
    """
    return np.nansum(f) / np.nanmean(np.diff(w))

import globals
globals.initialize()

def fitbb_to_spec(a, mspec):
    """
    Objective function to fit blackbody to m dwarf spectrum

    Parameters
    -----------
    a: float
        blackbody scale factor
    mspec: ndarray
        m dwarf spectrum

    Returns
    -----------
    float:
        sum of absolute differences between blackbody and m dwarf spectrum
    """
    bb = make_bb(WAVELENGTH, 3000) * 1e27 * a
    relevant_w = np.argmin(np.abs(WAVELENGTH - WMAX))
    indices = range(relevant_w-50, relevant_w)
    x = np.abs(((bb[indices] - mspec[indices])).sum())
    return x

def gen_mdspec(mdname, filename, extended=True):
    """
    Loads an m dwarf spectrum, fits a blackbody to extend the spectrum past 9200 A, 
    and saves m dwarf spectrum as .npy file

    Parameters  
    -----------
    mdname: str
        mdwarf spectrum .fits filename
    filename: str
        filename to save .npy spectrum as
    extended: bool
        if True, extend spectrum with blackbody fit

    Returns
    -----------
    None
    """
    mdf = mdwarf_interp(mdname)
    md = mdf(WAVELENGTH)

    if extended:
        amplitude = 1
        res = scipy.optimize.minimize(fitbb_to_spec, [amplitude], args=(3000, md))
        md[WAVELENGTH >= WMAX] = (make_bb(WAVELENGTH, 3000) * 1e27 * res.x)[WAVELENGTH >= WMAX]

    np.save(filename, md)

def compspec(temp, md, ff, balmer_ratio = 1, lorentz_lines=False, linefrac=0.0, band='g'):
    """
    Creates composite blackbody + m dwarf spectrum

    Parameters
    -----------
    temp: int
        blackbody temperature in Kelvin

    mdname: str
        mdwarf spectrum .fits filename

    ff: float
        blackbody fill factor

    Returns
    -----------
    ndarray:
        composite spectrum
    """

    bandf = filt_interp(band=band)(WAVELENGTH)
    maxpt = np.where(bandf == bandf.max())[0][0]
    band_edges = (WAVELENGTH[bandf == bandf[(bandf > 0) * (WAVELENGTH < maxpt)].min()][0], 
                        WAVELENGTH[bandf == bandf[(bandf > 0) * (WAVELENGTH > maxpt)].min()][0])


    bb = make_bb(WAVELENGTH, temp) * globals.BBnorm
    sed_plain = np.copy(bb + md)
    ff = ff / globals.FF #change to "units" of global FF

    balmer_step = np.ones_like(WAVELENGTH, dtype=float)
    balmer_step[WAVELENGTH < 3700] = balmer_ratio

    if lorentz_lines:

        sed_sum = sed_plain[band_edges[0]:band_edges[1]].sum()
        #print('Sum under blackbody = {}'.format(sed_sum))

        lf = linefrac[0]
        l = lorentzian(WAVELENGTH, *linedict[linenames[0]]) +  lorentzian(WAVELENGTH, *linedict[linenames[1]])
        lnew = l * (sed_sum / l.sum()) * lf 
        bb += lnew
        #print('Sum under Ca lines = {0} ({1}% of blackbody)'.format(lnew.sum(), (lnew.sum() / sed_sum)*100))

        lf = linefrac[1]
        l = lorentzian(WAVELENGTH, *linedict[linenames[2]]) +  lorentzian(WAVELENGTH, *linedict[linenames[3]]) + lorentzian(WAVELENGTH, *linedict[linenames[4]])
        lnew = l * (sed_sum / l.sum()) * lf 
        bb += lnew
        #print('Sum under H lines = {0} ({1}% of blackbody)'.format(lnew.sum(), (lnew.sum() / sed_sum)*100))

    return md + ((bb * ff**2) * balmer_step)


def filt_interp(band,plotit=False, survey='DES', path=ROOTDIR):

    """
    Imports and interpolates LSST or DES filter

    Parameters
    -----------
    band: string
        which LSST band to use (u,g,r,i,z,y)

    Returns
    -----------
    interp1d(filtx,filty, bound_error = False, fill_value = 0.0) : function
        function that interpolates filtx,filty
    """


    if survey == 'LSST':
        from rubin_sim.photUtils import Bandpass
        lsst = {}
        lsst[band] = Bandpass()
        lsst[band].readThroughput(path + '/baseline/total_' + band + '.dat')
        lsst[band].readThroughput(path + '/baseline/total_' + band + '.dat')

        sb, w = lsst[band].sb, lsst[band].wavelen*10 #scale flux, conv nm to A

        if plotit:
            plt.plot(w,sb)

        return interp1d(w, sb, bounds_error=False, fill_value=0.0)

    if survey == 'DES':
        w = np.loadtxt(ROOTDIR + 'des_g.txt')[:,0]
        sb = np.loadtxt(ROOTDIR + 'des_g.txt')[:,1]

    return interp1d(w, sb, bounds_error=False, fill_value=0.0)

def lamb_eff_md(band, temp, mdpath = ROOTDIR + quiescent_spectranpy["m7"], ff=globals.FF, balmer_ratio = 1.0,
                lorentz_lines=False, linefrac=0.0, WAVELENGTH=WAVELENGTH, 
                compplot=False, ax=None, ax2=None, returnFlux=False):

    """
    Calculates the effective wavelength in Angstroms for md + BB sed

    Parameters
    -----------
    band: string
        which LSST band to use

    temp: float
        BB effective temperature in Kelvin

    mdname: string
       filename of mdwarf spectra .fits file

    ff: float
        filling factor for mdwarf + (BB*ff)

    Returns
    -----------
    float
        effective wavelength in Angstroms
    """

    #Create composite spectrum
    wave = WAVELENGTH
    mdspec = np.load(mdpath, allow_pickle=True)
    mdbb = compspec(temp, md=mdspec, ff=ff, balmer_ratio=balmer_ratio, lorentz_lines=False, linefrac=linefrac)

    #Import filter
    f = filt_interp(band=band)
    interpolated_filt = f(wave)

    #Create left slice ind
    for i,s in enumerate(interpolated_filt):
        if s == 0:
            pass
        else:
            s_left = wave[i]
            break
        
    #Create right slice ind
    for i,s in reversed(list(enumerate(interpolated_filt))):
        if s == 0:
            pass
        else:
            s_right = wave[i]
            break

    #take slice where band is non-zero
    BBleft = np.where(np.abs(wave - s_left) == np.abs(wave - s_left).min())[0][0]
    BBright = np.where(np.abs(wave - s_right) == np.abs(wave - s_right).min())[0][0]
    
    #Slice spectrum
    mdbb_band = mdbb[BBleft:BBright]
    wave_band = wave[BBleft:BBright]

    #if verbose:
        #print("Calculating BB at T = {} K".format(temp))
        
    #Calc effective lambda
    w_eff = calc_w_eff(mdbb_band, interpolated_filt[BBleft:BBright], wave_band)
    #np.exp(np.sum(mdbb_band * interpolated_filt[BBleft:BBright] * np.log(wave_band)) / 
            #       np.sum(mdbb_band * interpolated_filt[BBleft:BBright]))

    if lorentz_lines:

        mdbb_lines = compspec(temp, md=mdspec, ff=ff, balmer_ratio=balmer_ratio, lorentz_lines=lorentz_lines, linefrac=linefrac)
        mdq = compspec(temp=0, md=mdspec, ff=ff, balmer_ratio=balmer_ratio, lorentz_lines=False, linefrac=linefrac)

        mdbb_lines_band = mdbb_lines[BBleft:BBright]
        mdq_band = mdq[BBleft:BBright]

        w_eff_lines = np.exp(np.sum(mdbb_lines_band * interpolated_filt[BBleft:BBright] * np.log(wave_band)) / 
                   np.sum(mdbb_lines_band * interpolated_filt[BBleft:BBright]))
    
        w_effq = calc_w_eff(mdq_band, interpolated_filt[BBleft:BBright], wave_band)
        #np.exp(np.sum(mdq_band * interpolated_filt[BBleft:BBright] * np.log(wave_band)) / 
                 #   np.sum(mdq_band * interpolated_filt[BBleft:BBright]))

        if compplot:
            if not ax:
                fig, axs = plt.subplots(2)
                ax, ax2 = axs
            q_factor = np.nanmax(mdbb_lines) / np.nanmean(mdq)
            print("Quiescent scale factor = {}".format(q_factor))
            ax.plot(WAVELENGTH, mdq * q_factor, label="dM only", color='C3')
            ax.plot(WAVELENGTH, mdbb, label="dM + blackbody", color='C2')
            ax.plot(WAVELENGTH, mdbb_lines, label="dM + blackbody + lines", color='C0')
            ax.vlines(w_effq, -np.nanmax(mdq), np.nanmax(mdbb_lines) * 2, color='C3', ls='--', label=r'$\lambda_\mathrm{eff, quiescent}$')
            ax.vlines(w_eff, -np.nanmax(mdq), np.nanmax(mdbb_lines) * 2, color='C2', ls='--', label=r'$\lambda_\mathrm{eff, flare}$')
            ax.vlines(w_eff_lines, -np.nanmax(mdq), np.nanmax(mdbb_lines) * 2, color='C0', ls='--', label=r'$\lambda_\mathrm{eff, flare + lines}$')
            ax.set_xlabel(r'Wavelength $(\AA)$', fontsize=16)
            ax.set_ylabel(r'$F_\lambda$ (arb. units)', fontsize=16)
            ax.xaxis.set_tick_params(labelsize=12)
            ax.yaxis.set_tick_params(labelsize=12)

            ax2.set_ylabel('Filter Throughput', fontsize=16)
            ax2.tick_params(axis ='y')
            ax2.yaxis.set_tick_params(labelsize=12)
            ax2.plot(WAVELENGTH, interpolated_filt, color='k', alpha=0.4)

            ax.set_xlim(BBleft, BBright)
            ax.set_ylim(None, np.nanmax(mdbb_lines_band))
            #ax.legend()

            if returnFlux:
                g_flux = (np.nansum(interpolated_filt[BBleft:BBright] * mdq_band) / np.nansum(interpolated_filt[BBleft:BBright])) * (2*np.pi*u.sr) * (Rstar.to('cm') **2) / ((dist.to('cm'))**2)
                g_flux_flare = (np.nansum(interpolated_filt[BBleft:BBright] * mdbb_band) / np.nansum(interpolated_filt[BBleft:BBright])) * (2*np.pi*u.sr) * (Rstar.to('cm') **2) / ((dist.to('cm'))**2)
                g_flux_flare_lines = (np.nansum(interpolated_filt[BBleft:BBright] * mdbb_lines_band) / np.nansum(interpolated_filt[BBleft:BBright])) * (2*np.pi*u.sr) * (Rstar.to('cm') **2) / ((dist.to('cm'))**2)
                g_mag = 22.5 - 2.5 * np.log10(g_flux.value)
                g_mag_flare = 22.5 - 2.5 * np.log10(g_flux_flare.value)
                g_mag_flare_lines = 22.5 - 2.5 * np.log10(g_flux_flare_lines.value)
                delta_g = g_mag_flare - g_mag
                delta_g_lines = g_mag_flare_lines - g_mag
        
                return w_eff_lines, w_eff, w_effq, delta_g, delta_g_lines, g_mag, g_mag_flare, g_mag_flare_lines
            
        return w_eff, w_eff_lines, w_effq
    
    else:
        return w_eff

def lamb_eff_BB(band, temp, verbose=False):

    """
    Calculates the effective wavelength in angstroms for blackbody-only sed

    Parameters
    -----------
    band: string
        which LSST band to use

    temp: float
        BB effective temperature in Kelvin

    Returns
    -----------
    float
        effective wavelength in Angstroms
    """

    #Create BB
    BBwave = WAVELENGTH
    BBflux = make_bb(BBwave, temp) / globals.BBnorm

    #Import filter
    f = filt_interp(band=band)
    interpolated_filt = f(BBwave)

    #Create left slice ind
    for i,s in enumerate(interpolated_filt):
        if s == 0:
            pass
        else:
            s_left = BBwave[i]
            break
        
    #Create right slice ind
    for i,s in reversed(list(enumerate(interpolated_filt))):
        if s == 0:
            pass
        else:
            s_right = BBwave[i]
            break

    #take slice where band is non-zero
    BBleft = np.where(np.abs(BBwave - s_left) == np.abs(BBwave - s_left).min())[0][0]
    BBright = np.where(np.abs(BBwave - s_right) == np.abs(BBwave - s_right).min())[0][0]
    
    #Slice BB
    BBfluxc = BBflux[BBleft:BBright]
    BBwavec = BBwave[BBleft:BBright]

    #if verbose:
        #print("Calculating w_eff")
    
    return calc_w_eff(BBfluxc,  interpolated_filt[BBleft:BBright],  BBwavec)

def R0(w_eff):
    #Docstring
    w_effn = np.copy(w_eff) / 1e4 #Convert angstrom to micron

    #Calc index of refr
    n = (10**-6 * (64.328 + (29498.1 / (146-(1/w_effn**2))) + (255.4 / (41 - (1/w_effn**2))))) + 1

    #Calc R_0
    return (n**2 - 1) / (2 * n**2)

def dcr_offset(w_eff, airmass, coord = None, header = None, chrDistCorr=False):

    """
    Calculates the DCR offset in arcsec

    Parameters
    -----------
    w_eff: float
        effective wavelength in angstroms

    airmass: float
        airmass value

    Returns
    -----------
    float
        DCR offset in arcsec
    """

    #w_effn = np.copy(w_eff) / 1e4 #Convert angstrom to micron

    #Calc index of refr
    #n = (10**-6 * (64.328 + (29498.1 / (146-(1/w_effn**2))) + (255.4 / (41 - (1/w_effn**2))))) + 1

    #Calc R_0
    #R_0 = (n**2 - 1) / (2 * n**2)

    R_0 = R0(w_eff)

    Z = np.arccos(1/airmass)

    R = R_0*np.tan(Z)

    if chrDistCorr:

        corr = chrDistCorr(w_eff, coord, header)

        return np.rad2deg(R) * 3600 * corr

    return np.rad2deg(R) * 3600 

def dcr_offset_inverse(w_eff_1, w_eff_0, dcr):
    """
    Given flaring and quiescent effective wavelengths and a delta dcr in arcsec,
    this function calculates the airmass needed to produce observed delta dcr.

    Parameters
    -----------
    w_eff_1: float
        flaring effective wavelength in angstroms
    w_eff_0: float
        quiescent effective wavelength in angstroms
    dcr: float  
        delta dcr in arcsec

    Returns
    -----------
    float
        airmass needed to produce observed delta dcr
    """

    q = np.deg2rad(dcr / 3600)

    R0_1 = R0(w_eff_1)
    R0_0 = R0(w_eff_0)

    z_crit = np.arctan(q / (R0_1 - R0_0))

    return 1 / np.cos(z_crit)


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    #plt.register_cmap(cmap=newcmap)

    return newcmap

def getsciimg(filefracday,paddedfield,filtercode,paddedccdid,imgtypecode,qid):
    """
    Downloads ZTF science image from IPAC
    
    Parameters
    -----------
    filefracday: str
        filefracday string from ZTF filename (e.g. '20210930000234')
    paddedfield: str
        field number, zero-padded to 4 digits (e.g. '0123')
    filtercode: str
        filter code ('zg','zr','zi')
    paddedccdid: str
        ccdid, zero-padded to 2 digits (e.g. '01')
    imgtypecode: str
        image type code ('o','s','c')
    qid: str
        quadrant id ('1','2','3','4')

    Returns
    -----------
    None
    """

    year = filefracday[0:4]
    month = filefracday[4:6]
    day = filefracday[6:8]
    fracday = filefracday[8:]

    url = 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/'+year+'/'+month+day+'/'+fracday+'/ztf_'+filefracday+'_'+paddedfield+'_'+filtercode+'_c'+paddedccdid+'_'+imgtypecode+'_q'+qid+'_'+'sciimg.fits'
    print("Querying: "+url)
    hdu = fits.open(url)
    hdu.writeto('srcext/'+filefracday+'_'+paddedfield+'_sciimg'+'.fits', overwrite=True)

def getrefimg(paddedfield,filtercode,paddedccdid,qid):
    """
    Downloads ZTF reference image from IPAC
    
    Parameters
    -----------
    paddedfield: str
        field number, zero-padded to 4 digits (e.g. '0123')
    filtercode: str
        filter code ('zg','zr','zi')
    paddedccdid: str
        ccdid, zero-padded to 2 digits (e.g. '01')
    qid: str
        quadrant id ('1','2','3','4')

    Returns
    -----------
    None
    """

    fieldprefix = paddedfield[0:3]

    url = 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/ref/'+fieldprefix+'/field'+paddedfield+'/'+filtercode+'/ccd'+paddedccdid+'/q'+qid+'/ztf_'+paddedfield+'_'+filtercode+'_c'+paddedccdid+'_q'+qid+'_refimg.fits'
    print("Querying: "+url)
    hdu = fits.open(url)
    hdu.writeto('srcext/'+str(paddedfield)+'_refimg'+'.fits', overwrite=True)

def srcext(file, det_thresh, ana_thresh, catname):
    """
    Runs SExtractor on a given file to create a catalog.
    
    Parameters
    -----------
    file: str
        Path to the input image file.
    det_thresh: float
        Detection threshold for SExtractor.
    ana_thresh: float
        Analysis threshold for SExtractor.
    catname: str
        Name of the output catalog file.

    Returns
    -----------
    pandas.DataFrame:
        DataFrame containing the SExtractor catalog.
    """

    if os.getcwd().endswith('srcext') == False:
        print('Changing directory to srcext...')
        os.chdir('srcext')

    print('Making SExtractor catalog of '+file+'...')

    if os.path.isfile(catname) == True:
        print('This catalogue already exists, moving on...')
    else:
        os.system('sex ' + file + ' -c default.sex' + ' -DETECT_THRESH ' + str(det_thresh) + ' -ANALYSIS_THRESH ' 
                  + str(ana_thresh) + ' -CATALOG_NAME ' + str(catname))

    cata_df = pd.read_table(catname, names=['NUMBER',
    'X_IMAGE',
    'Y_IMAGE',
    'XWIN_IMAGE',
    'YWIN_IMAGE',
    'XMODEL_IMAGE',
    'YMODEL_IMAGE',
    'FLUX_AUTO',
    'FLUX_MODEL',
    'MAG_AUTO',
    'MAG_MODEL',
    'FLUX_RADIUS',
    'FLAGS',
    'NITER_MODEL',
    'ALPHA_SKY',
    'DELTA_SKY',
    'THETA_WORLD',
    'ELLIPTICITY'], index_col=0, comment='#', delim_whitespace=True)

    os.chdir('..')

    return cata_df

def xmatch(cat1, cat2):
    """
    Cross-matches two catalogs using astropy.coordinates.SkyCoord.
    
    Parameters
    -----------
    cat1: pandas.DataFrame
        First catalog with 'ALPHA_SKY' and 'DELTA_SKY' columns.
    cat2: pandas.DataFrame
        Second catalog with 'ALPHA_SKY' and 'DELTA_SKY' columns.

    Returns
    -----------
    tuple:
        Indices of the matched sources in cat2, on-sky distances, and 3D distances.
    """

    print("Matching catalogs...")
    c1 = SkyCoord(ra=cat1["ALPHA_SKY"]*u.degree, dec=cat1["DELTA_SKY"]*u.degree)
    c2 = SkyCoord(ra=cat2["ALPHA_SKY"]*u.degree, dec=cat2["DELTA_SKY"]*u.degree)

    if len(c1) < len(c2):
        idx, d2d, d3d = c1.match_to_catalog_sky(c2)
    
    else:
        idx, d2d, d3d = c2.match_to_catalog_sky(c1)

    return idx, d2d, d3d

def circle_cut(imgcen_ra, imgcen_dec, cat, radius):
    """
    Returns indices from cat within radius of image center

    Parameters
    -----------
    imgcen_ra: float
        Image center right ascension in degrees
    imgcen_dec: float
        Image center declination in degrees
    cat: pandas.DataFrame
        Catalog with 'ra' and 'dec' columns in degrees
    radius: astropy.units.Quantity
        Radius for circular cut (e.g., 0.5*u.degree)

    Returns
    -----------
    numpy.ndarray:
        Boolean array indicating which sources are within the specified radius

    """
    
    c1 = SkyCoord(ra=imgcen_ra*u.degree, dec=imgcen_dec*u.degree)
    c2 = SkyCoord(ra=cat["ra"].values*u.degree, dec=cat["dec"].values*u.degree)
    ind = c1.separation(c2) < radius

    return ind

def calc_zenith(date, site):
    """
    Calculates the zenith position at a given date and location 
    Parameters
    -----------
    date: float 
        MJD date
    site: str
        site name (e.g. 'Cerro Pachon')

    Returns
    -----------
    astropy.coordinates.SkyCoord:
        ICRS coordinates of the zenith at the given date and location
    """

    mtn = EarthLocation.of_site(site)
    mjd = Time(date, format='mjd')

    zenith = SkyCoord(AltAz(alt=90 * u.degree, az=0 * u.degree, obstime = mjd, location=mtn)).transform_to(ICRS())

    return zenith

def plot_shifts(ref_ra, ref_dec, sci_ra, sci_dec, zen_ra, zen_dec, flr_ind, mjd, am, centered=False):
    """
    Plots the DCR shifts of sources in RA and Dec
    
    Parameters  
    -----------
    ref_ra: ndarray
        reference catalog right ascensions in degrees
    ref_dec: ndarray
        reference catalog declinations in degrees
    sci_ra: ndarray
        science catalog right ascensions in degrees
    sci_dec: ndarray
        science catalog declinations in degrees
    zen_ra: float
        zenith right ascension in degrees
    zen_dec: float
        zenith declination in degrees
    flr_ind: int
        index of the flare star in the catalog
    mjd: float
        MJD date of observation
    am: float
        airmass of observation
    centered: bool
        if True, center axes on centroid of points; if False, center on (0,0)

    Returns
    -----------
    None
    """

    #Calc delta coords
    d_ra = (ref_ra - sci_ra) * 3600
    d_dec = (ref_dec - sci_dec) * 3600

    #Calc zenith delta coord
    d_zra = (ref_ra.mean() - zen_ra) * 3600
    d_zdec = (ref_dec.mean() - zen_dec) * 3600

    #Calc centroid
    centroid = (sum(d_ra) / len(d_ra), sum(d_dec) / len(d_dec))

    #Plotting
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(d_ra, d_dec, alpha=0.75)
    ax.scatter(d_ra[flr_ind], d_dec[flr_ind], color='red', s=100, marker='*', label='Flare star')

    if not centered:
        ax.scatter(centroid[0], centroid[1], color='k', marker='x', label="Centroid")

    ax.plot([0,d_zra],[0,d_zdec], c='gray',ls='--', label="to zenith")

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)

    if centered:
        ax.spines['left'].set_position(('data', centroid[0]))
        ax.spines['bottom'].set_position(('data', centroid[1]))
    else:
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xlabel(r"$\Delta$ RA (arcsec)", labelpad=150)
    ax.set_ylabel(r"$\Delta$ Dec (arcsec)", labelpad=150)
    ax.set_xticks([-0.4,-0.2,0.2,0.4])
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)
    ax.set_title("MJD: {0}, Airmass = {1}".format(mjd, am))
    ax.legend()
    ax.grid(False)
    
    plt.gca().set_aspect('equal')

def pa(h, phi, d):

    '''
    PA equation from Astronomical Algorithms

    Parameters
    -------------
    h: float
        hour angle in hours
    phi: float
        Geographic latitude of observatory in degrees
    d: float
        Declination in degrees

    Returns
    -------------
    float
        Parallactic angle in degrees
    '''

    q = np.arctan2(np.sin(h * ha2deg * deg2rad), \
        np.cos(d * deg2rad) * np.tan(phi * deg2rad) - \
        np.sin(d * deg2rad) * np.cos(h * ha2deg * deg2rad))

    return q / deg2rad

def celest_to_pa(ra, dec, time, loc, delra=None, deldec=None, round_lmst = False, verbose = False):

    '''
    Convert celestial coordinates to a parallactic angle given
    a observation time and observatory location

    Parameters
    -------------
    ra: float
        Right Ascension in degrees
    dec: float
        Declination in degrees
    time: float
        astropy.time.Time object
    location: astropy.coordinates.EarthLocation object
        EarthLocation object of observing site

    Returns
    -------------
    astropy.Quantity object
        Parallactic angle quantity
    '''

    t = time
    lat = loc.lat.deg
    lon = loc.lon.deg
    scoord = SkyCoord(ra=ra * u.deg, dec = dec * u.deg)
    lst = t.sidereal_time('mean', longitude=lon)

    if round_lmst:
        lst = (lst * 60).round() / 60

    ha = lst.hour - scoord.ra.hour

    if verbose:
        print('Location = Lon:{0:.3f}, Lat:{1:.3f}'.format(loc.lon, loc.lat))
        print('RA = {0}, Dec = {1}'.format(scoord.ra.hms, scoord.dec.dms))
        print('time = {}'.format(t))
        print('LMST = {}'.format(lst.hms))
        print('ha = {}'.format(ha))

    if delra is not None and deldec is not None:

        delh = ha2deg * delra
        dpa = pa_error(ha, dec, loc.lat.deg, delh, deldec)

        return pa(ha, lat, dec), dpa
    
    else:

        return pa(ha, lat, dec) 

def celest_to_ha(ra, dec, time, loc, round_lmst = False, verbose = False):

    '''
    Convert celestial coordinates to a parallactic angle given
    a observation time and observatory location

    Parameters
    -------------
    ra: float
        Right Ascension in degrees
    dec: float
        Declination in degrees
    time: float
        astropy.time.Time object
    location: astropy.coordinates.EarthLocation object
        EarthLocation object of observing site

    Returns
    -------------
    astropy.Quantity object
        Parallactic angle quantity
    '''

    t = time
    lat = loc.lat.deg
    lon = loc.lon.deg
    scoord = SkyCoord(ra=ra * u.deg, dec = dec * u.deg)
    lst = t.sidereal_time('mean', longitude=lon)

    if round_lmst:
        lst = (lst * 60).round() / 60

    ha = lst.hour - scoord.ra.hour

    if verbose:
        print('Location = Lon:{0:.3f}, Lat:{1:.3f}'.format(loc.lon, loc.lat))
        print('RA = {0}, Dec = {1}'.format(scoord.ra.hms, scoord.dec.dms))
        print('time = {}'.format(t))
        print('LMST = {}'.format(lst.hms))
        print('ha = {}'.format(ha))
    return ha

def dpar(dra, ddec, pa2, delra = None, deldec = None, delpa2 = None):

    '''
    Compute component of positional offset parallel to zenith direction

    Parameters
    -------------
    dra: float
        Change in right Ascension in degrees
    ddec: float
        Change in declination in degrees
    pa2: float
        Parallactic angle of second position in degreees

    Returns
    -------------
    float 
        zenith-parallel component in degree
    '''

    dparallel = np.sqrt(dra**2 + ddec**2) * np.cos((np.pi/2) - np.deg2rad(pa2) - np.arctan2(ddec, dra))

    if delra is not None and deldec is not None and delpa2 is not None:

        dparallel_err, ddra, dddec, ddpar2 = dpar_error(dra, ddec, pa2, delra, deldec, delpa2)

        return dparallel, dparallel_err, ddra, dddec, ddpar2
    
    else:

        return dparallel

def dtan(dra, ddec, pa2):

    '''
    Compute component of positional offset perpendicular to zenith direction

    Parameters
    -------------
    dra: float
        Change in right Ascension in degrees
    ddec: float
        Change in declination in degrees
    pa2: float
        Parallactic angle of second position in degreees

    Returns
    -------------
    float 
        zenith-parallel component
    '''

    return np.sqrt(dra**2 + ddec**2) * np.sin((np.pi/2) - np.deg2rad(pa2) - np.arctan(ddec/dra))

def gcd(lat1, lat2, lon1, lon2, haversine=False):
    """
    Calculate the great-circle distance between two points on the Earth specified in radians.
    
    Parameters
    -----------
    lat1: float
        Latitude of the first point in radians
    lat2: float
        Latitude of the second point in radians
    lon1: float
        Longitude of the first point in radians
    lon2: float
        Longitude of the second point in radians
    haversine: bool, optional
        If True, use the haversine formula; otherwise, use the spherical law of cosines. Default is False.

    Returns
    -----------
    float
        Great-circle distance in radians
    """
    dlat = np.abs(lat2 - lat1)
    dlon = np.abs(lon2 - lon1)
    dsig = np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon))
    
    if haversine:
        dsig = 2 * np.arcsin(np.sqrt(np.sin(dlat / 2))**2 + (1 - np.sin(dlat/2)**2 - np.sin((lat1 + lat2) / 2)**2) * np.sin(dlon / 2)**2)
        
    return dsig

def pa_error(h, dec, phi, dh, ddec):
    """
    Calculate the error in parallactic angle given errors in hour angle and declination
    
    Parameters
    -----------
    h: float
        hour angle in hours
    dec: float
        declination in degrees
    phi: float
        latitude in degrees
    dh: float
        error in hour angle in hours
    ddec: float
        error in declination in degrees

    Returns
    -----------
    float
        error in parallactic angle in degrees
    """

    h = h * ha2deg * deg2rad
    dec = dec * deg2rad
    phi = phi * deg2rad
    dh = dh * deg2rad
    ddec = ddec * deg2rad

    dPdh = (-(np.cos(dec) * np.cos(h) * np.tan(phi)) + (np.sin(dec) * np.sin(h)**2) + (np.sin(dec) * np.cos(h)**2)) / ((-2 * np.sin(dec) * np.cos(dec) * np.cos(h) * np.tan(phi)) + (np.sin(dec)**2 * np.cos(h)**2) + (np.cos(dec)**2 * np.tan(phi)**2) + np.sin(h)**2)
    
    dPdd = (np.sin(h) * (np.cos(h) * np.cos(dec) + np.sin(dec) * np.tan(phi))) / ((np.cos(h) * np.sin(dec) - np.cos(dec) * np.tan(phi))**2 + np.sin(h)**2)

    err = np.sqrt( (dPdh * np.rad2deg(dh))**2 + (dPdd * np.rad2deg(ddec))**2 )

    return err

def dpar_error(dra, ddec, pa2, delra, deldec, delpa2):
    """
    Calculate the error in the zenith-parallel component of positional offset given errors in RA, Dec, and PA
    
    Parameters
    -----------
    dra: float
        Change in right Ascension in degrees
    ddec: float
        Change in declination in degrees
    pa2: float
        Parallactic angle of second position in degrees
    delra: float
        Error in change in right Ascension in degrees
    deldec: float
        Error in change in declination in degrees
    delpa2: float
        Error in parallactic angle of second position in degrees

    Returns
    -----------
    float
        Error in zenith-parallel component of positional offset in degrees
    """

    ddpar_ddra = (dra * np.sin(np.arctan(ddec/dra) + np.deg2rad(pa2)) - ddec * np.cos(np.arctan(ddec/dra) + np.deg2rad(pa2))) / np.sqrt(dra**2 + ddec**2)
    ddpar_dddec = (ddec * np.sin(np.arctan(ddec/dra) + np.deg2rad(pa2)) + dra * np.cos(np.arctan(ddec/dra) + np.deg2rad(pa2))) / np.sqrt(dra**2 + ddec**2)
    ddpar_dpa2 = np.sqrt(dra**2 + ddec**2) * np.cos(np.arctan(ddec/dra) + np.deg2rad(pa2))

    err = np.sqrt( (ddpar_ddra * delra)**2 + (ddpar_dddec * deldec)**2 + (ddpar_dpa2 * delpa2)**2 )

    return err, ddpar_ddra, ddpar_dddec, ddpar_dpa2

def obj2(T, weff, ff, linefrac):
    """
    Objective function to minimize the difference between a given weff and the weff calculated from a given Teff

    Parameters
    -----------
    T: float
        Effective temperature in Kelvin
    weff: float
        Effective wavelength in Angstroms
    ff: float
        Filling factor
    linefrac: list
        List of two floats representing the fraction of the flare energy in H and Ca lines, respectively

    Returns
    -----------
    float
        Absolute difference between the given weff and the weff calculated from the given Teff
    """

    weff_test,_ ,_ = lamb_eff_md(band = 'g', temp = T, ff = ff, mdpath = 'sdsstemplates/m7.active.ha.na.k_ext.npy', 
                            lorentz_lines=True, linefrac=linefrac)
    
    return abs(weff - weff_test)

def obj1(weff, dpar_0, dcr_q, airmass, batoid_a, batoid_b, batoid_c, theta):
    """
    Objective function to minimize the difference between a calculated dpar and a given dpar
    Parameters
    -----------
    weff: float
        Effective wavelength in Angstroms
    dpar_0: float
        Given dpar value in arcseconds
    dcr_q: float
        Quiescent dcr value in arcseconds
    airmass: float
        Airmass value
    batoid_a: float
        Batoid parameter a
    batoid_b: float
        Batoid parameter b
    batoid_c: float
        Batoid parameter c
    theta: float
        Parallactic angle in degrees

    Returns
    -----------
    float
        Absolute difference between the calculated dpar and the given dpar
    """
    
    R_0 = R0(weff)
    Z = np.arccos(1/airmass)

    dpar = ((np.rad2deg(R_0 * np.tan(Z)) * 3600 - dcr_q) - (((batoid_a * weff**2) + (batoid_b * weff) + batoid_c) * np.cos(theta)))
  
    return abs(dpar - dpar_0)
    
Nfeval = 1

def callbackF(Xi):
    """
    Callback function for the optimization process to print the current evaluation number and parameters.

    Parameters
    -----------
    Xi: ndarray
        Current parameters being evaluated in the optimization process.
    """
    global Nfeval
    print(Nfeval, obj1(Xi), Xi)
    Nfeval += 1

def inverse_Teff(delta_dcr, quiescent_dcr, airmass, ff, source_coord=None, source_header=None, callback=False, return_weff = False, linefrac=[0.0, 0.0]):
    """
    Inverse calculation of effective temperature from delta DCR, quiescent DCR, airmass, filling factor, and source coordinates.

    Parameters
    -----------
    delta_dcr: float
        Delta DCR in arcseconds
    quiescent_dcr: float
        Quiescent DCR in arcseconds
    airmass: float
        Airmass value
    ff: float
        Filling factor
    source_coord: astropy.coordinates.SkyCoord, optional
        Source coordinates (RA, Dec) in degrees. If not provided, theta will be calculated from source_header.
    source_header: astropy.io.fits.Header, optional
        FITS header containing 'CENTRA' and 'CENTDEC' for calculating theta. Required if source_coord is not provided.
    callback: bool, optional
        If True, enables callback function to print optimization progress.
    return_weff: bool, optional
        If True, returns the effective wavelength along with the temperature.
    linefrac: list, optional
        List of two floats representing the fraction of the flare energy in H and Ca lines, respectively.

    Returns
    -----------
    float or tuple:
        If return_weff is True, returns a tuple containing the effective temperature in Kelvin, effective wavelength in angstroms, and theta in degrees.
        If return_weff is False, returns only the effective temperature in Kelvin.
    """
    
    dpar_0 = delta_dcr
    dcr_q = quiescent_dcr
    batoid_a, batoid_b, batoid_c = batoid_params
    theta = chrDistAng(source_coord, source_header)

    init_guess_1 = 4500
    if callback:
        result_1 =  minimize(obj1, init_guess_1, args = (dpar_0, dcr_q, airmass, batoid_a, batoid_b, batoid_c, theta), callback=callbackF, method='Nelder-Mead', options = {'gtol':1e-3})
    else:
        result_1 =  minimize(obj1, init_guess_1, args = (dpar_0, dcr_q, airmass, batoid_a, batoid_b, batoid_c, theta), method='Nelder-Mead', options = {'gtol':1e-3})
    weff = result_1.x

    init_guess_2 = 2800.0
    if callback:
        result_2 = minimize(obj2, init_guess_2, args=(weff, ff, linefrac), callback=callbackF, method='Nelder-Mead', options = {'disp':True, 'gtol':1e-2})
    else:
        result_2 = minimize(obj2, init_guess_2, args=(weff, ff, linefrac), method='Nelder-Mead', options = {'gtol':1e-2})

    if return_weff:
        return result_2.x, weff, theta
    else:
        return result_2.x
    
def inverseWeff(delta_dcr, quiescent_dcr, airmass, theta):
    """
    Inverse calculation of effective wavelength from delta DCR, quiescent DCR, airmass, and theta.

    Parameters
    -----------
    delta_dcr: float
        Delta DCR in arcseconds
    quiescent_dcr: float
        Quiescent DCR in arcseconds
    airmass: float
        Airmass value
    theta: float
        Zenith-anti field center angle in degrees

    Returns
    -----------
    float:
        Effective wavelength in angstroms.
    """
    
    dpar_0 = delta_dcr
    dcr_q = quiescent_dcr
    batoid_a, batoid_b, batoid_c = batoid_params

    init_guess_1 = 4500
    result =  minimize(obj1, init_guess_1, args = (dpar_0, dcr_q, airmass, batoid_a, batoid_b, batoid_c, theta), method='Nelder-Mead', options = {'gtol':1e-3})

    return result.x

def inverseTeff(weff, ff, linefrac, callback=False):
    """
    Inverse calculation of effective temperature from effective wavelength, filling factor, and line fractions.
    Unlike inverse_Teff, this function takes effective wavelength as input, skipping the weff calculation step.

    Parameters
    -----------
    weff: float
        Effective wavelength in angstroms
    ff: float
        Filling factor
    linefrac: list
        List of two floats representing the fraction of the flare energy in H and Ca lines, respectively.

    Returns
    -----------
    float:
        Effective temperature in Kelvin.
    """

    init_guess_2 = 2800.0
    if callback:
        result_2 = minimize(obj2, init_guess_2, args=(weff, ff, linefrac), callback=callbackF, method='Nelder-Mead', options = {'disp':True, 'gtol':1e-2})
    else:
        result_2 = minimize(obj2, init_guess_2, args=(weff, ff, linefrac), method='Nelder-Mead', options = {'gtol':1e-2})
    
    return result_2.x

###DMTN-037 refraction calculations

def R(l, Z):
    """
    Calculate the refraction angle in arcseconds for a given wavelength and zenith angle.

    Parameters
    -----------
    l: float
        Wavelength in angstroms
    Z: float
        Zenith angle in degrees

    Returns
    -----------
    float:
        Refraction angle in arcseconds
    """

    chi = CHI
    beta = BETA
    n = n_0(l)

    return (chi * (n - 1) * (1 - beta) * np.tan(np.deg2rad(Z)) - chi * (1 - n) * (beta - ((n - 1) / 2)) * np.tan(np.deg2rad(Z))**3) * 3600

def n_0(l):
    """
    Calculate the refractive index of air at a given wavelength.

    Parameters
    -----------
    l: float
        Wavelength in angstroms

    Returns
    -----------
    float:
        Refractive index of air
    """

    sigma = 1e4 / l
    dn_s = (2371.34 + (683939.7 / (130 - sigma**2)) + (4547.3 / (38.9 - sigma**2))) * D_S * 1e-8
    dn_w = (6487.31 + 58.058 * sigma**2 - 0.7115 * sigma**4 + 0.08851 * sigma**6) * D_W * 1e-8

    return 1 + dn_s + dn_w

###

def chrDistAng(coord, header):
    """
    Calculate the zenith-anti field center angle for a given source coordinate and header.

    Parameters
    -----------
    coord: astropy.coordinates.SkyCoord
        Source coordinates (RA, Dec) in degrees.
    header: astropy.io.fits.Header
        FITS header containing 'CENTRA' and 'CENTDEC' for calculating the zenith-anti field center angle.

    Returns
    -----------
    float:
        Zenith-anti field center angle in radians.
    """

    source = np.array([coord.ra.value, coord.dec.value])
    zenith = np.zeros_like(source)
    center = np.zeros_like(source)

    zenith[0] = SkyCoord(AltAz(alt=90 * u.degree, az=0 * u.degree, obstime = coord.obstime, location=EarthLocation.of_site('Cerro Tololo'))).transform_to(ICRS()).ra.value
    zenith[1] = SkyCoord(AltAz(alt=90 * u.degree, az=0 * u.degree, obstime = coord.obstime, location=EarthLocation.of_site('Cerro Tololo'))).transform_to(ICRS()).dec.value
    center[0] = header['CENTRA']
    center[1] = header['CENTDEC']

    a = gcd(np.deg2rad(center[1]), np.deg2rad(zenith[1]), np.deg2rad(center[0]), np.deg2rad(zenith[0]))
    b = gcd(np.deg2rad(source[1]), np.deg2rad(center[1]), np.deg2rad(source[0]), np.deg2rad(center[0]))
    c = gcd(np.deg2rad(zenith[1]), np.deg2rad(source[1]), np.deg2rad(zenith[0]), np.deg2rad(source[0]))

    A = np.arccos( (np.cos(a) - np.cos(b) * np.cos(c))  / (np.sin(b) * np.sin(c)) ) 

    theta = np.pi - A

    return theta

def chrDistCorr(wavelength, coord, header):
    """
    Calculate the chromatic distortion correction for a given wavelength, source coordinate, and header.
    
    Parameters
    -----------
    wavelength: float
        Wavelength in angstroms.
    coord: astropy.coordinates.SkyCoord
        Source coordinates (RA, Dec) in degrees.
    header: astropy.io.fits.Header
        FITS header containing field center coordinates 'CENTRA' and 'CENTDEC'
    
    Returns
    -----------
    float:
        Chromatic distortion correction in arcseconds.
    """

    source = np.array([coord.ra.value, coord.dec.value])
    zenith = np.zeros_like(source)
    center = np.zeros_like(source)

    zenith[0] = SkyCoord(AltAz(alt=90 * u.degree, az=0 * u.degree, obstime = coord.obstime, location=EarthLocation.of_site('Cerro Tololo'))).transform_to(ICRS()).ra.value
    zenith[1] = SkyCoord(AltAz(alt=90 * u.degree, az=0 * u.degree, obstime = coord.obstime, location=EarthLocation.of_site('Cerro Tololo'))).transform_to(ICRS()).dec.value
    center[0] = header['CENTRA']
    center[1] = header['CENTDEC']

    a = gcd(np.deg2rad(center[1]), np.deg2rad(zenith[1]), np.deg2rad(center[0]), np.deg2rad(zenith[0]))
    b = gcd(np.deg2rad(source[1]), np.deg2rad(center[1]), np.deg2rad(source[0]), np.deg2rad(center[0]))
    c = gcd(np.deg2rad(zenith[1]), np.deg2rad(source[1]), np.deg2rad(zenith[0]), np.deg2rad(source[0]))

    A = np.arccos( (np.cos(a) - np.cos(b) * np.cos(c))  / (np.sin(b) * np.sin(c)) ) 

    theta = np.pi - A

    pixpermm = 153 / 2.3
    arcsecperpix = 0.2637

    new_w = np.arange(batoid_trace[0][0],batoid_trace[0][-1],1)
    f = interp1d(x = batoid_trace[0], y=batoid_trace[1] * 1e-3 * pixpermm * arcsecperpix, kind='quadratic')

    dist_mag = f(new_w)[np.where(abs(new_w - wavelength) == abs(new_w - wavelength).min())[0]]

    return dist_mag * np.cos(theta)
