# Sylvia Dee
# PSM d18O Speleothem Calcite
# SENSOR SUB-MODEL
# Created 07/02/2014 <sdee@usc.edu>
# Modified 10/28/2014 <julieneg@usc.edu>

# import necessary modules
import numpy as np
import scipy.integrate as si
from scipy.special import erf
from scipy.signal import butter, lfilter, filtfilt
import p2k


def butter_lowpass(cutoff, fs, order=3):
    """ butter lowpass

    Created on Tue Nov  4 14:23:45 2014
    @author: jeg
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def bwf_filter(x, cutoff, fs=1.0, order=3):
    """ butter lowpass

    Created on Tue Nov  4 14:23:45 2014
    @author: jeg
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    
    m = np.mean(x)
    y = filtfilt(b, a, x-m) + m
    return y


def adv_disp_transit(tau,tau0,Pe):
    #  advection-dispersion model [3]
    z0 = -0.5*np.sqrt(Pe*tau/tau0)   #(Eq 12a)
    zL = np.sqrt(Pe/tau*tau0) + z0   #(Eq 12b)
    h  = np.sqrt(1/(Pe*4*np.pi*tau/tau0))*np.exp(-0.25*Pe*tau/tau0)*(1 - np.exp(z0**2-zL**2)) + 0.25*(erf(zL) - erf(z0))  # Eq(11)
    return h


def speleo_sensor(year, d18Op, tas, pr, model='Adv-Disp', tau0=0.5, Pe=1.0):
    '''
    Speleothem Calcite [Sensor] Model
    Converts environmental signals to d18O calcite/dripwater using
    various models of karst aquifer recharge and calcification.
    INPUTS:
        year (1d array: time): time axis [year in float]
        d18Op (1d array: time): d18O of precipitation [permil]
        pr (2d array: location, time): precipitation rate [kg m-2 s-1]
        # dt        time step [years]                         (int or float, 1=1 year, 1./12. for months. etc.)
        # T        Average Annual Temperature     [K]         (numpy array, length n)
        # d18O    delta-18-O (precip or soil water) [permil]  (numpy array, length n)
        NB: make sure these are precipitation weighted
    MODEL PARAMETERS
    ================
        model:  aquifer recharge model. possible values 'Well-Mixed'[1,2] or 'Adv-Disp' [3]
        tau0: mean transit time, [years]    (default = 0.5)
        Pe: Peclet number [non-dimensional] (default = 1.0) ('Adv-Disp' only)
     OUTPUTS:
         d18Oc  cave dripwater delta-18-O
         d18OK  delta-18-O of water after passage through karst
         h      transit time distribution
    REFERNCES:
        [1] Gelhar and Wilson: Ground Water Quality Modeling, Ground Water, 12(6):399--408
        [2] Partin et al., Geology, 2013, doi:10.1130/G34718.1  (SI)
        [3] Kirchner, J. W., X. Feng, and C. Neal (2001), Catchment-scale advection and dispersion as a mechanism for fractal scaling in stream tracer concentrations, Journal of Hydrology, 254 (1-4), 82-101, doi:10.1016/S0022-1694(01)00487-5.
        [4] Wackerbarth et al. Modelling the d18O value of cave drip water and speleothem calcite, EPSL, 2010    doi:10.1016/j.epsl.2010.09.019
    '''
    #======================================================================
    # 1. Karst Model
    #======================================================================
    # ensure that tau0 is scaled correctly
    # (n.b. if in units of months, scale to years, *1/12)
    d18O, year_int = p2k.annualize(d18Op, year, weights=pr)

    dt = np.mean(np.diff(year))

    T, year_int = p2k.annualize(tas, year)

    tau0=tau0*dt

	# Check to make sure mean residence time does not exceed 0.5 x [time series length]
	# It makes little sense to compute mixing with a long residence time with an extremely short timeseries.

    timeseries_len=len(d18O)*dt
    if (tau0>=(0.5*timeseries_len)):
        print("Warning: mean residence time (tau0) is too close to half of the length of the timeseries.")

	# establish kernel length for transit time distribution
	# tau=np.arange(20.*tau0) {SDEE: revised 11/17/15}

    h_s=1E-4						# ensure kernel approaches 0
    tau_s=-tau0*np.log(tau0*h_s)
    tau=np.arange(tau_s)

	# np.convolve method will not function correctly if the length of the kernel approaches that of the series.

    if (len(tau)>=(0.5*timeseries_len)):
        print("Error: kernel length is too close to half of the length of the timeseries. This will result in spurious numerical effects: the effect of a kernel of length L is roughly felt over 2*L+1. Consider revising.")

    # define the transit time distribution h(tau) owing to several models
    if model == 'Well-Mixed':  # well-mixed reservoir model as in [1,2]
         h = (1/tau0) * np.exp(-tau/tau0)
    elif model == 'Adv-Disp':  #  advection-dispersion model [3]
        h = adv_disp_transit(tau,tau0,Pe)
        h[0] = adv_disp_transit(1e-3,tau0,Pe) # fix the pathological case at tau = 0
    else:
        print("Error: model " + model + " not found")

    # obtain normalization constant using Simpson's rule
    hint = si.simps(h,tau)
    # normalize transit time distribution
    h = h/hint

    # Remove mean to avoid boundary effects. (then put it back)
    d18Om = np.mean(d18O)
    #convolve d18O input with the transit time distribution (Green's function of the problem)
    # this yields the d18O of aquifer water, after passage through the karst
    d18OK = np.convolve(h,d18O - d18Om,mode='same') + d18Om

    #======================================================================
    #   2. Dripwater Model
    #======================================================================
    # NB: right now this is only temperature-Dependent Fractionation, but one could
    #    add any number of relevant processes here, as in [4]
    # NB: we assume yearly data, so T is the yearly average temperature for each time step.
    #   the "average temperature" used for the thermal fractionation calculation is
    #   somewhat arbitrarily the decadal average.
    #
    avg_period = 10.0
    Tm = np.mean(T)  # extract mean
    # lowpass filter the  temperature
    Tl = bwf_filter(T-Tm,1.0/avg_period) + Tm
    # apply thermal fractionation (Eq 11 in [4])
    d18Oc = (d18OK + 1000)/1.03086*np.exp(2780/Tl**2 - 0.00289)-1000.0

    return d18Oc, d18OK, h