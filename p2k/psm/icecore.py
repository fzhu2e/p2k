#!/usr/bin/env python3
''' Icecore PSM

Adapted from Sylvia's PRYSM code (https://github.com/sylvia-dee/PRYSM) with precipitation weighting added.
'''

import numpy as np
import p2k
from scipy import integrate, signal
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm

#  import time
#  from IPython import embed


def ice_sensor(year, d18Op, pr, alt_diff=0.):
    ''' Icecore sensor model

    The ice core sensor model calculates precipitation-weighted del18OP (i.e. isotope ratio is weighted by
    the amount of precipitation that accumulates) and corrects for temperature and altitude bias between model
    and site ([Yurtsever, 1975], 0.3/100m [Vogel et al., 1975]).

    Args:
        year (1d array: time): time axis [year in float]
        d18Op (2d array: location, time): d18O of precipitation [permil]
        pr (2d array: location, time): precipitation rate [kg m-2 s-1]
        alt_diff 12d array: location): actual Altitude-Model Altitude [meters]

    Returns:
        d18Oice (2d array: location, year in int): annualizd d18O of ice [permil]

    References:
        Yurtsever, Y., Worldwide survey of stable isotopes in precipitation., Rep. Isotope Hydrology Section, IAEA, 1975.

    '''
    # Altitude Effect: cooling and precipitation of heavy isotopes.
    # O18 ~0.15 to 0.30 permil per 100m.

    alt_eff = -0.25
    alt_corr = (alt_diff/100.)*alt_eff

    d18Op_weighted, year_int = p2k.annualize(d18Op, year, weights=pr)
    d18O_ice = d18Op_weighted + alt_corr

    return d18O_ice


def diffusivity(rho, T=250, P=0.9, rho_d=822, b=1.3):
    '''
    DOCSTRING: Function 'diffusivity'
    Description: Calculates diffusivity (in m^2/s) as a function of density.

    Inputs:
    P: Ambient Pressure in Atm
    T: Temperature in K
    rho: density profile (kg/m^3)
    rho_d: 822 kg/m^2 [default], density at which ice becomes impermeable to diffusion

    Defaults are available for all but rho, so only one argument need be entered.

    Note values for diffusivity in air:

    D16 = 2.1e-5*(T/273.15)^1.94*1/P
    D18 = D16/1.0285
    D2 = D16/1.0251
    D17 = D16/((D16/D18)^0.518)

    Reference: Johnsen et al. (2000): Diffusion of Stable isotopes in polar firn and ice:
    the isotope effect in firn diffusion

    '''

    # Set Constants

    R = 8.314478                                                # Gas constant
    m = 18.02e-3                                                # molar weight of water (in kg)
    alpha18 = np.exp(11.839/T-28.224e-3)                    # ice-vapor fractionation for oxygen 18
    p = np.exp(9.5504+3.53*np.log(T)-5723.265/T-0.0073*T)     # saturation vapor pressure
    Po = 1.                                                 # reference pressure, atmospheres
    rho_i = 920.  # kg/m^3, density of solid ice

    # Set diffusivity in air (units of m^2/s)

    Da = 2.1e-5*np.power((T/273.15), 1.94)*(Po/P)
    Dai = Da/1.0285

    # Calculate Tortuosity

    invtau = np.zeros(len(rho))
    #  for i in range(len(rho)):
    #      if rho[i] <= rho_i/np.sqrt(b):
    #          # invtau[i]=1.-1.3*np.power((rho[i]/rho_d),2)
    #          invtau[i] = 1.-1.3*np.power((rho[i]/rho_i), 2)
    #      else:
    #          invtau[i] = 0.

    selector =rho <= rho_i/np.sqrt(b)
    invtau[selector] = 1.-1.3*(rho[selector]/rho_i)**2

    D = m*p*invtau*Dai*(1/rho-1/rho_d)/(R*T*alpha18)

    return D


def densification(Tavg, bdot, rhos, z):  # ,model='hljohnsen'):
    ''' Calculates steady state snow/firn depth density profiles using Herron-Langway type models.



    Args:
        Tavg: 10m temperature in celcius ## CELCIUS!
        bdot: accumulation rate in mwe/yr or (kg/m2/yr)
        rhos: surface density in kg/m3
        z: depth in true_metres

        model can be: {'HLJohnsen' 'HerronLangway' 'LiZwally' 'Helsen' 'NabarroHerring'}
        default is herronlangway. (The other models are tuned for non-stationary modelling (Read Arthern et al.2010 before applying in steady state).

    Returns:
        rho: density (kg/m3) for all z-values.
        zieq: ice equivalent depth for all z-values.
        t: age for all z-values (only taking densification into account.)

        Example usage:
        z=0:300
        [rho,zieq,t]=densitymodel(-31.5,177,340,z,'HerronLangway')
        plot(z,rho)

    References:
        Herron-Langway type models. (Arthern et al. 2010 formulation).
        Aslak Grinsted, University of Copenhagen 2010
        Adapted by Sylvia Dee, Brown University, 2017
        Optimized by Feng Zhu, University of Southern California, 2017

    '''
    rhoi = 920.
    rhoc = 550.
    rhow = 1000.
    rhos = 340.
    R = 8.314

    # Tavg=248.
    # bdot=0.1
    # Herron-Langway with Johnsen et al 2000 corrections.
    # Small corrections to HL model which are not in Arthern et al. 2010

    c0 = 0.85*11*(bdot/rhow)*np.exp(-10160./(R*Tavg))
    c1 = 1.15*575*np.sqrt(bdot/rhow)*np.exp(-21400./(R*Tavg))

    k0 = c0/bdot  # ~g4
    k1 = c1/bdot

    # critical depth at which rho=rhoc
    zc = (np.log(rhoc/(rhoi-rhoc))-np.log(rhos/(rhoi-rhos)))/(k0*rhoi)  # g6

    ix = z <= zc  # find the z's above and below zc
    upix = np.where(ix)  # indices above zc
    dnix = np.where(~ix)  # indices below zc

    q = np.zeros((z.shape))  # pre-allocate some space for q, rho
    rho = np.zeros((z.shape))

    # test to ensure that this will not blow up numerically if you have a very very long core.
    # manually set all super deep layers to solid ice (rhoi=920)
    NUM = k1*rhoi*(z-zc)+np.log(rhoc/(rhoi-rhoc))

    numerical = np.where(NUM <= 100.0)
    blowup = np.where(NUM > 100.0)

    q[dnix] = np.exp(k1*rhoi*(z[dnix]-zc)+np.log(rhoc/(rhoi-rhoc)))  # g7
    q[upix] = np.exp(k0*rhoi*z[upix]+np.log(rhos/(rhoi-rhos)))  # g7

    rho[numerical] = q[numerical]*rhoi/(1+q[numerical])  # [g8] modified by fzhu to fix inconsistency of array size
    rho[blowup] = rhoi

    # only calculate this if you want zieq
    tc = (np.log(rhoi-rhos)-np.log(rhoi-rhoc))/c0  # age at rho=rhoc [g17]
    t = np.zeros((z.shape))  # pre allocate a vector for age as a function of z
    t[upix] = (np.log(rhoi-rhos)-np.log(rhoi-rho[upix]))/c0  # [g16] above zc
    t[dnix] = (np.log(rhoi-rhoc)-np.log(rhoi+0.0001-rho[dnix]))/c1 + tc  # [g16] below zc
    tdiff = np.diff(t)

    # make sure time keeps increasing even after we reach the critical depth.
    if np.any(tdiff == 0.00):
        inflection = np.where(tdiff == 0.0)
        lineardepth_change = t[inflection][0]

        for i in range(len(t)):
            if t[i] > lineardepth_change:
                t[i] = t[i-1] + 1e-5

    zieq = t*bdot/rhoi  # [g15]

    return rho, zieq, t


def ice_archive(d18Oice, pr_ann, tas_ann, psl_ann, nproc=8):
    ''' Accounts for diffusion and compaction in the firn.

    Args:
        d18Oice (1d array: year in int): annualizd d18O of ice [permil]
        pr_ann (1d array: year in int): precipitation rate [kg m-2 s-1]
        tas_ann (1d array: year in int): annualizd atomspheric temerature [K]
        psl_ann (1d array: year in int): annualizd sea level pressure [Pa]
        nproc (int): the number of processes for multiprocessing

    Returns:
        ice_diffused (1d array: year in int): archived ice d18O [permil]

    '''
    # ======================================================================
    # A.0: Initialization
    # ======================================================================
    # accumulation rate [m/yr]
    # note that the unit of pr_ann is [kg m-2 s-1], so need to divide by density [kg m-3] and convert the time
    yr2sec_factor = 3600*24*365.25
    accum = pr_ann/1000*yr2sec_factor

    # depth horizons (accumulation per year corresponding to depth moving down-core)
    bdown = accum[::-1]
    bmean = np.mean(bdown)
    depth = np.sum(bdown)
    depth_horizons = np.cumsum(bdown)
    dz = np.min(depth_horizons)/10.  # step in depth [m]

    Tmean = np.mean(tas_ann)  # unit in [K]
    Pmean = np.mean(psl_ann)*9.8692e-6  # unit in [Atm]

    # contants
    rho_s = 300.  # kg/m^3, surface density
    rho_d = 822.  # kg/m^2, density at which ice becomes impermeable to diffusion
    rho_i = 920.  # kg/m^3, density of solid ice

    # ======================================================================
    # A.1: Compaction Model
    # ======================================================================
    z = np.arange(0, depth, dz) + dz  # linear depth scale

    # set density profile by calling densification function
    rho, zieq, t = densification(Tmean, bmean, rho_s, z)

    rho = rho[:len(z)]  # cutoff the end
    time_d = np.cumsum(dz/bmean*rho/rho_i)
    ts = time_d*yr2sec_factor  # convert time in years to ts in seconds

    # integrate diffusivity along the density gradient to obtain diffusion length
    D = diffusivity(rho, Tmean, Pmean, rho_d, bmean)

    D = D[:-1]
    rho = rho[:-1]
    diffs = np.diff(z)/np.diff(time_d)
    diffs = diffs[:-1]

    # Integration using the trapezoidal method

    # IMPORTANT: once the ice reaches crtiical density (solid ice), there will no longer
    # be any diffusion. There is also numerical instability at that point. Set Sigma=1E-13 for all
    # points below that threshold.

    # Set to 915 to be safe.
    solidice = np.where(rho >= rho_d-5.0)
    diffusion = np.where(rho < rho_d-5.0)

    dt = np.diff(ts)
    sigma_sqrd_dummy = 2*np.power(rho, 2)*dt*D
    sigma_sqrd = integrate.cumtrapz(sigma_sqrd_dummy)
    diffusion_array = diffusion[0]
    diffusion_array = diffusion_array[diffusion_array < len(sigma_sqrd)]  # fzhu: to avoid the boundary index error
    diffusion = np.asarray(diffusion_array)

    #  rho=rho[0:-1] # modified by fzhu to fix inconsistency of array size
    #  sigma=np.zeros((len(rho)+1)) # modified by fzhu to fix inconsistency of array size
    sigma = np.zeros((len(rho)))
    sigma[diffusion] = np.sqrt(1/np.power(rho[diffusion],2)*sigma_sqrd[diffusion]) # modified by fzhu to fix inconsistency of array size
    #sigma[solidice]=np.nanmax(sigma) #max diffusion length in base of core // set in a better way. max(sigma)
    sigma[solidice] = sigma[diffusion][-1]
    sigma = sigma[:-1]

    # ======================================================================
    # A.2. Diffusion Profile
    # ======================================================================
    # Load water isotope series
    del18 = np.flipud(d18Oice)  # NOTE YOU MIGHT NOT NEED FLIP UD here. Our data goes forward in time.

    # interpolate over depths to get an array of dz values corresponding to isotope values for convolution/diffusion
    iso_interp = np.interp(z, depth_horizons, del18)

    # Return a warning if the kernel length is approaching 1/2 that of the timeseries.
    # This will result in spurious numerical effects.

    zp = np.arange(-100, 100, dz)
    if (len(zp) >= 0.5*len(z)):
        print("Warning: convolution kernal length (zp) is approaching that of half the length of timeseries. Kernal being clipped.")
        bound = 0.20*len(z)*dz
        zp = np.arange(-bound, bound, dz)

    #  print('start for loop ...')
    #  start_time = time.time()

    rm = np.nanmean(iso_interp)
    cdel = iso_interp-rm

    diffused_final = np.zeros(len(iso_interp))
    if nproc == 1:
        for i in tqdm(range(len(sigma))):
            sig = sigma[i]
            part1 = 1./(sig*np.sqrt(2.*np.pi))
            part2 = np.exp(-zp**2/(2*sig**2))
            G = part1*part2
            #  diffused = np.convolve(G, cdel, mode='same')*dz  # fzhu: this is way too slow
            diffused = signal.fftconvolve(cdel, G, mode='same')*dz  # put cdel in the front to keep the same length as before
            diffused += rm  # remove mean and then put back
            diffused_final[i] = diffused[i]

    else:
        #  print('Multiprocessing: nproc = {}'.format(nproc))

        def conv(sig, i):
            part1 = 1./(sig*np.sqrt(2.*np.pi))
            part2 = np.exp(-zp**2/(2*sig**2))
            G = part1*part2
            diffused = signal.fftconvolve(cdel, G, mode='same')*dz
            diffused += rm  # remove mean and then put back

            return diffused[i]

        res = Pool(nproc).map(conv, sigma, range(len(sigma)))
        diffused_final[:len(res)] = np.asarray(res)

    #  print('for loop: {:0.2f} s'.format(time.time()-start_time))

    # take off the first few and last few points used in convolution
    diffused_timeseries = diffused_final[0:-3]

    # Now we need to pack our data back into single year data units based on the depths and year interpolated data
    final_iso = np.interp(depth_horizons, z[0:-3], diffused_timeseries)
    ice_diffused = final_iso

    return ice_diffused
