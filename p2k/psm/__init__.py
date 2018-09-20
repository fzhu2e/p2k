from . import icecore
from . import tree
from . import coral
from . import lake
from . import speleo
import p2k
import numpy as np


def forward(proxy, lat_obs, lon_obs, lat_model, lon_model, time_model,
    tas=None, pr=None, psl=None, d18Opr=None, d18Ocoral=None, nproc=8,
    Rlib_path='/Library/Frameworks/R.framework/Versions/3.4/Resources/library',
    T1=8, T2=23, M1=0.01, M2=0.05):

    ''' Forward environmental variables to proxy variables

    This is a major wrapper of the PSMs.

    Args:
        proxy (str): options are `coral_d18O`, `ice_d18O`, `tree_trw`
        lat_obs, lon_obs (float): the location of the proxy site
        lat_model, lon_model (1-D/2-D array): the grid points of the model simulation
        tas (3-D array): surface air temperature in (time, lat, lon) [K]
        pr (3-D array): precipitation rate in (time, lat, lon) [kg/m2/s]
        psl (3-D array): sea-level pressure in (time, lat, lon) [Pa]
        d18Opr (3-D array): precipitation d18O in (time, lat, lon) [permil]
        nproc (int): number of threads; only works for `coral_d18O`

    Returns:
        pseudo_value (1-D array): pseudoproxy timeseries
        pseudo_time (1-D array): the time axis of the pseudoproxy timeseries

    '''
    if proxy == 'coral_d18O':
        print('p2k >>> forward to {} ...'.format(proxy))
        lat_ind, lon_ind = p2k.find_closest_loc(lat_model, lon_model, lat_obs, lon_obs, mode='mesh')
        print('p2k >>> Target: ({}, {}) >>> Found: ({}, {})'.format(
            lat_obs, lon_obs, lat_model[lat_ind, lon_ind], lon_model[lat_ind, lon_ind]))
    
        pseudo_value = d18Ocoral[:, lat_ind, lon_ind]
        pseudo_value[pseudo_value>1e5] = np.nan
        pseudo_time = time_model

    elif proxy == 'ice_d18O':
        print('p2k >>> forward to {} ...'.format(proxy))
        lat_ind, lon_ind = p2k.find_closest_loc(lat_model, lon_model, lat_obs, lon_obs, mode='latlon')
        print('p2k >>> Target: ({}, {}) >>> Found: ({}, {})'.format(
            lat_obs, lon_obs, lat_model[lat_ind], lon_model[lon_ind]))

        # annualize the data
        tas_ann, year_int = p2k.annualize(tas, time_model)
        psl_ann, year_int = p2k.annualize(psl, time_model)
        d18O_ann, year_int = p2k.annualize(d18Opr, time_model)
        pr_ann, year_int = p2k.annualize(pr, time_model)

        nyr = len(year_int)

        # sensor model
        d18O_ice = p2k.psm.icecore.ice_sensor(time_model, d18Opr, pr)
        d18O_ice = d18O_ice[:, lat_ind, lon_ind]
        # diffuse model
        ice_diffused = p2k.psm.icecore.ice_archive(d18O_ice,
            pr_ann[:, lat_ind, lon_ind], tas_ann[:, lat_ind, lon_ind],
            psl_ann[:, lat_ind, lon_ind], nproc=nproc)

        pseudo_value = ice_diffused[::-1]
        pseudo_time = year_int

    elif proxy == 'tree_trw':
        print('p2k >>> forward to {} ...'.format(proxy))
        lat_ind, lon_ind = p2k.find_closest_loc(lat_model, lon_model, lat_obs, lon_obs, mode='latlon')
        print('p2k >>> Target: ({}, {}) >>> Found: ({}, {})'.format(
            lat_obs, lon_obs, lat_model[lat_ind], lon_model[lon_ind]))

        syear, eyear = int(np.floor(time_model[0])), int(np.floor(time_model[-1]))  # start and end year
        nyr = eyear - syear + 1
        phi = lat_obs

        pseudo_value = p2k.psm.tree.vslite(
            syear, eyear, phi, tas[:, lat_ind, lon_ind], pr[:, lat_ind, lon_ind],
            Rlib_path=Rlib_path, T1=T1, T2=T2, M1=M1, M2=M2)
        pseudo_time = np.linspace(syear, eyear, nyr)

    else:
        print('p2k >>> ERROR: Proxy type not supported!')
        pseudo_value, pseudo_time =  None, None

    return pseudo_value, pseudo_time