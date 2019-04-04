''' API for PRYSM

Feng Zhu (fengzhu@usc.edu)
2018-12-15 16:12:22
'''
from . import icecore
from . import tree
from . import coral
from . import lake
from . import speleo
import p2k
import numpy as np


def forward(psm_name, lat_obs, lon_obs, lat_model, lon_model, time_model,
            prior_vars, verbose=False, **psm_params):

    ''' Forward environmental variables to proxy variables

    This is a major wrapper of the PSMs.
    It assumes that VS-Lite has been installed in R via:
        ```R
        install.packages("devtools")
        devtools::install_github("fzhu2e/VSLiteR")
        ```

    Args:
        psm_name (str): options are `coral_d18O`, `ice_d18O`, `tree_trw`
        lat_obs, lon_obs (float): the location of the proxy site
        lat_model, lon_model (1-D/2-D array): the grid points of the model simulation
        prior_vars (dict): the dictionary that stores the prior variables, including
            - tas (3-D array): surface air temperature in (time, lat, lon) [K]
            - pr (3-D array): precipitation rate in (time, lat, lon) [kg/m2/s]
            - psl (3-D array): sea-level pressure in (time, lat, lon) [Pa]
            - d18Opr (3-D array): precipitation d18O in (time, lat, lon) [permil]
            - d18Ocoral (3-D array): coral d18O in (time, lat, lon) [permil]
            - d18Osw (3-D array): seawater d18O in (time, lat, lon) [permil]
            - sst (3-D array): sea-surface temperature in (time, lat, lon) [K]
            - sss (3-D array): sea-surface salinity in (time, lat, lon) [permil]
        psm_params (kwargs): the specific parameters for certain PSMs

    Returns:
        pseudo_value (1-D array): pseudoproxy timeseries
        pseudo_time (1-D array): the time axis of the pseudoproxy timeseries

    '''
    prior_vars_dict = {
        'tas': None,
        'pr': None,
        'psl': None,
        'd18Opr': None,
        'd18Ocoral': None,
        'd18Osw': None,
        'sst': None,
        'sss': None,
    }
    prior_vars_dict.update(prior_vars)

    psm_params_dict = {
        # for coral d18O
        'annualize_coral': True,
        'species': 'default',
        'b1': 0.3007062,
        'b2': 0.2619054,
        'b3': 0.436509,
        'b4': 0.1552032,
        'b5': 0.15,

        # for ice d18O
        'nproc': 8,

        # for vslite
        'Rlib_path': '/Library/Frameworks/R.framework/Versions/3.4/Resources/library',
        'T1': 8,
        'T2': 23,
        'M1': 0.01,
        'M2': 0.05,
    }
    psm_params_dict.update(psm_params)

    def run_psm_for_coral_d18O():
        lat_ind, lon_ind = p2k.find_closest_loc(lat_model, lon_model, lat_obs, lon_obs, mode='mesh')
        if verbose:
            print(f'PRYSM >>> Target: ({lat_obs}, {lon_obs}); Found: ({lat_model[lat_ind, lon_ind]:.2f}, {lon_model[lat_ind, lon_ind]:.2f})')

        d18Ocoral = prior_vars_dict['d18Ocoral']
        sst = prior_vars_dict['sst']
        sss = prior_vars_dict['sss']
        d18Osw = prior_vars_dict['d18Osw']

        annualize_coral = psm_params_dict['annualize_coral']

        if d18Ocoral is None:
            species = psm_params_dict['species']
            b1 = psm_params_dict['b1']
            b2 = psm_params_dict['b2']
            b3 = psm_params_dict['b3']
            b4 = psm_params_dict['b4']
            b5 = psm_params_dict['b5']

            pseudo_value = coral.pseudocoral(lat_obs, lon_obs, sst, sss,
                                             d18O=d18Osw, species=species,
                                             b1=b1, b2=b2, b3=b3, b4=b4, b5=b5)

        else:
            pseudo_value = np.asarray(d18Ocoral[:, lat_ind, lon_ind])
            pseudo_value[pseudo_value>1e5] = np.nan  # replace missing values with nan
            while np.all(np.isnan(pseudo_value)):
                for lat_fix in [0, -1, 1, -2, 2, -3, 3]:
                    for lon_fix in [0, -1, 1, -2, 2, -3, 3]:
                        pseudo_value = np.asarray(d18Ocoral[:, lat_ind+lat_fix, lon_ind+lon_fix])

            pseudo_value[pseudo_value>1e5] = np.nan  # replace missing values with nan

        if annualize_coral:
            pseudo_value, pseudo_time = p2k.annualize_ts(pseudo_value, time_model)
        else:
            pseudo_time = time_model

        return pseudo_value, pseudo_time

    def run_psm_for_ice_d18O():
        tas = prior_vars_dict['tas']
        pr = prior_vars_dict['pr']
        psl = prior_vars_dict['psl']
        d18Opr = prior_vars_dict['d18Opr']

        if tas is None or pr is None or psl is None or d18Opr is None:
            raise TypeError

        nproc = psm_params_dict['nproc']

        lat_ind, lon_ind = p2k.find_closest_loc(lat_model, lon_model, lat_obs, lon_obs, mode='latlon')
        if verbose:
            print(f'PRYSM >>> Target: ({lat_obs}, {lon_obs}); Found: ({lat_model[lat_ind]:.2f}, {lon_model[lon_ind]:.2f})')

        tas_sub = np.asarray(tas[:, lat_ind, lon_ind])
        pr_sub = np.asarray(pr[:, lat_ind, lon_ind])
        psl_sub = np.asarray(psl[:, lat_ind, lon_ind])

        # annualize the data
        tas_ann, year_int = p2k.annualize(tas_sub, time_model)
        psl_ann, year_int = p2k.annualize(psl_sub, time_model)
        pr_ann, year_int = p2k.annualize(pr_sub, time_model)

        # sensor model
        d18O_ice = icecore.ice_sensor(time_model, d18Opr, pr)
        # diffuse model
        ice_diffused = icecore.ice_archive(d18O_ice[:, lat_ind, lon_ind], pr_ann, tas_ann, psl_ann, nproc=nproc)

        pseudo_value = ice_diffused[::-1]
        pseudo_time = year_int

        return pseudo_value, pseudo_time

    def run_psm_for_tree_trw():
        tas = prior_vars_dict['tas']
        pr = prior_vars_dict['pr']

        if tas is None or pr is None:
            raise TypeError

        T1 = psm_params_dict['T1']
        T2 = psm_params_dict['T2']
        M1 = psm_params_dict['M1']
        M2 = psm_params_dict['M2']
        Rlib_path = psm_params_dict['Rlib_path']
        if verbose:
            print(f'PRYSM >>> Using R libs from: {Rlib_path}')

        lat_ind, lon_ind = p2k.find_closest_loc(lat_model, lon_model, lat_obs, lon_obs, mode='latlon')
        if verbose:
            print(f'PRYSM >>> Target: ({lat_obs}, {lon_obs}); Found: ({lat_model[lat_ind]:.2f}, {lon_model[lon_ind]:.2f})')

        syear, eyear = int(np.floor(time_model[0])), int(np.floor(time_model[-1]))  # start and end year
        nyr = eyear - syear + 1
        phi = lat_obs

        tas_sub = np.asarray(tas[:, lat_ind, lon_ind])
        pr_sub = np.asarray(pr[:, lat_ind, lon_ind])

        pseudo_value = tree.vslite(
            syear, eyear, phi, tas_sub, pr_sub,
            Rlib_path=Rlib_path, T1=T1, T2=T2, M1=M1, M2=M2
        )
        pseudo_time = np.linspace(syear, eyear, nyr)

        return pseudo_value, pseudo_time

    print(f'PRYSM >>> forward with {psm_name} ...')

    psm_func = {
        'prysm.coral.d18O': run_psm_for_coral_d18O,
        'prysm.ice.d18O': run_psm_for_ice_d18O,
        'prysm.vslite': run_psm_for_tree_trw,
    }

    pseudo_value, pseudo_time = psm_func[psm_name]()

    return pseudo_value, pseudo_time
