#!/usr/bin/env python3
__author__ = 'Feng Zhu'
__email__ = 'fengzhu@usc.edu'
__version__ = '0.4.5'

import os
import pandas as pd
import numpy as np
from scipy import spatial
from scipy.stats.mstats import mquantiles
import xarray as xr
import netCDF4
from datetime import datetime
from scipy.interpolate import interp1d
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from matplotlib.colors import Normalize, ListedColormap
from matplotlib import gridspec
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import nitime.algorithms as tsa

from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm
import pickle
import warnings

from pyleoclim import Spectral, Timeseries


class PAGES2k(object):
    ''' A bunch of PAGES2k style settings
    '''
    archive_types = ['bivalve',
                     'borehole',
                     'coral',
                     'documents',
                     'glacier ice',
                     'hybrid',
                     'lake sediment',
                     'marine sediment',
                     'sclerosponge',
                     'speleothem',
                     'tree',
                     ]
    markers = ['p', 'p', 'o', 'v', 'd', '*', 's', 's', '8', 'D', '^']
    markers_dict = dict(zip(archive_types, markers))
    colors = [np.array([ 1.        ,  0.83984375,  0.        ]),
              np.array([ 0.73828125,  0.71484375,  0.41796875]),
              np.array([ 1.        ,  0.546875  ,  0.        ]),
              np.array([ 0.41015625,  0.41015625,  0.41015625]),
              np.array([ 0.52734375,  0.8046875 ,  0.97916667]),
              np.array([ 0.        ,  0.74609375,  1.        ]),
              np.array([ 0.25390625,  0.41015625,  0.87890625]),
              np.array([ 0.54296875,  0.26953125,  0.07421875]),
              np.array([ 1         ,           0,           0]),
              np.array([ 1.        ,  0.078125  ,  0.57421875]),
              np.array([ 0.1953125 ,  0.80078125,  0.1953125 ])]
    colors_dict = dict(zip(archive_types, colors))


def lipd2df(lipd_dirpath, pkl_filepath, col_str=[
            'paleoData_pages2kID',
            'dataSetName', 'archiveType',
            'geo_meanElev', 'geo_meanLat', 'geo_meanLon',
            'year', 'yearUnits',
            'paleoData_variableName',
            'paleoData_units',
            'paleoData_values',
            'paleoData_proxy']):

    ''' Convert a bunch of PAGES2k LiPD files to a pickle file of Pandas DataFrame to boost data loading

    Args:
        lipd_dirpath (str): the path of the PAGES2k LiPD files
        pkl_filepath (str): the path of the converted pickle file
        col_str (list of str): the name string of the variables to extract from the LiPD files

    Returns:
        df (Pandas DataFrame): the converted Pandas DataFrame
    '''
    import lipd

    # save the current working directory for later use, as the LiPD utility will change it in the background
    work_dir = os.getcwd()

    # LiPD utility requries the absolute path, so let's get it
    lipd_dirpath = os.path.abspath(lipd_dirpath)

    # load LiPD files from the given directory
    lipds = lipd.readLipd(lipd_dirpath)

    # extract timeseries from the list of LiDP objects
    ts_list = lipd.extractTs(lipds)

    # recover the working directory
    os.chdir(work_dir)

    # create an empty pandas dataframe with the number of rows to be the number of the timeseries (PAGES2k records),
    # and the columns to be the variables we'd like to extract
    df_tmp = pd.DataFrame(index=range(len(ts_list)), columns=col_str)

    # loop over the timeseries and pick those for global temperature analysis
    i = 0
    for ts in ts_list:
        if 'paleoData_useInGlobalTemperatureAnalysis' in ts.keys() and \
            ts['paleoData_useInGlobalTemperatureAnalysis'] == 'TRUE':
            for name in col_str:
                try:
                    df_tmp.loc[i, name] = ts[name]
                except:
                    df_tmp.loc[i, name] = np.nan
            i += 1

    # drop the rows with all NaNs (those not for global temperature analysis)
    df = df_tmp.dropna(how='all')

    # save the dataframe to a pickle file for later use
    save_path = os.path.abspath(pkl_filepath)
    print(f'Saving pickle file at: {save_path}')
    df.to_pickle(save_path)

    return df


def find_closest_loc(lat, lon, target_lat, target_lon, mode=None, verbose=False):
    ''' Find the closet model sites (lat, lon) based on the given target (lat, lon) list

    Args:
        lat, lon (array): the model latitude and longitude arrays
        target_lat, target_lon (array): the target latitude and longitude arrays
        mode (str):
        + latlon: the model lat/lon is a 1-D array
        + mesh: the model lat/lon is a 2-D array

    Returns:
        lat_ind, lon_ind (array): the indices of the found closest model sites

    '''

    if mode is None:
        if len(np.shape(lat)) == 1:
            mode = 'latlon'
        elif len(np.shape(lat)) == 2:
            mode = 'mesh'
        else:
            raise ValueError('ERROR: The shape of the lat/lon cannot be processed !!!')

    if mode is 'latlon':
        # model locations
        mesh = np.meshgrid(lon, lat)

        list_of_grids = list(zip(*(grid.flat for grid in mesh)))
        model_lon, model_lat = zip(*list_of_grids)

    elif mode is 'mesh':
        model_lat = lat.flatten()
        model_lon = lon.flatten()

    elif mode is 'list':
        model_lat = lat
        model_lon = lon

    model_locations = []

    for m_lat, m_lon in zip(model_lat, model_lon):
        model_locations.append((m_lat, m_lon))

    # target locations
    if np.size(target_lat) > 1:
        #  target_locations_dup = list(zip(target_lat, target_lon))
        #  target_locations = list(set(target_locations_dup))  # remove duplicated locations
        target_locations = list(zip(target_lat, target_lon))
        n_loc = np.shape(target_locations)[0]
    else:
        target_locations = [(target_lat, target_lon)]
        n_loc = 1

    lat_ind = np.zeros(n_loc, dtype=int)
    lon_ind = np.zeros(n_loc, dtype=int)

    # get the closest grid
    for i, target_loc in (enumerate(tqdm(target_locations)) if verbose else enumerate(target_locations)):
        X = target_loc
        Y = model_locations
        distance, index = spatial.KDTree(Y).query(X)
        closest = Y[index]
        nlon = np.shape(lon)[-1]

        lat_ind[i] = index // nlon
        lon_ind[i] = index % nlon

        #  if np.size(target_lat) > 1:
            #  df_ind[i] = target_locations_dup.index(target_loc)

    if np.size(target_lat) > 1:
        #  return lat_ind, lon_ind, df_ind
        return lat_ind, lon_ind
    else:
        return lat_ind[0], lon_ind[0]


def nearest_loc(lats, lons, target_loc):
    ''' Find the nearest location in paris of lat and lon based on target location

    Args:
        lats, lons (array): the model latitude and longitude arrays
        target_loc (tuple): the target location (lat, lon)

    Returns:
        lat_ind, lon_ind (array): the indices of the found closest model sites

    '''

    # model locations
    model_locations = []

    for m_lat, m_lon in zip(lats, lons):
        model_locations.append((m_lat, m_lon))

    # get the closest grid
    X = target_loc
    Y = model_locations
    distance, index = spatial.KDTree(Y).query(X)

    return index


def load_netCDF4(path, var_list):
    var_fields = []
    ncfile = netCDF4.Dataset(path, 'r')

    for var in var_list:

        if var is 'year':
            time = ncfile.variables['time']
            time_convert = netCDF4.num2date(time[:], time.units, time.calendar)

            def datetime2year(dt):
                dt = datetime(year=dt.year, month=dt.month, day=dt.day)
                year_part = dt - datetime(year=dt.year, month=1, day=1)
                year_length = datetime(year=dt.year+1, month=1, day=1) - datetime(year=dt.year, month=1, day=1)
                return dt.year + year_part/year_length

            nt = np.shape(time_convert)[0]
            year = np.zeros(nt)
            for i, day in enumerate(time_convert):
                year[i] = datetime2year(day)

            var_fields.append(year)

        else:
            if var is 'lon':
                if np.min(ncfile.variables[var]) >= 0:
                    field = np.mod(ncfile.variables[var]+180, 360) - 180  # convert from range (0, 360) to (-180, 180)
                else:
                    field = np.asarray(ncfile.variables[var])
            elif var is 'lat':
                field = np.asarray(ncfile.variables[var])
            else:
                field = ncfile.variables[var]

            var_fields.append(field)

    if len(var_list) == 1:
        var_fields = var_fields[0]

    return var_fields


def load_CESM_netcdf(path, var_list, decode_times=False):
    ''' Load CESM NetCDF file

    Args:
        path (str): the path of the CESM NetCDF file
        var_list (list of str): the names of the variables to load
            - if the variable name is `year`, then the unit would be in year
            - if the variable name is `lon`, then it would be shifted by -180 degree
              to be consistent with PAGES2k

    Returns:
        var_fields (list of 3d array): the 3D field of the variable

    '''

    handle = xr.open_dataset(path, decode_times=decode_times)

    var_fields = []

    for var in var_list:

        if var is 'year':
            ncfile = netCDF4.Dataset(path, 'r')
            time = ncfile.variables['time']
            time_convert = netCDF4.num2date(time[:], time.units, time.calendar)

            def datetime2year(dt):
                dt = datetime(year=dt.year, month=dt.month, day=dt.day)
                year_part = dt - datetime(year=dt.year, month=1, day=1)
                year_length = datetime(year=dt.year+1, month=1, day=1) - datetime(year=dt.year, month=1, day=1)
                return dt.year + year_part/year_length

            nt = np.shape(time_convert)[0]
            year = np.zeros(nt)
            for i, day in enumerate(time_convert):
                year[i] = datetime2year(day)

            var_fields.append(year)

        else:
            if var is 'lon' and np.min(handle[var].values) >= 0:
                field = np.mod(handle[var].values+180, 360) - 180  # convert from range (0, 360) to (-180, 180)
            else:
                field = handle[var].values

            var_fields.append(field)

    handle.close()

    if len(var_fields) == 1:
        var_fields = np.asarray(var_fields)[0]

    return var_fields


def annualize(var_field, year, weights=None):
    ''' Annualize the variable field, whose first axis must be time

    Args:
        var_field (3d array): the 3D field of the variable
        year (array): time axis [year in float]

    Returns:
        var_ann (3d arrary): the annualized field
        year_int (array): the set of the years in integers [year in int]
    '''
    year_int = list(set(np.floor(year)))
    year_int = np.sort(list(map(int, year_int)))
    n_year = len(year_int)
    var_ann = np.ndarray(shape=(n_year, *var_field.shape[1:]))

    year_int_pad = list(year_int)
    year_int_pad.append(np.max(year_int)+1)

    for i in range(n_year):
        t_start = year_int_pad[i]
        t_end = year_int_pad[i+1]
        t_range = (year>=t_start) & (year<t_end)

        if weights is None:
            var_ann[i] = np.average(var_field[t_range], axis=0, weights=None)
        else:
            var_ann[i] = np.average(var_field[t_range], axis=0, weights=weights[t_range])


    return var_ann, year_int


def annualize_ts(ys, ts):
    year_int = list(set(np.floor(ts)))
    year_int = np.sort(list(map(int, year_int)))
    n_year = len(year_int)
    year_int_pad = list(year_int)
    year_int_pad.append(np.max(year_int)+1)
    ys_ann = np.zeros(n_year)

    for i in range(n_year):
        t_start = year_int_pad[i]
        t_end = year_int_pad[i+1]
        #  t_range = (ts >= t_start) & (ts < t_end)
        t_range = (ts > t_start) & (ts <= t_end)
        value_between = ys[t_range]
        if np.size(value_between) == 0:
            ys_ann[i] = np.nan
        else:
            ys_ann[i] = np.average(value_between[~np.isnan(value_between)], axis=0)

    ys_tmp = np.copy(ys_ann)
    ys_ann = ys_ann[~np.isnan(ys_tmp)]
    year_int = year_int[~np.isnan(ys_tmp)]

    return ys_ann, year_int


def smooth_ts(ys, ts, bin_vector=None, bin_width=10):
    if bin_vector is None:
        bin_vector = np.arange(
            bin_width*(np.min(ts)//bin_width),
            bin_width*(np.max(ts)//bin_width+2),
            step=bin_width)

    ts_bin = (bin_vector[1:] + bin_vector[:-1])/2
    n_bin = np.size(ts_bin)
    ys_bin = np.zeros(n_bin)

    for i in range(n_bin):
        t_start = bin_vector[i]
        t_end = bin_vector[i+1]
        t_range = (ts >= t_start) & (ts < t_end)
        if sum(t_range*1) == 1:
            ys_bin[i] = ys[t_range]
        else:
            ys_bin[i] = np.average(ys[t_range], axis=0)

    return ys_bin, ts_bin, bin_vector


def bin_ts(ys, ts, bin_vector=None, bin_width=10, resolution=1):
    ys_smooth, ts_smooth, bin_vector = smooth_ts(ys, ts, bin_vector=bin_vector, bin_width=bin_width)

    bin_vector_finer = np.arange(np.min(bin_vector), np.max(bin_vector)+1, step=resolution)
    bin_value = np.zeros(bin_vector_finer.size)

    n_bin = np.size(ts_smooth)
    for i in range(n_bin):
        t_start = bin_vector[i]
        t_end = bin_vector[i+1]
        t_range = (bin_vector_finer >= t_start) & (bin_vector_finer <= t_end)
        bin_value[t_range] = ys_smooth[i]

    return bin_value, bin_vector_finer


def df2psd(df, freqs=None, value_name='paleoData_values', time_name='year', save_path=None,
           standardize=False, gaussianize=False):
    ''' Calculate the power spectral densities of a Pandas DataFrame PAGES2k dataset using WWZ method

    Args:
        df (Pandas DataFrame): a Pandas DataFrame
        freqs (array): frequency vector for spectral analysis
        save_path (str): if set, save the PSD result to the given path

    Returns:
        psds (2d array): the scaling exponents

    '''
    paleoData_values = df[value_name].values
    year = df[time_name].values

    n_series = len(year)
    n_freqs = np.size(freqs)
    psd_list = []
    freqs_list = []

    for k in tqdm(range(n_series), desc='Processing time series'):
        Xo = np.asarray(paleoData_values[k], dtype=np.float)
        to = np.asarray(year[k], dtype=np.float)
        Xo, to = Timeseries.clean_ts(Xo, to)
        if np.mean(np.diff(to)) < 1:
            warnings.warn('The time series will be annualized due to mean of dt less than one year.')
            Xo, to = annualize_ts(Xo, to)
        tau = np.linspace(np.min(to), np.max(to), np.min([np.size(to), 501]))
        res_wwz = Spectral.wwz_psd(Xo, to, freqs=freqs, tau=tau, c=1e-3, nproc=16, nMC=0,
                                   standardize=standardize, gaussianize=gaussianize)
        psd_list.append(res_wwz.psd)
        freqs_list.append(res_wwz.freqs)

    if save_path:
        print('p2k >>> Saving pickle file at: {}'.format(save_path))

        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        with open(save_path, 'wb') as f:
            pickle.dump([psd_list, freqs_list], f)

        print('p2k >>> DONE')

    return psd_list, freqs_list


def df2psd_mtm(df, value_name='paleoData_values', time_name='year', save_path=None, standardize=False):
    ''' Calculate the power spectral densities of a Pandas DataFrame PAGES2k dataset using WWZ method

    Args:
        df (Pandas DataFrame): a Pandas DataFrame
        freqs (array): frequency vector for spectral analysis
        save_path (str): if set, save the PSD result to the given path

    Returns:
        psds (2d array): the scaling exponents

    '''
    paleoData_values = df[value_name].values
    year = df[time_name].values

    n_series = len(year)
    freqs = []
    psds = []

    for k in tqdm(range(n_series), desc='Processing time series'):
        Xo = np.asarray(paleoData_values[k], dtype=np.float)
        to = np.asarray(year[k], dtype=np.float)
        Xo, to = Timeseries.clean_ts(Xo, to)

        if standardize:
            Xo, _, _ = Timeseries.standardize(Xo)

        Xo_ann, to_ann = annualize(Xo, to)

        interp_f = interp1d(to_ann, Xo_ann)
        to_interp = np.arange(np.min(to_ann), np.max(to_ann)+0.5, dtype=int)
        Xo_interp = interp_f(to_interp)

        freq_mtm, psd_mtm, nu = tsa.multi_taper_psd(Xo_interp, adaptive=False, jackknife=False,
                                                    NW=1, Fs=1)
        freqs.append(freq_mtm)
        psds.append(psd_mtm)

    if save_path:
        print('p2k >>> Saving pickle file at: {}'.format(save_path))

        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        with open(save_path, 'wb') as f:
            pickle.dump(psds, freqs, f)

        print('p2k >>> DONE')

    return psds, freqs


def df2composite(df, value_name='paleoData_values', time_name='year',
                 bin_width=10,
                 gaussianize=False, standardize=False,
                 n_bootstraps=10000, stat_func=np.nanmean, nproc=8):
    ''' Performs binning and averaging of the proxy timeseries, and bootstrap for uncertainty estimation
    '''
    df_comp = pd.DataFrame(columns=['time', 'value', 'min', 'max', 'mean', 'median', 'bootstrap_stats'])
    df_comp.set_index('time', inplace=True)

    for index, row in df.iterrows():
        Xo, to = row2ts(row, clean_ts=True, value_name=value_name, time_name=time_name)
        if np.shape(Xo)[0] == 0:
            continue

        wa = Spectral.WaveletAnalysis()
        if gaussianize:
            Xo = wa.preprocess(Xo, to, gaussianize=True)
        if standardize:
            Xo = wa.preprocess(Xo, to, standardize=True)

        Xo_bin, to_bin, _ = smooth_ts(Xo, to, bin_width=bin_width)
        for i, t in enumerate(to_bin):
            ind_list = list(df_comp.index)
            if t not in ind_list:
                df_comp.at[t, 'value'] = np.asarray([Xo_bin[i]])
            else:
                list_tmp = df_comp.loc[t, 'value']
                list_tmp = np.append(list_tmp, Xo_bin[i])
                df_comp['value'] = df_comp['value'].astype(object)
                df_comp.at[t, 'value'] = list_tmp

    df_comp = df_comp.sort_index()
    for index, row in df_comp.iterrows():
        samples = row['value']
        row['min'] = np.nanmin(samples)
        row['max'] = np.nanmax(samples)
        row['mean'] = np.nanmean(samples)
        row['median'] = np.nanmedian(samples)

        if samples.shape[0] > 1:
            row['bootstrap_stats'] = bootstrap(samples, n_bootstraps=n_bootstraps, stat_func=stat_func, nproc=nproc)
        else:
            if stat_func is np.nanmedian:
                row['bootstrap_stats'] = row['median']
            elif stat_func is np.nanmean:
                row['bootstrap_stats'] = row['mean']

    return df_comp


def df2quantile_median(df):
    ts = np.asarray(df.index)
    n_ts = ts.shape[0]
    ys_bootstrap = df['bootstrap_stats'].values
    ys_quantile_median = np.zeros(n_ts)
    for i, ys_boot in enumerate(ys_bootstrap):
        ys_quantile_median[i] = mquantiles(ys_boot, 0.5)

    return ys_quantile_median, ts


def bootstrap(samples, n_bootstraps=1000, stat_func=np.nanmean, nproc=8):
    n_samples = np.shape(samples)[0]

    if nproc == 1 :
        stats = np.zeros(n_bootstraps)
        for i in tqdm(range(n_bootstraps)):
            rand_ind = np.random.randint(n_samples, size=n_samples)
            stats[i] = stat_func(samples[rand_ind])
    else:
        def one_bootstrap(i):
            rand_ind = np.random.randint(n_samples, size=n_samples)
            stat = stat_func(samples[rand_ind])

            return stat

        with Pool(nproc) as pool:
            stats = pool.map(one_bootstrap, range(n_bootstraps))

    return stats


def df2comp_ols(df, inst_temp_path, stat_func=np.nanmean,
                value_name='paleoData_values', time_name='year',
                netcdf_lat_name='latitude',
                netcdf_lon_name='longitude',
                netcdf_time_name='year',
                netcdf_temp_name='temperature_anomaly',
                ensemble_num=None,
                bin_width=10):
    df_proxy = df_append_nearest_obs(df, inst_temp_path,
                                     lat_name=netcdf_lat_name,
                                     lon_name=netcdf_lon_name,
                                     time_name=netcdf_time_name,
                                     temp_name=netcdf_temp_name,
                                     ensemble_num=ensemble_num,
                                     )
    for index, row in df_proxy.iterrows():
        if row['obs_temp'].shape[0] < 1:
            df_proxy = df_proxy.drop(index)

    df_comp_proxy = df2composite(df_proxy, bin_width=bin_width,
                           gaussianize=True, standardize=True, stat_func=stat_func,
                           value_name=value_name, time_name=time_name)
    df_comp_obs = df2composite(df_proxy, bin_width=bin_width,
                               gaussianize=False, standardize=False, stat_func=stat_func,
                               value_name='obs_temp', time_name='obs_year')
    #  ys_proxy, ts_proxy = df2quantile_median(df_comp_proxy)
    #  ys_obs, ts_obs = df2quantile_median(df_comp_obs)
    if stat_func is np.nanmedian:
        ys_proxy, ts_proxy = df_comp_proxy['median'].values, np.asarray(df_comp_proxy.index)
        ys_obs, ts_obs = df_comp_obs['median'].values, np.asarray(df_comp_obs.index)
    elif stat_func is np.nanmean:
        ys_proxy, ts_proxy = df_comp_proxy['mean'].values, np.asarray(df_comp_proxy.index)
        ys_obs, ts_obs = df_comp_obs['mean'].values, np.asarray(df_comp_obs.index)

    # OLS
    ys_proxy_overlap, ys_obs_overlap, time_overlap = overlap_ts(ys_proxy, ts_proxy, ys_obs, ts_obs)

    if time_overlap.shape[0] <= 1:
        intercept, slope, R2 = np.nan, np.nan, np.nan

    else:
        model = ols_ts(ys_proxy_overlap, time_overlap, ys_obs_overlap, time_overlap)
        results = model.fit()
        R2 = results.rsquared
        intercept = results.params[0]
        slope = results.params[1]

    return ys_proxy, ts_proxy, ys_obs, ts_obs, intercept, slope, R2


def clean_composite(df, lower_bd=10):
    for index, row in df.iterrows():
        if row['value'].shape[0] < lower_bd:
            df = df.drop(index)

    return df


def fill_nan_composite(df, lower_bd=10):
    for index, row in df.iterrows():
        if row['value'].shape[0] < lower_bd:
            df.loc[index, 'min'] = np.nan
            df.loc[index, 'max'] = np.nan
            df.loc[index, 'mean'] = np.nan
            df.loc[index, 'median'] = np.nan

    return df


def row2ts(row, value_name='paleoData_values', time_name='year', clean_ts=True):
    to = np.asarray(row[time_name])
    Xo = np.asarray(row[value_name])
    if clean_ts:
        Xo, to = Timeseries.clean_ts(Xo, to)
    return Xo, to


def row2latlon(row):
    lat = np.asarray(row['geo_meanLat'])
    lon = np.asarray(row['geo_meanLon'])
    return lat, lon


def clean_df(df):
    #  df['dt_median'] = 1.0
    #  df['dt_mean'] = 1.0
    #  df['evenly_spaced'] = False
    for index, row in df.iterrows():
        Xo, to = row2ts(row, clean_ts=True)
        df.at[index, 'paleoData_values'] = Xo
        df.at[index, 'year'] = to
        #  dt_median = np.median(np.diff(to))
        #  df.at[index, 'dt_median'] = dt_median
        #  dt_mean = np.mean(np.diff(to))
        #  df.at[index, 'dt_mean'] = dt_mean
        #  if len(set(np.diff(to))) == 1:
        #      df.at[index, 'evenly_spaced']=True

    return df


def overlap_ts(ys_proxy, ts_proxy, ys_obs, ts_obs):
    ys_proxy = np.asarray(ys_proxy, dtype=np.float)
    ts_proxy = np.asarray(ts_proxy, dtype=np.float)
    ys_obs = np.asarray(ys_obs, dtype=np.float)
    ts_obs = np.asarray(ts_obs, dtype=np.float)

    overlap_proxy = (ts_proxy >= np.min(ts_obs)) & (ts_proxy <= np.max(ts_obs))
    overlap_obs = (ts_obs >= np.min(ts_proxy)) & (ts_obs <= np.max(ts_proxy))

    ys_proxy_overlap, ts_proxy_overlap = ys_proxy[overlap_proxy], ts_proxy[overlap_proxy]
    ys_obs_overlap, ts_obs_overlap = ys_obs[overlap_obs], ts_obs[overlap_obs]

    time_overlap = np.intersect1d(ts_proxy_overlap, ts_obs_overlap)
    ind_proxy = list(i for i, t in enumerate(ts_proxy_overlap) if t in time_overlap)
    ind_obs = list(i for i, t in enumerate(ts_obs_overlap) if t in time_overlap)
    ys_proxy_overlap = ys_proxy_overlap[ind_proxy]
    ys_obs_overlap = ys_obs_overlap[ind_obs]

    return ys_proxy_overlap, ys_obs_overlap, time_overlap


def ols_ts(ys_proxy, ts_proxy, ys_obs, ts_obs):
    ys_proxy = np.asarray(ys_proxy, dtype=np.float)
    ts_proxy = np.asarray(ts_proxy, dtype=np.float)
    ys_obs = np.asarray(ys_obs, dtype=np.float)
    ts_obs = np.asarray(ts_obs, dtype=np.float)

    ys_proxy_overlap, ys_obs_overlap, time_overlap = overlap_ts(ys_proxy, ts_proxy, ys_obs, ts_obs)

    # calculate the linear regression
    X = sm.add_constant(ys_proxy_overlap)
    ols_model = sm.OLS(ys_obs_overlap, X, missing='drop')

    return ols_model


def df_append_nearest_obs(df, inst_temp_path,
                          lat_name='latitude',
                          lon_name='longitude',
                          time_name='year',
                          temp_name='temperature_anomaly',
                          ensemble_num=None,
                          ):
    # load instrumental temperature data
    lat, lon, year, temp = load_CESM_netcdf(
        inst_temp_path, [lat_name, lon_name, time_name, temp_name], decode_times=False
    )

    # preprocess df
    df = clean_df(df)
    df['obs_temp'] = np.nan
    df['obs_temp'] = df['obs_temp'].astype(object)
    df['obs_year'] = np.nan
    df['obs_year'] = df['obs_year'].astype(object)

    for index, row in df.iterrows():
        tgt_lat, tgt_lon = row2latlon(row)
        lat_ind, lon_ind = find_closest_loc(lat, lon, tgt_lat, tgt_lon)
        if ensemble_num is None:
            df.at[index, 'obs_temp'], df.at[index, 'obs_year'] = Timeseries.clean_ts(temp[:, lat_ind, lon_ind], year)
        else:
            df.at[index, 'obs_temp'], df.at[index, 'obs_year'] = Timeseries.clean_ts(temp[:, ensemble_num,
                                                                                          lat_ind, lon_ind], year)

    return df


def df_append_converted_temp(df, inst_temp_path, bin_width=10, yr_range=None,
                             lat_name='latitude',
                             lon_name='longitude',
                             time_name='year',
                             temp_name='temperature_anomaly'):
    # load instrumental temperature data
    lat, lon, year, temp = load_CESM_netcdf(
        inst_temp_path, [lat_name, lon_name, time_name, temp_name], decode_times=False
    )

    # preprocess df
    df = clean_df(df)
    #  df['dt'] = 1.0
    df['conversion_factor'] = np.nan
    df['R2'] = np.nan
    df['converted_temperature'] = np.nan
    df['converted_temperature'] = df['converted_temperature'].astype(object)
    df['converted'] = False

    # loop over df
    c_NaN = 0
    c_overlap = 0
    for index, row in df.iterrows():
        Xo, to = row2ts(row, clean_ts=True)

        # focus on time series over the defined year range
        if yr_range:
            selector = (to >= yr_range[0]) & (to <= yr_range[1])
            Xo, to = Xo[selector], to[selector]
            if to.size == 0:
                c_overlap += 1
                continue

        Xo_bin, to_bin = bin_ts(Xo, to, bin_width=bin_width)

        tgt_lat, tgt_lon = row2latlon(row)
        lat_ind, lon_ind = find_closest_loc(lat, lon, tgt_lat, tgt_lon)
        temp_nearest, year_nearest = Timeseries.clean_ts(temp[:, lat_ind, lon_ind], year)
        if np.size(year_nearest) == 0:
            c_NaN += 1
            continue

        temp_bin, year_bin = bin_ts(temp_nearest, year_nearest, bin_width=bin_width)
        if row['paleoData_variableName'] == 'temperature':
            df.at[index, 'converted_temperature'] = Xo
            df.at[index, 'converted'] = True
            continue

        #  overlap_proxy = (to_bin >= np.min(year_bin)) & (to_bin <= np.max(year_bin))
        #  overlap_obs = (year_bin >= np.min(to_bin)) & (year_bin <= np.max(to_bin))
        Xo_bin_overlap, temp_bin_overlap, time_overlap = overlap_ts(Xo_bin, to_bin, temp_bin, year_bin)

        if np.sum(time_overlap*1) <= 10:
            c_overlap += 1
            continue

        # calculate the linear regression slope and R2
        X = sm.add_constant(Xo_bin_overlap)
        model = sm.OLS(temp_bin_overlap, X, missing='drop')
        results = model.fit()
        #  Y_reg = model.predict(results.params)

        df.at[index, 'conversion_factor'] = results.params[1]
        df.at[index, 'R2'] = results.rsquared
        df.at[index, 'converted_temperature'] = results.params[0] + results.params[1]*Xo
        df.at[index, 'converted'] = True

    print('{} times skip due to NaN at cloest location'.format(c_NaN))
    print('{} times skip due to short time range overlap'.format(c_overlap))

    return df


def df_append_beta(df, freqs=None, psd_list=None, freqs_list=None, save_path=None,
                   value_name='paleoData_values', time_name='year',
                   append_psd=False, standardize=False, gaussianize=False,
                   period_ranges=[(1/200, 1/20), (1/8, 1/2)], period_names=['beta_D', 'beta_I']
                   ):
    ''' Calculate the scaling exponent and add to a new column in the given DataFrame

    Args:
        df (Pandas DataFrame): a Pandas DataFrame
        freqs (array): frequency vector for spectral analysis

    Returns:
        df_new (Pandas DataFrame): the DataFrame with scaling exponents added on

    '''
    if psd_list is None or freqs_list is None:
        psd_list, freqs_list = df2psd(df, freqs=freqs, value_name=value_name, time_name=time_name,
                                      standardize=standardize, gaussianize=gaussianize)

    df_new = df.copy()
    df_new['psd'] = np.nan
    df_new['psd'] = df_new['psd'].astype(object)
    df_new['freqs'] = np.nan
    df_new['freqs'] = df_new['freqs'].astype(object)
    if append_psd:
        for index, row in df_new.iterrows():
            df_new.at[index, 'psd'] = psd_list[index]
            df_new.at[index, 'freqs'] = freqs_list[index]

    for i, period_range in enumerate(period_ranges):

        beta_list = []
        beta_err_list = []
        for j, psd in enumerate(psd_list):
            beta, f_binned, psd_binned, Y_reg, stderr = Spectral.beta_estimation(psd, freqs_list[j], period_range[0], period_range[1])
            beta_list.append(beta)
            beta_err_list.append(stderr)

        df_new[period_names[i]] = beta_list
        df_new[period_names[i]+'_err'] = beta_err_list

    if save_path:
        print('p2k >>> Saving pickle file at: {}'.format(save_path))
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        df_new.to_pickle(save_path)
        print('p2k >>> DONE')

    return df_new


def df_append_beta_mtm(df, psds=None, freqs=None, save_path=None, value_name='paleoData_values', time_name='year',
                       period_ranges=[(1/200, 1/20), (1/8, 1/2)], period_names=['beta_D', 'beta_I']
                       ):
    ''' Calculate the scaling exponent and add to a new column in the given DataFrame

    Args:
        df (Pandas DataFrame): a Pandas DataFrame
        freqs (array): frequency vector for spectral analysis

    Returns:
        df_new (Pandas DataFrame): the DataFrame with scaling exponents added on

    '''
    if psds is None or freqs is None:
        psds, freqs = df2psd_mtm(df, value_name=value_name, time_name=time_name)

    df_new = df.copy()
    for i, period_range in enumerate(period_ranges):

        beta_list = []
        for j, psd in enumerate(psds):
            beta, f_binned, psd_binned, Y_reg, stderr = Spectral.beta_estimation(psds[j], freqs[j], period_range[0], period_range[1])
            beta_list.append(beta)

        df_new[period_names[i]] = beta_list

    if save_path:
        print('p2k >>> Saving pickle file at: {}'.format(save_path))
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        df_new.to_pickle(save_path)
        print('p2k >>> DONE')

    return df_new


def calc_plot_psd(Xo, to, ntau=501, dcon=1e-3, standardize=False,
                  anti_alias=False, plot_fig=True, method='Kirchner_f2py', nproc=8,
                  period_ticks=[0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], color=None,
                  figsize=[10, 6], font_scale=2, lw=3, label='PSD', zorder=None,
                  xlim=None, ylim=None, loc='upper right', bbox_to_anchor=None):
    if color is None:
        color = sns.xkcd_rgb['denim blue']

    tau = np.linspace(np.min(to), np.max(to), ntau)
    res_psd = Spectral.wwz_psd(Xo, to, freqs=None, tau=tau, c=dcon, standardize=standardize, nMC=0,
                               method=method, anti_alias=anti_alias, nproc=nproc)
    if plot_fig:
        sns.set(style='ticks', font_scale=font_scale)
        fig, ax = plt.subplots(figsize=figsize)
        ax.loglog(1/res_psd.freqs, res_psd.psd, lw=lw, color=color, label=label,
                  zorder=zorder)
        ax.set_xticks(period_ticks)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.invert_xaxis()
        ax.set_ylabel('Spectral Density')
        ax.set_xlabel('Period (years)')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if ylim:
            ax.set_ylim(ylim)
        if xlim:
            ax.set_xlim(xlim)
        ax.legend(bbox_to_anchor=bbox_to_anchor, loc=loc, frameon=False)
        return fig, res_psd.psd, res_psd.freqs
    else:
        return res_psd.psd, res_psd.freqs


def gen_noise(alpha, t, f0=None, m=None):
    ''' Generate a colored noise timeseries

    Args:
        alpha (float): exponent of the 1/f^alpha noise
        t (float): time vector of the generated noise
        f0 (float): fundamental frequency
        m (int): maximum number of the waves, which determines the
            highest frequency of the components in the synthetic noise

    Returns:
        y (array): the generated 1/f^alpha noise

    References:
        Eq. (15) in Kirchner, J. W. Aliasing in 1/f(alpha) noise spectra: origins, consequences, and remedies.
            Phys Rev E Stat Nonlin Soft Matter Phys 71, 066110 (2005).
    '''
    n = np.size(t)  # number of time points
    y = np.zeros(n)

    if f0 is None:
        f0 = 1/n  # fundamental frequency
    if m is None:
        m = n

    k = np.arange(m) + 1  # wave numbers

    theta = np.random.rand(int(m))*2*np.pi  # random phase
    for j in range(n):
        coeff = (k*f0)**(-alpha/2)
        sin_func = np.sin(2*np.pi*k*f0*t[j] + theta)
        y[j] = np.sum(coeff*sin_func)

    return y


def plot_psds(psds, freqs, archive_type='glacier ice',
              period_ranges=[(1/200, 1/20), (1/8, 1/2)], period_names=[r'$\beta_D$', r'$\beta_I$'],
              period_ticks=[2, 5, 10, 20, 50, 100, 200, 500], title=None, legend_loc='best', legend_ncol=1,
              figsize=[8, 8], ax=None, ylim=None):
    ''' Plot PSDs with scaling slopes

    Args:
        psds (2d array): a Pandas DataFrame
        freqs (array): frequency vector for spectral analysis
        period_ranges (list of tuples): the period range over which to calculate the scaling slope

    Returns:
        ax (Axes): the PSD plot

    '''

    p = PAGES2k()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.set(style="darkgrid", font_scale=2)

    for i, psd in enumerate(psds):
        plt.plot(1/freqs, psd, color=p.colors_dict[archive_type], alpha=0.3)


    # plot the median of psds
    ax.set_xscale('log', nonposx='clip')
    ax.set_yscale('log', nonposy='clip')
    psd_med = np.nanmedian(psds, axis=0)
    ax.plot(1/freqs, psd_med, '-', color=sns.xkcd_rgb['denim blue'], label='Median')

    n_pr = len(period_ranges)
    f_binned_list = []
    Y_reg_list = []
    beta_list = []

    for period_range in period_ranges:
        beta, f_binned, psd_binned, Y_reg, stderr = Spectral.beta_estimation(psd_med, freqs, period_range[0], period_range[1])
        f_binned_list.append(f_binned)
        Y_reg_list.append(Y_reg)
        beta_list.append(beta)

    for i in range(n_pr):
        if i == 0:
            label = ''
            for j in range(n_pr):
                if j < n_pr-1:
                    label += period_names[j]+' = {:.2f}, '.format(beta_list[j])
                else:
                    label += period_names[j]+' = {:.2f}'.format(beta_list[j])

            ax.plot(1/f_binned_list[i], Y_reg_list[i], color='k', label=label)
        else:
            ax.plot(1/f_binned_list[i], Y_reg_list[i], color='k')

    if ylim:
        ax.set_ylim(ylim)

    if title:
        ax.set_title(title, fontweight='bold')

    ax.set_ylabel('Spectral Density')
    ax.set_xlabel('Period (years)')
    ax.set_xticks(period_ticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.set_xlim([np.min(period_ticks), np.max(period_ticks)])
    ax.invert_xaxis()
    ax.legend(loc=legend_loc, ncol=legend_ncol)

    return ax


def plot_psds_dist(psds, freqs, archive_type='glacier ice',
              period_ranges=[(1/200, 1/20), (1/8, 1/2)], period_names=[r'$\beta_D$', r'$\beta_I$'],
              period_ticks=[2, 5, 10, 20, 50, 100, 200, 500], title=None, legend_loc='best', legend_ncol=1,
              figsize=[8, 8], ax=None, ylim=None):
    ''' Plot PSDs with scaling slopes

    Args:
        psds (2d array): a Pandas DataFrame
        freqs (array): frequency vector for spectral analysis
        period_ranges (list of tuples): the period range over which to calculate the scaling slope

    Returns:
        ax (Axes): the PSD plot

    '''

    p = PAGES2k()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.set(style="darkgrid", font_scale=2)

    for i, psd in enumerate(psds):
        plt.plot(1/freqs, psd, color=p.colors_dict[archive_type], alpha=0.3)


    # plot the median of psds
    ax.set_xscale('log', nonposy='clip')
    ax.set_yscale('log', nonposy='clip')
    psd_med = np.nanmedian(psds, axis=0)
    ax.plot(1/freqs, psd_med, '-', color=sns.xkcd_rgb['denim blue'], label='Median')

    n_pr = len(period_ranges)
    f_binned_list = []
    Y_reg_list = []
    beta_list = []

    for period_range in period_ranges:
        beta, f_binned, psd_binned, Y_reg, stderr = Spectral.beta_estimation(psd_med, freqs, period_range[0], period_range[1])
        f_binned_list.append(f_binned)
        Y_reg_list.append(Y_reg)
        beta_list.append(beta)

    for i in range(n_pr):
        if i == 0:
            label = ''
            for j in range(n_pr):
                if j < n_pr-1:
                    label += period_names[j]+' = {:.2f}, '.format(beta_list[j])
                else:
                    label += period_names[j]+' = {:.2f}'.format(beta_list[j])

            ax.plot(1/f_binned_list[i], Y_reg_list[i], color='k', label=label)
        else:
            ax.plot(1/f_binned_list[i], Y_reg_list[i], color='k')

    if ylim:
        ax.set_ylim(ylim)

    if title:
        ax.set_title(title, fontweight='bold')

    ax.set_ylabel('Spectral Density')
    ax.set_xlabel('Period (years)')
    ax.set_xticks(period_ticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.set_xlim([np.min(period_ticks), np.max(period_ticks)])
    plt.gca().invert_xaxis()
    ax.legend(loc=legend_loc, ncol=legend_ncol)

    return ax


def plot_sites(df, title=None, lon_col='geo_meanLon', lat_col='geo_meanLat', archiveType_col='archiveType',
               title_size=20, title_weight='bold', figsize=[10, 8], projection=ccrs.Robinson(), markersize=50,
               plot_legend=True, legend_ncol=3, legend_anchor=(0, -0.4), legend_fontsize=15, frameon=False, ax=None):

    ''' Plot the location of the sites on a map

    Args:
        df (Pandas DataFrame): the Pandas DataFrame

    Returns:
        ax (Axes): the map plot of the sites

    '''
    p = PAGES2k()
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(projection=projection)

    sns.set(style="ticks", font_scale=2)

    # plot map
    if title:
        plt.title(title, fontsize=title_size, fontweight=title_weight)

    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)
    ax.gridlines(edgecolor='gray', linestyle=':')

    # plot markers by archive types
    s_plots = []
    type_names = []
    df_archiveType_set = np.unique(df[archiveType_col])
    for type_name in df_archiveType_set:
        selector = df[archiveType_col] == type_name
        type_names.append(f'{type_name} (n={len(df[selector])})')
        s_plots.append(
            ax.scatter(
                df[selector][lon_col], df[selector][lat_col], marker=p.markers_dict[type_name],
                c=p.colors_dict[type_name], edgecolor='k', s=markersize, transform=ccrs.Geodetic()
            )
        )

    # plot legend
    if plot_legend:
        plt.legend(
            s_plots, type_names,
            scatterpoints=1,
            bbox_to_anchor=legend_anchor,
            loc='lower left',
            ncol=legend_ncol,
            frameon=frameon,
            fontsize=legend_fontsize
        )

    return ax


def plot_beta_map(df_beta,
                  beta_I_name='beta_I', beta_I_title='Interannual Scaling Exponent',
                  beta_D_name='beta_D', beta_D_title='Decadal to Centennial Scaling Exponent',
                  hist_xlim=(-3.5, 3.5), nan_color='gray',
                  cmap="RdBu_r", vmin=-2, vmax=2, n_ticks=11, n_clr=10,
                  color_range_start=0, color_range_end=None):
    ''' Plot beta map for beta_I and beta_D with histogram plots

    Args:
        df_beta (Pandas DataFrame): the Pandas DataFrame with beta_I and beta_D

    Returns:
        fig (Figure): the beta plot

    '''
    color_norm = Normalize(vmin=vmin, vmax=vmax)
    tick_range = np.linspace(vmin, vmax, n_ticks)

    # set colormap
    palette = sns.color_palette(cmap, n_clr)
    palette_chosen = palette[color_range_start:color_range_end]
    sns_cmap = ListedColormap(palette_chosen)
    sns_cmap.set_bad(color='gray', alpha = 1.)

    p = PAGES2k()
    fig = plt.figure(figsize=[20, 10])
    sns.set(style="ticks", font_scale=1.5)

    # map 1
    map1 = plt.subplot(2, 10, (1, 7), projection=ccrs.Robinson())
    map1.set_title(beta_I_title)
    map1.set_global()
    map1.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)
    map1.gridlines(edgecolor='gray', linestyle=':')

    s_plots = []
    cbar_added = False
    for type_name in p.archive_types:

        selector = df_beta['archiveType'] == type_name
        color = df_beta[selector][beta_I_name].values
        good_idx = ~np.isnan(color)
        bad_idx = np.isnan(color)
        lon = df_beta[selector]['geo_meanLon'].values
        lat = df_beta[selector]['geo_meanLat'].values
        if np.size(color[good_idx]) > 0:
            sc = map1.scatter(
                lon[good_idx], lat[good_idx], marker=p.markers_dict[type_name],
                c=color[good_idx], edgecolor='k', s=100, transform=ccrs.Geodetic(), cmap=sns_cmap, norm=color_norm,
                zorder=99
            )


            if not cbar_added:
                cbar_added = True
                cbar = plt.colorbar(mappable=sc, ax=map1, drawedges=True,
                                    orientation='vertical', ticks=tick_range, fraction=0.05, pad=0.05)
                cbar.ax.tick_params(axis='y', direction='in')
                cbar.set_label(r'$\{}$'.format(beta_I_name))

        if np.size(color[bad_idx]) > 0:
            sc = map1.scatter(
                lon[bad_idx], lat[bad_idx], marker=p.markers_dict[type_name],
                c=nan_color, edgecolor='k', s=100, transform=ccrs.Geodetic(), cmap=sns_cmap
            )

        s_plots.append(sc)

    lgnd = plt.legend(s_plots, p.archive_types,
                      scatterpoints=1,
                      bbox_to_anchor=(-0.4, 1.1),
                      loc='upper left',
                      ncol=1,
                      fontsize=17)

    for i in range(len(p.archive_types)):
        lgnd.legendHandles[i].set_color(nan_color)

    # hist 1
    hist1 = plt.subplot(2,10,(9,10))
    hist1.set_title('Distribution', y=1.0)
    beta_I_data = np.asarray(df_beta[beta_I_name].dropna())
    hist1.set_title('median: {:.3f}'.format(np.median(beta_I_data)), loc='right', x=0.98, y=0.9)
    g1 = sns.distplot(beta_I_data, kde=True, color="gray")
    g1.set(xlim=hist_xlim)
    plt.grid()

    # map 2
    map2 = plt.subplot(2, 10, (11, 17), projection=ccrs.Robinson())
    map2.set_title('Decadal to Centennial Scaling Exponent')
    map2.set_global()
    map2.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)
    map2.gridlines(edgecolor='gray', linestyle=':')

    cbar_added = False
    for type_name in p.archive_types:

        selector = df_beta['archiveType'] == type_name
        color = df_beta[selector][beta_D_name].values
        good_idx = ~np.isnan(color)
        bad_idx = np.isnan(color)
        lon = df_beta[selector]['geo_meanLon'].values
        lat = df_beta[selector]['geo_meanLat'].values
        if np.size(color[good_idx]) > 0:
            sc = map2.scatter(
                    lon[good_idx], lat[good_idx], marker=p.markers_dict[type_name],
                    c=color[good_idx], edgecolor='k', s=100, transform=ccrs.Geodetic(), cmap=sns_cmap, norm=color_norm,
                    zorder=99
            )
            if not cbar_added:
                cbar_added = True
                cbar = plt.colorbar(mappable=sc, ax=map2, drawedges=True,
                                    orientation='vertical', ticks=tick_range, fraction=0.05, pad=0.05)
                cbar.ax.tick_params(axis='y', direction='in')
                cbar.set_label(r'$\{}$'.format(beta_D_name))

        if np.size(color[bad_idx]) > 0:
            map2.scatter(
                    lon[bad_idx], lat[bad_idx], marker=p.markers_dict[type_name],
                    c=nan_color, edgecolor='k', s=100, transform=ccrs.Geodetic(), cmap=sns_cmap
            )


    # hist 2
    hist2 = plt.subplot(2,10, (19,20))
    hist2.set_title('Distribution', y=1.0)
    beta_D_data = np.asarray(df_beta['beta_D'].dropna())
    hist2.set_title('median: {:.3f}'.format(np.median(beta_D_data)), loc='right', x=0.98, y=0.9)
    g2 = sns.distplot(beta_D_data, kde=True, color="gray")
    g2.set(xlim=hist_xlim)
    plt.grid()

    return fig


def plot_beta_hist(df_beta, archives,
                   beta_I_name='beta_I',
                   beta_D_name='beta_D',
                   figsize=[10, 10], xlim=None, xticks=None,
                   font_scale=1.5,
                   grids=(1, 1)):
    ''' Plot the histagram of beta_I and beta_D with KDE distributions for an archive

    Args:
        df_beta (Pandas DataFrame): the scaling exponents
        archives (list of str): the list of the archives to plot
        grids (a list of 3 integers): nRow, nCol, loc
        [bivalve, borehole, coral, documents, glacier ice, hybrid,
        lake sediment, marine sediment, sclerosponge, speleothem, tree]

    Returns:
        plot: a figure with the histagram of beta_I and beta_D with KDE distributions

    '''
    p = PAGES2k()
    sns.set(style='darkgrid', font_scale=font_scale)
    #plt.style.use('ggplot')
    fig = plt.figure(figsize=figsize)

    nRow, nCol = grids

    for i, type_name in enumerate(archives):
        print('Processing {}...'.format(type_name))

        selector = df_beta['archiveType'] == type_name
        color = p.colors_dict[type_name]

        nr = df_beta[selector].count()[0]
        hist = plt.subplot(nRow, nCol, i+1)
        hist.set_title('{}, {} records'.format(type_name, nr), y=1.0, fontweight="bold")

        beta_I = df_beta[selector][beta_I_name].dropna().values
        beta_D = df_beta[selector][beta_D_name].dropna().values
        med_I = np.median(beta_I)
        med_D = np.median(beta_D)
        n_I = np.size(beta_I)
        n_D = np.size(beta_D)

        g1 = sns.kdeplot(beta_D, shade=False, color=color, linestyle='-',
                         label=r'$\{}$ = {:.2f} ({} records)'.format(beta_D_name, med_D, n_D))
        g1.axvline(x=med_D, ymin=0, ymax=0.1, linewidth=1, color=color, linestyle='-')
        g2 = sns.kdeplot(beta_I, shade=False, color=color, linestyle='--',
                         label=r'$\{}$ = {:.2f} ({} records)'.format(beta_I_name, med_I, n_I))
        g2.axvline(x=med_I, ymin=0, ymax=0.1, linewidth=1, color=color, linestyle='--')

        if xlim:
            g1.set(xlim=xlim)
            g2.set(xlim=xlim)

        if xticks:
            g1.set(xticks=xticks)
            g2.set(xticks=xticks)

    fig.tight_layout()
    return fig


def plot_psd_betahist(dfs, figsize=None, period_names=['beta_D', 'beta_I'],
                      ax1_ylim=None,
                      lgd_loc='upper right', lgd_anchor=(1.4, 1),
                      period_ticks=[2, 5, 10, 20, 50, 100, 200, 500]
                      ):
    ''' Plot the PSD as well as the distribution of beta_I and beta_D

    Args:
        dfs (a list of Pandas DataFrame): dfs for different types

    Returns:
        plot: a figure with the histagram of beta_I and beta_D with KDE distributions

    '''
    p = PAGES2k()

    n_dfs = len(dfs)
    if figsize is None:
        figsize = [12,  4*n_dfs]

    fig = plt.figure(figsize=figsize)


    for i, df in enumerate(dfs):
        # PSD
        sns.set(style="darkgrid", font_scale=2)
        ax1 = plt.subplot(n_dfs, 2, 2*i+1)
        for index, row in df.iterrows():
            archive_type = row['archiveType']
            ax1.loglog(1/row['freqs'], row['psd'], color=p.colors_dict[archive_type], alpha=0.3)

        ax1.set_xticks(period_ticks)
        ax1.set_ylabel('Spectral Density')
        ax1.set_xlabel('Period (years)')
        ax1.set_xticks(period_ticks)
        if ax1_ylim is not None:
            ax1.set_ylim(ax1_ylim)
        ax1.get_xaxis().set_major_formatter(ScalarFormatter())
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax1.set_xlim([np.min(period_ticks), np.max(period_ticks)])
        ax1.invert_xaxis()
        ax1.set_title('{}, {} records'.format(archive_type, len(df)), fontweight='bold')

        # distribution
        sns.set(style="ticks", font_scale=2)
        ax2 = plt.subplot(n_dfs, 2, 2*i+2)
        beta_I_str = period_names[1]
        beta_D_str = period_names[0]
        beta_I = df[beta_I_str].dropna().values
        beta_D = df[beta_D_str].dropna().values
        med_I = np.median(beta_I)
        med_D = np.median(beta_D)
        n_I = np.size(beta_I)
        n_D = np.size(beta_D)
        g1 = sns.kdeplot(beta_D, shade=False, color=p.colors_dict[archive_type], linestyle='-', ax=ax2,
                         label=r'$\{}$ = {:.2f} ({} records)'.format(beta_D_str, med_D, n_D))
        g1.axvline(x=med_D, ymin=0, ymax=0.1, linewidth=1, color=p.colors_dict[archive_type], linestyle='-')
        g2 = sns.kdeplot(beta_I, shade=False, color=p.colors_dict[archive_type], linestyle='--', ax=ax2,
                         label=r'$\{}$ = {:.2f} ({} records)'.format(beta_I_str,  med_I, n_I))
        g2.axvline(x=med_I, ymin=0, ymax=0.1, linewidth=1, color=p.colors_dict[archive_type], linestyle='--')
        ax2.legend(fontsize=18, bbox_to_anchor=lgd_anchor, loc=lgd_loc, ncol=1)
        ax2.set_xlim([-3, 5])
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)

    fig.tight_layout()

    return fig


def plot_wavelet_summary(df_row, c1=1/(8*np.pi**2), c2=1e-3, nMC=200, nproc=8, detrend='no',
                         gaussianize=False, standardize=True, levels=None,
                         anti_alias=False, period_ticks=None,
                         psd_lmstyle='-', psd_lim=None, period_I=[1/8, 1/2], period_D=[1/200, 1/20]):
    """ Plot the time series with the wavelet analysis and psd

    Args:
        df_row (DateFrame): one row of a DataFrame
        freqs (array): vector of frequency
        tau (array): the evenly-spaced time points, namely the time shift for wavelet analysis
        c (float): the decay constant
        Neff (int): the threshold of the number of effective degree of freedom
        nproc (int): fake argument, just for convenience
        detrend (str): 'no' - the original time series is assumed to have no trend;
                       'linear' - a linear least-squares fit to `ys` is subtracted;
                       'constant' - the mean of `ys` is subtracted
        psd_lmstyle (str): the line style in the psd plot
        psd_lim (list): the limits for psd
        period_I, period_D (list): the ranges for beta estimation

    Returns:
        fig (figure): the summary plot

    """
    p = PAGES2k()
    title_font = {'fontname': 'Arial', 'size': '24', 'color': 'black', 'weight': 'normal', 'verticalalignment': 'bottom'}

    ys = np.asarray(df_row['paleoData_values'], dtype=np.float)
    ts = np.asarray(df_row['year'], dtype=np.float)
    ys, ts = Timeseries.clean_ts(ys, ts)
    dt_med = np.median(np.diff(ts))
    ylim_min = dt_med*2
    tau = np.linspace(np.min(ts), np.max(ts), np.min([np.size(ts)//2, 501]))
    #  freqs = np.linspace(1/1000, 1/2, 501)
    freqs = None

    if np.mean(np.diff(ts)) < 1:
        warnings.warn('The time series will be annualized due to mean of dt less than one year.')
        ys, ts = annualize_ts(ys, ts)

    if period_ticks is not None:
        period_ticks = np.asarray(period_ticks)
        gt_part = period_ticks[period_ticks >= ylim_min]
        period_ticks = np.concatenate(([np.floor(ylim_min)], gt_part))

    gs = gridspec.GridSpec(6, 12)
    gs.update(wspace=0, hspace=0)

    fig = plt.figure(figsize=(15, 15))

    # plot the time series
    sns.set(style="ticks", font_scale=1.5)
    ax1 = plt.subplot(gs[0:1, :-3])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.plot(ts, ys, '-o', color=p.colors_dict[df_row['archiveType']])

    plt.title(df_row['dataSetName']+' - '+df_row['archiveType'], **title_font)

    plt.xlim([np.min(ts), np.max(ts)])
    plt.ylabel('{} ({})'.format(df_row['paleoData_variableName'], df_row['paleoData_units']))

    #  plt.grid()
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

    # plot location
    sns.set(style="ticks", font_scale=1.5)
    ax_loc = plt.subplot(
        gs[0:1, -3:],
        projection=ccrs.NearsidePerspective(
            central_longitude=df_row['geo_meanLon'],
            central_latitude=df_row['geo_meanLat'],
            satellite_height=1.5*1e7,
        )
    )

    #  ax_loc = plt.subplot(gs[0:1, -3:], projection=ccrs.Robinson())

    ax_loc.set_global()
    ax_loc.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)
    ax_loc.gridlines(edgecolor='gray', linestyle=':')

    ax_loc.scatter(df_row['geo_meanLon'], df_row['geo_meanLat'], marker=p.markers_dict[df_row['archiveType']],
                   c=p.colors_dict[df_row['archiveType']], edgecolor='k', s=50, transform=ccrs.Geodetic())
    #  ax_loc.text(df_row['geo_meanLon']+40, df_row['geo_meanLat'], 'lon: {}\nlat:{}'.format(df_row['geo_meanLon'], df_row['geo_meanLat']), fontsize=15)

    # plot wwa
    sns.set(style="ticks", font_scale=1.5)
    ax2 = plt.subplot(gs[1:5, :-3])

    res_wwz = Spectral.wwz(ys, ts, freqs=freqs, tau=tau, c=c1, nMC=nMC, nproc=nproc, detrend=detrend,
                           gaussianize=gaussianize, standardize=standardize)

    lt_part = period_ticks[period_ticks <= np.max(res_wwz.coi)]
    period_ticks = np.concatenate((lt_part, [np.ceil(np.max(res_wwz.coi))]))

    period_tickslabel = list(map(str, period_ticks))
    period_tickslabel[0] = ''
    period_tickslabel[-1] = ''
    for i, label in enumerate(period_tickslabel):
        if label[-2:] == '.0':
            label =  label[:-2]
            period_tickslabel[i] = label

    Spectral.plot_wwa(res_wwz.wwa, res_wwz.freqs, res_wwz.tau, coi=res_wwz.coi, AR1_q=res_wwz.AR1_q,
                      yticks=period_ticks, yticks_label=period_tickslabel,
                      ylim=[np.min(period_ticks), np.max(res_wwz.coi)],
                      plot_cone=True, plot_signif=True, xlabel='Year ({})'.format(df_row['yearUnits']), ylabel='Period (years)',
                      ax=ax2, levels=levels,
                      cbar_orientation='horizontal', cbar_labelsize=15, cbar_pad=0.1, cbar_frac=0.15,
                      )

    # plot psd
    sns.set(style="ticks", font_scale=1.5)
    ax3 = plt.subplot(gs[1:4, -3:])
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    res_psd = Spectral.wwz_psd(ys, ts, freqs=None, tau=tau, c=c2, nproc=nproc, nMC=nMC,
                               detrend=detrend, gaussianize=gaussianize, standardize=standardize,
                               anti_alias=anti_alias)

    #  Spectral.plot_psd(psd, freqs, plot_ar1=True, psd_ar1_q95=psd_ar1_q95,
                      #  #  period_ticks=period_ticks[period_ticks < np.max(coi)],
                      #  period_ticks=period_ticks, period_tickslabel=period_tickslabel,
                      #  period_lim=[np.min(period_ticks), np.max(coi)], psd_lim=psd_lim,
                      #  color=p.colors_dict[df_row['archiveType']],
                      #  ar1_lmstyle='--', plot_gridlines=False,
                      #  lmstyle=psd_lmstyle, ax=ax3, period_label='',
                      #  label='Estimated spectrum', psd_label='', vertical=True)
    ax3.loglog(res_psd.psd, 1/res_psd.freqs, '-', label='Estimated spectrum', color=p.colors_dict[df_row['archiveType']])
    ax3.loglog(res_psd.psd_ar1_q95, 1/res_psd.freqs, '--', label='AR(1) 95%', color=sns.xkcd_rgb['pale red'])
    ax3.set_yticks(period_ticks)
    ax3.set_ylim([np.min(period_ticks), np.max(res_wwz.coi)])

    res_beta1 = Spectral.beta_estimation(res_psd.psd, res_psd.freqs, period_I[0], period_I[1])
    res_beta2 = Spectral.beta_estimation(res_psd.psd, res_psd.freqs, period_D[0], period_D[1])
    ax3.plot(res_beta1.Y_reg, 1/res_beta1.f_binned, color='k',
             label=r'$\beta_I$ = {:.2f}'.format(res_beta1.beta) + ', ' + r'$\beta_D$ = {:.2f}'.format(res_beta2.beta))
    #  ax3.plot(Y_reg_1, 1/f_binned_1, color='k')
    ax3.plot(res_beta2.Y_reg, 1/res_beta2.f_binned, color='k')

    #  if not np.isnan(beta_1):
        #  ax3.annotate(r'$\beta_I$ = {:.2f}'.format(beta_1),
                     #  xy=(0.1, 0.1),
                     #  arrowprops=dict(facecolor='black', shrink=0.05),
                     #  horizontalalignment='right', verticalalignment='top',
                     #  xycoords='axes fraction')
    #  if not np.isnan(beta_2):
        #  ax3.annotate(r'$\beta_D$ = {:.2f}'.format(beta_2),
                     #  xy=(0.1, 0.9),
                     #  arrowprops=dict(facecolor='black', shrink=0.05),
                     #  horizontalalignment='right', verticalalignment='top',
                     #  xycoords='axes fraction')

    plt.tick_params(axis='y', which='both', labelleft='off')
    #  plt.legend(fontsize=15, bbox_to_anchor=(0.1, -0.25), loc='lower left', ncol=1)
    plt.legend(fontsize=15, bbox_to_anchor=(0.1, -0.3), loc='lower left', ncol=1)

    return fig


def plot_ols(ys_proxy, ts_proxy, ys_obs, ts_obs, title='',
             proxy_label='proxy', obs_label='instrument'):
    ys_proxy_overlap, ys_obs_overlap, time_overlap = overlap_ts(ys_proxy, ts_proxy, ys_obs, ts_obs)

    ys_proxy_overlap_std, _, _ = Timeseries.standardize(ys_proxy_overlap)
    ys_obs_overlap_std, _, _ = Timeseries.standardize(ys_obs_overlap)

    model = ols_ts(ys_proxy_overlap_std, time_overlap, ys_obs_overlap_std, time_overlap)
    #  model = ols_ts(ys_proxy_overlap, time_overlap, ys_obs_overlap, time_overlap)
    results = model.fit()
    Y_reg = model.predict(results.params)

    # plot
    title_font = {'fontname': 'Arial', 'size': '24', 'color': 'black', 'weight': 'normal', 'verticalalignment': 'bottom'}

    gs = gridspec.GridSpec(6, 12)
    gs.update(wspace=2, hspace=2)

    fig = plt.figure(figsize=[12, 12])
    sns.set(style='ticks', font_scale=1.5)
    ax1 = plt.subplot(gs[0:2, :])
    ax1.plot(time_overlap, ys_proxy_overlap_std, '-o', label=proxy_label)
    ax1.plot(time_overlap, ys_obs_overlap_std, '-o', label=obs_label)
    ax1.set_ylabel('Standardized value')
    ax1.set_xlabel('Year (AD)')
    ax1.grid()
    #  ax1.legend(fontsize=15, bbox_to_anchor=(1.2, 1), loc='upper right', ncol=1)
    ax1.legend(ncol=2)
    ax1.set_title(title, **title_font)

    sns.set(style='darkgrid', font_scale=1.5)
    ax2 = plt.subplot(gs[2:5, 0:6])
    ax2.scatter(ys_proxy_overlap_std, ys_obs_overlap_std, alpha=0.5)
    ax2.set_xlabel(proxy_label)
    ax2.set_ylabel(obs_label)
    sort_order = np.argsort(ys_proxy_overlap_std)
    ax2.plot(ys_proxy_overlap_std[sort_order], Y_reg[sort_order], '--',
             label=r'R$^2$ = {:.2f}'.format(results.rsquared),
             #  label=r'R$^2$ = {:.2f}, slope = {:.2f}'.format(results.rsquared, results.params[1]),
             color= sns.xkcd_rgb["medium green"])
    ax2.set_title('Linear regression')
    ax2.legend()

    ax3 = plt.subplot(gs[2:5, 6:])
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    QQ = ProbPlot(results.resid)
    QQ.qqplot(line='45', alpha=0.5, color=sns.xkcd_rgb["denim blue"], lw=1, ax=ax3)
    ax3.set_title('QQ plot of residuals')

    return fig


def plot_composite(df, archive_type='coral',  title='', bin_width=10, lower_bd=10, stat_func=np.nanmedian,
                   intercept=0, slope=1, R2=np.nan, proxy_label='proxy', legend_loc='upper left',
                   plot_target=False, target_ys=None, target_ts=None, target_label='instrumental',
                   left_ylim=[-2, 2], xlim=[0, 2000], right_ylim=[0, 80],
                   n_right_yticks=None):
    title_font = {'fontname': 'Arial', 'size': '24', 'color': 'black', 'weight': 'normal', 'verticalalignment': 'bottom'}

    ts = np.asarray(df.index)
    n_ts = ts.shape[0]
    ys_bootstrap = df['bootstrap_stats'].values
    ys_quantile_median = np.zeros(n_ts)
    ys_quantile_low = np.zeros(n_ts)
    ys_quantile_high = np.zeros(n_ts)
    for i, ys_boot in enumerate(ys_bootstrap):
        #  ys_quantile_median[i] = mquantiles(ys_boot, 0.5)
        ys_quantile_low[i] = mquantiles(ys_boot, 0.025)
        ys_quantile_high[i] = mquantiles(ys_boot, 0.975)
        #  ys_quantile_median[i] = (ys_quantile_low[i]+ys_quantile_high[i])/2

    def convert(x, intercept, slope):
        if np.isnan(slope):
            return x
        else:
            return intercept + x * slope

    if stat_func is np.nanmean:
        ys_quantile_median = np.asarray(df['mean'].values)
    elif stat_func is np.nanmedian:
        ys_quantile_median = np.asarray(df['median'].values)

    ys_quantile_median = convert(ys_quantile_median, intercept, slope)
    ys_quantile_low = convert(ys_quantile_low, intercept, slope)
    ys_quantile_high = convert(ys_quantile_high, intercept, slope)

    n_value = np.asarray(df['value'].apply(len).values)

    # bin data for plot
    bin_quantile_median, bin_time = bin_ts(ys_quantile_median, ts, bin_width=bin_width)
    bin_quantile_low, bin_time = bin_ts(ys_quantile_low, ts, bin_width=bin_width)
    bin_quantile_high, bin_time = bin_ts(ys_quantile_high, ts, bin_width=bin_width)
    bin_nvalue, bin_time = bin_ts(n_value, ts, bin_width=bin_width)

    bin_target_ys, bin_target_ts = bin_ts(target_ys, target_ts, bin_width=bin_width)

    # plot
    p = PAGES2k()
    sns.set(style='ticks', font_scale=2)
    proxy_color = p.colors_dict[archive_type]
    num_rec_color = sns.xkcd_rgb['grey']

    fig, ax1 = plt.subplots(figsize=[12, 6])

    ax1.plot(bin_time, bin_quantile_median, ':', color=proxy_color)
    selector = bin_nvalue >= lower_bd
    ax1.plot(bin_time[selector], bin_quantile_median[selector], '-', color=proxy_color,
             label='{}, conversion factor = {:.3f}, r = {:.3f}'.format(proxy_label, slope, np.sqrt(R2)))
    ax1.fill_between(bin_time, bin_quantile_low, bin_quantile_high, color=proxy_color, alpha=0.2)
    ax1.plot(bin_target_ts, bin_target_ys, '-', color='red', label=target_label)

    ax1.yaxis.grid('on', color=proxy_color, alpha=0.5)
    ax1.set_ylabel('proxy', color=proxy_color)
    ax1.tick_params('y', colors=proxy_color)
    ax1.set_xlabel('year (AD)')
    ax1.spines['left'].set_color(proxy_color)
    ax1.spines['right'].set_color(num_rec_color)
    ax1.spines['bottom'].set_color(proxy_color)
    ax1.spines['bottom'].set_alpha(0.5)
    ax1.spines['top'].set_visible(False)
    ax1.set_ylim(left_ylim)
    ax1.set_xlim(xlim)
    ax1.set_yticks(np.linspace(np.min(left_ylim), np.max(left_ylim), 5))
    ax1.set_xticks(np.linspace(np.min(xlim), np.max(xlim), 5))
    ax1.set_title(title, **title_font)
    ax1.legend(loc=legend_loc, frameon=False)

    ax2 = ax1.twinx()
    ax2.bar(ts, n_value, bin_width*0.9, color=num_rec_color, alpha=0.2)
    ax2.set_ylabel('# records', color=num_rec_color)
    ax2.tick_params(axis='y', colors=num_rec_color)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylim(right_ylim)
    if n_right_yticks:
        ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], n_right_yticks))
    ax2.grid('off')

    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)

    return fig


def make_composite(df_proxy, inst_temp_path, fig_save_path=None,
                   bin_width=10, lower_bd=10, n_bootstraps=1000,
                   archive_type='coral',  title='', stat_func=np.nanmean,
                   intercept=0, slope=1, proxy_label='proxy', legend_loc='upper left',
                   plot_target=False, target_ys=None, target_ts=None, target_label='instrumental',
                   netcdf_lat_name='latitude',
                   netcdf_lon_name='longitude',
                   netcdf_time_name='year',
                   netcdf_temp_name='temperature_anomaly',
                   ensemble_num=None,
                   left_ylim=[-2, 2], xlim=[0, 2000], right_ylim=[0, 80],
                   n_right_yticks=None):
    ''' Make composites and plots from a given dataframe of proxy records.
    '''
    n_records = df_proxy.shape[0]

    # append the nearest observation to the dataframe of proxy records
    df_proxy_obs = df_append_nearest_obs(df_proxy, inst_temp_path,
                                         lat_name=netcdf_lat_name,
                                         lon_name=netcdf_lon_name,
                                         time_name=netcdf_time_name,
                                         temp_name=netcdf_temp_name,
                                         ensemble_num=ensemble_num,
                                         )

    # make composites of the proxy records
    df_comp_proxy = df2composite(df_proxy_obs, bin_width=bin_width, stat_func=stat_func,
                                 gaussianize=True, standardize=True,
                                 n_bootstraps=n_bootstraps)
    #  df_comp_obs = df2composite(df_proxy_obs, bin_width=bin_width, stat_func=stat_func,
    #                             value_name='obs_temp', time_name='obs_year',
    #                             gaussianize=False, standardize=False,
    #                             n_bootstraps=n_bootstraps)

    # calculate the linear regression statistics
    ys_proxy, ts_proxy, ys_obs, ts_obs, intercept, slope, R2 = df2comp_ols(df_proxy, inst_temp_path,
                                                                           netcdf_lat_name=netcdf_lat_name,
                                                                           netcdf_lon_name=netcdf_lon_name,
                                                                           netcdf_time_name=netcdf_time_name,
                                                                           netcdf_temp_name=netcdf_temp_name,
                                                                           ensemble_num=ensemble_num,
                                                                           stat_func=stat_func,
                                                                           bin_width=bin_width)

    # bin the observation timeseires, for plotting purposes
    bin_target_ys, bin_target_ts = bin_ts(ys_obs, ts_obs, bin_width=bin_width)

    # plot
    fig = plot_composite(df_comp_proxy, archive_type=archive_type, R2=R2, stat_func=stat_func,
                         slope=slope, intercept=intercept,
                         plot_target=True, target_ys=ys_obs, target_ts=ts_obs,
                         title=title, left_ylim=left_ylim, xlim=xlim, right_ylim=[0, 10*(n_records//10+1)],
                         bin_width=bin_width)

    if fig_save_path:
        fig.savefig(fig_save_path, bbox_inches='tight')
        plt.close(fig)
        return ys_proxy, ts_proxy, slope, R2

    else:
        return ys_proxy, ts_proxy, slope, R2, fig
