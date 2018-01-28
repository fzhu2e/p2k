#!/usr/bin/env python3
__author__ = """Feng Zhu"""
__email__ = 'fengzhu@usc.edu'
__version__ = '0.1.0'

import os
import lipd as lpd
import pandas as pd
import numpy as np
from scipy import spatial
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

from tqdm import tqdm
import pickle
import warnings

from pyleoclim import Spectral, Timeseries

from . import psm

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
    #  markers = ['D', 'v', 'o', '+', 'd', '*', 's', 's', '>', 'x', '^']
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


def lipd2pkl(lipd_file_dir, pkl_file_path):
    ''' Convert a bunch of PAGES2k LiPD files to a pickle file to boost the speed of loading data

    Args:
        lipd_file_dir (str): the path of the PAGES2k LiPD files
        pkl_file_path (str): the path of the converted pickle file

    Returns:
        df (Pandas DataFrame): the converted Pandas DataFrame

    '''
    lipd_file_dir = os.path.abspath(lipd_file_dir)
    pkl_file_path = os.path.abspath(pkl_file_path)

    lipds = lpd.readLipd(lipd_file_dir)
    ts_list = lpd.extractTs(lipds)

    col_str = ['dataSetName', 'archiveType',
               'geo_meanElev', 'geo_meanLat', 'geo_meanLon',
               'year', 'yearUnits',
               'paleoData_variableName',
               'paleoData_units',
               'paleoData_values',
               'paleoData_proxy']

    df_tmp = pd.DataFrame(index=range(len(ts_list)), columns=col_str)

    i = 0
    for ts in ts_list:
        if 'paleoData_useInGlobalTemperatureAnalysis' in ts.keys() and ts['paleoData_useInGlobalTemperatureAnalysis'] == 'TRUE':

            for name in col_str:
                df_tmp.loc[i, name] = ts[name]

            i += 1

    df = df_tmp.dropna(how='all')

    print('p2k >>> Saving pickle file at: {}'.format(pkl_file_path))
    dir_name = os.path.dirname(pkl_file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    df.to_pickle(pkl_file_path)
    print('p2k >>> DONE')

    return df


def find_closest_loc(lat, lon, target_lat, target_lon):
    ''' Find the closet model sites (lat, lon) based on the given target (lat, lon) list

    Args:
        lat, lon (array): the model latitude and longitude arrays
        target_lat, target_lon (array): the target latitude and longitude arrays

    Returns:
        lat_ind, lon_ind (array): the indices of the found closest model sites

    '''

    # model locations
    mesh = np.meshgrid(lat, lon)

    list_of_grids = list(zip(*(grid.flat for grid in mesh)))
    model_lat, model_lon = zip(*list_of_grids)
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

    if np.size(target_lat) > 1:
        df_ind = np.zeros(n_loc, dtype=int)

    # get the closest grid
    for i, target_loc in enumerate(target_locations):
        X = target_loc
        Y = model_locations
        distance, index = spatial.KDTree(Y).query(X)
        closest = Y[index]
        lat_ind[i] = list(lat).index(closest[0])
        lon_ind[i] = list(lon).index(closest[1])
        #  if np.size(target_lat) > 1:
            #  df_ind[i] = target_locations_dup.index(target_loc)

    if np.size(target_lat) > 1:
        #  return lat_ind, lon_ind, df_ind
        return lat_ind, lon_ind
    else:
        return lat_ind[0], lon_ind[0]


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
            if var is 'lon':
                field = handle[var].values - 180  # make the longitude consistent with PAGES2k
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
        t_range = (ts >= t_start) & (ts < t_end)
        ys_ann[i] = np.average(ys[t_range], axis=0)

    return ys_ann, year_int


def bin_ts(ys, ts, bin_vector=None, bin_width=10, clean_ts=True):
    if bin_vector is None:
        bin_vector = np.arange(
            bin_width*(np.min(ts)//bin_width),
            bin_width*(np.max(ts)//bin_width+1),
            step=bin_width)

    ts_bin = (bin_vector[1:] + bin_vector[:-1])/2
    n_bin = np.size(ts_bin)
    ys_bin = np.zeros(n_bin)

    for i in range(n_bin):
        t_start = bin_vector[i]
        t_end = bin_vector[i+1]
        t_range = (ts >= t_start) & (ts < t_end)
        ys_bin[i] = np.average(ys[t_range], axis=0)

    if clean_ts:
        ys_bin, ts_bin = Timeseries.clean_ts(ys_bin, ts_bin)

    return ys_bin, ts_bin


def df2psd(df, freqs, value_name='paleoData_values', time_name='year', save_path=None, standardize=False):
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
    psds = np.ndarray(shape=(n_series, n_freqs))

    for k in tqdm(range(n_series), desc='Processing time series'):
        Xo = np.asarray(paleoData_values[k], dtype=np.float)
        to = np.asarray(year[k], dtype=np.float)
        Xo, to = Timeseries.clean_ts(Xo, to)
        tau = np.linspace(np.min(to), np.max(to), 501)
        psds[k, :], _, _, _ = Spectral.wwz_psd(Xo, to, freqs=freqs, tau=tau, c=1e-3, nproc=16, nMC=0, standardize=standardize)

    if save_path:
        print('p2k >>> Saving pickle file at: {}'.format(save_path))

        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        with open(save_path, 'wb') as f:
            pickle.dump(psds, f)

        print('p2k >>> DONE')

    return psds


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


def df2composite(df, value_name='paleoData_values', time_name='year', bin_width=10, standardize=False):
    df_comp = pd.DataFrame(columns=['time', 'value', 'min', 'max', 'median', 'mean'])
    df_comp.set_index('time', inplace=True)

    for index, row in df.iterrows():
        Xo, to = row2ts(row, clean_ts=True, value_name=value_name, time_name=time_name)

        if standardize:
            wa = Spectral.WaveletAnalysis()
            Xo = wa.preprocess(Xo, to, standardize=True)

        Xo_bin, to_bin = bin_ts(Xo, to, bin_width=bin_width)
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
        row['min'] = np.nanmin(row['value'])
        row['max'] = np.nanmax(row['value'])
        row['mean'] = np.nanmean(row['value'])
        row['median'] = np.nanmedian(row['value'])

    return df_comp


def df2comp_ols(df, inst_temp_path,
                value_name='paleoData_values', time_name='year',
                bin_width=10, lower_bd=3):
    df_proxy = df_append_nearest_obs(df, inst_temp_path)
    df_comp = df2composite(df_proxy, bin_width=bin_width, value_name=value_name, time_name=time_name)
    df_comp = clean_composite(df_comp, lower_bd=lower_bd)
    df_comp_obs = df2composite(df_proxy, bin_width=bin_width,
                                   value_name='obs_temp', time_name='obs_year')
    df_comp_obs = clean_composite(df_comp_obs, lower_bd=lower_bd)

    ts_proxy = np.asarray(df_comp.index)
    ys_proxy = np.asarray(df_comp['median'])
    ts_obs = np.asarray(df_comp_obs.index)
    ys_obs = np.asarray(df_comp_obs['median'])

    # OLS
    ys_proxy_overlap, ys_obs_overlap, time_overlap = overlap_ts(ys_proxy, ts_proxy, ys_obs, ts_obs)

    model = ols_ts(ys_proxy_overlap, time_overlap, ys_obs_overlap, time_overlap)
    results = model.fit()
    R2 = results.rsquared
    slope = results.params[1]

    return ys_proxy, ts_proxy, ys_obs, ts_obs, slope, R2


def clean_composite(df, lower_bd=3):
    for index, row in df.iterrows():
        if row['value'].shape[0] < lower_bd:
            df = df.drop(index)

    return df


def fill_nan_composite(df, lower_bd=3):
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


def df_append_nearest_obs(df, inst_temp_path):
    # load instrumental temperature data
    lat, lon, year, temp = load_CESM_netcdf(
        inst_temp_path, ['latitude', 'longitude', 'year', 'temperature_anomaly'], decode_times=False
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
        df.at[index, 'obs_temp'], df.at[index, 'obs_year'] = Timeseries.clean_ts(temp[:, lat_ind, lon_ind], year)

    return df


def df_append_converted_temp(df, inst_temp_path, bin_width=10, yr_range=None):
    # load instrumental temperature data
    lat, lon, year, temp = load_CESM_netcdf(
        inst_temp_path, ['latitude', 'longitude', 'year', 'temperature_anomaly'], decode_times=False
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


def df_append_beta(df, freqs, psds=None, save_path=None, value_name='paleoData_values', time_name='year',
                   period_ranges=[(1/200, 1/20), (1/8, 1/2)], period_names=['beta_D', 'beta_I']
                   ):
    ''' Calculate the scaling exponent and add to a new column in the given DataFrame

    Args:
        df (Pandas DataFrame): a Pandas DataFrame
        freqs (array): frequency vector for spectral analysis

    Returns:
        df_new (Pandas DataFrame): the DataFrame with scaling exponents added on

    '''
    if psds is None:
        psds = df2psd(df, freqs, value_name=value_name, time_name=time_name)

    df_new = df.copy()
    for i, period_range in enumerate(period_ranges):

        beta_list = []
        for psd in psds:
            beta, f_binned, psd_binned, Y_reg = Spectral.beta_estimation(psd, freqs, period_range[0], period_range[1])
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
        psds, freqs = df2psd_mtm(df, freqs, value_name=value_name, time_name=time_name)

    df_new = df.copy()
    for i, period_range in enumerate(period_ranges):

        beta_list = []
        for j, psd in enumerate(psds):
            beta, f_binned, psd_binned, Y_reg = Spectral.beta_estimation(psds[j], freqs[j], period_range[0], period_range[1])
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
    ax.set_xscale('log', nonposy='clip')
    ax.set_yscale('log', nonposy='clip')
    psd_med = np.nanmedian(psds, axis=0)
    ax.plot(1/freqs, psd_med, '-', color=sns.xkcd_rgb['denim blue'], label='Median')

    n_pr = len(period_ranges)
    f_binned_list = []
    Y_reg_list = []
    beta_list = []

    for period_range in period_ranges:
        beta, f_binned, psd_binned, Y_reg = Spectral.beta_estimation(psd_med, freqs, period_range[0], period_range[1])
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
        beta, f_binned, psd_binned, Y_reg = Spectral.beta_estimation(psd_med, freqs, period_range[0], period_range[1])
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
               plot_legend=True, legend_ncol=4, legend_anchor=(0, -0.3), legend_fontsize=15, ax=None):
    ''' Plot the location of the sites on a map.

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

    s_plots = []
    df_archiveType_set = np.unique(df[archiveType_col])
    for type_name in df_archiveType_set:
        selector = df[archiveType_col] == type_name
        s_plots.append(
            ax.scatter(
                df[selector][lon_col], df[selector][lat_col], marker=p.markers_dict[type_name],
                c=p.colors_dict[type_name], edgecolor='k', s=markersize, transform=ccrs.Geodetic()
            )
        )

    if plot_legend:
        # plot legend
        lgnd = plt.legend(s_plots, df_archiveType_set,
                          scatterpoints=1,
                          bbox_to_anchor=legend_anchor,
                          loc='lower left',
                          ncol=legend_ncol,
                          fontsize=legend_fontsize)

    return ax


def plot_beta_map(df_beta,
                  beta_I_name='beta_I', beta_I_title='Interannual Scaling Exponent',
                  beta_D_name='beta_D', beta_D_title='Decadal to Centennial Scaling Exponent',
                  hist_xlim=(-3, 6), nan_color='black',
                  cmap="RdBu_r", vmin=-1.2, vmax=2.8, n_ticks=11, n_clr=15,
                  color_range_start=5, color_range_end=None):
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


def plot_wavelet_summary(df_row, freqs=None, tau=None, c1=1/(8*np.pi**2), c2=1e-3, nMC=200, nproc=8, detrend='no',
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
    tau = np.linspace(np.min(ts), np.max(ts), 501)
    freqs = np.linspace(1/1000, 1/2, 501)

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

    wwa, phase, AR1_q, coi, freqs, tau, Neffs, coeff = \
        Spectral.wwz(ys, ts, freqs=freqs, tau=tau, c=c1, nMC=nMC, nproc=nproc, detrend=detrend,
                     gaussianize=gaussianize, standardize=standardize)

    lt_part = period_ticks[period_ticks <= np.max(coi)]
    period_ticks = np.concatenate((lt_part, [np.ceil(np.max(coi))]))

    period_tickslabel = list(map(str, period_ticks))
    period_tickslabel[0] = ''
    period_tickslabel[-1] = ''
    for i, label in enumerate(period_tickslabel):
        if label[-2:] == '.0':
            label =  label[:-2]
            period_tickslabel[i] = label

    Spectral.plot_wwa(wwa, freqs, tau, coi=coi, AR1_q=AR1_q, yticks=period_ticks, yticks_label=period_tickslabel,
                      ylim=[np.min(period_ticks), np.max(coi)],
                      plot_cone=True, plot_signif=True, xlabel='Year ({})'.format(df_row['yearUnits']), ylabel='Period (years)',
                      ax=ax2, levels=levels,
                      cbar_orientation='horizontal', cbar_labelsize=15, cbar_pad=0.1, cbar_frac=0.15,
                      )

    # plot psd
    sns.set(style="ticks", font_scale=1.5)
    ax3 = plt.subplot(gs[1:4, -3:])
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    psd, freqs, psd_ar1_q95, psd_ar1 = Spectral.wwz_psd(ys, ts, freqs=freqs, tau=tau, c=c2, nproc=nproc, nMC=nMC,
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
    ax3.loglog(psd, 1/freqs, '-', label='Estimated spectrum', color=p.colors_dict[df_row['archiveType']])
    ax3.loglog(psd_ar1_q95, 1/freqs, '--', label='AR(1) 95%', color=sns.xkcd_rgb['pale red'])
    ax3.set_yticks(period_ticks)
    ax3.set_ylim([np.min(period_ticks), np.max(coi)])

    beta_1, f_binned_1, psd_binned_1, Y_reg_1 = Spectral.beta_estimation(psd, freqs, period_I[0], period_I[1])
    beta_2, f_binned_2, psd_binned_2, Y_reg_2 = Spectral.beta_estimation(psd, freqs, period_D[0], period_D[1])
    ax3.plot(Y_reg_1, 1/f_binned_1, color='k',
             label=r'$\beta_I$ = {:.2f}'.format(beta_1) + ', ' + r'$\beta_D$ = {:.2f}'.format(beta_2))
    #  ax3.plot(Y_reg_1, 1/f_binned_1, color='k')
    ax3.plot(Y_reg_2, 1/f_binned_2, color='k')

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


def plot_composite(df):
    ts = np.asarray(df.index)
    ys_median = np.asarray(df['median'])
    ys_min = np.asarray(df['min'])
    ys_max = np.asarray(df['min'])
    n_value = np.asarray(df['value'].shape[0])

    fig = plt.figure(figsize=[12, 12])
    sns.set(style='ticks', font_scale=1.5)
