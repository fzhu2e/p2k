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
    target_locations = list(zip(target_lat, target_lon))
    target_locations = list(set(target_locations))  # remove duplicated locations
    n_loc = np.shape(target_locations)[0]

    lat_ind = np.zeros(n_loc, dtype=int)
    lon_ind = np.zeros(n_loc, dtype=int)

    # get the closest grid
    for i, target_loc in enumerate(target_locations):
        X = target_loc
        Y = model_locations
        distance, index = spatial.KDTree(Y).query(X)
        closest = Y[index]
        lat_ind[i] = list(lat).index(closest[0])
        lon_ind[i] = list(lon).index(closest[1])

    return lat_ind, lon_ind


def load_CESM_netcdf(path, var_list):
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

    handle = xr.open_dataset(path, decode_times=False)

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


def annualize(var_field, year):
    ''' Annualize the variable field, whose first axis must be time

    Args:
        var_field (3d array): the 3D field of the variable

    Returns:
        var_ann (3d arrary): the annualized field
        year_set (array): the set of the years
    '''
    year_set = list(set(np.floor(year)))
    n_year = len(year_set)
    var_ann = np.ndarray(shape=(n_year, *var_field.shape[1:]))

    year_set_pad = list(year_set)
    year_set_pad.append(np.max(year_set)+1)

    for i in range(n_year):
        t_start = year_set_pad[i]
        t_end = year_set_pad[i+1]
        t_range = (year>=t_start) & (year<t_end)

        var_ann[i,:,:] = np.mean(var_field[t_range, :, :], axis=0)

    return var_ann, year_set
