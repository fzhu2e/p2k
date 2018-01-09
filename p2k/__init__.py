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

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from matplotlib.colors import Normalize, ListedColormap
from matplotlib import gridspec
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
    target_locations_dup = list(zip(target_lat, target_lon))
    target_locations = list(set(target_locations_dup))  # remove duplicated locations
    n_loc = np.shape(target_locations)[0]

    lat_ind = np.zeros(n_loc, dtype=int)
    lon_ind = np.zeros(n_loc, dtype=int)
    df_ind = np.zeros(n_loc, dtype=int)

    # get the closest grid
    for i, target_loc in enumerate(target_locations):
        X = target_loc
        Y = model_locations
        distance, index = spatial.KDTree(Y).query(X)
        closest = Y[index]
        lat_ind[i] = list(lat).index(closest[0])
        lon_ind[i] = list(lon).index(closest[1])
        df_ind[i] = target_locations_dup.index(target_loc)

    return lat_ind, lon_ind, df_ind


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
    year_int = np.asarray(list(map(int, year_int)))
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


def df2psd(df, freqs, value_name='paleoData_values', time_name='year', save_path=None):
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
        psds[k, :], _, _, _ = Spectral.wwz_psd(Xo, to, freqs=freqs, tau=tau, c=1e-3, nproc=16, nMC=0)

    if save_path:
        print('p2k >>> Saving pickle file at: {}'.format(save_path))

        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        with open(save_path, 'wb') as f:
            pickle.dump(psds, f)

        print('p2k >>> DONE')

    return psds


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
        psds = df2psd(df, freqs, value_name=value_name, time_name=value_name)

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


def plot_psds(psds, freqs, archive_type='glacier ice',
              period_ranges=[(1/200, 1/20), (1/8, 1/2)], period_names=[r'$\beta_D$', r'$\beta_I$'],
              period_ticks=[2, 5, 10, 20, 50, 100, 200, 500], title=None,
              figsize=[8, 8], ax=None, ylim=[1e-3, 1e4]):
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
    ax.legend()

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

        g1 = sns.kdeplot(beta_I, shade=False, color=color, linestyle='--', label=r'$\{}$ ({} records)'.format(beta_I_name, n_I))
        g1.axvline(x=med_I, ymin=0, ymax=0.1, linewidth=1, color=color, linestyle='--')
        g2 = sns.kdeplot(beta_D, shade=False, color=color, linestyle='-', label=r'$\{}$ ({} records)'.format(beta_D_name, n_D))
        g2.axvline(x=med_D, ymin=0, ymax=0.1, linewidth=1, color=color, linestyle='-')

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
        ys, ts = Timeseries.annualize(ys, ts)

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

    Spectral.plot_psd(psd, freqs, plot_ar1=True, psd_ar1_q95=psd_ar1_q95,
                      #  period_ticks=period_ticks[period_ticks < np.max(coi)],
                      period_ticks=period_ticks, period_tickslabel=period_tickslabel,
                      period_lim=[np.min(period_ticks), np.max(coi)], psd_lim=psd_lim,
                      color=p.colors_dict[df_row['archiveType']],
                      ar1_lmstyle='--', plot_gridlines=False,
                      lmstyle=psd_lmstyle, ax=ax3, period_label='',
                      label='Estimated spectrum', psd_label='', vertical=True)

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
