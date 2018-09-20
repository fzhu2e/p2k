#!/usr/bin/env python3
''' The python version of the simple varve model (in R lang) by Nick McKay

Adapted by Feng Zhu
'''
import scipy.stats as ss
import numpy as np
import fbm


def simpleVarveModel(signal, H, shape=1.5, mean=1, SNR=.25):
    ''' Simple PSM for varves

    Args:
        signal (array): the signal matrix, each row is a time series
        H (float): Hurst index, should be in (0, 1)
        '''

    # first, gammify the input
    gamSig = gammify(signal, shape=shape, mean=mean)

    # create gammafied autocorrelated fractal brownian motion series
    #     wa = Spectral.WaveletAnalysis()
    #     fBm_ts = wa.fBMsim(N=np.size(signal), H=H)
    fBm_ts = fbm.fgn(np.size(signal), H)
    gamNoise = gammify(fBm_ts, shape=shape, mean=mean)

    # combine the signal with the noise, based on the SNR
    varves = (gamSig*SNR + gamNoise*(1/SNR)) / (SNR + 1/SNR)

    return varves


def gammify(X, shape=1.5, mean=1, jitter=False):
    ''' Transform each **row** of data matrix X to a gaussian distribution using the inverse Rosenblatt transform
    '''
    X = np.matrix(X)
    n = np.shape(X)[0]  # number of rows
    p = np.shape(X)[1]  # number of columns
    if jitter:
        # add tiny random numbers to aviod ties
        X += np.random.normal(0, np.std(X)/1e6, n*p).reshape(n, p)

    Xn = np.matrix(np.zeros(n*p).reshape(n, p))
    for j in range(n):
        # sort the data in ascending order and retain permutation indices
        R = ss.rankdata(X[j, :])
        # the cumulative distribution function
        CDF = R/p - 1/(2*p)
        # apply the inverse Rosenblatt transformation
        rate = shape/mean
        Xn[j, :] = ss.gamma.ppf(CDF, shape, scale=1/rate)  # Xn is now gamma distributed

    return Xn


def moving_avg(ys, ts, start=0, end=11):
    ys_out = []
    ts = np.asarray(list(set(np.floor(ts))))
    ts_out = ts[:-1]
    nt = np.size(ts_out)
    for i in range(nt):
        ys_out.append(np.mean(ys[12*i+start:12*i+end]))

    ys_out = np.asarray(ys_out)
    return ys_out, ts_out

