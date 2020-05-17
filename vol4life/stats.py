import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats


def get_ecdf(series_):
    return lambda x: (series_.sort_values() < x).astype(int).mean()


def get_sample_acvf(x, h):
    """
    x: series
    h: shift param
    return: autocovariance estimator
    """
    n = x.shape[0]
    shift_x = x.shift(h)
    mean = x.mean()
    result = (x - mean) * (shift_x - mean)
    return result.sum() / n


def autocovariance_f(x, nlags):
    """
    x: series
    nlags: range of lags param
    return: array of autocovariance estimators
    """
    results = np.array([get_sample_acvf(x, h) for h in range(nlags)])
    return results


def autocorrelation_f(x, nlags):
    """
    x: series
    nlags: range of lags param
    return: array of autocorrelation estimators
    """
    gammas = autocovariance_f(x, nlags)
    gamma_0 = get_sample_acvf(x, 0)
    return gammas / gamma_0


def rank_acf(x, nlags):
    """
    x: series
    nlags: range of lags param
    return: array of autocorrelation estimators
            using Spearman rank-order correlation
    """
    results = [sp.stats.spearmanr(x.shift(h), x, nan_policy='omit')[
        0] for h in range(nlags)]
    return np.array(results)


def get_sample_ccvf(x, y, h):
    """
    x: series
    y: series
    h: shift param
    return: cross-covariance estimator
    """
    n = x.shape[0]
    shift_x = x.shift(h)
    mean_x = x.mean()
    mean_y = y.mean()
    result = (shift_x - mean_x) * (y - mean_y)
    return result.sum() / n


def crosscorrelation_f(x, y, nlags):
    """
    x: series
    y: series
    nlags: range of lags param
    return: array of cross-correlation estimators
    """
    results = np.array([get_sample_ccvf(x, y, h) for h in range(nlags)])
    gamma_x_0 = get_sample_acvf(x, 0)
    gamma_y_0 = get_sample_acvf(y, 0)
    denominator = np.sqrt(gamma_x_0 * gamma_y_0)
    return results / denominator


def stats_ccf(x, y, nlags):
    return stats.ccf(y, x, unbiased=False)[:nlags]


def rank_sample_ccf(x, y, h):
    """
    x: series that we will perform the lag
    y: series
    h: lag param
    return: cross-correlation estimator
            using Spearman rank-order correlation
    """
    x_h = x.shift(h)
    return sp.stats.spearmanr(x_h, y, nan_policy='omit')[0]


def rank_ccf(x, y, nlags):
    """
    x: series
    y: series
    nlags: range of lags param
    return: array of cross-correlation estimators
    """
    results = [rank_sample_ccf(x, y, h) for h in range(nlags)]
    return np.array(results)
