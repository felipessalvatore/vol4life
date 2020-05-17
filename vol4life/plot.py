import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def plot_acf(x, lag_range, out_path,
             acf_function, reverse=True, figsize=(12, 5),
             title_fontsize=15, xlabel_fontsize=16, ylabel_fontsize=16):
    """
    plot autocorrelation of series x
    :param x: series that we will perform the lag
    :type x: pd.Series
    :param lag_range: range of lag
    :type lag_range: int
    :param out_path: path to save figure
    :type out_path: str
    :param ccf: cross-correlation function
    :type ccf: function
    :param reverse: param to reverse lags
    :type reverse: boolean
    :param figsize: figure size
    :type figsize: tuple
    :param title_fontsize: title font size
    :type title_fontsize: int
    :param xlabel_fontsize: x axis label size
    :type xlabel_fontsize: int
    :param ylabel_fontsize: y axis label size
    :type ylabel_fontsize: int
    """

    title = "{}".format(x.name)
    lags = range(lag_range)
    ac = acf_function(x, nlags=lag_range)
    sigma = 1 / np.sqrt(x.shape[0])
    fig, ax = plt.subplots(figsize=figsize)
    ax.vlines(lags, [0], ac)
    plt.plot(lags, [0] * len(lags), c="black", linewidth=1.0)
    plt.plot(lags, [2 * sigma] * len(lags), '-.', c="blue", linewidth=0.6)
    plt.plot(lags, [-2 * sigma] * len(lags), '-.', c="blue", linewidth=0.6)
    ax.set_xlabel('Lag', fontsize=xlabel_fontsize)
    ax.set_ylabel('autocorrelation', fontsize=ylabel_fontsize)
    fig.suptitle(title, fontsize=title_fontsize, fontweight='bold', y=0.93)
    if out_path is not None:
        plt.savefig(out_path)


def plot_ccf(x, y, lag_range, out_path,
             ccf, reverse=True, figsize=(12, 5),
             title_fontsize=15, xlabel_fontsize=16, ylabel_fontsize=16):
    """
    plot cross-correlation between series x and y
    :param x: series that we will perform the lag
    :type x: pd.Series
    :param y: series
    :type y: pd.Series
    :param lag_range: range of lag
    :type lag_range: int
    :param out_path: path to save figure
    :type out_path: str
    :param ccf: cross-correlation function
    :type ccf: function
    :param reverse: param to reverse lags
    :type reverse: boolean
    :param figsize: figure size
    :type figsize: tuple
    :param title_fontsize: title font size
    :type title_fontsize: int
    :param xlabel_fontsize: x axis label size
    :type xlabel_fontsize: int
    :param ylabel_fontsize: y axis label size
    :type ylabel_fontsize: int
    """

    title = "{} & {}".format(x.name, y.name)
    lags = range(-lag_range, lag_range + 1)
    left = ccf(x, y, lag_range + 1)
    rigt = ccf(y, x, lag_range)
    left = left[1:][::-1]
    cc = np.concatenate([left, rigt])

    sigma = 1 / np.sqrt(x.shape[0])
    fig, ax = plt.subplots(figsize=figsize)
    ax.vlines(lags, [0], cc)
    plt.plot(lags, [0] * len(lags), c="black", linewidth=1.0)
    plt.plot(lags, [2 * sigma] * len(lags), '-.', c="blue", linewidth=0.6)
    plt.plot(lags, [-2 * sigma] * len(lags), '-.', c="blue", linewidth=0.6)
    ax.set_xlabel('Lag', fontsize=xlabel_fontsize)
    ax.set_ylabel('cross-correlation', fontsize=ylabel_fontsize)
    fig.suptitle(title, fontsize=title_fontsize, fontweight='bold', y=0.93)
    if out_path is not None:
        plt.savefig(out_path)


def plot_simple_lr(x,
                   y,
                   lag,
                   out_path,
                   top_space=0.88,
                   height=8,
                   title_fontsize=20,
                   axis_fontsize=18):
    data = pd.concat([x.shift(lag), y], 1).dropna()
    x_name = x.name
    y_name = y.name
    r_2 = stats.pearsonr(data[x_name].values, data[y_name].values)[0] ** 2
    title = "{} x {}\n(lag = {}, $R^2$ = {:.5f})".format(
        x_name, y_name, lag, r_2)

    g = sns.jointplot(x_name, y_name, data=data, kind="reg", height=height)
    plt.subplots_adjust(top=top_space)
    g.fig.suptitle(title, fontsize=title_fontsize)
    g.set_axis_labels(x_name, y_name, fontsize=axis_fontsize)
    if out_path is not None:
        plt.savefig(out_path)
    plt.show()
