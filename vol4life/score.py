import numpy as np


def sharpe_ratio(returns,
                 rfr=0,
                 period='daily',
                 annualization=True):
    """
    Sharpe ratio of returns

    references:
    # https://augmentedtrader.com/2015/09/02/why-multiply-by-sqrt252-to-compute-the-sharpe-ratio/
    # https://www.investopedia.com/terms/s/sharperatio.asp

    :param returns: daily returns
    :type returns: pd.Series
    :param rfr: risk-free-rate throughout the period of the returns
    :type classes: float
    :param period: periodicity of the 'returns' data for annulazing
    :type period: str
    :param annualization: param to control annualizing version of sr
    :type annualization: Boolean
    :return: sharpe ratio
    :rtype: float
    """

    returns = returns - rfr

    ann_factors = {'monthly': 12,
                   'weekly': 52,
                   'daily': 252}
    ann_factor = ann_factors[period]

    if not annualization:
        ann_factor = 1

    raw_sr = np.divide(np.mean(returns), np.std(returns, ddof=1))
    ann_sr = raw_sr * np.sqrt(ann_factor)
    return ann_sr
