import pandas as pd


def ccf_x_leads_y(x, y, ccf, nlags, head):
    """
    using the ccf function show if x leads y
    :param x: series that we will perform the lag
    :type x: pd.Series
    :param y: series
    :type y: pd.Series
    :param n_lags: range of lag
    :type n_lags: int
    """
    results = ccf(x, y, nlags)
    name = "{} leads {}".format(x.name, y.name)
    dict_ = {name: results}
    df = pd.DataFrame(dict_)
    df.index.name = "lags"
    sort_index = df[name].map(np.abs).sort_values(ascending=False).index
    return df.loc[sort_index].head(head)