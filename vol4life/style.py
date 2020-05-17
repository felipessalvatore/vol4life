import pandas as pd


def color_negative_red_positive_green(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, green otherwise.
    :param val: input value
    :type val: float
    :return: css property
    :rtype: str
    """
    color = 'red' if val < 0 else 'green'
    return 'color: %s' % color


def show_red_green(df, round_format, display_format="%"):
    """
    show df using the color_negative_red_positive_green
    function
    :param df: input dataframe
    :type sdf: pd.DataFrame
    :param round_format: number of digits to round float number
    :type round_format: int
    :return: sharpe ratio styler
    :rtype: pd.io.formats.style.Styler
    """
    float_format = '{:,.' + '{}'.format(round_format) + display_format + "}"
    return df.style.format(float_format).applymap(
        color_negative_red_positive_green)