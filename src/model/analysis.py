from pandas.core.series import Series
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import yfinance as yf


def get_shifted_log_rets(series: Series, shift_val: int = 1) -> Series:
    return np.log(series / series.shift(shift_val))


def download_price_dict(
    tickers: tuple, start: str, end: str
) -> dict[str, DataFrame]:
    output = {}
    for tic in tickers:
        output[tic] = yf.download(
            tickers=tic, start=start, end=end, progress=False
        )

    return output


def get_series_dict(
    df_dict: dict[str, DataFrame], column: str
) -> dict[str, Series]:
    output = {}
    for key, val in df_dict.items():
        output[key] = val[column]

    return output


def get_shifted_log_dict(
    series_dict: dict[str, Series], shift_val: int = 1
) -> dict[str, Series]:
    output = {}
    for key, series in series_dict.items():
        output[key] = get_shifted_log_rets(series, shift_val)

    return output


def plot_box_and_hist(target: str, series_dict: dict[str, Series]):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    [plot_box, plot_hist] = ax

    plot_box.set_title(f"New {target} box plot")
    plot_hist.set_title(f"{target} histogram with normal distribution")

    plot_box.boxplot(series_dict[target])

    mu = np.mean(series_dict[target])
    sigma = np.std(series_dict[target])
    x_points = np.linspace(start=mu - 5 * sigma, stop=mu + 5 * sigma, num=1000)
    pdf = st.norm.pdf(x=x_points, loc=mu, scale=sigma)
    plot_hist.plot(x_points, pdf, linewidth=1, color="black")

    for ticker in series_dict.keys():
        (color, alpha) = ("red", 0.8) if ticker == target else ("grey", 0.3)
        plot_hist.hist(series_dict[ticker], bins=40, color=color, alpha=alpha)


def load_test():
    print("load----a")
