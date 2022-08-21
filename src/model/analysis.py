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


def get_cumprod_dict(series_dict: dict[str, Series]) -> dict[str, Series]:
    output = {}
    for key, series in series_dict.items():
        output[key] = (1 + series).cumprod()

    return output


def plot_series(target: str, series_dict: dict[str, Series]) -> None:
    plt.figure(figsize=(15, 5))
    plt.title(f"{target} returns over time compared to other tech stocks")
    for ticker, series in series_dict.items():
        (color, alpha) = ("red", 0.8) if ticker == target else ("grey", 0.3)
        plt.plot(
            series,
            color=color,
            alpha=alpha,
        )

    tickers = list(series_dict.keys())
    plt.legend(tickers)


def plot_box_and_hist(
    target: str,
    box_series_dict: dict[str, Series],
    hist_series_dict: dict[str, Series],
):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    [ax_box, ax_hist] = ax

    ax_box.set_title(f"{target} box plot")
    ax_box.boxplot(box_series_dict[target])

    ax_hist = plot_hist(ax_hist, target, hist_series_dict)


def plot_hist(
    axes_subplot: plt.Axes, target: str, hist_series_dict: dict[str, Series]
) -> plt.Axes:
    axes_subplot.set_title(f"{target} histogram with normal distribution")

    mu = np.mean(hist_series_dict[target])
    sigma = np.std(hist_series_dict[target])
    x_points = np.linspace(start=mu - 5 * sigma, stop=mu + 5 * sigma, num=1000)
    pdf = st.norm.pdf(x=x_points, loc=mu, scale=sigma)
    axes_subplot.plot(x_points, pdf, linewidth=1, color="black")

    for ticker in hist_series_dict.keys():
        (color, alpha) = ("red", 0.8) if ticker == target else ("grey", 0.3)
        axes_subplot.hist(
            hist_series_dict[ticker], bins=40, color=color, alpha=alpha
        )

    return axes_subplot


def add_features(df: DataFrame) -> DataFrame:
    df_out = df.copy(deep=True)
    for n_roll in range(2, 8):
        df_out[f"Adj_Close_RM{n_roll}"] = df["Adj Close"].rolling(n_roll).mean()
        df_out[f"Adj_Close_RSTD{n_roll}"] = df["Adj Close"].rolling(n_roll).std()
        df_out[f"Adj_Close_RMAX{n_roll}"] = df["Adj Close"].rolling(n_roll).max()
        df_out[f"Adj_Close_RMIN{n_roll}"] = df["Adj Close"].rolling(n_roll).min()
        df_out[f"Adj_Close_RQ{n_roll}"] = (
            df["Adj Close"].rolling(n_roll).quantile(1)
        )

        df_out[f"Volume_RM{n_roll}"] = df["Volume"].rolling(n_roll).mean()

        df_out[f"Low_RSTD{n_roll}"] = df["Low"].rolling(n_roll).std()

        df_out[f"High_RSTD{n_roll}"] = df["High"].rolling(n_roll).std()

    for n_shift in range(2, 8):
        df_out[f"Close_S{n_shift}"] = df["Close"].shift(n_shift)

    df_out["Day"] = df.index.day
    df_out["Month"] = df.index.month
    df_out["Year"] = df.index.year
    df_out["Day_Year"] = df.index.day_of_year
    df_out["Weekday"] = df.index.weekday

    df_out["Upper_Shape"] = df["High"] - np.maximum(df["Open"], df["Close"])
    df_out["Lower_Shape"] = np.minimum(df["Open"], df["Close"]) - df["Low"]

    df_out.rename(columns={"Close": "Close_y"}, inplace=True)
    df_out.dropna(axis=0, inplace=True)

    return df_out
