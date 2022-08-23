from datetime import timedelta
from pandas.core.series import Series
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import yfinance as yf
from typing import Union


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
        df_out[f"Adj_Close_RM{n_roll}"] = (
            df["Adj Close"].rolling(n_roll).mean()
        )
        df_out[f"Adj_Close_RSTD{n_roll}"] = (
            df["Adj Close"].rolling(n_roll).std()
        )
        df_out[f"Adj_Close_RMAX{n_roll}"] = (
            df["Adj Close"].rolling(n_roll).max()
        )
        df_out[f"Adj_Close_RMIN{n_roll}"] = (
            df["Adj Close"].rolling(n_roll).min()
        )
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

    df_out.dropna(axis=0, inplace=True)

    return df_out


def put_column_to_last(df: DataFrame, col: str) -> DataFrame:
    col_list = list(df.columns)
    col_list.remove(col)
    col_list.append(col)

    return df[col_list]


def train_test_split(df: DataFrame, window: int) -> dict[str, DataFrame]:
    return {"train": df.iloc[:-window], "test": df.iloc[-window:]}


def train_validation_split(
    df: DataFrame, percentage: float
) -> dict[str, np.ndarray]:
    threshold = int(len(df) * percentage)

    return {
        "train": np.array(df.iloc[:threshold]),
        "validation": np.array(df.iloc[threshold:]),
    }


def windowing(
    data: np.ndarray, window: int, prediction_scope: int
) -> dict[str, np.ndarray]:
    input_data, target_data = [], []

    for i in range(len(data) - (window + prediction_scope)):
        input_data.append(np.array(data[i : i + window, :-1]))
        target_data.append(np.array(data[i + window + prediction_scope, -1]))

    return {"input": np.array(input_data), "target": np.array(target_data)}


def plot_concat_data(
    y_val: np.ndarray,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    df_for_x: DataFrame,
    mae: np.float64,
    window: int,
    prediction_scope: int,
):
    "plot comcatenated data: y-val + y-test + pred_test"
    plt.figure(figsize=(16, 8))

    plot_values = np.concatenate([y_val, y_test, pred_test])
    num_plot_values = len(plot_values)

    plot_list(
        data=y_val,
        color="blue",
        start_idx=0,
        start_val=None,
    )

    y_test_start_idx = len(y_val) - 1
    plot_list(
        data=y_test,
        color="orange",
        start_idx=y_test_start_idx,
        start_val=y_val[-1],
    )

    pred_test_start_idx = y_test_start_idx + len(y_test)
    plot_list(
        data=pred_test,
        color="red",
        start_idx=pred_test_start_idx,
        start_val=y_test[-1],
    )

    upper_band = plot_values + mae
    lower_band = plot_values - mae
    plt.plot(upper_band, color="grey", alpha=0.3)
    plt.plot(lower_band, color="grey", alpha=0.3)

    plt.fill_between(
        x=list(range(0, num_plot_values)),
        y1=upper_band,
        y2=lower_band,
        color="grey",
        alpha=0.1,
    )

    pred_value = round(pred_test[-1], 2)
    x_pos_delta = -0.5
    y_pos_delta = 2
    pred_test_last_idx = pred_test_start_idx + len(pred_test)
    plt.text(
        x=pred_test_last_idx + x_pos_delta,
        y=pred_test[-1] + y_pos_delta,
        s=str(pred_value) + "$",
        size=11,
        color="red",
    )

    dates_str = df_for_x.index[-(num_plot_values - 1) :].strftime("%y-%b-%d")
    predict_date = df_for_x.index[-1] + timedelta(prediction_scope + 1)
    predict_date_str = predict_date.strftime("%y-%b-%d")
    x_ticks = list(dates_str) + [predict_date_str]

    plt.xticks(
        ticks=list(range(0, num_plot_values)), labels=x_ticks, rotation=60
    )
    plt.title(
        f"Target date: {predict_date_str}, Predict value: {pred_value}, "
        f"MAE = {round(mae,2)}\n"
        f"To predict {prediction_scope + 1} days after, "
        f"used last {window} days data.",
        size=15,
    )
    plt.legend(
        ["Validation", "Testing Set (input for Prediction)", "Prediction"]
    )
    plt.show()


def plot_list(
    data: np.ndarray,
    color: str,
    start_idx: int,
    start_val: np.float64 = None,
) -> None:
    plot_values = data
    last_idx = len(data) - 1

    if start_idx:
        plot_values = np.insert(arr=data, obj=0, values=start_val)
        last_idx = start_idx + len(data)

    plt.plot(
        list(range(start_idx, last_idx + 1)),
        plot_values,
        marker=".",
        color=color,
    )


# target: "AAPL"
# ref: "SPY"
# start_date: "2001-11-30"
# end_date: "2022-08-19"
# target_col: "Close"
def data_prep_for_fitting(
    target_tic: str,
    target_col: str,
    ref_tic: str,
    start_date: str,
    end_date: str,
    window: int,
    percentage: float,
    prediction_scope: int,
) -> Union[dict[str, np.ndarray], dict[str, DataFrame]]:
    target_price_dict = download_price_dict(
        tickers=(target_tic,), start=start_date, end=end_date
    )
    target_price = target_price_dict[target_tic]
    target_price_feat = add_features(target_price)

    ref_price_dict = download_price_dict(
        tickers=(ref_tic,), start=start_date, end=end_date
    )

    ref_column_price_dict = get_series_dict(ref_price_dict, target_col)
    ref_close_price = ref_column_price_dict[ref_tic]

    target_price_feat[ref_tic] = ref_close_price
    target_col_new_name = target_col + "_y"
    target_price_feat.rename(
        columns={target_col: target_col_new_name}, inplace=True
    )
    target_price_feat_reordered = put_column_to_last(
        target_price_feat, target_col_new_name
    )

    train_test_dict = train_test_split(
        target_price_feat_reordered, window=window
    )
    train_reg = train_test_dict["train"]
    test_reg = train_test_dict["test"]

    train_val_dict = train_validation_split(train_reg, percentage=percentage)
    train_set_reg = train_val_dict["train"]
    validation_set_reg = train_val_dict["validation"]

    input_target_train_dict = windowing(
        train_set_reg, window=window, prediction_scope=prediction_scope
    )
    x_train_reg = input_target_train_dict["input"]
    y_train_reg = input_target_train_dict["target"]
    input_target_val_dict = windowing(
        validation_set_reg, window=window, prediction_scope=prediction_scope
    )
    x_val_reg = input_target_val_dict["input"]
    y_val_reg = input_target_val_dict["target"]

    x_train_reg_1d = x_train_reg.reshape(x_train_reg.shape[0], -1)
    x_val_reg_1d = x_val_reg.reshape(x_val_reg.shape[0], -1)

    x_test_reg_1d = np.array(test_reg.iloc[:, :-1])
    y_test_reg_1d = np.array(test_reg.iloc[:, -1])

    x_test_reg_1d_rs = x_test_reg_1d.reshape(1, -1)

    return {
        "features": target_price_feat_reordered,
        "x_train": x_train_reg_1d,
        "y_train": y_train_reg,
        "x_val": x_val_reg_1d,
        "y_val": y_val_reg,
        "x_test": x_test_reg_1d_rs,
        "y_test": y_test_reg_1d,
    }
