from datetime import timedelta
import json
from pathlib import Path, PosixPath
from typing import Union
import pandas as pd
from pandas.core.series import Series
from pandas.core.frame import DataFrame
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import tensorflow as tf
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
        print(f"Downloaded size for {tic}: {len(output[tic])}")

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


def add_features(df: DataFrame, roll_max: int, shift_max: int) -> DataFrame:
    df_out = df.copy(deep=True)

    for n_roll in range(2, roll_max + 1):
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

    for n_shift in range(1, shift_max + 1):
        df_out[f"Close_S{n_shift}"] = df["Close"].shift(n_shift)

    df_out["Day"] = df.index.day
    df_out["Month"] = df.index.month
    df_out["Year"] = df.index.year
    df_out["Day_Year"] = df.index.day_of_year
    df_out["Weekday"] = df.index.weekday

    df_out["Upper_Shape"] = df["High"] - np.maximum(df["Open"], df["Close"])
    df_out["Lower_Shape"] = np.minimum(df["Open"], df["Close"]) - df["Low"]

    return df_out


def put_column_to_last(df: DataFrame, col: str) -> DataFrame:
    col_list = list(df.columns)
    col_list.remove(col)
    col_list.append(col)

    return df[col_list]


def windowing(
    data: np.ndarray, window: int, prediction_scope: int
) -> dict[str, np.ndarray]:
    """
    Condition to get not empty list: len(data) > window + prediction_scope
    Input:
        data:
            [
                [11, 12, 13],
                [21, 22, 23],
                [31, 32, 33],
                [41, 42, 43],
                [51, 52, 53],
            ]
        window: 2
        prediction_scope: 0
    Output:
        "input":
            [
                [
                    [11, 12],
                    [21, 22]
                ],
                [
                    [21, 22],
                    [31, 32]
                ]
            ]
        "target":
            [43, 53]

    The len(Input) and len(Output) should be same.
    len(Input) == len(Output) == len(data) - window - prediction_scope
    """
    try:
        output_length = len(data) - window - prediction_scope
        assert output_length > 0
    except AssertionError as ae:
        print(
            f"Not enough data. Data length({len(data)}) should be more than "
            f"{window + prediction_scope}, "
            f"window: {window}, prediction scope: {prediction_scope}"
        )
        raise ae

    input_data, target_data = [], []

    for i in range(output_length):
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


def data_prep_for_fitting(
    target_tic: str,
    target_col: str,
    ref_tic: str,
    start_date: str,
    end_date: str,
    window: int,
    percentage: float,
    prediction_scope: int,
    is_lstm: bool,
    roll_max: int,
    shift_max: int,
    target_suffix: str = "_y",
) -> Union[dict[str, np.ndarray], dict[str, DataFrame]]:
    price_dict = download_price_dict(
        tickers=(target_tic, ref_tic), start=start_date, end=end_date
    )
    target_price = price_dict[target_tic]

    check_required_download_size(
        target_price,
        window,
        percentage,
        prediction_scope,
        roll_max,
        shift_max,
        is_lstm,
    )

    target_price_feat = target_price
    if not is_lstm:
        target_price_feat = add_features(
            target_price, roll_max=roll_max, shift_max=shift_max
        )
        target_price_feat.dropna(axis=0, inplace=True)

    target_price_feat[ref_tic] = price_dict[ref_tic][target_col]

    target_price_feat_reordered = rename_target_col_and_put_it_last(
        df=target_price_feat, target_col=target_col, suffix=target_suffix
    )

    train_val_test = make_train_val_test_set(
        target_price_feat_reordered, window, percentage
    )
    train_set_data = train_val_test["train"]
    validation_set_data = train_val_test["val"]
    test_data = train_val_test["test"]

    train_set_data_scaled = train_set_data
    validation_set_data_scaled = validation_set_data
    test_data_scaled = test_data

    if is_lstm:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(target_price_feat_reordered.to_numpy())

        train_set_data_scaled = scaler.transform(train_set_data)
        validation_set_data_scaled = scaler.transform(validation_set_data)
        test_data_scaled = scaler.transform(test_data)

    x_y_train_dict = windowing(
        train_set_data_scaled, window=window, prediction_scope=prediction_scope
    )
    x_train_data = x_y_train_dict["input"]
    y_train_data = x_y_train_dict["target"]

    x_y_val_dict = windowing(
        validation_set_data_scaled,
        window=window,
        prediction_scope=prediction_scope,
    )
    x_val_data = x_y_val_dict["input"]
    y_val_data = x_y_val_dict["target"]

    x_test_data_matrix = test_data_scaled[:, :-1]
    y_test_data_col_vector = test_data_scaled[:, -1]

    target_name = target_col + target_suffix
    target_data = target_price_feat_reordered[target_name].to_numpy()
    target_min = np.amin(target_data)
    target_max = np.amax(target_data)

    if is_lstm:
        return {
            "features": target_price_feat_reordered,
            "x_train": x_train_data,
            "y_train": y_train_data,
            "x_val": x_val_data,
            "y_val": y_val_data,
            "x_test": np.expand_dims(x_test_data_matrix, axis=0),
            "y_test": y_test_data_col_vector,
            "target_min": target_min,
            "target_max": target_max,
        }

    x_train_data_reshaped = x_train_data.reshape(x_train_data.shape[0], -1)
    x_val_data_reshaped = x_val_data.reshape(x_val_data.shape[0], -1)
    x_test_data_reshaped = x_test_data_matrix.reshape(1, -1)

    return {
        "features": target_price_feat_reordered,
        "x_train": x_train_data_reshaped,
        "y_train": y_train_data,
        "x_val": x_val_data_reshaped,
        "y_val": y_val_data,
        "x_test": x_test_data_reshaped,
        "y_test": y_test_data_col_vector,
        "target_min": target_min,
        "target_max": target_max,
    }


def make_train_val_test_set(
    df: DataFrame, window: int, percentage: float
) -> dict[str, np.ndarray]:
    #                   d = len(target_price)
    #               d - window          | window
    # target_price -> nontest_data      | test_data
    #     percentage | 100 - percentage |
    #  train_set     |   val_set        |
    test = df.iloc[-window:].to_numpy()
    nontest = df.iloc[:-window]

    threshold = int(len(nontest) * percentage)

    train = np.array(nontest.iloc[:threshold])
    val = np.array(nontest.iloc[threshold:])

    return {"train": train, "val": val, "test": test}


def rename_target_col_and_put_it_last(
    df: DataFrame, target_col: str, suffix: str
) -> DataFrame:
    target_col_new_name = target_col + suffix
    df.rename(columns={target_col: target_col_new_name}, inplace=True)
    return put_column_to_last(df, target_col_new_name)


def check_required_download_size(
    target_price: DataFrame,
    window: int,
    percentage: float,
    prediction_scope: int,
    shift_max: int,
    roll_max: int,
    is_lstm: bool,
) -> None:
    if is_lstm:
        data_size = len(target_price)
    else:  # because of rolling and shift, there are NaN in data.
        data_size = len(target_price) - max(shift_max, roll_max - 1)
    try:
        val_window_data_size = get_val_window_data_size(
            data_size, window, percentage, prediction_scope
        )

        assert val_window_data_size > 0
        print("Expected val data size:", val_window_data_size)
    except AssertionError as ae:
        print(
            f"Not enough data size, {len(target_price)} for "
            f"validation window data, ({val_window_data_size} <= 0) "
            f"for window {window} and percentage {percentage}, "
            f"prediction scope {prediction_scope}"
        )
        raise ae


def save_df_json(
    df: DataFrame, orient: str, save_dir: PosixPath, filename: str
) -> None:
    df_json = df.to_json(orient=orient, date_format="iso")
    loaded_json = json.loads(df_json)
    save_json(loaded_json, save_dir, filename)


def save_json(json_data: dict, save_dir: PosixPath, filename: str) -> None:
    output = save_dir / filename

    with output.open("w", encoding="UTF-8") as file:
        json.dump(
            json_data, file, ensure_ascii=False, indent=2, sort_keys=False
        )


def df_json_reader(json_file: PosixPath, orient: str) -> DataFrame:
    try:
        loaded_df = pd.read_json(json_file, orient=orient, dtype=False)
    except ValueError as no_file:
        raise FileNotFoundError(
            f"Input json {json_file} is not found."
        ) from no_file

    loaded_df.index = pd.to_datetime(loaded_df.index).strftime("%Y-%m-%d")
    loaded_df.index = loaded_df.index.astype("datetime64[ns]")
    loaded_df.index.name = "Date"

    return loaded_df


def get_val_window_data_size(
    data_size: int,
    window: int,
    percentage: float,
    prediction_scope: int,
) -> int:
    train_size = data_size - window
    val_size = train_size - int(train_size * percentage)
    val_window_size = val_size - window - prediction_scope

    return val_window_size


def transform_min_max_scaler(
    target: np.ndarray, base: np.ndarray, feature_range: tuple = (0, 1)
) -> np.ndarray:
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(base)

    return scaler.transform(target)


def setup_lstm_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epoch_n: int,
    batch_size: int,
    threshold: float,
):
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (val_mae := logs.get("val_mae")) < threshold:
                print("keys in logs", list(logs.keys()))
                print(
                    f"MAE for Validation, {val_mae} is lower than "
                    f"{threshold}. So cancelling training"
                )
                self.model.stop_training = True

    callbacks = myCallback()

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.LSTM(
                units=32,
                input_shape=x_train.shape[1:],
                return_sequences=True,
            ),
            tf.keras.layers.LSTM(units=32, return_sequences=False),
            tf.keras.layers.Dense(units=1),
        ]
    )

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 0.228 * 10 ** (-epoch / 20)
    )
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.228, momentum=0.85)
    optimizer = tf.keras.optimizers.SGD(momentum=0.85)
    model.compile(
        loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics="mae"
    )
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epoch_n,
        callbacks=[callbacks, lr_schedule],
        # callbacks=[callbacks],
        validation_data=[x_val, y_val],
        verbose=1,
    )

    return model
