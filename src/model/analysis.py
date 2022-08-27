from datetime import timedelta
import pandas as pd
from pandas.core.series import Series
from pandas.core.frame import DataFrame
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import tensorflow as tf
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
        print(f"Data for {tic} was downloaded with size, {len(output[tic])}")

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


def nontest_test_split(df: DataFrame, threshold: int) -> dict[str, DataFrame]:
    return {"nontest": df.iloc[:-threshold], "test": df.iloc[-threshold:]}


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
    """
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
        prediction_scope: 1
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


def data_prep_for_fitting(
    target_tic: str,
    target_col: str,
    ref_tic: str,
    start_date: str,
    end_date: str,
    window: int,
    percentage: float,
    prediction_scope: int,
    is_lstm: bool = False,
) -> Union[dict[str, np.ndarray], dict[str, DataFrame]]:
    try:
        price_dict = download_price_dict(
            tickers=(target_tic, ref_tic), start=start_date, end=end_date
        )
        target_price = price_dict[target_tic]
        val_window_data_size = get_val_window_data_size(
            target_price, window, percentage, prediction_scope
        )

        assert val_window_data_size > 0
    except AssertionError as ae:
        print(
            f"Not enough data size, {len(target_price)} for "
            f"validation window data, ({val_window_data_size} <= 0) "
            f"for window {window} and percentage {percentage}, "
            f"prediction scope {prediction_scope}"
        )
        raise ae

    target_price_feat = add_features(target_price)

    target_price_feat[ref_tic] = price_dict[ref_tic][target_col]

    target_col_new_name = target_col + "_y"
    target_price_feat.rename(
        columns={target_col: target_col_new_name}, inplace=True
    )
    target_price_feat_reordered = put_column_to_last(
        target_price_feat, target_col_new_name
    )

    nontest_test_dict = nontest_test_split(
        target_price_feat_reordered, threshold=window
    )
    nontest_data = nontest_test_dict["nontest"]
    test_data = nontest_test_dict["test"].to_numpy()

    train_val_dict = train_validation_split(
        nontest_data, percentage=percentage
    )
    train_set_data = train_val_dict["train"]
    validation_set_data = train_val_dict["validation"]

    if is_lstm:
        # train_set_data = transform_min_max_scaler(
        #     target=train_set_data, base=train_set_data
        # )
        # validation_set_data = transform_min_max_scaler(
        #     target=validation_set_data, base=train_set_data
        # )
        # test_data = transform_min_max_scaler(
        #     target=test_data, base=train_set_data
        # )
        train_set_data = transform_min_max_scaler(
            target=train_set_data, base=target_price_feat_reordered.to_numpy()
        )
        validation_set_data = transform_min_max_scaler(
            target=validation_set_data,
            base=target_price_feat_reordered.to_numpy(),
        )
        test_data = transform_min_max_scaler(
            target=test_data, base=target_price_feat_reordered.to_numpy()
        )

    x_y_train_dict = windowing(
        train_set_data, window=window, prediction_scope=prediction_scope
    )
    x_train_data = x_y_train_dict["input"]
    y_train_data = x_y_train_dict["target"]

    x_y_val_dict = windowing(
        validation_set_data, window=window, prediction_scope=prediction_scope
    )
    x_val_data = x_y_val_dict["input"]
    y_val_data = x_y_val_dict["target"]

    x_test_data_matrix = test_data[:, :-1]
    y_test_data_col_vector = test_data[:, -1]

    if is_lstm:
        return {
            "features": target_price_feat_reordered,
            "x_train": x_train_data,
            "y_train": y_train_data,
            "x_val": x_val_data,
            "y_val": y_val_data,
            "x_test": x_test_data_matrix,
            "y_test": y_test_data_col_vector,
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
    }


def get_val_window_data_size(
    target_price: DataFrame,
    window: int,
    percentage: float,
    prediction_scope: int,
) -> int:
    data_size = len(target_price)
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
        def on_epoch_end(self, epoch, logs={}):
            if logs.get("val_mae") < threshold:
                print("\n Accuracy % so cancelling training")
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


def inverse_transformation(
    X: np.ndarray, y: np.ndarray, y_hat: np.ndarray, scaler: MinMaxScaler
) -> dict[str, DataFrame]:
    if X.shape[1] > 1:  # for validation and training data with windowing
        new_X = []

        for i in range(len(X)):
            # add every first time data because other time
            # data are duplicated in the next row
            new_X.append(X[i][0])

        new_X = np.array(new_X)
        new_X = pd.DataFrame(new_X)

        y = np.expand_dims(y, axis=1)  # covert row -> col vector
        y = pd.DataFrame(y)
        y_hat = pd.DataFrame(y_hat)

    else:  # for testing data without windowing
        # (30, 1, 67) -> (30, 67)
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        new_X = pd.DataFrame(X)

        y = pd.DataFrame(y)
        y_hat = pd.DataFrame(y_hat)
        y_hat = pd.concat((y, y_hat))   # for y_hat no matching features
        y_hat.index = range(len(y_hat))

    real_val = np.array(pd.concat((new_X, y), axis=1))
    pred_val = np.array(pd.concat((new_X, y_hat), axis=1))

    real_val = pd.DataFrame(scaler.inverse_transform(real_val))
    pred_val = pd.DataFrame(scaler.inverse_transform(pred_val))

    return {"real": real_val, "pred": pred_val}
