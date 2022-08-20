from datetime import timedelta
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
import seaborn as sns
import tensorflow as tf
import xgboost as xgb
import yfinance as yf

# ParameterGrid for Gridsearch without CV
# import warnings
# warnings.filterwarnings("ignore")


def feature_engineering(
    data: DataFrame, SPY: Series, predictions: np.ndarray = np.array([None])
) -> DataFrame:

    """
    The function applies feature engineering to the data in order
    to get more information out of the inserted data.
    The commented code below is used when we are trying to append
    the predictions of the model as a new input feature to train it again.
    In this case it performed slightly better, however depending
    on the parameter optimization this gain can be vanished.
    """
    # if predictions.any() == True:
    #     data = yf.download("AAPL", start="2001-11-30")
    #     SPY = yf.download("SPY", start="2001-11-30")["Close"]
    #     data = features(data, SPY)
    #     print(data.shape)
    #     data["Predictions"] = predictions
    #     data["Close"] = data["Close_y"]
    #     data.drop("Close_y", 1, inplace=True)
    #     data.dropna(0, inplace=True)
    # else:
    #     pass

    print("No model yet")
    return features(data, SPY)


def features(data: DataFrame, SPY: Series) -> DataFrame:
    for i in range(2, 8):
        # Rolling Mean
        data[f"Adj_Close{i}"] = data["Adj Close"].rolling(i).mean()
        data[f"Volume{i}"] = data["Volume"].rolling(i).mean()

        # Rolling Standart Deviation
        data[f"Low_std{i}"] = data["Low"].rolling(i).std()
        data[f"High_std{i}"] = data["High"].rolling(i).std()
        data[f"Adj_CLose{i}"] = data["Adj Close"].rolling(i).std()

        # Stock return for the next i days
        data[f"Close{i}"] = data["Close"].shift(i)

        # Rolling Maximum and Minimum
        data[f"Adj_Close{i}"] = data["Adj Close"].rolling(i).max()
        data[f"Adj_Close{i}"] = data["Adj Close"].rolling(i).min()

        # Rolling Quantile
        data[f"Adj_Close{i}"] = data["Adj Close"].rolling(i).quantile(1)

    data["SPY"] = SPY
    # Decoding the time of the year
    data["Day"] = data.index.day
    data["Month"] = data.index.month
    data["Year"] = data.index.year
    data["day_year"] = data.index.day_of_year
    data["Weekday"] = data.index.weekday

    # Upper and Lower shade
    data["Upper_Shape"] = data["High"] - np.maximum(
        data["Open"], data["Close"]
    )
    data["Lower_Shape"] = np.minimum(data["Open"], data["Close"]) - data["Low"]

    data["Close_y"] = data["Close"]
    # data.drop(labels="Close", axis=1, inplace=True)  # Rename?
    data.drop("Close", 1, inplace=True)  # Rename?
    #data.dropna(axis=0, inplace=True)
    data.dropna(0, inplace=True)
    return data


def windowing(
    train: np.ndarray, val: np.ndarray, WINDOW: int, PREDICTION_SCOPE: int
):
    """
    Divides the inserted data into a list of lists. Where the shape
    of the data becomes and additional axe, which is time.
    Basically gets as an input shape of (X, Y) and gets returned a list
    which contains 3 dimensions (X, Z, Y) being Z, time.

    Input:
        - Train Set
        - Validation Set
        - WINDOW: the desired window
        - PREDICTION_SCOPE: The period in the future you want to analyze

    Output:
        - X_train: Explanatory variables for training set
        - y_train: Target variable training set
        - X_test: Explanatory variables for validation set
        - y_test:  Target variable validation set
    """

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in range(len(train) - (WINDOW + PREDICTION_SCOPE)):
        X, y = np.array(train[i : i + WINDOW, :-1]), np.array(
            train[i + WINDOW + PREDICTION_SCOPE, -1]
        )
        X_train.append(X)
        y_train.append(y)

    for i in range(len(val) - (WINDOW + PREDICTION_SCOPE)):
        X, y = np.array(val[i : i + WINDOW, :-1]), np.array(
            val[i + WINDOW + PREDICTION_SCOPE, -1]
        )
        X_test.append(X)
        y_test.append(y)

    return [X_train, y_train, X_test, y_test]


def train_test_split(data: DataFrame, WINDOW: int) -> list[DataFrame]:
    """
    Divides the training set into train and validation set depending
    on the percentage indicated.
    Note this could also be done through the sklearn traintestsplit() function.

    Input:
        - The data to be splitted (stock data in this case)
        - The size of the window used that will be taken as an input in order
        to predict the t+1

    Output:
        - Train/Validation Set
        - Test Set
    """
    train = data.iloc[:-WINDOW]
    test = data.iloc[-WINDOW:]

    return [train, test]


def train_validation_split(train: DataFrame, percentage: float) -> list[np.array]:
    """
    Divides the training set into train and validation set depending
    on the percentage indicated
    """
    threshold = int(len(train) * percentage)
    train_set = np.array(train.iloc[: threshold])
    validation_set = np.array(train.iloc[threshold :])

    return [train_set, validation_set]


def plotting(
    y_val, y_test, pred_test, mae, WINDOW: int, PREDICTION_SCOPE: int
):
    """
    This function returns a graph where:
    - Validation Set
    - Test Set
    - Future Prediction
    - Upper Bound
    - Lower Bound
    """
    ploting_pred = [y_test[-1], pred_test]
    ploting_test = [y_val[-1]] + list(y_test)

    time = (len(y_val) - 1) + (len(ploting_test) - 1) + (len(ploting_pred) - 1)

    test_time_init = time - (len(ploting_test) - 1) - (len(ploting_pred) - 1)
    test_time_end = time - (len(ploting_pred) - 1) + 1

    pred_time_init = time - (len(ploting_pred) - 1)
    pred_time_end = time + 1

    x_ticks = list(stock_prices.index[-time:]) + [
        stock_prices.index[-1] + timedelta(PREDICTION_SCOPE + 1)
    ]

    values_for_bounds = list(y_val) + list(y_test) + list(pred_test)
    upper_band = values_for_bounds + mae
    lower_band = values_for_bounds - mae

    print(f"For used windowed data: {WINDOW}")
    print(
        f"Prediction scope for date {x_ticks[-1]} / {PREDICTION_SCOPE+1} days"
    )
    print(f"The predicted price is {str(round(ploting_pred[-1][0],2))}$")
    print(f"With a spread of MAE is {round(mae,2)}")
    print()

    plt.figure(figsize=(16, 8))

    plt.plot(
        list(range(test_time_init, test_time_end)),
        ploting_test,
        marker="$m$",
        color="orange",
    )
    plt.plot(
        list(range(pred_time_init, pred_time_end)),
        ploting_pred,
        marker="$m$",
        color="red",
    )
    plt.plot(y_val, marker="$m$")

    plt.plot(upper_band, color="grey", alpha=0.3)
    plt.plot(lower_band, color="grey", alpha=0.3)

    plt.fill_between(
        list(range(0, time + 1)),
        upper_band,
        lower_band,
        color="grey",
        alpha=0.1,
    )

    plt.xticks(list(range(0 - 1, time)), x_ticks, rotation=45)
    plt.text(
        time - 0.5,
        ploting_pred[-1] + 2,
        str(round(ploting_pred[-1][0], 2)) + "$",
        size=11,
        color="red",
    )
    plt.title(
        f"Target price for date {x_ticks[-1]} / {PREDICTION_SCOPE+1} days, "
        f"with used past data of {WINDOW} days and a MAE of {round(mae,2)}",
        size=15,
    )
    plt.legend(
        ["Testing Set (input for Prediction)", "Prediction", "Validation"]
    )
    plt.show()


def inverse_transformation(X: np.ndarray, y: np.ndarray, y_hat):
    """
    This function serves to inverse the rescaled data.
    There are two ways in which this can happen:
        - There could be the conversion for the validation data to see it
        on the plotting.
        - There could be the conversion for the testing data,
        to see it plotted.
    """
    if X.shape[1] > 1:
        new_X = []

        for i in range(len(X)):
            new_X.append(X[i][0])

        new_X = np.array(new_X)
        y = np.expand_dims(y, 1)

        new_X = pd.DataFrame(new_X)
        y = pd.DataFrame(y)
        y_hat = pd.DataFrame(y_hat)

        real_val = np.array(pd.concat((new_X, y), 1))
        pred_val = np.array(pd.concat((new_X, y_hat), 1))

        real_val = pd.DataFrame(scaler.inverse_transform(real_val))
        pred_val = pd.DataFrame(scaler.inverse_transform(pred_val))

    else:
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

        new_X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        y_hat = pd.DataFrame(y_hat)
        y_hat = pd.concat((y, y_hat))
        y_hat.index = range(len(y_hat))

        real_val = np.array(pd.concat((new_X, y), 1))
        pred_val = np.array(pd.concat((new_X, y_hat), 1))

        pred_val = pd.DataFrame(scaler.inverse_transform(pred_val))
        real_val = pd.DataFrame(scaler.inverse_transform(real_val))

    return real_val, pred_val


def window_optimization(plots: dict):
    """
    Returns the key that contains the most optimal window
    (respect to mae) for t+1
    """
    rank = []
    m = []
    for i in plots.keys():
        if not rank:
            rank.append(plots[i])
            m.append(i)
        elif plots[i][3] < rank[0][3]:
            rank.clear()
            m.clear()
            rank.append(plots[i])
            m.append(i)

    return rank, m


def predictions(mae_lstm, mae_xgboost, prediction_xgb, prediction_lstm):
    """
    Returns the prediction at t+1 weighted by the respective mae.
    Giving a higher weight to the one which is lower
    """
    prediction = (
        1 - (mae_xgboost / (mae_lstm + mae_xgboost))
    ) * prediction_xgb + (
        1 - (mae_lstm / (mae_lstm + mae_xgboost))
    ) * prediction_lstm
    return prediction


def sp500_log_rets(tickers: list[str]) -> dict[str, Series]:
    """
    Returns the logarithmic returns from the SP500
    """

    stock_prices = yf.download(tickers, start="2015-11-30", end="2021-11-30")[
        "Close"
    ]
    log_rets = {}
    for ticker in tickers:
        log_rets[ticker] = np.log(
            stock_prices[ticker] / stock_prices[ticker].shift(1)
        )
    return log_rets


def annualized_rets(r: Series) -> np.float64:
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compounded_growth ** (255 / n_periods) - 1


def xgb_model_fun(X_train, y_train, X_val, y_val, plotting=False):
    """
    Trains a preoptimized XGBoost model and
    returns the Mean Absolute Error a plot if needed
    """
    xgb_model = xgb.XGBRegressor(gamma=1, n_estimators=200)
    xgb_model.fit(X_train, y_train)

    pred_val = xgb_model.predict(X_val)
    mae = mean_absolute_error(y_val, pred_val)

    if plotting is True:

        plt.figure(figsize=(15, 6))

        sns.set_theme(style="white")
        sns.lineplot(x=range(len(y_val)), y=y_val, color="grey", alpha=0.4)
        sns.lineplot(x=range(len(y_val)), y=pred_val, color="red")

        plt.xlabel("Time")
        plt.ylabel("AAPL stock price")
        plt.title(f"The MAE for this period is: {round(mae, 3)}")

    return mae, xgb_model


def lstm_model_fun(
    X_train, y_train, X_val, y_val, EPOCH, BATCH_SIZE, CALLBACK, plotting=False
):
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get("val_mae") < CALLBACK:
                print("\n Accuracy % so cancelling training")
                self.model.stop_training = True

    callbacks = myCallback()

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.LSTM(
                32,
                input_shape=(X_train.shape[1], X_train.shape[2]),
                return_sequences=True,
            ),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1),
        ]
    )

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 0.228 * 10 ** (epoch / 20)
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.228, momentum=0.85)
    model.compile(
        loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics="mae"
    )
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
        callbacks=[callbacks],
        validation_data=[X_val, y_val],
        verbose=1,
    )

    if plotting == True:
        plt.figure(figsize=(18, 6))

        lrs = 1e-5 * (10 ** (np.arange(len(history.history["loss"])) / 20))
        plt.semilogx(lrs, history.history["loss"])
        plt.xticks(size=14)
        plt.show()

    return model
