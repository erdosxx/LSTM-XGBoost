import pytest
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas import testing as tm
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import yfinance as yf
from src.model.analysis import (
    get_shifted_log_rets,
    download_price_dict,
    get_series_dict,
    get_shifted_log_dict,
    plot_box_and_hist,
    plot_series,
    get_cumprod_dict,
    add_features,
    train_test_split,
    train_validation_split,
    windowing,
    put_column_to_last,
    plot_concat_data,
)


class TestBuildData:
    def test_df_check_nan_value_should_return_expected(self):
        data = {
            "C1": [1, 2, np.nan, 4, 5],
            "C2": [1, 2, 3, 4, 5],
        }
        df = pd.DataFrame(data)
        assert df.isnull().values.any() is np.True_
        assert df["C2"].isnull().values.any() is np.False_

    def test_get_apple_data_should_be_expected(self):
        apple_price_dict = download_price_dict(
            tickers=("AAPL",), start="1980-12-13", end="2022-08-19"
        )
        apple_price = apple_price_dict["AAPL"]
        assert isinstance(apple_price, DataFrame) is True
        assert apple_price.isnull().values.any() is np.False_
        assert isinstance(
            apple_price.index, pd.core.indexes.datetimes.DatetimeIndex
        )
        assert isinstance(apple_price.index[0], pd.Timestamp) is True
        ordered_column = [
            "Open",
            "High",
            "Low",
            "Adj Close",
            "Volume",
            "Close",
        ]
        assert set(apple_price.columns) == set(ordered_column)
        assert apple_price.index.min() == dt.datetime(1980, 12, 12, 0, 0, 0)
        assert apple_price.index.max() == dt.datetime(2022, 8, 18, 0, 0, 0)
        assert len(apple_price) == 10510
        last_column = apple_price.iloc[:, -1]
        assert last_column.name == "Volume"
        ordered_apple_price = apple_price[ordered_column]
        new_last_column = ordered_apple_price.iloc[:, -1]
        assert new_last_column.name == "Close"

    def test_get_spy_data_should_be_expected(self):
        spy_price_dict = download_price_dict(
            tickers=("SPY",), start="2001-11-30", end="2022-08-19"
        )
        assert spy_price_dict["SPY"].isnull().values.any() is np.False_
        spy_close_price_dict = get_series_dict(spy_price_dict, "Close")
        spy_close_price = spy_close_price_dict["SPY"]

        assert isinstance(spy_close_price, Series) is True
        assert spy_close_price.name == "Close"
        assert spy_close_price.index.min() == dt.datetime(
            2001, 11, 29, 0, 0, 0
        )
        assert spy_close_price.index.max() == dt.datetime(2022, 8, 18, 0, 0, 0)
        assert len(spy_close_price) == 5216

    def test_pd_shift_should_return_expected(self):
        pd_series = pd.Series(
            {
                "2001-12-01": 1,
                "2001-12-02": 2,
                "2001-12-03": 3,
                "2001-12-04": 4,
                "2001-12-05": 5,
            }
        )
        shifted_series = pd_series.shift(1)
        expected_series = pd.Series(
            {
                "2001-12-01": np.NaN,
                "2001-12-02": 1,
                "2001-12-03": 2,
                "2001-12-04": 3,
                "2001-12-05": 4,
            }
        )
        assert tm.assert_series_equal(shifted_series, expected_series) is None

    def test_np_log_should_return_expected(self):
        pd_series = pd.Series(
            {
                "2001-12-01": np.e,
                "2001-12-02": np.e**2,
                "2001-12-03": np.e**3,
                "2001-12-04": np.e**4,
                "2001-12-05": np.e**5,
            }
        )
        shifted_series = pd_series.shift(1)
        log_series = np.log(pd_series / shifted_series)
        expected_series = pd.Series(
            {
                "2001-12-01": np.NaN,
                "2001-12-02": 1.0,
                "2001-12-03": 1.0,
                "2001-12-04": 1.0,
                "2001-12-05": 1.0,
            }
        )
        assert tm.assert_series_equal(log_series, expected_series) is None

    def test_get_shifted_log_rets_should_return_expected(self):
        pd_series = pd.Series(
            {
                "2001-12-01": np.e,
                "2001-12-02": np.e**2,
                "2001-12-03": np.e**3,
                "2001-12-04": np.e**4,
                "2001-12-05": np.e**5,
            }
        )
        shifted_log_val = get_shifted_log_rets(pd_series, shift_val=1)
        expected_series = pd.Series(
            {
                "2001-12-01": np.NaN,
                "2001-12-02": 1.0,
                "2001-12-03": 1.0,
                "2001-12-04": 1.0,
                "2001-12-05": 1.0,
            }
        )
        assert tm.assert_series_equal(shifted_log_val, expected_series) is None

    def test_get_log_rets_dict_should_return_expected(self):
        pd_series = pd.Series(
            {
                "2001-12-01": np.e,
                "2001-12-02": np.e**2,
                "2001-12-03": np.e**3,
                "2001-12-04": np.e**4,
                "2001-12-05": np.e**5,
            }
        )

        close_dict = {"DEMO": pd_series}
        log_dict = get_shifted_log_dict(close_dict, shift_val=1)
        expected_series = pd.Series(
            {
                "2001-12-01": np.NaN,
                "2001-12-02": 1.0,
                "2001-12-03": 1.0,
                "2001-12-04": 1.0,
                "2001-12-05": 1.0,
            }
        )
        assert (
            tm.assert_series_equal(log_dict["DEMO"], expected_series) is None
        )

    def test_get_comprod_dict_should_return_expected(self):
        pd_series = pd.Series(
            {
                "2001-12-01": 0,
                "2001-12-02": 1,
                "2001-12-03": 2,
                "2001-12-04": 3,
                "2001-12-05": 4,
            }
        )

        close_dict = {"DEMO": pd_series}
        cumprod_dict = get_cumprod_dict(close_dict)
        expected_series = pd.Series(
            {
                "2001-12-01": 0 + 1,
                "2001-12-02": 1 * (1 + 1),
                "2001-12-03": 1 * 2 * (2 + 1),
                "2001-12-04": 1 * 2 * 3 * (3 + 1),
                "2001-12-05": 1 * 2 * 3 * 4 * (4 + 1),
            }
        )
        assert (
            tm.assert_series_equal(cumprod_dict["DEMO"], expected_series)
            is None
        )

    def test_np_any_should_return_expected(self):
        all_false_array = np.array([False, False])
        assert all_false_array.any() == np.False_
        assert all_false_array.any() is not False

        any_true_array = np.array([False, True])
        assert any_true_array.any() == np.True_  # or True
        assert any_true_array.any() == True
        assert any_true_array.any() is not True

    def test_add_add_features_should_return_expected(self):
        tickers = ("AAPL",)
        stocks_price_dict = download_price_dict(
            tickers=tickers, start="2015-11-30", end="2022-08-19"
        )
        apple_price = stocks_price_dict["AAPL"]
        assert apple_price.isnull().values.any() is np.False_
        apple_price_feat = add_features(apple_price)

        assert len(apple_price_feat.columns) == 6 + 9 * 6 + 7
        assert "Close" in apple_price_feat.columns
        assert "Close_y" not in apple_price_feat.columns

    def test_rolling_window_calc_should_return_expected(self):
        df = pd.DataFrame({"Data": [0, 1, 2, 3, 4, np.nan]})
        df_rolling_sum = df.rolling(2).sum()
        df_expected_rolling_sum = pd.DataFrame(
            {"Data": [np.nan, 1, 3, 5, 7, np.nan]}
        )
        tm.assert_frame_equal(df_rolling_sum, df_expected_rolling_sum) is None

        df_rolling_quantile = df.rolling(2).quantile(1)
        df_expected_rolling_quantile = pd.DataFrame(
            {"Data": [np.nan, 1, 2, 3, 4, np.nan]}
        )
        tm.assert_frame_equal(
            df_rolling_quantile, df_expected_rolling_quantile
        ) is None

    def test_pd_timestamp_index_should_work_as_expected(self):
        df = pd.Series(
            range(3), index=pd.date_range("2000", freq="D", periods=3)
        ).to_frame()
        df["Day"] = df.index.day
        df["Month"] = df.index.month
        df["Year"] = df.index.year
        df["day_year"] = df.index.day_of_year
        df["Weekday"] = df.index.weekday
        df_expected = pd.DataFrame(
            {
                0: [0, 1, 2],
                "Day": [1, 2, 3],
                "Month": [1, 1, 1],
                "Year": [2000, 2000, 2000],
                "day_year": [1, 2, 3],
                "Weekday": [5, 6, 0],
            },
            index=pd.date_range("2000", freq="D", periods=3),
        )
        tm.assert_frame_equal(df, df_expected) is None

    def test_train_test_split_should_return_expected(self):
        pd_series = pd.Series(
            {
                "2001-12-01": 1,
                "2001-12-02": 2,
                "2001-12-03": 3,
                "2001-12-04": 4,
                "2001-12-05": 5,
            }
        )
        df = pd_series.to_frame()
        train_test_dict = train_test_split(df, window=2)
        train = train_test_dict["train"]
        train_expected = pd.Series(
            {
                "2001-12-01": 1,
                "2001-12-02": 2,
                "2001-12-03": 3,
            }
        ).to_frame()
        assert tm.assert_frame_equal(train, train_expected) is None
        test = train_test_dict["test"]
        test_expected = pd.Series(
            {
                "2001-12-04": 4,
                "2001-12-05": 5,
            }
        ).to_frame()
        assert tm.assert_frame_equal(test, test_expected) is None

    def test_train_validation_split_should_return_expected(self):
        pd_series = pd.Series(
            {
                "2001-12-01": 1,
                "2001-12-02": 2,
                "2001-12-03": 3,
                "2001-12-04": 4,
                "2001-12-05": 5,
            }
        )
        df = pd_series.to_frame()
        train_val_dict = train_validation_split(df, percentage=0.995)

        train = train_val_dict["train"]
        train_expected = np.array([[1], [2], [3], [4]])
        assert assert_array_equal(train, train_expected) is None

        val = train_val_dict["validation"]
        val_expected = np.array([[5]])
        assert assert_array_equal(val, val_expected) is None

    def test_windowing_should_return_expected(self):
        data = np.array(
            [
                [11, 12, 13],
                [21, 22, 23],
                [31, 32, 33],
                [41, 42, 43],
                [51, 52, 53],
            ]
        )
        input_target_dict = windowing(data, window=2, prediction_scope=0)
        input_data = input_target_dict["input"]
        input_data_expected = np.array(
            [[[11, 12], [21, 22]], [[21, 22], [31, 32]], [[31, 32], [41, 42]]]
        )
        assert assert_array_equal(input_data, input_data_expected) is None

        target_data = input_target_dict["target"]
        target_data_expected = np.array([33, 43, 53])
        assert assert_array_equal(target_data, target_data_expected) is None

        input_target_dict = windowing(data, window=2, prediction_scope=1)
        input_data = input_target_dict["input"]
        input_data_expected = np.array(
            [[[11, 12], [21, 22]], [[21, 22], [31, 32]]]
        )
        assert assert_array_equal(input_data, input_data_expected) is None

        target_data = input_target_dict["target"]
        target_data_expected = np.array([43, 53])
        assert assert_array_equal(target_data, target_data_expected) is None

    def test_make_1d_array_should_return_expected(self):
        input_data_expected = np.array(
            [[[11, 12], [21, 22]], [[21, 22], [31, 32]], [[31, 32], [41, 42]]]
        )
        input_1d = input_data_expected.reshape(
            input_data_expected.shape[0], -1
        )
        input_1d_expected = np.array(
            [[11, 12, 21, 22], [21, 22, 31, 32], [31, 32, 41, 42]]
        )
        assert assert_array_equal(input_1d, input_1d_expected) is None

        input_data_expected = np.array(
            [[[11, 12], [21, 22]], [[21, 22], [31, 32]]]
        )
        input_1d = input_data_expected.reshape(
            input_data_expected.shape[0], -1
        )
        input_1d_expected = np.array([[11, 12, 21, 22], [21, 22, 31, 32]])
        assert assert_array_equal(input_1d, input_1d_expected) is None

    def test_reorder_colum_to_last_should_return_expected(self):
        df = pd.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            columns=["a", "b", "c"],
        )
        df_reordered = put_column_to_last(df, "a")
        assert df_reordered.iloc[:, -1].name == "a"
        assert df.iloc[:, -1].name == "c"

    def test_regression_fitting_should_return_expected(self):
        apple_price_dict = download_price_dict(
            tickers=("AAPL",), start="2001-11-30", end="2022-08-19"
        )
        apple_price = apple_price_dict["AAPL"]
        apple_price_feat = add_features(apple_price)

        spy_price_dict = download_price_dict(
            tickers=("SPY",), start="2001-11-30", end="2022-08-19"
        )

        spy_close_price_dict = get_series_dict(spy_price_dict, "Close")
        spy_close_price = spy_close_price_dict["SPY"]

        apple_price_feat["SPY"] = spy_close_price
        apple_price_feat.rename(columns={"Close": "Close_y"}, inplace=True)
        apple_price_feat_reordered = put_column_to_last(
            apple_price_feat, "Close_y"
        )

        train_test_dict = train_test_split(
            apple_price_feat_reordered, window=2
        )
        train_reg = train_test_dict["train"]
        test_reg = train_test_dict["test"]

        train_val_dict = train_validation_split(train_reg, percentage=0.995)
        train_set_reg = train_val_dict["train"]
        validation_set_reg = train_val_dict["validation"]

        input_target_train_dict = windowing(
            train_set_reg, window=2, prediction_scope=0
        )
        x_train_reg = input_target_train_dict["input"]
        y_train_reg = input_target_train_dict["target"]
        input_target_val_dict = windowing(
            validation_set_reg, window=2, prediction_scope=0
        )
        x_val_reg = input_target_val_dict["input"]
        y_val_reg = input_target_val_dict["target"]

        x_train_reg_1d = x_train_reg.reshape(x_train_reg.shape[0], -1)
        x_val_reg_1d = x_val_reg.reshape(x_val_reg.shape[0], -1)

        x_test_reg_1d = np.array(test_reg.iloc[:, :-1])
        y_test_reg_1d = np.array(test_reg.iloc[:, -1])

        x_test_reg_1d_rs = x_test_reg_1d.reshape(1, -1)

        lr = LinearRegression()
        lr.fit(x_train_reg_1d, y_train_reg)

        y_hat_lr = lr.predict(x_val_reg_1d)

        mae_lr = mean_absolute_error(y_val_reg, y_hat_lr)
        mse_lr = np.mean((y_hat_lr - y_val_reg) ** 2)

        print(f"Linear Regression MSE: {mse_lr}")
        print(f"Linear Regression MAE: {mae_lr}")

        rf = RandomForestRegressor()
        rf.fit(x_train_reg_1d, y_train_reg)

        y_hat_rf = rf.predict(x_val_reg_1d)
        mae_rf = mean_absolute_error(y_val_reg, y_hat_rf)
        mse_rf = np.mean((y_hat_rf - y_val_reg) ** 2)

        print(f"Random Forest MSE: {mse_rf}")
        print(f"Random Forest MAE: {mae_rf}")

        # y_hat_lr
        # np.ravel(y_hat_lr)
        # y_hat_rf
        # np.ravel(y_hat_rf)
        # y_val_reg np.ravel(y_val_reg)

        assert assert_array_equal(y_hat_lr, np.ravel(y_hat_lr)) is None
        assert assert_array_equal(y_hat_rf, np.ravel(y_hat_rf)) is None
        assert assert_array_equal(y_val_reg, np.ravel(y_val_reg)) is None

        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(y_val_reg, color="red")
        ax.plot(y_hat_lr, color="orange")
        ax.plot(y_hat_rf, color="grey", alpha=0.2)
        ax.legend(["True Returns", "Linear Regression", "Random Forest"])

        pred_test_lr = lr.predict(x_test_reg_1d_rs)

        plot_concat_data(
            y_val=y_val_reg,
            y_test=y_test_reg_1d,
            pred_test=pred_test_lr,
            df_for_x=apple_price_feat_reordered,
            mae=mae_lr,
            window=2,
            prediction_scope=0,
        )


    @pytest.mark.skip("Need to implement for testing")
    def test_plot(self):
        tickers = (
            "AAPL",
            "MSFT",
            "TSLA",
            "AMZN",
            "SPY",
        )
        stocks_price_dict = download_price_dict(
            tickers=tickers, start="2015-11-30", end="2022-08-19"
        )
        stocks_close_price_dict = get_series_dict(stocks_price_dict, "Close")
        stocks_close_shift_log_dict = get_shifted_log_dict(
            stocks_close_price_dict
        )
        plot_box_and_hist(
            target="AAPL",
            box_series_dict=stocks_close_price_dict,
            hist_series_dict=stocks_close_shift_log_dict,
        )
        stocks_cumprod_close_shift_log_dict = get_cumprod_dict(
            stocks_close_shift_log_dict
        )
        plot_series(
            target="AAPL", series_dict=stocks_cumprod_close_shift_log_dict
        )
        plot_series(target="AAPL", series_dict=stocks_close_price_dict)
