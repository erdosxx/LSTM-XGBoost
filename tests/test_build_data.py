import datetime as dt
from pathlib import Path
import pytest
import joblib
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pandas as pd
from pandas import testing as tm
from pandas.core.frame import DataFrame
from pandas.core.series import Series
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tqdm import tqdm
from xgboost import XGBRegressor, plot_importance
from src.model.analysis import (
    get_shifted_log_rets,
    download_price_dict,
    get_series_dict,
    get_shifted_log_dict,
    plot_box_and_hist,
    plot_series,
    get_cumprod_dict,
    add_features,
    make_train_val_test_set,
    windowing,
    put_column_to_last,
    plot_concat_data,
    data_prep_for_fitting,
    transform_min_max_scaler,
    setup_lstm_model,
    save_df_json,
    df_json_reader,
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
        apple_price_feat = add_features(apple_price, roll_max=7, shift_max=7)

        assert len(apple_price_feat.columns) == 6 + 8 * 6 + 1 * 7 + 7
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

        data_list = [1, 1, 1, 1, 5, 1, 5, 1]
        df = pd.DataFrame({"Data": data_list})
        df_rolling_mean = df.rolling(4).mean()
        df_expected_rolling_mean = pd.DataFrame(
            {"Data": [np.nan, np.nan, np.nan, 1, 2, 2, 3, 3]}
        )
        assert (
            tm.assert_frame_equal(df_rolling_mean, df_expected_rolling_mean)
            is None
        )

        np_rolling_mean = (
            np.convolve(np.array(data_list), np.ones(4), mode="valid") / 4
        )
        np_rolling_mean_expected = np.array([1, 2, 2, 3, 3])
        assert (
            assert_array_equal(np_rolling_mean, np_rolling_mean_expected)
            is None
        )

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

    def test_make_train_val_test_should_return_expected(self):
        pd_series = pd.Series(
            {
                "2001-12-01": 1,
                "2001-12-02": 2,
                "2001-12-03": 3,
                "2001-12-04": 4,
                "2001-12-06": 5,
                "2001-12-07": 6,
                "2001-12-08": 7,
            }
        )
        df = pd_series.to_frame()
        train_val_test = make_train_val_test_set(df, window=2, percentage=0.8)
        train = train_val_test["train"]
        train_expected = np.array([[1], [2], [3], [4]])
        assert assert_array_equal(train, train_expected) is None

        val = train_val_test["val"]
        val_expected = np.array([[5]])
        assert assert_array_equal(val, val_expected) is None

        test = train_val_test["test"]
        test_expected = np.array([[6], [7]])
        assert assert_array_equal(test, test_expected) is None

    def test_windowing_for_size_2_pred_scope_0_should_return_expected(self):
        data = np.array(
            [
                [11, 12, 13],
                [21, 22, 23],
                [31, 32, 33],
                [41, 42, 43],
                [51, 52, 53],
            ]
        )
        window_size = 2
        prediction_scope = 0
        input_target_dict = windowing(
            data, window=window_size, prediction_scope=0
        )
        input_data = input_target_dict["input"]
        input_data_expected = np.array(
            [[[11, 12], [21, 22]], [[21, 22], [31, 32]], [[31, 32], [41, 42]]]
        )
        assert len(input_data) == len(data) - window_size - prediction_scope
        assert assert_array_equal(input_data, input_data_expected) is None

        target_data = input_target_dict["target"]
        target_data_expected = np.array([33, 43, 53])
        assert len(input_data) == len(target_data)
        assert assert_array_equal(target_data, target_data_expected) is None

    def test_windowing_for_size_2_pred_scope_1_should_return_expected(self):
        data = np.array(
            [
                [11, 12, 13],
                [21, 22, 23],
                [31, 32, 33],
                [41, 42, 43],
                [51, 52, 53],
            ]
        )
        window_size = 2
        prediction_scope = 1
        input_target_dict = windowing(
            data, window=window_size, prediction_scope=prediction_scope
        )
        input_data = input_target_dict["input"]
        input_data_expected = np.array(
            [[[11, 12], [21, 22]], [[21, 22], [31, 32]]]
        )
        assert len(input_data) == len(data) - window_size - prediction_scope
        assert assert_array_equal(input_data, input_data_expected) is None

        target_data = input_target_dict["target"]
        target_data_expected = np.array([43, 53])
        assert len(input_data) == len(target_data)
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

    def test_save_and_load_df_for_lstm_should_return_same_df(self, tmp_path):
        data_dict = data_prep_for_fitting(
            target_tic="AAPL",
            target_col="Close",
            ref_tic="SPY",
            start_date="2022-07-27",
            end_date="2022-08-19",
            window=2,
            percentage=0.75,
            prediction_scope=0,
            is_lstm=True,
            roll_max=7,
            shift_max=7,
        )
        df_feat = data_dict["features"]
        x_train = data_dict["x_train"]
        y_train = data_dict["y_train"]
        x_val = data_dict["x_val"]
        y_val = data_dict["y_val"]
        x_test = data_dict["x_test"]
        y_test = data_dict["y_test"]

        # tmp_path = Path("./tests")
        file_name = "df_feat_lstm.json"

        save_df_json(
            df=df_feat, orient="index", save_dir=tmp_path, filename=file_name
        )

        json_file = tmp_path / file_name
        loaded_df = df_json_reader(json_file, orient="index")

        assert tm.assert_frame_equal(df_feat, loaded_df) is None

        name_array_list = [
            {"filename": "x_train_lstm.npy", "array": x_train},
            {"filename": "y_train_lstm.npy", "array": y_train},
            {"filename": "x_val_lstm.npy", "array": x_val},
            {"filename": "y_val_lstm.npy", "array": y_val},
            {"filename": "x_test_lstm.npy", "array": x_test},
            {"filename": "y_test_lstm.npy", "array": y_test},
        ]

        for name_array in name_array_list:
            np.save(tmp_path / name_array["filename"], name_array["array"])
            loaded = np.load(tmp_path / name_array["filename"])
            assert assert_array_equal(name_array["array"], loaded) is None

    def test_save_and_load_df_for_regression_should_return_same_df(
        self, tmp_path
    ):
        data_dict = data_prep_for_fitting(
            target_tic="AAPL",
            target_col="Close",
            ref_tic="SPY",
            start_date="2022-07-27",
            end_date="2022-08-19",
            window=2,
            percentage=0.75,
            prediction_scope=0,
            is_lstm=False,
            roll_max=7,
            shift_max=7,
        )
        df_feat = data_dict["features"]
        x_train = data_dict["x_train"]
        y_train = data_dict["y_train"]
        x_val = data_dict["x_val"]
        y_val = data_dict["y_val"]
        x_test = data_dict["x_test"]
        y_test = data_dict["y_test"]

        # tmp_path = Path("./tests")
        file_name = "df_feat_reg.json"

        save_df_json(
            df=df_feat, orient="index", save_dir=tmp_path, filename=file_name
        )

        json_file = tmp_path / file_name
        loaded_df = df_json_reader(json_file, orient="index")

        assert tm.assert_frame_equal(df_feat, loaded_df) is None

        name_array_list = [
            {"filename": "x_train_reg.npy", "array": x_train},
            {"filename": "y_train_reg.npy", "array": y_train},
            {"filename": "x_val_reg.npy", "array": x_val},
            {"filename": "y_val_reg.npy", "array": y_val},
            {"filename": "x_test_reg.npy", "array": x_test},
            {"filename": "y_test_reg.npy", "array": y_test},
        ]

        for name_array in name_array_list:
            np.save(tmp_path / name_array["filename"], name_array["array"])
            loaded = np.load(tmp_path / name_array["filename"])
            assert assert_array_equal(name_array["array"], loaded) is None

    def test_data_prep_for_regression_lstm_should_return_expected(self):
        """
        Download size: 18
        number of NaN rows by shift: shift_max
        number of NaN rows by rolling: roll_max - 1
        """

        shift_max = 7
        roll_max = 7
        data_dict_reg = data_prep_for_fitting(
            target_tic="AAPL",
            target_col="Close",
            ref_tic="SPY",
            start_date="2022-07-27",
            end_date="2022-08-19",
            window=2,
            percentage=0.75,
            prediction_scope=0,
            is_lstm=False,
            roll_max=roll_max,
            shift_max=shift_max,
        )
        df_feat_reg = data_dict_reg["features"]
        x_train_reg = data_dict_reg["x_train"]
        y_train_reg = data_dict_reg["y_train"]
        x_val_reg = data_dict_reg["x_val"]
        y_val_reg = data_dict_reg["y_val"]
        x_test_reg = data_dict_reg["x_test"]
        y_test_reg = data_dict_reg["y_test"]

        print("val data size:", len(x_val_reg))

        download_size = 18
        assert len(df_feat_reg) == download_size - max(shift_max, roll_max - 1)

        json_file = Path("./tests") / "df_feat_reg.json"
        loaded_df_reg = df_json_reader(json_file, orient="index")

        assert tm.assert_frame_equal(df_feat_reg, loaded_df_reg) is None

        name_array_list = [
            {"filename": "x_train_reg.npy", "array": x_train_reg},
            {"filename": "y_train_reg.npy", "array": y_train_reg},
            {"filename": "x_val_reg.npy", "array": x_val_reg},
            {"filename": "y_val_reg.npy", "array": y_val_reg},
            {"filename": "x_test_reg.npy", "array": x_test_reg},
            {"filename": "y_test_reg.npy", "array": y_test_reg},
        ]

        for name_array in name_array_list:
            loaded_reg = np.load(Path("./tests") / name_array["filename"])
            assert assert_array_equal(name_array["array"], loaded_reg) is None

        data_dict_lstm = data_prep_for_fitting(
            target_tic="AAPL",
            target_col="Close",
            ref_tic="SPY",
            start_date="2022-07-27",
            end_date="2022-08-19",
            window=2,
            percentage=0.75,
            prediction_scope=0,
            is_lstm=True,
            roll_max=roll_max,
            shift_max=shift_max,
        )
        df_feat_lstm = data_dict_lstm["features"]
        x_train_lstm = data_dict_lstm["x_train"]
        y_train_lstm = data_dict_lstm["y_train"]
        x_val_lstm = data_dict_lstm["x_val"]
        y_val_lstm = data_dict_lstm["y_val"]
        x_test_lstm = data_dict_lstm["x_test"]
        y_test_lstm = data_dict_lstm["y_test"]

        print("val data size:", len(x_val_lstm))

        json_file = Path("./tests") / "df_feat_lstm.json"
        loaded_df_lstm = df_json_reader(json_file, orient="index")

        assert tm.assert_frame_equal(df_feat_lstm, loaded_df_lstm) is None

        name_array_list = [
            {"filename": "x_train_lstm.npy", "array": x_train_lstm},
            {"filename": "y_train_lstm.npy", "array": y_train_lstm},
            {"filename": "x_val_lstm.npy", "array": x_val_lstm},
            {"filename": "y_val_lstm.npy", "array": y_val_lstm},
            {"filename": "x_test_lstm.npy", "array": x_test_lstm},
            {"filename": "y_test_lstm.npy", "array": y_test_lstm},
        ]

        for name_array in name_array_list:
            loaded_lstm = np.load(Path("./tests") / name_array["filename"])
            assert assert_array_equal(name_array["array"], loaded_lstm) is None

    def test_regression_fitting_should_return_expected(self):
        data_dict = data_prep_for_fitting(
            target_tic="AAPL",
            target_col="Close",
            ref_tic="SPY",
            start_date="2001-11-30",
            end_date="2022-08-19",
            window=2,
            percentage=0.995,
            prediction_scope=0,
            is_lstm=False,
            roll_max=7,
            shift_max=7,
        )
        x_train = data_dict["x_train"]
        y_train = data_dict["y_train"]
        x_val = data_dict["x_val"]
        y_val = data_dict["y_val"]
        x_test = data_dict["x_test"]
        y_test = data_dict["y_test"]
        features = data_dict["features"]

        lr = LinearRegression()
        lr.fit(x_train, y_train)

        y_hat_lr = lr.predict(x_val)

        mae_lr = mean_absolute_error(y_val, y_hat_lr)
        mse_lr = np.mean((y_hat_lr - y_val) ** 2)

        print(f"Linear Regression MSE: {mse_lr}")
        print(f"Linear Regression MAE: {mae_lr}")

        assert assert_array_equal(y_hat_lr, np.ravel(y_hat_lr)) is None

        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(y_val, color="red")
        ax.plot(y_hat_lr, color="orange")
        ax.legend(["True Returns", "Linear Regression"])

        pred_test_lr = lr.predict(x_test)

        plot_concat_data(
            y_val=y_val,
            y_test=y_test,
            pred_test=pred_test_lr,
            df_for_x=features,
            mae=mae_lr,
            window=2,
            prediction_scope=0,
        )

    def test_random_forest_fitting_should_return_expected(self):
        data_dict = data_prep_for_fitting(
            target_tic="AAPL",
            target_col="Close",
            ref_tic="SPY",
            start_date="2001-11-30",
            end_date="2022-08-19",
            window=2,
            percentage=0.995,
            prediction_scope=0,
            is_lstm=False,
            roll_max=7,
            shift_max=7,
        )
        x_train = data_dict["x_train"]
        y_train = data_dict["y_train"]
        x_val = data_dict["x_val"]
        y_val = data_dict["y_val"]
        x_test = data_dict["x_test"]
        y_test = data_dict["y_test"]
        features = data_dict["features"]

        rf = RandomForestRegressor()
        rf.fit(x_train, y_train)

        y_hat_rf = rf.predict(x_val)
        mae_rf = mean_absolute_error(y_val, y_hat_rf)
        mse_rf = np.mean((y_hat_rf - y_val) ** 2)

        print(f"Random Forest MSE: {mse_rf}")
        print(f"Random Forest MAE: {mae_rf}")

        assert assert_array_equal(y_hat_rf, np.ravel(y_hat_rf)) is None
        assert assert_array_equal(y_val, np.ravel(y_val)) is None

        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(y_val, color="red")
        ax.plot(y_hat_rf, color="grey", alpha=0.2)
        ax.legend(["True Returns", "Random Forest"])

    def test_xgboost_fitting_should_return_expected(self):
        data_dict = data_prep_for_fitting(
            target_tic="AAPL",
            target_col="Close",
            ref_tic="SPY",
            start_date="2001-11-30",
            end_date="2022-08-19",
            window=2,
            percentage=0.995,
            prediction_scope=0,
            is_lstm=False,
            roll_max=7,
            shift_max=7,
        )
        x_train = data_dict["x_train"]
        y_train = data_dict["y_train"]
        x_val = data_dict["x_val"]
        y_val = data_dict["y_val"]
        x_test = data_dict["x_test"]
        y_test = data_dict["y_test"]
        features = data_dict["features"]

        xgb_model = XGBRegressor(gamma=1, n_estimators=200)
        xgb_model.fit(x_train, y_train)
        y_hat_xgb = xgb_model.predict(x_val)
        mae_xgb = mean_absolute_error(y_val, y_hat_xgb)
        mse_xgb = np.mean((y_hat_xgb - y_val) ** 2)

        pred_test_xgb = xgb_model.predict(x_test)

        plot_concat_data(
            y_val=y_val,
            y_test=y_test,
            pred_test=pred_test_xgb,
            df_for_x=features,
            mae=mae_xgb,
            window=2,
            prediction_scope=0,
        )

        print(f"XGBoost MSE: {mse_xgb}")
        print(f"XGBoost MAE: {mae_xgb}")

        plt.figure(figsize=(15, 6))
        sns.set_theme(style="white")
        sns.lineplot(x=range(len(y_val)), y=y_val, color="grey", alpha=0.4)
        sns.lineplot(x=range(len(y_val)), y=y_hat_xgb, color="red")

        plt.xlabel("Time")
        plt.ylabel("AAPL stock price")
        plt.title(f"The MAE for this period is: {round(mae_xgb, 3)}")

        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        plot_importance(xgb_model, ax=ax, height=0.5, max_num_features=10)
        plt.show()

        joblib.dump(xgb_model, "XGBoost.pkl")

        ##  Add the predictions (if needed)
        # Transpose row vector -> column vector
        y_hat_train = np.expand_dims(xgb_model.predict(x_train), axis=1)

        # initialize array with np.nan
        array = np.empty((features.shape[0] - y_hat_train.shape[0], 1))
        array[:] = np.nan

        # is this correct? predictions = np.concatenate((y_hat_train, array))
        predictions = np.concatenate((array, y_hat_train))

    @pytest.mark.skip("Takes long time")
    def test_preped_data_with_xgb_should_work_expected(self):
        for percentage in tqdm([0.995, 0.99, 0.98, 0.97, 0.95, 0.92]):
            for window in [35, 30, 25, 20, 10, 7, 6, 5, 4, 3, 2, 1]:
                data_dict = data_prep_for_fitting(
                    target_tic="AAPL",
                    target_col="Close",
                    ref_tic="SPY",
                    start_date="1993-11-30",
                    end_date="2022-08-19",
                    window=window,
                    percentage=percentage,
                    prediction_scope=0,
                    is_lstm=False,
                    roll_max=7,
                    shift_max=7,
                )

                x_train = data_dict["x_train"]
                y_train = data_dict["y_train"]
                x_val = data_dict["x_val"]
                y_val = data_dict["y_val"]
                x_test = data_dict["x_test"]
                y_test = data_dict["y_test"]
                features = data_dict["features"]

                xgb_model = XGBRegressor(gamma=1)
                xgb_model.fit(x_train, y_train)

                pred_val = xgb_model.predict(x_val)
                mae = mean_absolute_error(y_val, pred_val)

                pred_test = xgb_model.predict(x_test)

    def test_minmaxscaler_with_out_range_data_should_return_expected(self):
        data_list = [[1, 20], [5, 60], [7, 80], [10, 100]]
        data = np.array(data_list)
        scaler = MinMaxScaler()
        scaler.fit(data)
        out_range_data = np.array([[20, 200]])
        out_range_trans = scaler.transform(out_range_data)
        out_range_trans_expected = np.array(
            [[(20 - 1) / (10 - 1), (200 - 20) / (100 - 20)]]
        )
        assert (
            assert_allclose(out_range_trans, out_range_trans_expected) is None
        )
        out_range_restore = scaler.inverse_transform(out_range_trans)
        assert assert_allclose(out_range_restore, out_range_data) is None

    def test_minmaxscaler_should_return_in_range_data(self):
        data_list = [[1, 20], [5, 60], [7, 80], [10, 100]]
        data = np.array(data_list)
        scaler = MinMaxScaler()
        scaler.fit(data)
        data_transformed = scaler.transform(data_list)
        condition = (0 <= data_transformed) & (data_transformed <= 1)

        expected = np.ones((4, 2), dtype=bool)
        assert assert_array_equal(np.asarray(condition), expected) is None

    def test_minmaxscaler_should_return_same_transformed_data(self):
        data_list = [[1, 20], [5, 60], [7, 80], [10, 100]]
        data = np.array(data_list)
        scaler = MinMaxScaler()
        scaler.fit(data)
        assert assert_array_equal(scaler.data_min_, np.array([1, 20])) is None
        assert (
            assert_array_equal(scaler.data_max_, np.array([10, 100])) is None
        )
        assert (
            assert_array_equal(np.amin(data, axis=0), np.array([1, 20]))
            is None
        )
        assert (
            assert_array_equal(np.amax(data, axis=0), np.array([10, 100]))
            is None
        )

        target = np.array([[8, 50]])
        data_transformed = scaler.transform(target)
        expected = np.array(
            [
                [
                    (8 - scaler.data_min_[0])
                    / (scaler.data_max_[0] - scaler.data_min_[0]),
                    (50 - scaler.data_min_[1])
                    / (scaler.data_max_[1] - scaler.data_min_[1]),
                ]
            ]
        )
        assert assert_allclose(data_transformed, expected) is None

    def test_transform_minmaxscaler_should_return_expected(self):
        data_list = [[1, 20], [5, 60], [7, 80], [10, 100]]
        data = np.array(data_list)
        data_transformed = transform_min_max_scaler(target=data, base=data)
        transform_expected = np.array(
            [
                [(1 - 1) / (10 - 1), (20 - 20) / (100 - 20)],
                [(5 - 1) / (10 - 1), (60 - 20) / (100 - 20)],
                [(7 - 1) / (10 - 1), (80 - 20) / (100 - 20)],
                [(10 - 1) / (10 - 1), (100 - 20) / (100 - 20)],
            ]
        )
        assert assert_allclose(data_transformed, transform_expected) is None

        base_list = data_list + [[0, 0], [20, 200]]
        base = np.array(base_list)
        data_transformed = transform_min_max_scaler(target=data, base=base)
        transform_expected = np.array(
            [
                [(1 - 0) / (20 - 0), (20 - 0) / (200 - 0)],
                [(5 - 0) / (20 - 0), (60 - 0) / (200 - 0)],
                [(7 - 0) / (20 - 0), (80 - 0) / (200 - 0)],
                [(10 - 0) / (20 - 0), (100 - 0) / (200 - 0)],
            ]
        )
        assert assert_allclose(data_transformed, transform_expected) is None

    def test_lstm_fitting_works_as_expected(self):
        data_dict = data_prep_for_fitting(
            target_tic="AAPL",
            target_col="Close",
            ref_tic="SPY",
            start_date="1993-11-30",
            end_date="2022-08-19",
            window=30,
            percentage=0.98,
            prediction_scope=0,
            is_lstm=True,
            roll_max=7,
            shift_max=7,
        )

        x_train = data_dict["x_train"]
        y_train = data_dict["y_train"]
        x_val = data_dict["x_val"]
        y_val = data_dict["y_val"]
        x_test = data_dict["x_test"]
        y_test = data_dict["y_test"]
        features = data_dict["features"]
        target_min = data_dict["target_min"]
        target_max = data_dict["target_max"]

        model_lstm = setup_lstm_model(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            epoch_n=50,
            batch_size=20,
            threshold=0.031,
        )
        y_hat_val = model_lstm.predict(x_val)

        y_val_restored = y_val * (target_max - target_min) + target_min
        y_hat_val_restored = y_hat_val * (target_max - target_min) + target_min
        mae_val_scaled = mean_absolute_error(y_val, y_hat_val)

        mae_val_restored = mean_absolute_error(
            y_val_restored, y_hat_val_restored
        )
        assert mae_val_restored == pytest.approx(
            (target_max - target_min) * mae_val_scaled, 0.0000001
        )

        y_hat_train = model_lstm.predict(x_train)
        mean_absolute_error(y_train, y_hat_train)

        y_hat_test = model_lstm.predict(x_test)
        y_hat_test_restored = (
            y_hat_test * (target_max - target_min) + target_min
        )

    def test_mean_absolute_error_should_work_regardless_of_row_or_col_vector(
        self,
    ):
        y_true = np.array([1, 1, 1, 1, 1])
        y_pred = np.array([2, 2, 2, 2, 2])
        assert mean_absolute_error(y_true, y_pred) == 1.0

        y_true_t = np.expand_dims(y_true, axis=1)
        y_pred = np.array([2, 2, 2, 2, 2])
        assert mean_absolute_error(y_true_t, y_pred) == 1.0

        y_pred_t = np.expand_dims(y_pred, axis=1)
        assert mean_absolute_error(y_true, y_pred_t) == 1.0

        assert mean_absolute_error(y_true_t, y_pred_t) == 1.0

    def test_inverse_transform_return_expected(self):
        data_dict = data_prep_for_fitting(
            target_tic="AAPL",
            target_col="Close",
            ref_tic="SPY",
            start_date="1993-11-30",
            end_date="2022-08-19",
            window=30,
            percentage=0.98,
            prediction_scope=0,
            is_lstm=True,
            roll_max=7,
            shift_max=7,
        )

        x_train = data_dict["x_train"]
        y_train = data_dict["y_train"]
        x_val = data_dict["x_val"]
        y_val = data_dict["y_val"]
        x_test = data_dict["x_test"]
        y_test = data_dict["y_test"]
        features = data_dict["features"]

        model_lstm = setup_lstm_model(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            epoch_n=3,
            batch_size=20,
            threshold=0.031,
        )
        y_val_hat_lstm = model_lstm.predict(x_val)
        y_val_hat_lstm.shape

        x_val.shape
        x_val[0][0].shape
        y_val.shape  # (114,)

        pred_test_lstm = model_lstm.predict(x_test)

        pred_test_lstm.shape  # (1, 1)
        y_test.shape  # (30,)

    @pytest.mark.skip("Need to fix bug")
    def test_keras_lstm_should_return_expected(self):
        inputs = tf.random.normal([32, 10, 8])
        lstm_m1 = tf.keras.layers.LSTM(units=4)
        output = lstm_m1(inputs)
        output_shape_expected = tf.TensorShape([32, 4])
        assert output.shape == output_shape_expected

        lstm_m2 = tf.keras.layers.LSTM(
            units=4, return_sequences=True, return_state=True
        )
        whole_seq_output, final_memory_state, final_carry_state = lstm_m2(
            inputs
        )
        assert whole_seq_output.shape == tf.TensorShape([32, 10, 4])
        assert final_memory_state.shape == tf.TensorShape([32, 4])
        assert final_carry_state.shape == tf.TensorShape([32, 4])

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
