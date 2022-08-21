import pytest
import datetime as dt
import numpy as np
import pandas as pd
from pandas import testing as tm
from pandas.core.frame import DataFrame
from pandas.core.series import Series
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
        assert "Close" not in apple_price_feat.columns
        assert "Close_y" in apple_price_feat.columns

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
