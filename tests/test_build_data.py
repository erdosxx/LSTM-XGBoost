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
)


class TestBuildData:
    def test_get_apple_data_should_be_expected(self):
        apple_price_dict = download_price_dict(
            tickers=("AAPL",), start="1980-12-13", end="2022-08-19"
        )
        apple_price = apple_price_dict["AAPL"]
        assert isinstance(apple_price, DataFrame) is True
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
