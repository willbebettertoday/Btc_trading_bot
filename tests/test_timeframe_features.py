"""The multi-timeframe momentum features must be causal and span what they claim.

Two defects lived here together. Resampling labelled each bucket at its
start while storing the bucket's last value, so the forward fill published a
day's closing price during that same day's opening hours. And the change was
taken after the result had been spread onto the hourly index, so a shift of
N counted N hours rather than N buckets.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import create_features

DAILY_STEP = 0.01


def _frame(close, index):
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(len(close), 10.0),
        },
        index=index,
    )


def _daily_ramp(n_days):
    """Hourly bars whose close rises by exactly DAILY_STEP each calendar day."""
    n = n_days * 24
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 100.0 * (1.0 + DAILY_STEP) ** (np.arange(n) // 24)
    return _frame(close, index)


class TestSpan:
    def test_seven_day_momentum_spans_seven_days(self):
        out = create_features(_daily_ramp(40))
        assert out["momentum_daily_7d"].iloc[-1] == pytest.approx(
            (1 + DAILY_STEP) ** 7 - 1, rel=1e-6
        )

    def test_thirty_day_momentum_spans_thirty_days(self):
        out = create_features(_daily_ramp(60))
        assert out["momentum_daily_30d"].iloc[-1] == pytest.approx(
            (1 + DAILY_STEP) ** 30 - 1, rel=1e-6
        )

    def test_four_hour_momentum_spans_six_buckets_not_six_hours(self):
        """Six 4-hour buckets is a full day, so on a 1%-per-day ramp the
        change across them is one day's step, not a quarter of one."""
        out = create_features(_daily_ramp(40))
        assert out["momentum_4h_agg"].iloc[-1] == pytest.approx(DAILY_STEP, rel=1e-3)

    def test_seven_day_momentum_is_not_almost_always_zero(self):
        """Shifting seven rows on an hourly index kept both ends inside the
        same forward-filled daily step for most of every day, so the column
        read exactly 0.0 roughly seventeen hours out of twenty-four."""
        out = create_features(_daily_ramp(40))
        tail = out["momentum_daily_7d"].iloc[24 * 10 :]
        zero_fraction = float((tail == 0.0).mean())
        assert zero_fraction < 0.10, f"{zero_fraction:.0%} of rows are exactly zero"


class TestCausality:
    def test_a_days_closing_spike_is_invisible_earlier_that_day(self):
        """This is the leak, stated directly.

        The series is flat except for one huge jump in the final hour of day
        30. Nothing computed during day 30's earlier hours may move, because
        that jump has not happened yet.
        """
        n_days = 40
        n = n_days * 24
        index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        close = np.full(n, 100.0)
        spike_hour = 30 * 24 + 23
        close[spike_hour] = 500.0

        out = create_features(_frame(close, index))

        for col in ("momentum_4h_agg", "momentum_daily_7d", "momentum_daily_30d"):
            during = out[col].iloc[30 * 24 : spike_hour]
            assert (during.abs() < 1e-12).all(), (
                f"{col} moved before the spike that caused it; "
                f"max |value| was {during.abs().max():.6g}"
            )

    def test_aggregates_do_not_change_when_later_rows_arrive(self):
        df = _daily_ramp(40)
        full = create_features(df.copy())
        truncated = create_features(df.iloc[: 30 * 24].copy())
        for col in ("momentum_4h_agg", "momentum_daily_7d", "momentum_daily_30d"):
            pd.testing.assert_series_equal(
                full[col].iloc[20 * 24 : 30 * 24].reset_index(drop=True),
                truncated[col].iloc[20 * 24 :].reset_index(drop=True),
                check_exact=False,
                rtol=1e-9,
            )
