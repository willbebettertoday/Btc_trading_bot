"""Tests for scripts/backtest.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backtest import performance_summary, walk_forward_splits  # noqa: E402


class TestWalkForwardSplits:
    def test_test_window_always_follows_its_train_window(self):
        for train, test in walk_forward_splits(1000, train_size=400, test_size=100):
            assert max(train) < min(test)

    def test_test_windows_never_overlap(self):
        splits = walk_forward_splits(1000, train_size=400, test_size=100)
        seen = set()
        for _, test in splits:
            assert not seen & set(test)
            seen |= set(test)

    def test_train_window_expands(self):
        splits = walk_forward_splits(1000, train_size=400, test_size=100)
        sizes = [len(train) for train, _ in splits]
        assert sizes == sorted(sizes)
        assert sizes[-1] > sizes[0]

    def test_too_little_data_yields_no_splits(self):
        assert walk_forward_splits(300, train_size=400, test_size=100) == []


class TestPerformanceSummary:
    def test_sharpe_of_a_constant_series_is_infinite_not_a_crash(self):
        out = performance_summary(np.full(50, 0.01))
        assert np.isinf(out["sharpe"])

    def test_no_trades_returns_zeros_rather_than_nan(self):
        out = performance_summary(np.array([]))
        assert out["n_trades"] == 0
        assert out["sharpe"] == 0.0
        assert out["win_rate"] == 0.0

    def test_win_rate_counts_positive_returns(self):
        out = performance_summary(np.array([0.01, -0.01, 0.02, -0.005]))
        assert out["win_rate"] == pytest.approx(0.5)
        assert out["n_trades"] == 4

    def test_max_drawdown_is_reported_as_a_negative_fraction(self):
        out = performance_summary(np.array([0.10, -0.50, 0.10]))
        assert out["max_drawdown"] < 0
        assert out["max_drawdown"] >= -1.0

    def test_sharpe_is_annualised_from_the_period_count(self):
        returns = np.array([0.01, -0.005, 0.015, 0.0, -0.01, 0.02])
        hourly = performance_summary(returns, periods_per_year=8760)
        daily = performance_summary(returns, periods_per_year=365)
        assert abs(hourly["sharpe"]) > abs(daily["sharpe"])
