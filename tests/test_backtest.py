"""Tests for scripts/backtest.py."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backtest import apply_costs, performance_summary, walk_forward_splits  # noqa: E402


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

    def test_sharpe_is_annualised_by_the_period_count_exactly(self):
        returns = np.array([0.01, -0.005, 0.015, 0.0, -0.01, 0.02])
        mean, std = returns.mean(), returns.std(ddof=1)

        hourly = performance_summary(returns, periods_per_year=8760)
        assert hourly["sharpe"] == pytest.approx(mean / std * math.sqrt(8760))

        daily = performance_summary(returns, periods_per_year=365)
        assert daily["sharpe"] == pytest.approx(mean / std * math.sqrt(365))

    def test_sharpe_is_unchanged_when_the_series_is_lengthened_by_repetition(self):
        # Repeating the same underlying pattern must not systematically move
        # the annualised Sharpe just because there are more bars. The ddof=1
        # sample correction on std causes a small, bounded drift as the
        # sample grows, but nothing close to the ~4x shrink the previous
        # (incorrect) formula produced by dividing periods_per_year by the
        # sample size.
        returns = np.array([0.01, -0.005, 0.015, 0.0, -0.01, 0.02])
        short = performance_summary(returns)["sharpe"]
        long = performance_summary(np.tile(returns, 20))["sharpe"]
        assert long == pytest.approx(short, rel=0.15)


class TestApplyCosts:
    def test_costs_are_subtracted_as_a_round_trip(self):
        pnl = apply_costs(0.05, cost_bps=10, slippage_bps=5)
        assert pnl == pytest.approx(0.05 - 2 * (10 + 5) / 10_000)

    def test_zero_costs_leave_the_return_unchanged(self):
        assert apply_costs(0.05, cost_bps=0, slippage_bps=0) == pytest.approx(0.05)
