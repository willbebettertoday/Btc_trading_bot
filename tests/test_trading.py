"""Tests for src/trading.py. Config values come from the stub in conftest."""

import numpy as np
import pytest

from src import trading


class TestIsSignal:
    @pytest.mark.parametrize(
        "percentile,expected",
        [(0.80, True), (0.75, True), (0.50, False), (0.26, False), (0.25, True), (0.05, True)],
    )
    def test_thresholds_are_inclusive(self, percentile, expected):
        assert trading.is_signal(percentile) is expected


class TestGetConfidence:
    @pytest.mark.parametrize(
        "percentile,expected",
        [(0.75, 0.0), (1.00, 1.0), (0.875, 0.5), (0.25, 0.0), (0.00, 1.0), (0.50, 0.0)],
    )
    def test_confidence_scales_from_threshold_to_extreme(self, percentile, expected):
        assert trading.get_confidence(percentile) == pytest.approx(expected)


class TestCalculateTpSl:
    def test_long_puts_tp_above_and_sl_below(self):
        tp, sl = trading.calculate_tp_sl("LONG", expected_return=0.02, confidence=0.0)
        assert tp == pytest.approx(0.04)
        assert sl == pytest.approx(-0.02)

    def test_short_inverts_the_signs(self):
        tp, sl = trading.calculate_tp_sl("SHORT", expected_return=-0.02, confidence=0.0)
        assert tp == pytest.approx(0.04)
        assert sl == pytest.approx(0.02)

    def test_stop_is_never_tighter_than_the_minimum(self):
        tp, sl = trading.calculate_tp_sl("LONG", expected_return=0.001, confidence=0.0)
        assert sl == pytest.approx(-0.005)
        assert tp == pytest.approx(0.002)


class TestGenerateSignal:
    def test_returns_none_outside_the_signal_band(self):
        assert trading.generate_signal(0.50, np.linspace(-0.05, 0.05, 101)) is None

    def test_long_signal_is_fully_specified(self):
        sig = trading.generate_signal(0.80, np.linspace(-0.05, 0.05, 101))
        assert sig["direction"] == "LONG"
        assert sig["confidence"] == pytest.approx(0.2)
        assert sig["expected_return"] == pytest.approx(0.03)
        assert sig["take_profit"] == pytest.approx(0.06)
        assert sig["stop_loss"] == pytest.approx(-0.03)
        assert sig["rr_ratio"] == pytest.approx(2.0)

    def test_low_confidence_signals_are_skipped(self, monkeypatch):
        # trading.py binds config values at import time, so patch the module.
        monkeypatch.setattr(trading, "MIN_CONFIDENCE", 0.5)
        assert trading.generate_signal(0.80, np.linspace(-0.05, 0.05, 101)) is None


class TestPositionSize:
    def test_size_is_risk_divided_by_stop_distance(self):
        assert trading.calculate_position_size(-0.02) == pytest.approx(0.5)

    def test_zero_stop_gives_zero_size_instead_of_dividing_by_zero(self):
        assert trading.calculate_position_size(0) == 0


class TestCheckExit:
    LONG = {"entry_price": 100.0, "direction": "LONG", "tp_price": 104.0, "sl_price": 98.0}
    SHORT = {"entry_price": 100.0, "direction": "SHORT", "tp_price": 96.0, "sl_price": 102.0}

    def test_long_take_profit(self):
        bar = {"high": 105.0, "low": 99.5, "close": 104.5}
        assert trading.check_exit(self.LONG, bar, 1) == (True, "TP", pytest.approx(0.04))

    def test_long_stop_loss(self):
        bar = {"high": 100.5, "low": 97.0, "close": 97.5}
        assert trading.check_exit(self.LONG, bar, 1) == (True, "SL", pytest.approx(-0.02))

    def test_ambiguous_bar_books_the_loss_not_the_win(self):
        """An hourly bar hides the order of the two touches.

        Booking the win here is what inflates a backtest's Sharpe ratio, so
        the conservative assumption is that the stop filled first.
        """
        bar = {"high": 105.0, "low": 97.0, "close": 100.0}
        exited, reason, pnl = trading.check_exit(self.LONG, bar, 1)
        assert (exited, reason) == (True, "SL")
        assert pnl == pytest.approx(-0.02)

    def test_ambiguous_bar_books_the_loss_for_shorts_too(self):
        bar = {"high": 103.0, "low": 95.0, "close": 100.0}
        exited, reason, pnl = trading.check_exit(self.SHORT, bar, 1)
        assert (exited, reason) == (True, "SL")
        assert pnl == pytest.approx(-0.02)

    def test_time_exit_uses_the_close(self):
        bar = {"high": 101.0, "low": 99.0, "close": 101.0}
        assert trading.check_exit(self.LONG, bar, 48) == (True, "TIME", pytest.approx(0.01))

    def test_open_trade_stays_open(self):
        bar = {"high": 101.0, "low": 99.0, "close": 100.0}
        assert trading.check_exit(self.LONG, bar, 1) == (False, "", 0.0)
