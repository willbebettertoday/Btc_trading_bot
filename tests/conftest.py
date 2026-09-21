"""Test fixtures. Injects a stub `config` module so `src` is importable."""

import sys
import types

import numpy as np
import pandas as pd
import pytest


def _build_config_stub():
    cfg = types.ModuleType("config")
    cfg.TOP_PERCENTILE = 0.75
    cfg.BOTTOM_PERCENTILE = 0.25
    cfg.MIN_CONFIDENCE = 0.0
    cfg.TP_PARAMS = {"base": 2.0, "confidence": 0.0}
    cfg.SL_PARAMS = {"base": 1.0, "confidence": 0.0, "minimum": 0.005}
    cfg.RISK_PER_TRADE = 0.01
    cfg.MAX_HOLD_HOURS = 48
    cfg.MIN_HOURS_BETWEEN_TRADES = 8
    cfg.FEATURE_PARAMS = {
        "momentum_windows": [6, 12, 24],
        "volatility_windows": [24],
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "correlation_window": 48,
        "external_shift": 24,
    }
    cfg.CACHE_DIR = "/tmp/btc-test-cache"
    cfg.CACHE_DAILY = "/tmp/btc-test-cache-daily"
    cfg.START_DATE = "2019-09-01"
    cfg.D_MODEL, cfg.N_HEADS, cfg.N_LAYERS, cfg.D_FF, cfg.DROPOUT = 32, 4, 2, 64, 0.1
    return cfg


sys.modules.setdefault("config", _build_config_stub())


@pytest.fixture
def make_ohlcv():
    """Return a factory for deterministic hourly OHLCV frames."""

    def _make(n=400, seed=0):
        rng = np.random.default_rng(seed)
        close = 20000.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
        high = close * (1 + rng.uniform(0.0005, 0.004, n))
        low = close * (1 - rng.uniform(0.0005, 0.004, n))
        open_ = np.concatenate([[close[0]], close[:-1]])
        return pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(high, np.maximum(open_, close)),
                "low": np.minimum(low, np.minimum(open_, close)),
                "close": close,
                "volume": rng.uniform(10, 100, n),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        )

    return _make
