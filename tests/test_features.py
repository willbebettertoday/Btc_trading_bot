"""Invariant tests for src/features.py.

These do not depend on which indicators are implemented. They check the
properties that make a backtest trustworthy.
"""

import numpy as np
import pandas as pd

from src.features import create_features


def test_output_keeps_one_row_per_input_row(make_ohlcv):
    df = make_ohlcv(400)
    out = create_features(df.copy())
    assert len(out) == len(df)
    assert out.index.equals(df.index)


def test_no_feature_uses_future_data(make_ohlcv):
    """A feature at time t must not change when later rows are appended.

    This is the test that makes the README's performance figures
    defensible. If it fails, every backtest number in this repo is
    optimistic and the cause is here, not in the model.
    """
    df = make_ohlcv(400)
    full = create_features(df.copy())
    truncated = create_features(df.iloc[:300].copy())

    shared = [c for c in full.columns if c in truncated.columns]
    assert shared, "create_features produced no comparable columns"

    pd.testing.assert_frame_equal(
        full[shared].iloc[200:300].reset_index(drop=True),
        truncated[shared].iloc[200:300].reset_index(drop=True),
        check_exact=False,
        rtol=1e-9,
        atol=1e-12,
    )


def test_features_are_finite_after_the_warmup_window(make_ohlcv):
    df = make_ohlcv(600)
    out = create_features(df.copy())
    numeric = out.select_dtypes(include=[np.number]).iloc[400:]
    bad = numeric.columns[~np.isfinite(numeric.to_numpy()).all(axis=0)]
    assert list(bad) == [], f"non-finite values after warmup in: {list(bad)}"
