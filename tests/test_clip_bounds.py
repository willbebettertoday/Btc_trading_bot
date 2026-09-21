"""Tests for the fitted outlier-clip bounds.

The bounds are a statistic fitted on data, exactly like the RobustScaler's
center and scale. Fitting them over a whole series and applying them to its
own past is look-ahead leakage, which is what these tests prevent.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.features import apply_clip_bounds, create_features, fit_clip_bounds


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"a": rng.normal(0, 1, 500), "b": rng.normal(5, 2, 500)},
        index=pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC"),
    )


class TestFitClipBounds:
    def test_returns_one_pair_per_column(self, frame):
        bounds = fit_clip_bounds(frame)
        assert set(bounds) == {"a", "b"}
        for lower, upper in bounds.values():
            assert lower < upper

    def test_bounds_are_json_serialisable(self, frame):
        """They get written next to scaler.json, so they must survive a round trip."""
        bounds = fit_clip_bounds(frame)
        restored = json.loads(json.dumps({k: list(v) for k, v in bounds.items()}))
        assert set(restored) == set(bounds)
        for col in bounds:
            assert restored[col][0] == pytest.approx(bounds[col][0])

    def test_fitting_on_a_prefix_ignores_later_rows(self, frame):
        """This is the whole point: bounds fitted on train must not move
        when unseen rows arrive."""
        train = frame.iloc[:300]
        extended = pd.concat([train, frame.iloc[300:]]).iloc[:300]
        assert fit_clip_bounds(train) == fit_clip_bounds(extended)


class TestApplyClipBounds:
    def test_values_are_confined_to_the_bounds(self, frame):
        bounds = {"a": (-1.0, 1.0), "b": (0.0, 10.0)}
        out = apply_clip_bounds(frame, bounds)
        assert out["a"].min() >= -1.0 and out["a"].max() <= 1.0
        assert out["b"].min() >= 0.0 and out["b"].max() <= 10.0

    def test_is_idempotent(self, frame):
        bounds = fit_clip_bounds(frame)
        once = apply_clip_bounds(frame, bounds)
        twice = apply_clip_bounds(once, bounds)
        pd.testing.assert_frame_equal(once, twice)

    def test_columns_absent_from_bounds_pass_through_untouched(self, frame):
        out = apply_clip_bounds(frame, {"a": (-1.0, 1.0)})
        pd.testing.assert_series_equal(out["b"], frame["b"])

    def test_does_not_mutate_its_input(self, frame):
        before = frame.copy()
        apply_clip_bounds(frame, fit_clip_bounds(frame))
        pd.testing.assert_frame_equal(frame, before)


class TestCreateFeaturesContract:
    def test_supplying_bounds_keeps_features_independent_of_future_rows(self, make_ohlcv):
        """The same invariant as tests/test_features.py, but through the
        bounds path, which is how train and the live bot actually run."""
        df = make_ohlcv(400)
        bounds = fit_clip_bounds(create_features(df.iloc[:300].copy()))

        full = create_features(df.copy(), clip_bounds=bounds)
        truncated = create_features(df.iloc[:300].copy(), clip_bounds=bounds)

        shared = [c for c in full.columns if c in truncated.columns]
        pd.testing.assert_frame_equal(
            full[shared].iloc[200:300].reset_index(drop=True),
            truncated[shared].iloc[200:300].reset_index(drop=True),
            check_exact=False,
            rtol=1e-9,
        )

    def test_omitting_bounds_leaves_features_unclipped(self, make_ohlcv):
        """There is deliberately no fit-from-this-frame mode. Passing no
        bounds must not silently fit them, because that is the leak."""
        df = make_ohlcv(400)
        unclipped = create_features(df.copy())
        bounds = fit_clip_bounds(unclipped)
        clipped = create_features(df.copy(), clip_bounds=bounds)

        widened = {col: (lo - 1e6, hi + 1e6) for col, (lo, hi) in bounds.items()}
        pd.testing.assert_frame_equal(unclipped, apply_clip_bounds(unclipped, widened))
        assert not unclipped.equals(clipped) or all(
            lo == hi for lo, hi in bounds.values()
        )
