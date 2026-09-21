"""Tests for the fitted outlier-clip bounds.

The bounds are a statistic fitted on data, exactly like the RobustScaler's
center and scale. Fitting them over a whole series and applying them to its
own past is look-ahead leakage, which is what these tests prevent.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.data import load_clip_bounds
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

    def test_fit_depends_on_the_rows_it_is_given(self, frame):
        """This is why fit_clip_bounds is called on the training split
        alone and its result passed, unchanged, to every other split: if
        bounds fitted on train and on train-plus-outliers were identical,
        the whole train-only discipline would be pointless."""
        train = frame.iloc[:300]
        outliers = pd.DataFrame(
            {"a": [1e6, -1e6] * 25, "b": [1e6, -1e6] * 25},
            index=pd.date_range("2030-01-01", periods=50, freq="h", tz="UTC"),
        )
        extended = pd.concat([train, outliers])

        train_bounds = fit_clip_bounds(train)
        extended_bounds = fit_clip_bounds(extended)

        assert train_bounds != extended_bounds
        for col in train_bounds:
            assert extended_bounds[col][1] > train_bounds[col][1], (
                f"{col}'s upper bound did not move after outliers were appended"
            )
            assert extended_bounds[col][0] < train_bounds[col][0], (
                f"{col}'s lower bound did not move after outliers were appended"
            )


class TestLoadClipBounds:
    def test_round_trips_through_the_real_save_and_load_path(self, frame, tmp_path):
        """train.py writes clip_bounds.json as {col: list(pair)}; make sure
        load_clip_bounds (not a reimplementation of it) reads that back
        correctly, including that pairs come back as float tuples rather
        than the lists JSON stores."""
        bounds = fit_clip_bounds(frame)
        path = tmp_path / "clip_bounds.json"
        with open(path, "w") as f:
            json.dump({col: list(pair) for col, pair in bounds.items()}, f, indent=2)

        loaded = load_clip_bounds(path)

        assert loaded == bounds
        for pair in loaded.values():
            assert isinstance(pair, tuple)
            assert all(isinstance(v, float) for v in pair)


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
