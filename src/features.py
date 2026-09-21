"""
Feature engineering - calculate indicators from price data
"""

import numpy as np
import pandas as pd
from config import FEATURE_PARAMS


def create_features(df_btc, df_eth=None, df_gold=None, df_hashrate=None,
                    df_funding=None, df_fear_greed=None, clip_bounds=None):
    """
    Create features from raw OHLCV data

    Returns dataframe with all the indicators, unclipped unless
    `clip_bounds` is supplied. There is no fit-from-this-frame mode:
    fit outlier bounds on the training split with `fit_clip_bounds`
    and pass the result in here for every other split, or the bounds
    become look-ahead leakage.
    """
    p = FEATURE_PARAMS
    features = pd.DataFrame(index=df_btc.index)

    price = df_btc['close']
    returns = price.pct_change()
    volume = df_btc['volume']

    # --- MOMENTUM ---
    # how much price changed over different periods
    for window in p['momentum_windows']:
        features[f'momentum_{window}h'] = price.pct_change(window)

    # --- VOLATILITY ---
    # how much price moves around
    for window in p['volatility_windows']:
        min_periods = window // 2
        features[f'vol_{window}h'] = returns.rolling(window, min_periods=min_periods).std()

    # volume compared to average
    if len(p['volatility_windows']) > 1:
        vol_window = p['volatility_windows'][1]
    else:
        vol_window = p['volatility_windows'][0]
    vol_ma = volume.rolling(vol_window, min_periods=vol_window//2).mean()
    features['volume_ratio'] = volume / (vol_ma + 0.00000001)

    # candle range
    features['range'] = (df_btc['high'] - df_btc['low']) / (price + 0.00000001)

    # --- MULTI TIMEFRAME ---
    # 4 hour data
    #
    # label='right', closed='right' dates each bucket at the moment it is
    # complete. Pandas labels a bucket at its start by default, and the
    # forward fill below would then publish the bucket's closing price
    # across that same bucket's earlier hours. For this 4-hour bucket, the
    # old default could publish a value up to about 3 hours before it was
    # actually known.
    #
    # The change is taken on the resampled series and only then spread onto
    # the hourly index, so a shift of N means N buckets. Taking it after the
    # reindex shifts N rows, which on an hourly index means N hours.
    df_4h = df_btc.resample('4h', label='right', closed='right').agg({'close': 'last'}).ffill()
    features['momentum_4h_agg'] = (
        df_4h['close'].pct_change(6).reindex(df_btc.index, method='ffill')
    )

    # daily data
    #
    # Same fix as above, but the skew it masked was much larger here: at
    # 01:00 the old default could publish the close from 23:00 that same
    # day, 22 hours ahead of when it was actually known, for every row.
    df_daily = df_btc.resample('1D', label='right', closed='right').agg({'close': 'last'}).ffill()
    daily_close = df_daily['close']
    features['momentum_daily_7d'] = (
        daily_close.pct_change(7).reindex(df_btc.index, method='ffill')
    )
    features['momentum_daily_30d'] = (
        daily_close.pct_change(30).reindex(df_btc.index, method='ffill')
    )

    # --- RSI ---
    delta = price.diff()
    gain = delta.copy()
    loss = delta.copy()

    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = -loss

    avg_gain = gain.rolling(p['rsi_period'], min_periods=p['rsi_period']//2).mean()
    avg_loss = loss.rolling(p['rsi_period'], min_periods=p['rsi_period']//2).mean()

    rs = avg_gain / (avg_loss + 0.00000001)
    features['rsi'] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema_fast = price.ewm(span=p['macd_fast'], adjust=False).mean()
    ema_slow = price.ewm(span=p['macd_slow'], adjust=False).mean()
    features['macd'] = (ema_fast - ema_slow) / (price + 0.00000001)

    # --- ETH CORRELATION ---
    shift = p['external_shift']
    corr_window = p['correlation_window']

    if df_eth is not None and len(df_eth) > 0:
        eth_price = df_eth['close'].reindex(df_btc.index, method='ffill')
        features['eth_momentum'] = eth_price.pct_change(24)

        eth_returns = eth_price.pct_change()
        features['btc_eth_corr'] = returns.rolling(corr_window).corr(eth_returns)

    # --- GOLD CORRELATION ---
    if df_gold is not None and len(df_gold) > 0:
        gold_price = df_gold['close'].reindex(df_btc.index, method='ffill')
        features['gold_momentum'] = gold_price.pct_change(24)

        gold_returns = gold_price.pct_change()
        features['btc_gold_corr'] = returns.rolling(corr_window).corr(gold_returns)

    # --- HASH RATE ---
    if df_hashrate is not None and len(df_hashrate) > 0:
        hr = df_hashrate.reindex(df_btc.index, method='ffill')
        hr = hr.shift(shift)  # shift to avoid lookahead

        if len(hr.columns) > 0:
            hr_series = hr.iloc[:, 0]
            features['hashrate_momentum'] = hr_series.pct_change(168)

            hr_ma = hr_series.rolling(720, min_periods=360).mean()
            features['hashrate_ratio'] = hr_series / (hr_ma + 0.00000001)

    # --- FUNDING RATE ---
    if df_funding is not None and 'fundingRate' in df_funding.columns:
        funding = df_funding['fundingRate'].reindex(df_btc.index, method='ffill')
        funding = funding.shift(shift)
        features['funding'] = funding.fillna(0)

    # --- FEAR & GREED ---
    if df_fear_greed is not None and 'value' in df_fear_greed.columns:
        fg = df_fear_greed['value'].reindex(df_btc.index, method='ffill')
        fg = fg.shift(shift)
        features['fear_greed'] = (fg / 100).fillna(0.5)

    # --- CLEAN UP ---
    # fill missing values
    features = features.ffill()
    features = features.fillna(0)

    # replace infinity with 0
    features = features.replace([np.inf, -np.inf], 0)

    if clip_bounds is not None:
        features = apply_clip_bounds(features, clip_bounds)

    return features


def fit_clip_bounds(features, lower_q=0.005, upper_q=0.995):
    """Fit per-column outlier bounds.

    Call this on the training split only, then pass the result to
    create_features for every other split. Fitting over a whole series and
    applying the result to its own past is look-ahead leakage: it lets a
    feature value at time t depend on rows after t. This is the same
    discipline the RobustScaler in scripts/train.py already follows.
    """
    return {
        str(col): (float(features[col].quantile(lower_q)), float(features[col].quantile(upper_q)))
        for col in features.columns
    }


def apply_clip_bounds(features, bounds):
    """Clip each column to its fitted bounds, leaving unlisted columns alone."""
    out = features.copy()
    for col, (lower, upper) in bounds.items():
        if col in out.columns:
            out[col] = out[col].clip(lower, upper)
    return out


def returns_to_percentiles(returns, window=720):
    """
    Convert returns to percentile ranks

    For each return, calculate what percentile it is compared to history
    """
    percentiles = pd.Series(index=returns.index, dtype=float)

    for i in range(len(returns)):
        if i < 10:
            percentiles.iloc[i] = 0.5
            continue

        # get history
        start = max(0, i - window)
        history = returns.iloc[start:i]

        # what percent of history is less than current return
        current_return = returns.iloc[i]
        pct = (history < current_return).sum() / len(history)
        percentiles.iloc[i] = pct

    return percentiles
