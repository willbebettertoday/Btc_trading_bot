"""
Feature engineering - calculate indicators from price data
"""

import numpy as np
import pandas as pd

from config import FEATURE_PARAMS


def create_features(df_btc, df_eth=None, df_gold=None, df_hashrate=None, 
                    df_funding=None, df_fear_greed=None):
    """
    Create features from raw OHLCV data
    
    Returns dataframe with all the indicators
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
    df_4h = df_btc.resample('4H').agg({'close': 'last'}).ffill()
    price_4h = df_4h['close'].reindex(df_btc.index, method='ffill')
    features['momentum_4h_agg'] = price_4h.pct_change(6)
    
    # daily data
    df_daily = df_btc.resample('1D').agg({'close': 'last'}).ffill()
    price_daily = df_daily['close'].reindex(df_btc.index, method='ffill')
    features['momentum_daily_7d'] = price_daily.pct_change(7)
    features['momentum_daily_30d'] = price_daily.pct_change(30)
    
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
    
    # clip outliers
    for col in features.columns:
        lower = features[col].quantile(0.005)
        upper = features[col].quantile(0.995)
        features[col] = features[col].clip(lower, upper)
    
    return features


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