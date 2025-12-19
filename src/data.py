"""
Data loading functions
"""

import os
import json
import pandas as pd
import numpy as np
import ccxt
from sklearn.preprocessing import RobustScaler

from config import CACHE_DIR, CACHE_DAILY


def fetch_ohlcv(symbol, timeframe, limit):
    """Download OHLCV data from Binance"""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        df = pd.DataFrame(ohlcv)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        return df
    
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def get_current_price(symbol='BTC/USDT'):
    """Get current price"""
    try:
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except:
        return None


def load_cached_data():
    """Load all the cached CSV files"""
    data = {}
    
    # helper function to load csv safely
    def load_csv(path, name):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df[df.index.notna()]
            df = df[~df.index.duplicated(keep='last')]
            df = df.sort_index()
            return df
        except Exception as e:
            print(f"Warning loading {name}: {e}")
            return None
    
    # load all files
    data['btc'] = load_csv(f"{CACHE_DIR}/btc_1h.csv", "BTC")
    data['eth'] = load_csv(f"{CACHE_DIR}/eth_1h.csv", "ETH")
    data['gold'] = load_csv(f"{CACHE_DIR}/gold_1h.csv", "Gold")
    data['hashrate'] = load_csv(f"{CACHE_DIR}/hash_rate.csv", "Hash Rate")
    data['funding'] = load_csv(f"{CACHE_DIR}/funding_1h.csv", "Funding")
    data['fear_greed'] = load_csv(f"{CACHE_DAILY}/fear_greed.csv", "Fear & Greed")
    
    return data


def load_scaler(path):
    """Load the saved scaler"""
    with open(path, 'r') as f:
        params = json.load(f)
    
    scaler = RobustScaler()
    scaler.center_ = np.array(params['center_'])
    scaler.scale_ = np.array(params['scale_'])
    scaler.n_features_in_ = params['n_features_in_']
    
    feature_names = params['feature_names']
    
    return scaler, feature_names