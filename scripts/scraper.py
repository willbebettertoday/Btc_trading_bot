"""
Data scraper - download market data
Run this before training
"""

import os
import sys
import time
from datetime import datetime

import ccxt
import pandas as pd
import requests

# add parent folder to path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import CACHE_DIR, CACHE_DAILY, START_DATE

# create folders
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CACHE_DAILY, exist_ok=True)


def download_hashrate():
    """Download hash rate from mempool.space"""
    print("Downloading hash rate...", end=' ')
    
    filepath = f"{CACHE_DIR}/hash_rate.csv"
    
    try:
        url = "https://mempool.space/api/v1/mining/hashrate/3m"
        response = requests.get(url, timeout=15)
        data = response.json()
        
        hashrates = data.get('hashrates', [])
        
        # convert to dataframe
        records = []
        for item in hashrates:
            timestamp = pd.to_datetime(item['timestamp'], unit='s')
            value = item['avgHashrate'] / 1e12  # convert to TH/s
            records.append({'timestamp': timestamp, 'value': value})
        
        df = pd.DataFrame(records)
        df = df.set_index('timestamp')
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]
        
        # resample to hourly
        df = df.resample('h').ffill()
        
        # merge with existing data
        if os.path.exists(filepath):
            existing = pd.read_csv(filepath, index_col=0, parse_dates=True)
            existing = existing.sort_index()
            
            # keep old data, add new
            cutoff = df.index[0]
            old_data = existing[existing.index < cutoff]
            df = pd.concat([old_data, df])
            df = df[~df.index.duplicated(keep='last')]
        
        df.to_csv(filepath)
        print(f"OK - {len(df):,} rows")
        
    except Exception as e:
        print(f"FAILED - {e}")


def download_fear_greed():
    """Download Fear & Greed index"""
    print("Downloading Fear & Greed...", end=' ')
    
    filepath = f"{CACHE_DAILY}/fear_greed.csv"
    
    try:
        url = "https://api.alternative.me/fng/?limit=0"
        response = requests.get(url, timeout=10)
        data = response.json()['data']
        
        # convert to dataframe
        records = []
        for item in data:
            timestamp = pd.to_datetime(int(item['timestamp']), unit='s')
            value = int(item['value'])
            records.append({'timestamp': timestamp, 'value': value})
        
        df = pd.DataFrame(records)
        df = df.set_index('timestamp')
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]
        
        df.to_csv(filepath)
        print(f"OK - {len(df):,} rows")
        
    except Exception as e:
        print(f"FAILED - {e}")


def download_ohlcv(symbol, filepath, is_futures=True):
    """Download OHLCV data from Binance"""
    print(f"Downloading {symbol}...", end=' ')
    
    # create exchange
    if is_futures:
        exchange = ccxt.binanceusdm({'enableRateLimit': True})
    else:
        exchange = ccxt.binance({'enableRateLimit': True})
    
    try:
        # check if we have existing data
        if os.path.exists(filepath):
            existing = pd.read_csv(filepath, index_col=0, parse_dates=True)
            since_ms = int(existing.index[-1].timestamp() * 1000)
            all_data = existing
        else:
            since_ms = int(datetime.strptime(START_DATE, '%Y-%m-%d').timestamp() * 1000)
            all_data = pd.DataFrame()
        
        # download in batches
        while True:
            # fetch 1000 candles
            ohlcv = exchange.fetch_ohlcv(symbol, '1h', since_ms, 1000)
            
            if len(ohlcv) == 0:
                break
            
            # convert to dataframe
            df = pd.DataFrame(ohlcv)
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # append
            all_data = pd.concat([all_data, df])
            
            # next batch
            since_ms = ohlcv[-1][0] + (1000 * 60 * 60)  # +1 hour
            
            # check if we're at current time
            if since_ms > datetime.now().timestamp() * 1000:
                break
            
            # rate limit
            time.sleep(exchange.rateLimit / 1000)
        
        # clean up
        all_data = all_data[~all_data.index.duplicated(keep='last')]
        all_data = all_data.sort_index()
        
        all_data.to_csv(filepath)
        print(f"OK - {len(all_data):,} rows")
        
    except Exception as e:
        print(f"FAILED - {e}")


def download_funding():
    """Download funding rates"""
    print("Downloading funding rates...", end=' ')
    
    filepath = f"{CACHE_DIR}/funding_1h.csv"
    filepath_raw = f"{CACHE_DIR}/funding_raw.csv"
    
    exchange = ccxt.binanceusdm({'enableRateLimit': True})
    
    try:
        # check for existing data
        if os.path.exists(filepath_raw):
            existing = pd.read_csv(filepath_raw, index_col=0, parse_dates=True)
            since_ms = int(existing.index[-1].timestamp() * 1000) + 1
            all_data = existing
        else:
            since_ms = int(datetime.strptime(START_DATE, '%Y-%m-%d').timestamp() * 1000)
            all_data = pd.DataFrame()
        
        # download
        while True:
            history = exchange.fetch_funding_rate_history('BTCUSDT', since_ms, limit=1000)
            
            if len(history) == 0:
                break
            
            df = pd.DataFrame(history)[['timestamp', 'fundingRate']]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            all_data = pd.concat([all_data, df])
            
            since_ms = int(df.index[-1].timestamp() * 1000) + 1
            
            if since_ms > datetime.now().timestamp() * 1000:
                break
            
            time.sleep(exchange.rateLimit / 1000)
        
        # clean up
        all_data = all_data[~all_data.index.duplicated(keep='last')]
        all_data = all_data.sort_index()
        
        # save raw
        all_data.to_csv(filepath_raw)
        
        # resample to hourly
        hourly = all_data.resample('h').ffill()
        hourly.to_csv(filepath)
        
        print(f"OK - {len(hourly):,} rows")
        
    except Exception as e:
        print(f"FAILED - {e}")


def main():
    print("=" * 50)
    print("DATA SCRAPER")
    print("=" * 50)
    print()
    
    download_hashrate()
    download_fear_greed()
    download_ohlcv('BTCUSDT', f"{CACHE_DIR}/btc_1h.csv", is_futures=True)
    download_ohlcv('ETHUSDT', f"{CACHE_DIR}/eth_1h.csv", is_futures=True)
    download_ohlcv('PAXG/USDT', f"{CACHE_DIR}/gold_1h.csv", is_futures=False)
    download_funding()
    
    print()
    print("=" * 50)
    print("DONE!")
    print("=" * 50)


if __name__ == "__main__":
    main()