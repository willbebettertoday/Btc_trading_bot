"""
Bitcoin Data Scraper
====================
Downloads and caches cryptocurrency market data for ML trading bot.

Data Sources:
- Hash rate: mempool.space API (real-time, <1h lag)
- OHLCV data: Binance API
- Fear & Greed Index: alternative.me API  
- Funding rates: Binance Futures API

Usage:
    python scraper.py

Recommended: Run hourly via cron
    5 * * * * cd /path/to/project && python3 scraper.py >> scraper.log 2>&1

Author: Your Name
License: MIT
"""

import os
import time
from datetime import datetime

import ccxt
import pandas as pd
import requests


# ============================================================================
# CONFIGURATION
# ============================================================================

# Data storage paths
CACHE_DIR = "/root/btc_bot/binance_hourly_cache"
CACHE_DAILY = "/root/btc_bot/binance_daily_cache"

# Historical data start date
START_DATE = '2019-09-01'

# API settings
API_TIMEOUT = 15  # seconds
RETRY_DELAY = 1.0  # seconds between retries

# Create directories if they don't exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CACHE_DAILY, exist_ok=True)


# ============================================================================
# HASH RATE DATA (Primary Feature - 7% Model Importance)
# ============================================================================

def fetch_hashrate_mempool():
    """
    Fetch Bitcoin network hash rate from mempool.space API.
    
    This is the PRIMARY source as it provides real-time data (<1h lag).
    Hash rate is a top-3 feature in our model (7% importance).
    
    Returns:
        bool: True if successful, False if failed
        
    Data format:
        - Converted from hash/s to TH/s (terahashes per second)
        - Resampled to hourly frequency
        - Forward-filled for consistency
    """
    print("🔄 Hash Rate (mempool.space)...", end=' ')
    file_path = f"{CACHE_DIR}/hash_rate.csv"
    
    try:
        # Fetch 3 months of hash rate history
        url = "https://mempool.space/api/v1/mining/hashrate/3m"
        response = requests.get(url, timeout=API_TIMEOUT)
        
        if response.status_code != 200:
            print(f"❌ HTTP {response.status_code}")
            return False
        
        data = response.json()
        hashrates = data.get('hashrates', [])
        
        if not hashrates:
            print("❌ No data returned")
            return False
        
        # Parse and convert to DataFrame
        records = []
        for item in hashrates:
            timestamp = pd.to_datetime(item['timestamp'], unit='s')
            # Convert hash/s to TH/s for compatibility with trained model
            value_ths = item['avgHashrate'] / 1e12
            records.append({'timestamp': timestamp, 'value': value_ths})
        
        df = pd.DataFrame(records)
        df = df.set_index('timestamp').sort_index()
        df = df[~df.index.duplicated(keep='last')]
        
        # Resample to hourly frequency
        df_hourly = df.resample('h').ffill().sort_index()
        
        # Merge with existing historical data
        if os.path.exists(file_path):
            try:
                existing = pd.read_csv(file_path, index_col=0, parse_dates=True)
                existing = existing.sort_index()
                
                # Keep old historical data, add new fresh data
                cutoff_date = df_hourly.index[0]
                historical = existing[existing.index < cutoff_date]
                
                combined = pd.concat([historical, df_hourly])
                combined = combined[~combined.index.duplicated(keep='last')]
                df_hourly = combined.sort_index()
            except Exception as e:
                print(f"⚠️ Merge warning: {e}")
        
        # Save to CSV
        df_hourly.to_csv(file_path)
        
        # Calculate data freshness
        age_hours = (pd.Timestamp.now() - df_hourly.index[-1]).total_seconds() / 3600
        latest_ehs = df_hourly['value'].iloc[-1] / 1e6  # Display in EH/s
        
        print(f"✅ {len(df_hourly):,} rows (age: {age_hours:.1f}h, {latest_ehs:.2f} EH/s)")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def fetch_hashrate_fallback(file_path):
    """
    Fallback hash rate source from blockchain.info.
    
    Note: May have 10-15 day lag. Use only when mempool.space fails.
    """
    print("🔄 Hash Rate (blockchain.info fallback)...", end=' ')
    
    try:
        url = "https://api.blockchain.info/charts/hash-rate?timespan=all&format=json"
        response = requests.get(url, timeout=API_TIMEOUT)
        
        data = response.json()['values']
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['x'], unit='s')
        df = df.rename(columns={'y': 'value'})
        df = df.set_index('timestamp')[['value']].sort_index()
        
        # Resample to hourly
        df_hourly = df.resample('h').ffill().sort_index()
        df_hourly.to_csv(file_path)
        
        age_hours = (pd.Timestamp.now() - df_hourly.index[-1]).total_seconds() / 3600
        print(f"✅ {len(df_hourly):,} rows (age: {age_hours:.1f}h)")
        
    except Exception as e:
        print(f"❌ Error: {e}")


# ============================================================================
# FEAR & GREED INDEX (Sentiment Indicator)
# ============================================================================

def fetch_fear_greed():
    """
    Fetch Bitcoin Fear & Greed Index from alternative.me.
    
    Values range from 0 (Extreme Fear) to 100 (Extreme Greed).
    Updates daily.
    
    Used as a contrarian indicator in the model.
    """
    print("🔄 Fear & Greed Index...", end=' ')
    file_path = f"{CACHE_DAILY}/fear_greed.csv"
    
    try:
        # Fetch all available historical data
        url = "https://api.alternative.me/fng/?limit=0"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ HTTP {response.status_code}")
            return
        
        data = response.json()['data']
        
        # Parse records
        records = []
        for item in data:
            try:
                timestamp = pd.to_datetime(int(item['timestamp']), unit='s')
                value = int(item['value'])
                records.append({'timestamp': timestamp, 'value': value})
            except:
                # Skip malformed records
                continue
        
        if not records:
            print("❌ No valid data")
            return
        
        # Create and save DataFrame
        df = pd.DataFrame(records)
        df = df.set_index('timestamp').sort_index()
        df = df[~df.index.duplicated(keep='last')]
        df.to_csv(file_path)
        
        date_range = f"{df.index[0].date()} to {df.index[-1].date()}"
        print(f"✅ {len(df):,} rows ({date_range})")
        
    except Exception as e:
        print(f"❌ Error: {e}")


# ============================================================================
# BINANCE MARKET DATA (OHLCV)
# ============================================================================

def fetch_binance_ohlcv(symbol, timeframe, file_path, is_futures=True):
    """
    Fetch OHLCV (Open, High, Low, Close, Volume) candle data from Binance.
    
    Args:
        symbol (str): Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
        timeframe (str): Candle interval (e.g., '1h', '4h', '1d')
        file_path (str): Where to save CSV file
        is_futures (bool): True for futures market, False for spot
        
    Updates incrementally - only fetches new candles since last update.
    """
    print(f"🔄 {symbol}...", end=' ')
    
    # Initialize exchange
    if is_futures:
        exchange = ccxt.binanceusdm({'enableRateLimit': True})
    else:
        exchange = ccxt.binance({'enableRateLimit': True})
    
    batch_size = 1000  # Max candles per request
    all_data = pd.DataFrame()

    try:
        # Load existing data if available
        if os.path.exists(file_path):
            existing = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            if existing.empty:
                # Start from configured date
                since_ms = int(datetime.strptime(START_DATE, '%Y-%m-%d').timestamp() * 1000)
            else:
                # Resume from last timestamp
                since_ms = int(existing.index[-1].timestamp() * 1000)
                all_data = existing
        else:
            # No existing data, start from configured date
            since_ms = int(datetime.strptime(START_DATE, '%Y-%m-%d').timestamp() * 1000)

        # Fetch data in batches
        while True:
            # Fetch one batch
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since_ms, batch_size)
            
            if not ohlcv:
                break
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Append to existing data
            all_data = pd.concat([all_data, df])
            
            # Calculate next batch start time
            since_ms = ohlcv[-1][0] + (1000 * 60 * 60)  # +1 hour in milliseconds
            
            # Stop if we've caught up to present
            if since_ms > datetime.now().timestamp() * 1000:
                break
            
            # Rate limiting (respect exchange limits)
            time.sleep(exchange.rateLimit / 1000)

        # Clean data and save
        all_data = all_data[~all_data.index.duplicated(keep='last')].sort_index()
        all_data.to_csv(file_path)
        
        print(f"✅ {len(all_data):,} rows")
    
    except Exception as e:
        print(f"❌ Error: {e}")


# ============================================================================
# FUNDING RATES (Futures Market Sentiment)
# ============================================================================

def fetch_funding_rates():
    """
    Fetch historical funding rates for BTC perpetual futures.
    
    Funding rates indicate long/short bias in the futures market.
    - Positive rate: Longs pay shorts (bullish sentiment)
    - Negative rate: Shorts pay longs (bearish sentiment)
    
    Updates every 8 hours on Binance.
    """
    print("🔄 Funding rates...", end=' ')
    
    symbol = 'BTCUSDT'
    file_path = f"{CACHE_DIR}/funding_1h.csv"
    file_path_raw = file_path.replace('_1h.csv', '_raw.csv')
    
    exchange = ccxt.binanceusdm({'enableRateLimit': True})
    all_data = pd.DataFrame()

    try:
        # Load existing data if available
        if os.path.exists(file_path_raw):
            existing = pd.read_csv(file_path_raw, index_col=0, parse_dates=True)
            since_ms = int(existing.index[-1].timestamp() * 1000) + 1
            all_data = existing
        else:
            since_ms = int(datetime.strptime(START_DATE, '%Y-%m-%d').timestamp() * 1000)

        # Fetch in batches
        while True:
            history = exchange.fetch_funding_rate_history(symbol, since_ms, limit=1000)
            
            if not history:
                break
            
            df = pd.DataFrame(history)[['timestamp', 'fundingRate']]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            all_data = pd.concat([all_data, df])
            
            since_ms = int(df.index[-1].timestamp() * 1000) + 1
            
            if since_ms > datetime.now().timestamp() * 1000:
                break
            
            time.sleep(exchange.rateLimit / 1000)
        
        # Save raw data
        all_data = all_data[~all_data.index.duplicated(keep='last')].sort_index()
        all_data.to_csv(file_path_raw)
        
        # Resample to hourly (forward-fill for consistency)
        hourly_data = all_data.resample('h').ffill().sort_index()
        hourly_data.to_csv(file_path)
        
        print(f"✅ {len(hourly_data):,} rows")

    except Exception as e:
        print(f"❌ Error: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main data collection routine.
    
    Fetches all required data sources and saves to CSV files.
    Designed to be idempotent - safe to run multiple times.
    """
    print("=" * 70)
    print("📥 BTC Data Scraper v2.2")
    print("=" * 70)
    start_time = time.time()
    
    # Fetch hash rate (try mempool.space first, fallback to blockchain.info)
    success = fetch_hashrate_mempool()
    if not success:
        print("   ⚠️ Mempool.space unavailable, using blockchain.info fallback...")
        fetch_hashrate_fallback(f"{CACHE_DIR}/hash_rate.csv")
    
    # Fetch market sentiment indicator
    fetch_fear_greed()
    
    # Fetch OHLCV data for multiple assets
    fetch_binance_ohlcv(
        symbol='BTCUSDT',
        timeframe='1h',
        file_path=f"{CACHE_DIR}/btc_1h.csv",
        is_futures=True
    )
    
    fetch_binance_ohlcv(
        symbol='ETHUSDT',
        timeframe='1h',
        file_path=f"{CACHE_DIR}/eth_1h.csv",
        is_futures=True
    )
    
    fetch_binance_ohlcv(
        symbol='PAXG/USDT',  # Gold (spot market)
        timeframe='1h',
        file_path=f"{CACHE_DIR}/gold_1h.csv",
        is_futures=False
    )
    
    # Fetch funding rates
    fetch_funding_rates()
    
    # Summary
    elapsed_time = time.time() - start_time
    print("=" * 70)
    print(f"✅ Data collection complete in {elapsed_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()