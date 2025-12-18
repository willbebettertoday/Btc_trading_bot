"""
Bitcoin Transformer Trading Bot
================================
Automated 24/7 cryptocurrency trading bot using Transformer neural networks.

Features:
- Real-time market data integration
- Dynamic position sizing based on model confidence
- Automated TP/SL management
- Telegram notifications
- SQLite trade logging
- Multi-timeframe analysis

Usage:
    python bot.py

For VPS deployment:
    screen -S btc_bot
    python bot.py
    # Ctrl+A, D to detach

Author: Your Name
License: MIT
"""

import json
import math
import os
import sqlite3
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

# ML imports
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler

# Exchange API
import ccxt

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

try:
    # Try to import from config.py (local, not in git)
    from config import *
except ImportError:
    # Fallback to example configuration
    print("⚠️ config.py not found, using example configuration")
    print("   Copy config_example.py to config.py and edit with your settings")
    
    # Telegram settings (REQUIRED)
    TELEGRAM_BOT_TOKEN = "YOUR_TOKEN_FROM_BOTFATHER"
    TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"
    
    # Paths
    RESULTS_DIR = "/root/btc_bot/btc_PRODUCTION_NO_TX"
    CACHE_DIR = "/root/btc_bot/binance_hourly_cache"
    CACHE_DAILY = "/root/btc_bot/binance_daily_cache"
    DATABASE_FILE = "/root/btc_bot/trades.db"
    
    # ⚠️ DEMO PARAMETERS (not optimal!)
    TOP_PERC = 0.85
    BOTTOM_PERC = 0.15
    MIN_CONFIDENCE = 0.50
    RISK_PER_TRADE = 0.005
    MIN_HOURS_BETWEEN_TRADES = 6

# Display configuration
print("=" * 70)
print("🤖 BTC TRANSFORMER BOT v3.2")
print("=" * 70)

# Load model configuration
CONFIG_PATH = f"{RESULTS_DIR}/config.json"
MODEL_PATH = f"{RESULTS_DIR}/best_model.pth"
SCALER_PATH = f"{RESULTS_DIR}/scaler.json"

# Check required files exist
if not all(os.path.exists(p) for p in [CONFIG_PATH, MODEL_PATH, SCALER_PATH]):
    print(f"❌ Model files not found in '{RESULTS_DIR}'")
    print("   Run train.py first to generate model files")
    exit(1)

# Load model hyperparameters
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

LOOKBACK_1H = config.get('lookback', 168)
D_MODEL = config.get('d_model', 128)
N_HEADS = config.get('n_heads', 8)
N_LAYERS = config.get('n_layers', 3)
D_FF = config.get('d_ff', 512)
DROPOUT = config.get('dropout', 0.15)

print(f"✅ Model config loaded: {config.get('n_features', 23)} features")
print("=" * 70)

# Device selection (CPU for production, GPU for training)
device = torch.device('cpu')


# ============================================================================
# TRANSFORMER MODEL ARCHITECTURE
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the sequence.
    Uses sinusoidal functions for position encoding.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create position encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """
    Transformer-based time series prediction model.
    
    Architecture:
    - Input projection to d_model dimensions
    - Positional encoding
    - Transformer encoder (multi-head self-attention)
    - Output projection to percentile (0-1)
    """
    def __init__(self, input_dim):
        super().__init__()
        
        # Input projection
        self.proj = nn.Linear(input_dim, D_MODEL)
        self.norm1 = nn.LayerNorm(D_MODEL)
        
        # Positional encoding
        self.pos = PositionalEncoding(D_MODEL)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_FF,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, N_LAYERS)
        
        # Output layers
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(D_MODEL, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.normal_(self.fc.weight, 0, 0.02)
        nn.init.constant_(self.fc.bias, 0.5)
    
    def forward(self, x):
        # Project input to model dimension
        x = self.norm1(self.proj(x))
        
        # Add positional encoding
        x = self.pos(x)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Take last timestep and project to output
        x = self.dropout(self.norm2(x[:, -1, :]))
        
        # Output percentile (clamped to 0-1)
        return torch.clamp(self.fc(x), 0, 1)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(df_btc, df_eth=None, df_gold=None, df_hash_rate=None):
    """
    Engineer trading features from raw OHLCV data.
    
    Args:
        df_btc: Bitcoin OHLCV DataFrame
        df_eth: Ethereum OHLCV DataFrame (optional)
        df_gold: Gold OHLCV DataFrame (optional)
        df_hash_rate: Hash rate DataFrame (optional)
        
    Returns:
        DataFrame with 23 engineered features
        
    Features include:
    - Momentum indicators (various timeframes)
    - Volatility metrics
    - Volume analysis
    - Technical indicators (RSI, MACD)
    - Cross-asset correlations
    - On-chain metrics (hash rate)
    """
    df = df_btc.copy()
    features = pd.DataFrame(index=df.index)
    
    price = df['close']
    returns = price.pct_change()
    volume = df['volume']
    
    # --- MOMENTUM INDICATORS ---
    for horizon in [1, 4, 24, 168]:
        features[f'momentum_{horizon}h'] = price.pct_change(horizon)
    
    # --- VOLATILITY METRICS ---
    features['vol_24h'] = returns.rolling(24, min_periods=12).std()
    features['vol_7d'] = returns.rolling(168, min_periods=84).std()
    features['vol_30d'] = returns.rolling(720, min_periods=360).std()
    
    # Volume analysis
    volume_ma = volume.rolling(168, min_periods=84).mean()
    features['volume_ratio'] = volume / (volume_ma + 1e-8)
    
    # Intra-candle range
    features['range_1h'] = (df['high'] - df['low']) / (price + 1e-8)
    
    # --- MULTI-TIMEFRAME AGGREGATIONS ---
    # 4-hour aggregated data
    df_4h = df.resample('4H').agg({'close': 'last'}).ffill()
    price_4h = df_4h['close'].reindex(df.index, method='ffill')
    features['momentum_4h_agg'] = price_4h.pct_change(6)
    features['vol_4h'] = price_4h.pct_change().rolling(42, min_periods=21).std()
    
    # Daily aggregated data
    df_daily = df.resample('1D').agg({'close': 'last'}).ffill()
    price_daily = df_daily['close'].reindex(df.index, method='ffill')
    features['momentum_daily_7d'] = price_daily.pct_change(7)
    features['momentum_daily_30d'] = price_daily.pct_change(30)
    
    # --- TECHNICAL INDICATORS ---
    # RSI (Relative Strength Index)
    delta = price.diff()
    gain = delta.where(delta > 0, 0).rolling(336, min_periods=168).mean()
    loss = -delta.where(delta < 0, 0).rolling(336, min_periods=168).mean()
    features['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    
    # MACD
    ema_12 = price.ewm(span=288, adjust=False).mean()
    ema_26 = price.ewm(span=624, adjust=False).mean()
    features['macd'] = (ema_12 - ema_26) / (price + 1e-8)
    
    # --- CROSS-ASSET CORRELATIONS ---
    if df_eth is not None and len(df_eth) > 0:
        eth_price = df_eth['close'].reindex(df.index, method='ffill')
        features['eth_momentum_24h'] = eth_price.pct_change(24)
        features['btc_eth_correlation'] = returns.rolling(168).corr(
            eth_price.pct_change()
        )
    
    if df_gold is not None and len(df_gold) > 0:
        gold_price = df_gold['close'].reindex(df.index, method='ffill')
        features['gold_momentum_24h'] = gold_price.pct_change(24)
        features['btc_gold_correlation'] = returns.rolling(168).corr(
            gold_price.pct_change()
        )
    
    # --- ON-CHAIN METRICS ---
    if df_hash_rate is not None and len(df_hash_rate) > 0:
        # Shift by 48h to avoid look-ahead bias
        hash_rate = df_hash_rate.reindex(df.index, method='ffill').shift(48)
        
        if len(hash_rate.columns) > 0:
            hr_series = hash_rate.iloc[:, 0]
            
            # Hash rate momentum (7-day change)
            features['hashrate_momentum_7d'] = hr_series.pct_change(168)
            
            # Hash rate ratio to 30-day MA
            hr_ma = hr_series.rolling(720, min_periods=360).mean()
            features['hashrate_ratio'] = hr_series / (hr_ma + 1e-8)
    
    # --- CLEANUP ---
    # Forward-fill NaN values, then fill remaining with 0
    features = features.fillna(method='ffill').fillna(0)
    
    # Remove infinite values
    features = features.replace([np.inf, -np.inf], 0)
    
    # Clip outliers (0.5th and 99.5th percentile)
    for col in features.columns:
        if col in features:
            lower = features[col].quantile(0.005)
            upper = features[col].quantile(0.995)
            features[col] = features[col].clip(lower, upper)
    
    return features


def calculate_trading_levels(percentile, historical_returns, window=720):
    """
    Calculate dynamic TP/SL levels based on percentile prediction.
    
    Args:
        percentile: Model prediction (0-1)
        historical_returns: Array of historical returns
        window: Lookback window for statistics
        
    Returns:
        dict with direction, TP, SL, confidence, etc.
        None if no signal
        
    Logic:
    - Top 4% percentile → LONG signal
    - Bottom 4% percentile → SHORT signal
    - TP/SL dynamically sized based on confidence and expected return
    """
    # Check if prediction is in actionable range
    if not (percentile <= BOTTOM_PERC or percentile >= TOP_PERC):
        return None
    
    # Get recent historical returns for statistics
    recent_history = historical_returns[-window:] if len(historical_returns) > window else historical_returns
    
    if len(recent_history) == 0:
        return None
    
    # Calculate expected return at this percentile
    expected_return = np.percentile(recent_history, percentile * 100)
    
    # Determine direction and confidence
    if percentile > TOP_PERC:
        # LONG signal
        direction = "LONG"
        confidence = (percentile - TOP_PERC) / (1.0 - TOP_PERC)
        
        # Dynamic TP/SL based on confidence
        take_profit = expected_return * (1.5 + confidence * 1.5)
        stop_loss = -abs(expected_return) * (0.6 + confidence * 0.6)
    else:
        # SHORT signal
        direction = "SHORT"
        confidence = (BOTTOM_PERC - percentile) / BOTTOM_PERC
        
        # Dynamic TP/SL based on confidence
        take_profit = -expected_return * (1.5 + confidence * 1.5)
        stop_loss = abs(expected_return) * (0.6 + confidence * 0.6)
    
    # Minimum stop-loss (0.5%)
    if abs(stop_loss) < 0.005:
        stop_loss = -0.005 if direction == "LONG" else 0.005
    
    # Calculate risk/reward ratio
    rr_ratio = abs(take_profit / stop_loss) if stop_loss != 0 else 0
    
    return {
        'direction': direction,
        'percentile': percentile,
        'confidence': confidence,
        'expected_return': expected_return,
        'take_profit': take_profit,
        'stop_loss': stop_loss,
        'rr_ratio': rr_ratio
    }


# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

def init_database():
    """Initialize SQLite database for trade logging."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            open_time TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            tp_price REAL NOT NULL,
            sl_price REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'open'
        )
    ''')
    
    conn.commit()
    conn.close()


def add_trade(open_time, direction, entry_price, tp_price, sl_price):
    """Log new trade to database."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO trades (open_time, direction, entry_price, tp_price, sl_price, status) VALUES (?, ?, ?, ?, ?, 'open')",
            (open_time.isoformat(), direction, entry_price, tp_price, sl_price)
        )
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Database error: {e}")


def get_open_trades():
    """Retrieve all open trades from database."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM trades WHERE status = 'open'")
        trades = cursor.fetchall()
        
        conn.close()
        return trades
    except:
        return []


def close_trade(trade_id, reason):
    """Mark trade as closed in database."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE trades SET status = ? WHERE id = ?",
            (f"closed_{reason}", trade_id)
        )
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Database error: {e}")


# ============================================================================
# TELEGRAM NOTIFICATIONS
# ============================================================================

def send_telegram_message(message):
    """
    Send notification to Telegram.
    
    Args:
        message (str): Message text (supports Markdown)
        
    Returns:
        bool: True if sent successfully
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown',
        'disable_web_page_preview': True
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200 and response.json().get('ok'):
            print("   ✅ Telegram notification sent")
            return True
        else:
            print(f"   ❌ Telegram error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"   ❌ Telegram error: {e}")
        return False


# ============================================================================
# DATA LOADING
# ============================================================================

def fetch_live_ohlcv(symbol, timeframe, limit):
    """
    Fetch recent OHLCV data from Binance.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle interval (e.g., '1h')
        limit: Number of candles to fetch
        
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        return df
        
    except Exception as e:
        print(f"   ❌ {symbol} fetch error: {e}")
        return None


def get_current_price():
    """Get current BTC price from Binance."""
    try:
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker('BTC/USDT')
        return ticker['last']
    except:
        return None


def load_cached_data():
    """
    Load cached external data (hash rate, Fear & Greed, funding).
    
    Returns:
        dict with DataFrames for each data source
    """
    data = {}
    
    def safe_load_csv(path, name):
        """Safely load CSV with error handling."""
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df[df.index.notna()]  # Remove invalid timestamps
            df = df[~df.index.duplicated(keep='last')]  # Remove duplicates
            df = df.sort_index()  # Ensure chronological order
            return df
        except Exception as e:
            print(f"   ⚠️ {name} load warning: {e}")
            return None
    
    # Load external data sources
    data['funding'] = safe_load_csv(f"{CACHE_DIR}/funding_1h.csv", "Funding")
    data['fear_greed'] = safe_load_csv(f"{CACHE_DAILY}/fear_greed.csv", "Fear & Greed")
    data['hash_rate'] = safe_load_csv(f"{CACHE_DIR}/hash_rate.csv", "Hash Rate")
    
    return data


def load_model_and_scaler():
    """
    Load trained model and feature scaler from disk.
    
    Returns:
        tuple: (model, scaler, feature_names) or (None, None, None) if failed
    """
    print("🧠 Loading model...")
    
    # Load scaler
    try:
        with open(SCALER_PATH, 'r') as f:
            scaler_params = json.load(f)
        
        scaler = RobustScaler()
        scaler.center_ = np.array(scaler_params['center_'])
        scaler.scale_ = np.array(scaler_params['scale_'])
        scaler.n_features_in_ = scaler_params['n_features_in_']
        feature_names = scaler_params['feature_names']
        
        print(f"   ✅ Scaler loaded: {scaler.n_features_in_} features")
        
    except Exception as e:
        print(f"❌ Scaler load error: {e}")
        return None, None, None
    
    # Load model
    try:
        model = TransformerModel(scaler.n_features_in_).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        
        print(f"   ✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Model load error: {e}")
        return None, None, None
    
    return model, scaler, feature_names


# ============================================================================
# PREDICTION ENGINE
# ============================================================================

def generate_prediction(model, scaler, feature_names):
    """
    Generate trading prediction using live market data.
    
    Args:
        model: Trained Transformer model
        scaler: Feature scaler
        feature_names: List of feature column names
        
    Returns:
        dict with timestamp, price, percentile, and trading levels
        None if prediction failed
        
    Process:
    1. Fetch live market data
    2. Engineer features
    3. Normalize features
    4. Run model inference
    5. Calculate trading levels
    """
    print(f"\n[{datetime.now():%H:%M:%S}] 🔍 Generating prediction...")
    
    try:
        # Fetch live OHLCV data (need extra for feature calculation)
        btc_data = fetch_live_ohlcv('BTC/USDT', '1h', 1000 + LOOKBACK_1H)
        eth_data = fetch_live_ohlcv('ETH/USDT', '1h', 1000 + LOOKBACK_1H)
        
        # Load gold from cache (spot market, slower updates)
        try:
            gold_data = pd.read_csv(f"{CACHE_DIR}/gold_1h.csv", index_col=0, parse_dates=True)
        except:
            gold_data = None
        
        # Load external data
        cached_data = load_cached_data()
        
        # Validate BTC data
        if btc_data is None or len(btc_data) < LOOKBACK_1H + 200:
            print("   ❌ Insufficient BTC data")
            return None
        
        # --- FEATURE ENGINEERING ---
        features = create_features(
            df_btc=btc_data,
            df_eth=eth_data,
            df_gold=gold_data,
            df_hash_rate=cached_data['hash_rate']
        )
        
        # Add funding rate feature
        if cached_data['funding'] is not None:
            try:
                funding = cached_data['funding'].reindex(features.index, method='ffill').shift(48)
                if 'fundingRate' in funding.columns:
                    features['funding'] = funding['fundingRate'].fillna(0)
            except Exception as e:
                print(f"   ⚠️ Funding feature skipped: {e}")
        
        # Add Fear & Greed feature
        if cached_data['fear_greed'] is not None:
            try:
                fg = cached_data['fear_greed'].reindex(features.index, method='ffill').shift(48)
                if 'value' in fg.columns:
                    # Normalize to 0-1 range
                    features['fear_greed'] = (fg['value'] / 100).fillna(0.5)
            except Exception as e:
                print(f"   ⚠️ Fear & Greed feature skipped: {e}")
        
        # --- FEATURE ALIGNMENT ---
        # Ensure all expected features are present
        aligned_features = pd.DataFrame(columns=feature_names, index=features.index)
        
        for col in feature_names:
            if col in features.columns:
                aligned_features[col] = features[col]
            else:
                aligned_features[col] = 0  # Missing features = 0
        
        # Fill any remaining NaN
        aligned_features = aligned_features.fillna(method='ffill').fillna(0)
        
        # --- NORMALIZATION ---
        # Use scaler fitted during training
        normalized_features = scaler.transform(aligned_features)
        
        # Take last LOOKBACK_1H hours for model input
        X_live = normalized_features[-LOOKBACK_1H:]
        
        # Reshape to (batch=1, sequence=LOOKBACK_1H, features=23)
        X_tensor = torch.FloatTensor(X_live).reshape(1, LOOKBACK_1H, len(feature_names))
        X_tensor = X_tensor.to(device)
        
        # --- MODEL INFERENCE ---
        with torch.no_grad():
            prediction = model(X_tensor)
            percentile_pred = prediction.cpu().numpy()[0][0]
        
        print(f"   📈 Percentile prediction: {percentile_pred:.4f}")
        
        # --- CALCULATE TRADING LEVELS ---
        historical_returns = btc_data['close'].pct_change().dropna().values
        current_price = btc_data['close'].iloc[-1]
        
        trading_levels = calculate_trading_levels(
            percentile=percentile_pred,
            historical_returns=historical_returns
        )
        
        if trading_levels:
            print(f"   🎯 Signal: {trading_levels['direction']} (confidence: {trading_levels['confidence']:.2f})")
        else:
            print(f"   🧘 No signal (percentile {percentile_pred:.4f} not in actionable range)")
        
        return {
            'timestamp': btc_data.index[-1],
            'price': current_price,
            'percentile': percentile_pred,
            'levels': trading_levels
        }
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def check_for_signal(model, scaler, feature_names, last_trade_time):
    """
    Check for new trading signals.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        feature_names: List of feature names
        last_trade_time: Timestamp of last trade
        
    Returns:
        datetime: Updated last_trade_time
        
    Logic:
    1. Check trade spacing (minimum hours between trades)
    2. Generate prediction
    3. Check confidence threshold
    4. Send Telegram notification if signal
    5. Log trade to database
    """
    now = datetime.utcnow()
    
    # --- CHECK TRADE SPACING ---
    hours_since_last_trade = (now - last_trade_time).total_seconds() / 3600
    
    if hours_since_last_trade < MIN_HOURS_BETWEEN_TRADES:
        print(f"   [{now:%H:%M}] 🧘 Trade spacing: {hours_since_last_trade:.1f}h / {MIN_HOURS_BETWEEN_TRADES}h")
        return last_trade_time
    
    # --- GENERATE PREDICTION ---
    result = generate_prediction(model, scaler, feature_names)
    
    if result is None:
        return last_trade_time
    
    levels = result.get('levels')
    
    # No actionable signal
    if not levels:
        return last_trade_time
    
    # --- CHECK CONFIDENCE THRESHOLD ---
    confidence = levels['confidence']
    
    if confidence < MIN_CONFIDENCE:
        print(f"   🚫 Low confidence: {confidence:.2f} < {MIN_CONFIDENCE}")
        return last_trade_time
    
    # --- CALCULATE TRADE PARAMETERS ---
    current_price = result['price']
    direction = levels['direction']
    
    if direction == 'LONG':
        tp_price = current_price * (1 + levels['take_profit'])
        sl_price = current_price * (1 + levels['stop_loss'])
        tp_pct = levels['take_profit']
        sl_pct = levels['stop_loss']
    else:  # SHORT
        tp_price = current_price * (1 - levels['take_profit'])
        sl_price = current_price * (1 + levels['stop_loss'])
        tp_pct = -levels['take_profit']
        sl_pct = levels['stop_loss']
    
    # --- SEND TELEGRAM NOTIFICATION ---
    signal_emoji = "🟢 LONG" if direction == 'LONG' else "🔴 SHORT"
    
    message = (
        f"{signal_emoji} *NEW SIGNAL BTC/USDT*\n\n"
        f"💵 *Entry:* `${current_price:,.2f}`\n"
        f"🎯 *Take Profit:* `${tp_price:,.2f}` ({tp_pct*100:+.2f}%)\n"
        f"🛑 *Stop Loss:* `${sl_price:,.2f}` ({sl_pct*100:+.2f}%)\n\n"
        f"📊 *Risk/Reward:* `{levels['rr_ratio']:.2f}:1`\n"
        f"🔥 *Confidence:* `{confidence:.2f}`\n"
        f"📈 *Percentile:* `{result['percentile']:.4f}`"
    )
    
    if send_telegram_message(message):
        # Log trade to database
        add_trade(now, direction, current_price, tp_price, sl_price)
        return now  # Update last_trade_time
    
    return last_trade_time


# ============================================================================
# TRADE MONITORING
# ============================================================================

def monitor_open_trades():
    """
    Monitor open trades for TP/SL hits or time exits.
    
    Checks every 5 minutes:
    - Take profit hit
    - Stop loss hit
    - 24-hour time exit
    
    Sends Telegram notification on trade close.
    """
    open_trades = get_open_trades()
    
    if not open_trades:
        return
    
    print(f"   [{datetime.utcnow():%H:%M}] 🔍 Monitoring {len(open_trades)} open trades...")
    
    # Get current market price
    current_price = get_current_price()
    
    if current_price is None:
        print("   ❌ Unable to fetch current price")
        return
    
    now = datetime.utcnow()
    
    # Check each open trade
    for trade in open_trades:
        trade_id = trade['id']
        direction = trade['direction']
        entry_price = trade['entry_price']
        tp_price = trade['tp_price']
        sl_price = trade['sl_price']
        
        reason = None
        profit_pct = 0
        
        # --- CHECK LONG TRADE ---
        if direction == 'LONG':
            if current_price >= tp_price:
                reason = "TAKE_PROFIT"
                emoji = "✅"
                profit_pct = (current_price - entry_price) / entry_price
                
            elif current_price <= sl_price:
                reason = "STOP_LOSS"
                emoji = "🛑"
                profit_pct = (current_price - entry_price) / entry_price
        
        # --- CHECK SHORT TRADE ---
        elif direction == 'SHORT':
            if current_price <= tp_price:
                reason = "TAKE_PROFIT"
                emoji = "✅"
                profit_pct = (entry_price - current_price) / entry_price
                
            elif current_price >= sl_price:
                reason = "STOP_LOSS"
                emoji = "🛑"
                profit_pct = (entry_price - current_price) / entry_price
        
        # --- CHECK TIME EXIT (24 hours) ---
        if reason is None:
            open_time = datetime.fromisoformat(trade['open_time'])
            hours_open = (now - open_time).total_seconds() / 3600
            
            if hours_open >= 24:
                reason = "TIME_EXIT"
                emoji = "⏰"
                
                if direction == 'LONG':
                    profit_pct = (current_price - entry_price) / entry_price
                else:
                    profit_pct = (entry_price - current_price) / entry_price
        
        # --- CLOSE TRADE IF NEEDED ---
        if reason:
            message = (
                f"{emoji} *TRADE CLOSED*\n\n"
                f"Direction: {direction} (ID: {trade_id})\n"
                f"Reason: *{reason}*\n"
                f"Entry: `${entry_price:,.2f}`\n"
                f"Close: `${current_price:,.2f}`\n"
                f"Result: *{profit_pct*100:+.2f}%*"
            )
            
            if send_telegram_message(message):
                close_trade(trade_id, reason)


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    """
    Main bot execution loop.
    
    Runs 24/7:
    - Generates signals every hour (at :01 minute)
    - Monitors open trades every 5 minutes
    - Sends Telegram notifications
    - Logs all trades to database
    
    Error handling:
    - Recovers from API errors
    - Continues running on exceptions
    - Notifies via Telegram on fatal errors
    """
    
    # --- STARTUP CHECKS ---
    print("\n🔍 Verifying Telegram connection...")
    try:
        response = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe",
            timeout=10
        )
        
        if response.status_code == 200:
            bot_info = response.json()['result']
            print(f"   ✅ Telegram bot: @{bot_info['username']}")
        else:
            print("   ❌ Invalid Telegram token")
            return
            
    except Exception as e:
        print(f"   ❌ Telegram connection error: {e}")
        return
    
    # Initialize database
    init_database()
    print("🗃️  Database initialized")
    
    # Load model
    model, scaler, feature_names = load_model_and_scaler()
    
    if model is None:
        print("❌ Failed to load model, exiting")
        return
    
    print("=" * 70)
    
    # Send startup notification
    startup_message = (
        "🤖 *BTC Trading Bot Started*\n\n"
        "✅ Model loaded (23 features)\n"
        "🎯 Monitoring 24/7\n"
        "📊 Signals generated hourly\n"
        "🔍 Trade monitoring every 5 min"
    )
    send_telegram_message(startup_message)
    
    print("\n🔄 Starting 24/7 monitoring loop...\n")
    
    # Track last trade time (for spacing)
    last_trade_time = datetime.utcnow() - timedelta(hours=MIN_HOURS_BETWEEN_TRADES)
    
    # --- MAIN LOOP ---
    while True:
        try:
            now = datetime.utcnow()
            
            # Generate signals at :01 minute of every hour
            if now.minute == 1:
                print(f"[{now:%H:%M}] 🔔 Checking for signals...")
                last_trade_time = check_for_signal(model, scaler, feature_names, last_trade_time)
            
            # Monitor open trades every 5 minutes
            if now.minute % 5 == 0:
                monitor_open_trades()
            
            # Calculate sleep time until next minute
            next_run = (now + timedelta(minutes=1)).replace(second=1, microsecond=0)
            sleep_seconds = max(10, (next_run - datetime.utcnow()).total_seconds())
            
            # Status update
            if now.minute % 5 != 0:
                print(f"   [{now:%H:%M}] 💤 Waiting...", end="\r")
            
            time.sleep(sleep_seconds)
        
        except Exception as e:
            # Recover from errors
            print(f"\n❌ Error in main loop: {e}")
            send_telegram_message(f"❌ *Bot Error*\n`{e}`")
            time.sleep(60)  # Wait 1 minute before retry


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
        send_telegram_message("🤖 *Bot stopped*")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        send_telegram_message(f"❌ *Fatal Error*\n`{e}`")