"""
Live trading bot
Runs 24/7 and sends signals to Telegram
"""

import os
import sys
import time
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import torch

# add src folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import *
from src.model import load_model
from src.features import create_features
from src.trading import generate_signal, calculate_position_size, check_exit
from src.data import fetch_ohlcv, get_current_price, load_cached_data, load_scaler

# use CPU for inference (faster startup)
device = torch.device('cpu')


# ==================== DATABASE ====================

def init_database():
    """Create trades table if not exists"""
    conn = sqlite3.connect(DATABASE_FILE)
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            open_time TEXT,
            direction TEXT,
            entry_price REAL,
            tp_price REAL,
            sl_price REAL,
            status TEXT DEFAULT 'open'
        )
    ''')
    
    conn.commit()
    conn.close()


def add_trade(open_time, direction, entry_price, tp_price, sl_price):
    """Add new trade to database"""
    conn = sqlite3.connect(DATABASE_FILE)
    
    conn.execute(
        "INSERT INTO trades (open_time, direction, entry_price, tp_price, sl_price) VALUES (?, ?, ?, ?, ?)",
        (open_time.isoformat(), direction, entry_price, tp_price, sl_price)
    )
    
    conn.commit()
    conn.close()


def get_open_trades():
    """Get all open trades"""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.execute("SELECT * FROM trades WHERE status = 'open'")
    trades = cursor.fetchall()
    
    conn.close()
    return trades


def close_trade(trade_id, reason):
    """Close a trade"""
    conn = sqlite3.connect(DATABASE_FILE)
    
    conn.execute(
        "UPDATE trades SET status = ? WHERE id = ?",
        (f"closed_{reason}", trade_id)
    )
    
    conn.commit()
    conn.close()


# ==================== TELEGRAM ====================

def send_telegram(message):
    """Send message to Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    data = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    
    try:
        response = requests.post(url, json=data, timeout=10)
        return response.status_code == 200
    except:
        return False


# ==================== PREDICTION ====================

def make_prediction(model, scaler, feature_names):
    """Generate prediction from live data"""
    
    # fetch live data
    btc = fetch_ohlcv('BTC/USDT', '1h', 1000 + LOOKBACK)
    eth = fetch_ohlcv('ETH/USDT', '1h', 1000 + LOOKBACK)
    
    if btc is None:
        print("Could not fetch BTC data")
        return None
    
    if len(btc) < LOOKBACK + 200:
        print("Not enough BTC data")
        return None
    
    # load cached data
    cached = load_cached_data()
    
    # create features
    features = create_features(
        btc,
        eth,
        cached.get('gold'),
        cached.get('hashrate'),
        cached.get('funding'),
        cached.get('fear_greed')
    )
    
    # align features with what model expects
    aligned = pd.DataFrame(index=features.index)
    
    for col in feature_names:
        if col in features.columns:
            aligned[col] = features[col]
        else:
            aligned[col] = 0
    
    aligned = aligned.ffill().fillna(0)
    
    # normalize
    X = scaler.transform(aligned)
    
    # take last LOOKBACK rows
    X = X[-LOOKBACK:]
    
    # reshape for model: (1, lookback, features)
    X = X.reshape(1, LOOKBACK, len(feature_names))
    X = torch.FloatTensor(X)
    X = X.to(device)
    
    # predict
    model.eval()
    with torch.no_grad():
        prediction = model(X)
        percentile = prediction.cpu().numpy()[0][0]
    
    # get historical returns for signal calculation
    hist_returns = btc['close'].pct_change().dropna().values
    
    # generate signal
    signal = generate_signal(percentile, hist_returns)
    
    return {
        'timestamp': btc.index[-1],
        'price': btc['close'].iloc[-1],
        'percentile': percentile,
        'signal': signal
    }


# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("BTC TRADING BOT")
    print("=" * 60)
    
    # init database
    init_database()
    print("Database ready")
    
    # load model
    print("Loading model...")
    scaler, feature_names = load_scaler(f"{RESULTS_DIR}/scaler.json")
    model = load_model(f"{RESULTS_DIR}/best_model.pth", len(feature_names), device)
    print(f"Model loaded with {len(feature_names)} features")
    
    # send startup message
    send_telegram("🤖 *Bot Started*\n✅ Model loaded and ready")
    
    # track last trade time
    last_trade_time = datetime.utcnow() - timedelta(hours=MIN_HOURS_BETWEEN_TRADES)
    
    print("\nStarting main loop...")
    print("Press Ctrl+C to stop\n")
    
    # main loop
    while True:
        try:
            now = datetime.utcnow()
            
            # === CHECK FOR NEW SIGNALS ===
            # only check at minute 1 of each hour
            if now.minute == 1:
                
                # check if enough time since last trade
                hours_since_trade = (now - last_trade_time).total_seconds() / 3600
                
                if hours_since_trade >= MIN_HOURS_BETWEEN_TRADES:
                    print(f"[{now}] Checking for signals...")
                    
                    result = make_prediction(model, scaler, feature_names)
                    
                    if result is None:
                        print("Prediction failed, skipping...")
                    elif result['signal'] is not None:
                        sig = result['signal']
                        price = result['price']
                        
                        print(f"Signal: {sig['direction']} at ${price:.2f}")
                        
                        # calculate prices
                        if sig['direction'] == 'LONG':
                            tp_price = price * (1 + sig['take_profit'])
                            sl_price = price * (1 + sig['stop_loss'])
                        else:
                            tp_price = price * (1 - sig['take_profit'])
                            sl_price = price * (1 + sig['stop_loss'])
                        
                        # send telegram
                        if sig['direction'] == 'LONG':
                            emoji = "🟢 LONG"
                        else:
                            emoji = "🔴 SHORT"
                        
                        message = f"""{emoji} *NEW SIGNAL*

Entry: `${price:,.2f}`
TP: `${tp_price:,.2f}`
SL: `${sl_price:,.2f}`
Confidence: `{sig['confidence']:.2f}`"""
                        
                        if send_telegram(message):
                            # save to database
                            add_trade(now, sig['direction'], price, tp_price, sl_price)
                            last_trade_time = now
                            print("Trade logged!")
                    else:
                        print(f"No signal (percentile: {result['percentile']:.4f})")
            
            # === MONITOR OPEN TRADES ===
            # check every 5 minutes
            if now.minute % 5 == 0:
                trades = get_open_trades()
                
                if len(trades) > 0:
                    current_price = get_current_price()
                    
                    if current_price is not None:
                        for trade in trades:
                            # calculate how long trade has been open
                            open_time = datetime.fromisoformat(trade['open_time'])
                            hours_open = (now - open_time).total_seconds() / 3600
                            
                            # create current bar (simplified)
                            current_bar = {
                                'high': current_price * 1.001,
                                'low': current_price * 0.999,
                                'close': current_price
                            }
                            
                            # convert to dict
                            trade_dict = {
                                'entry_price': trade['entry_price'],
                                'tp_price': trade['tp_price'],
                                'sl_price': trade['sl_price'],
                                'direction': trade['direction']
                            }
                            
                            # check exit
                            should_exit, reason, pnl = check_exit(trade_dict, current_bar, hours_open)
                            
                            if should_exit:
                                # send notification
                                if pnl > 0:
                                    emoji = "✅"
                                else:
                                    emoji = "🛑"
                                
                                message = f"""{emoji} *TRADE CLOSED*

Reason: {reason}
PnL: `{pnl*100:+.2f}%`"""
                                
                                send_telegram(message)
                                close_trade(trade['id'], reason)
                                print(f"Trade {trade['id']} closed: {reason}")
            
            # wait until next minute
            seconds_to_wait = 60 - datetime.utcnow().second
            time.sleep(seconds_to_wait)
            
        except KeyboardInterrupt:
            print("\nStopping bot...")
            send_telegram("🛑 *Bot stopped*")
            break
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()