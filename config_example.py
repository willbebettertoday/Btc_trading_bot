"""
Config file - copy this to config.py and fill in your values
"""

# === PATHS ===
CACHE_DIR = "/path/to/data"
CACHE_DAILY = "/path/to/daily_data"
RESULTS_DIR = "/path/to/model"
DATABASE_FILE = "/path/to/trades.db"

# === TELEGRAM ===
TELEGRAM_BOT_TOKEN = "your_token"
TELEGRAM_CHAT_ID = "your_chat_id"

# === DATES ===
START_DATE = "2019-09-01"
TRAIN_END = "2025-08-01"
VAL_END = "2025-10-01"

# === MODEL ===
LOOKBACK = 168      # hours of history to look at
D_MODEL = 128       # size of embeddings
N_HEADS = 8         # attention heads
N_LAYERS = 3        # transformer layers
D_FF = 512          # feedforward size
DROPOUT = 0.15

# === TRAINING ===
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 128
EPOCHS = 80
EARLY_STOP_PATIENCE = 20

# === SIGNALS ===
# These are runnable defaults, not the tuned values behind the figures in
# the README. Tune them on your own validation split before trusting them.
TOP_PERCENTILE = 0.75       # go LONG at or above this predicted percentile
BOTTOM_PERCENTILE = 0.25    # go SHORT at or below it
MIN_CONFIDENCE = 0.15       # skip signals weaker than this

# === RISK ===
RISK_PER_TRADE = 0.01       # fraction of equity risked per trade

TP_PARAMS = {
    'base': 1.5,            # take profit as a multiple of expected return
    'confidence': 1.0,      # extra multiple scaled by signal confidence
}

SL_PARAMS = {
    'base': 1.0,            # stop loss as a multiple of expected return
    'confidence': 0.0,
    'minimum': 0.005,       # never place a stop tighter than 0.5%
}

MIN_HOURS_BETWEEN_TRADES = 8
MAX_HOLD_HOURS = 72

# === FEATURES ===
FEATURE_PARAMS = {
    'momentum_windows': [6, 12, 24, 72],
    'volatility_windows': [24, 72],
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'correlation_window': 168,
    'external_shift': 24,   # lag applied to external series to avoid look-ahead
}
