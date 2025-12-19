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

# === SIGNALS (tune these yourself!) ===
TOP_PERCENTILE = ...        # when to go LONG
BOTTOM_PERCENTILE = ...     # when to go SHORT
MIN_CONFIDENCE = ...        # minimum confidence

# === RISK (tune these yourself!) ===
RISK_PER_TRADE = ...

TP_PARAMS = {
    'base': ...,
    'confidence': ...,
}

SL_PARAMS = {
    'base': ...,
    'confidence': ...,
    'minimum': ...,
}

MIN_HOURS_BETWEEN_TRADES = ...
MAX_HOLD_HOURS = ...

# === FEATURES (tune these yourself!) ===
FEATURE_PARAMS = {
    'momentum_windows': [...],
    'volatility_windows': [...],
    'rsi_period': ...,
    'macd_fast': ...,
    'macd_slow': ...,
    'correlation_window': ...,
    'external_shift': ...,
}