import os
# Default stock ticker (overridden at runtime)
DEFAULT_TICKER = "AAPL"

# Default historical window
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE   = None          # None = today

# Yahoo Finance fetch interval
FETCH_INTERVAL = "1d"              # 1d, 1wk, 1mo

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR  = os.path.join(BASE_DIR, "data", "raw")
LOG_DIR       = os.path.join(BASE_DIR, "logs")


# Maximum allowed percentage of missing values per column
MAX_MISSING_PCT = 5.0              # 5%

# Minimum number of rows required for a valid dataset
MIN_REQUIRED_ROWS = 30

# Required columns that must exist in raw data
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# ──────────────────────────────────────────────
# DATA PROCESSING SETTINGS
# ──────────────────────────────────────────────

# Target column for prediction
TARGET_COLUMN = "Close"

# Missing value strategy: "ffill", "bfill", "drop", "interpolate"
MISSING_VALUE_STRATEGY = "ffill"

# Outlier detection: IQR multiplier (1.5 = standard, 3.0 = conservative)
OUTLIER_IQR_MULTIPLIER = 3.0

# Normalization method: "minmax" or "standard"
NORMALIZATION_METHOD = "minmax"

# Sequence window size (lookback period in trading days)
SEQUENCE_LENGTH = 60

# Train/Test split ratio
TRAIN_RATIO = 0.80

# Processed data storage
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# ──────────────────────────────────────────────
# FEATURE ENGINEERING SETTINGS
# ──────────────────────────────────────────────

# Moving average windows (in trading days)
MA_WINDOWS = [7, 21, 50]

# Bollinger Band window and standard deviation multiplier
BOLLINGER_WINDOW  = 20
BOLLINGER_STD_DEV = 2

# RSI period
RSI_PERIOD = 14

# MACD parameters
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9

# Volatility rolling window
VOLATILITY_WINDOW = 21

# Features data directory
FEATURES_DATA_DIR = os.path.join(BASE_DIR, "data", "features")

# ──────────────────────────────────────────────
# MODEL TRAINING SETTINGS
# ──────────────────────────────────────────────

# Models directory
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Linear Regression ──
LR_FIT_INTERCEPT = True

# ── ARIMA ──
ARIMA_ORDER = (5, 1, 0)          # (p, d, q)

# ── LSTM ──
LSTM_UNITS        = [128, 64]     # Units per LSTM layer
LSTM_DROPOUT      = 0.2
LSTM_EPOCHS       = 50
LSTM_BATCH_SIZE   = 32
LSTM_PATIENCE     = 10            # Early stopping patience
LSTM_LEARNING_RATE = 0.001

# ──────────────────────────────────────────────
# EVALUATION SETTINGS
# ──────────────────────────────────────────────

# Reports and plots output directory
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PLOTS_DIR   = os.path.join(BASE_DIR, "reports", "plots")

# ──────────────────────────────────────────────
# MODEL REGISTRY SETTINGS
# ──────────────────────────────────────────────

# Registry root directory
REGISTRY_DIR = os.path.join(BASE_DIR, "registry")

# Registry index file (JSON) — tracks all registered models
REGISTRY_INDEX_FILE = os.path.join(BASE_DIR, "registry", "registry_index.json")

# Metric used to select the best model
BEST_MODEL_METRIC = "RMSE ($)"     # lower is better