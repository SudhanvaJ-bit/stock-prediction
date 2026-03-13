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