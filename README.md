# Stock Price Trend Prediction Using Time Series Analysis

A modular, production-minded ML system that predicts stock price trends using historical Yahoo Finance data. Compares classical statistical models (ARIMA, Linear Regression) with deep learning (LSTM) through a layered, scalable pipeline architecture.

---

## Project Status

| Layer | Description | Status |
|-------|-------------|--------|
| Layer 1 | Data Ingestion | ✅ Complete |
| Layer 2 | Data Processing | ✅ Complete |
| Layer 3 | Feature Engineering | ✅ Complete |
| Layer 4 | Model Training | ✅ Complete |
| Layer 5 | Model Evaluation | 🔄 In Progress |
| Layer 6 | Model Registry | ⬜ Upcoming |
| Layer 7 | Prediction Service | ⬜ Upcoming |
| Layer 8 | Visualization / UI (Streamlit) | ⬜ Upcoming |

---

## Architecture Overview

```
Yahoo Finance API
      │
      ▼
Layer 1 — Data Ingestion
  ├── StockDataFetcher       → pulls OHLCV data via yfinance
  ├── DataValidator          → 6 quality checks + ValidationReport
  └── RawDataStorage         → saves raw CSVs to data/raw/
      │
      ▼
Layer 2 — Data Processing
  ├── DataCleaner            → handles missing values, caps outliers
  ├── TimeSeriesSplitter     → time-aware 80/20 split (no data leakage)
  ├── DataNormalizer         → MinMax/Standard scaling
  └── SequenceGenerator      → sliding window (X, y) pairs
      │
      ▼
Layer 3 — Feature Engineering
  ├── MovingAverageFeatures  → SMA, EMA, Price vs SMA (7, 21, 50 days)
  ├── RollingStatistics      → Rolling stats, Bollinger Bands
  └── ReturnsVolatility      → Daily/Log returns, RSI, MACD, Volatility
      │
      ▼
Layer 4 — Model Training
  ├── LinearRegression       → baseline ML model (sklearn)
  ├── ARIMA (5,1,0)          → statistical time-series (walk-forward)
  └── LSTM (128→64→32→1)     → deep learning with early stopping
      │
      ▼
Layer 5 — Model Evaluation     ← next
Layer 6 — Model Registry
Layer 7 — Prediction Service
Layer 8 — Streamlit UI
```

---

## Models

| Model | Type | Input | Notes |
|-------|------|-------|-------|
| Linear Regression | Classical ML | Flattened sequences (2640 features) | Baseline |
| ARIMA (5,1,0) | Statistical | Close price series only | Walk-forward validation |
| LSTM | Deep Learning | (samples, 60 timesteps, 44 features) | 2-layer + dropout + early stopping |

---

## Training Results (AAPL — 2020 to 2024)

| Model | RMSE | MAE | MAPE | R2 |
|-------|------|-----|------|----|
| LinearRegression | 0.0964 | 0.0764 | 6.99% | 0.6746 |
| ARIMA | 0.2253 | 0.2024 | 17.58% | -0.7780 |
| LSTM | 0.1675 | 0.1562 | 13.49% | 0.0165 |

> Metrics are on scaled data. Layer 5 will evaluate on original dollar values.

---

## Project Structure

```
stock_prediction/
│
├── config/
│   ├── config.py                    # Central configuration
│   └── logger.py                    # Shared logger (UTF-8 safe)
│
├── src/
│   ├── data_ingestion/
│   │   ├── fetcher.py               # Yahoo Finance data fetch
│   │   ├── validator.py             # Data quality checks
│   │   ├── storage.py               # Save/load raw CSVs
│   │   └── ingestion_pipeline.py    # Layer 1 entry point
│   │
│   ├── data_processing/
│   │   ├── cleaner.py               # Missing values + outlier capping
│   │   ├── splitter.py              # Time-aware train/test split
│   │   ├── normalizer.py            # MinMax / Standard scaling
│   │   ├── sequence_generator.py    # Sliding window sequences
│   │   └── processing_pipeline.py  # Layer 2 entry point
│   │
│   ├── feature_engineering/
│   │   ├── moving_averages.py       # SMA, EMA, Price vs SMA
│   │   ├── rolling_statistics.py    # Rolling stats, Bollinger Bands
│   │   ├── returns_volatility.py    # Returns, RSI, MACD, Volatility
│   │   └── feature_pipeline.py     # Layer 3 entry point
│   │
│   └── models/
│       ├── base_model.py            # Abstract interface for all models
│       ├── linear_regression_model.py
│       ├── arima_model.py
│       ├── lstm_model.py
│       └── training_pipeline.py    # Layer 4 entry point
│
├── data/
│   ├── raw/                         # Raw CSVs from Yahoo Finance
│   ├── processed/                   # Scaled train/test CSVs + scaler
│   └── features/                    # Feature-enriched CSVs
│
├── models/
│   └── AAPL/
│       ├── LinearRegression.pkl
│       ├── ARIMA.pkl
│       └── LSTM.keras
│
├── logs/
│   └── pipeline.log
│
├── run_ingestion.py                 # Test Layer 1
├── run_processing.py                # Test Layer 2
├── run_features.py                  # Test Layer 3
├── run_training.py                  # Test Layer 4
└── requirements.txt
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/SudhanvaJ-bit/stock-prediction.git
cd stock-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run each layer in order
```bash
python run_ingestion.py    # Layer 1 — fetch and save raw data
python run_processing.py   # Layer 2 — clean, split, normalize, sequence
python run_features.py     # Layer 3 — add technical indicators
python run_training.py     # Layer 4 — train all 3 models
```

---

## Requirements

```
yfinance
pandas
numpy
scikit-learn
statsmodels
tensorflow
```

---

## Key Design Principles

- **Modular design** — each layer is independent and replaceable
- **Separation of concerns** — fetch, validate, store, process, train are all separate
- **No data leakage** — scaler fitted on training data only; time-aware split
- **Common model interface** — all models implement `train()`, `predict()`, `evaluate()`, `save()`, `load()`
- **Scalable** — swap any ticker, date range, or model with config changes only

---

## Configuration

All settings are centralized in `config/config.py`:

```python
DEFAULT_TICKER     = "AAPL"
DEFAULT_START_DATE = "2018-01-01"
SEQUENCE_LENGTH    = 60        # lookback window in trading days
TRAIN_RATIO        = 0.80      # 80% train / 20% test
NORMALIZATION_METHOD = "minmax"
LSTM_EPOCHS        = 50
ARIMA_ORDER        = (5, 1, 0)
```

---

## Author

Built as a learning project to demonstrate end-to-end ML system design for time series forecasting.