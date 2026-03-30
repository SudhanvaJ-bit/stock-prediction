<div align="center">

# StockSense — ML Stock Price Prediction System

**An end-to-end machine learning system for stock price trend forecasting**  
built with a modular 8-layer pipeline architecture, comparing classical and deep learning models.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Yahoo Finance](https://img.shields.io/badge/Data-Yahoo%20Finance-6001D2?style=flat)](https://finance.yahoo.com)

</div>

---

## What This Project Demonstrates

This project is a **full production-grade ML system** — not just a model in a notebook. It demonstrates:

- End-to-end ML pipeline design from raw data to live predictions
- Model abstraction and a common interface across 3 different model types
- Time-series specific engineering — no data leakage, walk-forward validation
- 38 technical indicators engineered from raw OHLCV data
- A versioned model registry with JSON-backed persistence
- A real-time prediction service consuming live Yahoo Finance data
- A Bloomberg terminal-inspired Streamlit UI with Plotly charts

---

## Live Demo

```bash
pip install -r requirements.txt
python run_registry.py     # trains + registers all models
streamlit run app.py       # launches the UI
```

---

## System Architecture

The system is divided into **8 independent, reusable layers** following strict separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Yahoo Finance API                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  Layer 1 — Data Ingestion                                        │
│  ├── StockDataFetcher    → OHLCV data via yfinance               │
│  ├── DataValidator       → 6 automated quality checks            │
│  └── RawDataStorage      → versioned CSV storage                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  Layer 2 — Data Processing                                       │
│  ├── DataCleaner         → missing values + outlier capping      │
│  ├── TimeSeriesSplitter  → time-aware split (zero data leakage)  │
│  ├── DataNormalizer      → MinMax / Standard scaling             │
│  └── SequenceGenerator   → sliding window (X, y) pairs          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  Layer 3 — Feature Engineering                 (+38 features)    │
│  ├── MovingAverageFeatures  → SMA, EMA, Price vs SMA            │
│  ├── RollingStatistics      → Rolling stats, Bollinger Bands     │
│  └── ReturnsVolatility      → RSI, MACD, Log Returns, Volatility│
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  Layer 4 — Model Training                                        │
│  ├── LinearRegression    → sklearn baseline (flattened sequences)│
│  ├── ARIMA (5,1,0)       → statistical walk-forward validation   │
│  └── LSTM (128→64→32→1)  → deep learning, dropout, early stop   │
│  All models share: train() · predict() · evaluate() · save()    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  Layer 5 — Model Evaluation                                      │
│  ├── InverseTransformer  → scaled → real dollar values           │
│  ├── MetricsCalculator   → RMSE, MAE, MAPE, R², Direction Acc   │
│  ├── EvaluationPlotter   → 4 Matplotlib comparison charts        │
│  └── ReportGenerator     → CSV + TXT evaluation reports         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  Layer 6 — Model Registry                                        │
│  ├── ModelEntry          → structured metadata per model version │
│  ├── RegistryStore       → JSON-backed versioned index           │
│  ├── ModelLoader         → reconstructs any model from disk      │
│  └── ModelRegistry       → register · version · load best       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  Layer 7 — Prediction Service                                    │
│  ├── DataPreparator      → fetch live data → same pipeline       │
│  ├── Predictor           → single-step & multi-step forecasting  │
│  ├── ResultSaver         → CSV + JSON output                     │
│  └── PredictionService   → single public API for any UI          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  Layer 8 — Streamlit UI                                          │
│  ├── Dashboard           → live price, candlestick, technicals   │
│  ├── Prediction Engine   → forecast with any model, any horizon  │
│  ├── Model Analysis      → metrics, radar chart, LSTM history    │
│  ├── Historical Explorer → OHLCV data, period stats, CSV export  │
│  └── About               → architecture overview                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Models

| Model | Type | Architecture | Strength |
|-------|------|-------------|----------|
| **Linear Regression** | Classical ML | Flattened 60-day sequences → 2,640 features | Fast, interpretable baseline |
| **ARIMA (5,1,0)** | Statistical | Walk-forward validation on Close series | Time-series specific |
| **LSTM** | Deep Learning | Input(60,44) → LSTM(128) → LSTM(64) → Dense(32) → Output(1) | Long-range temporal patterns |

---

## Key Engineering Decisions

### No Data Leakage
The scaler is fitted **only on training data** and applied to the test set. The train/test split is strictly chronological — no shuffling. Features are computed from past data only.

### Common Model Interface
All models inherit from `BaseModel` and implement the same contract:
```python
model.train(X_train, y_train)
model.predict(X_test)
model.evaluate(y_true, y_pred)   # → {RMSE, MAE, MAPE, R2}
model.save(path)
model.load(path)
```
This means the evaluation, registry, and prediction layers are completely model-agnostic.

### Walk-Forward Validation for ARIMA
ARIMA uses the gold standard for time-series evaluation — predicting one step ahead at a time, using all available history, rather than training once and predicting all test points at once.

### Live Inference Pipeline
The prediction service fetches **real-time data** from Yahoo Finance, applies the same cleaning, feature engineering, and scaling used during training, then runs inference and outputs real dollar predictions.

---

## Evaluation Results (AAPL · Test Period: Jan 2024 – Dec 2024)

### Metrics on Real Dollar Values (Inverse-Transformed)

| Model | RMSE ($) | MAE ($) | MAPE (%) | R² | Direction Acc |
|-------|:--------:|:-------:|:--------:|:--:|:-------------:|
| Linear Regression | $13.69 | $10.85 | 5.11% | 0.675 | 48.2% |
| ARIMA | $31.99 | $28.74 | 13.02% | -0.778 | 40.3% |
| **LSTM** | **$18.99** | **$16.39** | **7.24%** | **0.373** | **54.5%** |

> **Key insight:** Linear Regression wins on raw error (RMSE) because it overfits to training patterns. LSTM wins on **Directional Accuracy (54.5%)** — meaning it correctly predicts whether the price will go UP or DOWN more than half the time. For any real trading application, direction matters more than absolute price error.

### Live Prediction (2026-03-30 · All Models · Next Day)

| Model | Last Known | Predicted | Signal |
|-------|:----------:|:---------:|:------:|
| Linear Regression | $248.80 | $227.02 | ▼ Bearish |
| ARIMA | $248.80 | $193.57 | ▼ Bearish |
| LSTM | $248.80 | $208.33 | ▼ Bearish |

---

## Evaluation Charts

### Predicted vs Actual Close Price
![Predictions vs Actual](reports/plots/AAPL_predictions_vs_actual.png)

### Model Metrics Comparison
![Metrics Comparison](reports/plots/AAPL_metrics_comparison.png)

### Residuals Analysis
![Residuals](reports/plots/AAPL_residuals.png)

### LSTM Training History
![LSTM Training](reports/plots/AAPL_lstm_training_history.png)

---

## Model Registry

Every model training run is automatically versioned:

```
registry/
├── registry_index.json          ← complete audit trail
└── AAPL/
    ├── LinearRegression/v1/     ← LinearRegression.pkl
    ├── ARIMA/v1/                ← ARIMA.pkl
    └── LSTM/v1/                 ← LSTM.keras
```

The registry tracks model name, version, ticker, timestamp, file path, scaler path, all metrics, and which model is currently marked as best — all in a single JSON index.

---

## Project Structure

```
stock_prediction/
│
├── app.py                           # Streamlit UI entry point
│
├── config/
│   ├── config.py                    # All settings in one place
│   └── logger.py                    # UTF-8 safe centralized logging
│
├── src/
│   ├── data_ingestion/
│   │   ├── fetcher.py               # Yahoo Finance via yfinance
│   │   ├── validator.py             # 6 automated data quality checks
│   │   ├── storage.py               # Raw CSV persistence
│   │   └── ingestion_pipeline.py   # Layer 1 orchestrator
│   │
│   ├── data_processing/
│   │   ├── cleaner.py               # Outlier capping, missing values
│   │   ├── splitter.py              # Time-aware chronological split
│   │   ├── normalizer.py            # Fit-on-train scaler
│   │   ├── sequence_generator.py   # Sliding window sequences
│   │   └── processing_pipeline.py  # Layer 2 orchestrator
│   │
│   ├── feature_engineering/
│   │   ├── moving_averages.py       # SMA, EMA (7/21/50 day)
│   │   ├── rolling_statistics.py   # Bollinger Bands, rolling stats
│   │   ├── returns_volatility.py   # RSI, MACD, log returns
│   │   └── feature_pipeline.py     # Layer 3 orchestrator
│   │
│   ├── models/
│   │   ├── base_model.py            # Abstract interface (ABC)
│   │   ├── linear_regression_model.py
│   │   ├── arima_model.py
│   │   ├── lstm_model.py
│   │   └── training_pipeline.py    # Layer 4 orchestrator
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # Real-dollar metric computation
│   │   ├── inverse_transformer.py  # Undo MinMax scaling
│   │   ├── plotter.py               # Plotly & Matplotlib charts
│   │   ├── report_generator.py     # CSV + TXT reports
│   │   └── evaluation_pipeline.py  # Layer 5 orchestrator
│   │
│   ├── registry/
│   │   ├── model_entry.py           # ModelEntry dataclass
│   │   ├── registry_store.py        # JSON index CRUD
│   │   ├── model_loader.py          # Deserializes any model type
│   │   └── model_registry.py       # Layer 6 orchestrator
│   │
│   ├── prediction_service/
│   │   ├── data_preparator.py       # Live data → model-ready input
│   │   ├── predictor.py             # Single + multi-step forecasting
│   │   ├── result_saver.py          # CSV + JSON output
│   │   └── prediction_service.py   # Layer 7 orchestrator
│   │
│   └── ui/
│       ├── styles.py                # Bloomberg terminal CSS theme
│       ├── charts.py                # Plotly chart builders
│       └── components.py            # Reusable HTML components
│
├── data/
│   ├── raw/                         # Raw OHLCV CSVs
│   ├── processed/                   # Scaled data + scaler (.pkl)
│   ├── features/                    # Feature-enriched CSVs
│   └── predictions/                 # Forecast outputs
│
├── models/AAPL/                     # Trained model files
├── registry/                        # Versioned registry + JSON index
├── reports/plots/                   # Evaluation charts (PNG)
├── logs/pipeline.log
│
├── run_ingestion.py                 # Run Layer 1 independently
├── run_processing.py                # Run Layer 2 independently
├── run_features.py                  # Run Layer 3 independently
├── run_training.py                  # Run Layer 4 independently
├── run_evaluation.py                # Run Layer 5 independently
├── run_registry.py                  # Run Layer 6 independently
├── run_prediction.py                # Run Layer 7 independently
└── requirements.txt
```

---

## Installation & Usage

### 1. Clone
```bash
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline (first time)
```bash
python run_ingestion.py    # fetch AAPL data from Yahoo Finance
python run_registry.py     # train, evaluate, and register all models
```

### 4. Launch the UI
```bash
streamlit run app.py
```

### 5. Predict from code
```python
from src.prediction_service import PredictionService

service = PredictionService()

# Next trading day — best model
result = service.predict(ticker="TSLA", horizon=1)

# Next 7 days — LSTM specifically
result = service.predict(ticker="MSFT", horizon=7, model_name="LSTM")

# Compare all models on next day
results = service.predict_all_models(ticker="AAPL", horizon=1)

print(result.summary())
```

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.11+ |
| **Deep Learning** | TensorFlow / Keras (LSTM) |
| **Classical ML** | Scikit-learn (Linear Regression) |
| **Statistical** | Statsmodels (ARIMA) |
| **Data** | Pandas, NumPy, yfinance |
| **Visualization** | Plotly, Matplotlib |
| **UI** | Streamlit |
| **Storage** | CSV, JSON, Pickle, Keras native format |

---

## Requirements

```
yfinance
pandas
numpy
scikit-learn
statsmodels
tensorflow
matplotlib
plotly
streamlit
```

---

## Design Principles

- **Modularity** — each layer is independently testable and replaceable
- **Single Responsibility** — every class does exactly one thing
- **No Data Leakage** — scaler fitted on train only; strictly chronological splits
- **Model Agnosticism** — evaluation, registry, and prediction layers work with any model
- **Real-world Evaluation** — metrics computed on inverse-transformed dollar values, not scaled outputs
- **Versioned Artifacts** — every training run is versioned, logged, and recoverable
- **Scalability** — swap tickers, date ranges, or models via config with zero code changes

---

## Configuration

All settings are centralized in `config/config.py` — no hardcoded values anywhere in the codebase:

```python
# Data
DEFAULT_TICKER       = "AAPL"
DEFAULT_START_DATE   = "2018-01-01"
TRAIN_RATIO          = 0.80

# Features
MA_WINDOWS           = [7, 21, 50]
RSI_PERIOD           = 14
BOLLINGER_WINDOW     = 20

# Model
SEQUENCE_LENGTH      = 60
NORMALIZATION_METHOD = "minmax"
ARIMA_ORDER          = (5, 1, 0)
LSTM_UNITS           = [128, 64]
LSTM_DROPOUT         = 0.2
LSTM_EPOCHS          = 50

# Registry & Prediction
BEST_MODEL_METRIC    = "RMSE ($)"
PREDICTION_HORIZON   = 7
```

---

<div align="center">

Built to demonstrate end-to-end ML system design · Not financial advice · Data sourced from Yahoo Finance

</div>