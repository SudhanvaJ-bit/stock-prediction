import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_ingestion import IngestionPipeline
from src.data_processing.cleaner import DataCleaner
from src.feature_engineering import FeaturePipeline
from src.data_processing import ProcessingPipeline
from src.models import TrainingPipeline


def main():
    TICKER     = "AAPL"
    START_DATE = "2020-01-01"
    END_DATE   = "2024-12-31"

    # Set to False to skip slow models during testing
    TRAIN_LR    = True
    TRAIN_ARIMA = True
    TRAIN_LSTM  = True     # Set False for quick test

    print(f"\n{'='*60}")
    print(f"  Stock Price Prediction — Model Training Layer")
    print(f"{'='*60}\n")

    # ── Step 1: Load raw data ─────────────────────────
    print(">> Step 1: Loading raw data...\n")
    ingestion   = IngestionPipeline(start_date=START_DATE, end_date=END_DATE)
    saved_files = ingestion.list_available(ticker=TICKER)
    df_raw      = ingestion.load_existing(saved_files[-1]) if saved_files \
                  else ingestion.run(ticker=TICKER)[0]

    # ── Step 2: Clean ─────────────────────────────────
    print(">> Step 2: Cleaning data...\n")
    df_clean = DataCleaner().clean(df_raw)

    # ── Step 3: Feature Engineering ───────────────────
    print(">> Step 3: Engineering features...\n")
    df_features = FeaturePipeline().run(df_clean, ticker=TICKER)

    # ── Step 4: Process (split, normalize, sequence) ──
    print(">> Step 4: Processing data...\n")
    processed = ProcessingPipeline().run(df_features, ticker=TICKER)

    # ── Step 5: Train all models ───────────────────────
    print(">> Step 5: Training models...\n")
    trainer = TrainingPipeline(
        train_lr    = TRAIN_LR,
        train_arima = TRAIN_ARIMA,
        train_lstm  = TRAIN_LSTM,
    )
    result = trainer.run(processed)

    # ── Step 6: Show detailed results ─────────────────
    print(f"\n>> Step 6: Detailed Results\n")
    for name, model_result in result.results.items():
        print(f"  {name}:")
        for metric, value in model_result.metrics.items():
            print(f"    {metric:<6} : {value}")
        print()

    print(f"  Best Model  : {result.best_model_name}")
    print(f"\n{'='*60}")
    print(f"  Model training layer working correctly.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()