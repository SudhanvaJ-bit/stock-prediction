import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_ingestion import IngestionPipeline
from src.data_processing.cleaner import DataCleaner
from src.feature_engineering import FeaturePipeline
from src.data_processing import ProcessingPipeline
from src.models import TrainingPipeline
from src.evaluation import EvaluationPipeline


def main():
    TICKER     = "AAPL"
    START_DATE = "2020-01-01"
    END_DATE   = "2024-12-31"
    TRAIN_LSTM = True     # Set False for a quicker test run

    print(f"\n{'='*60}")
    print(f"  Stock Price Prediction — Model Evaluation Layer")
    print(f"{'='*60}\n")

    # ── Step 1: Load raw data ─────────────────────────
    print(">> Step 1: Loading raw data...\n")
    ingestion   = IngestionPipeline(start_date=START_DATE, end_date=END_DATE)
    saved_files = ingestion.list_available(ticker=TICKER)
    df_raw      = ingestion.load_existing(saved_files[-1]) if saved_files \
                  else ingestion.run(ticker=TICKER)[0]

    # ── Step 2: Clean ─────────────────────────────────
    print(">> Step 2: Cleaning...\n")
    df_clean = DataCleaner().clean(df_raw)

    # ── Step 3: Feature Engineering ───────────────────
    print(">> Step 3: Feature engineering...\n")
    df_features = FeaturePipeline().run(df_clean, ticker=TICKER)

    # ── Step 4: Process ────────────────────────────────
    print(">> Step 4: Processing...\n")
    processed = ProcessingPipeline().run(df_features, ticker=TICKER)

    # ── Step 5: Train ──────────────────────────────────
    print(">> Step 5: Training models...\n")
    training_result = TrainingPipeline(
        train_lr    = True,
        train_arima = True,
        train_lstm  = TRAIN_LSTM,
    ).run(processed)

    # ── Step 6: Evaluate ───────────────────────────────
    print(">> Step 6: Evaluating on real dollar values...\n")
    eval_result = EvaluationPipeline().run(training_result, processed)

    # ── Step 7: Show outputs ───────────────────────────
    print("\n>> Generated Plots:")
    for name, path in eval_result.plot_paths.items():
        print(f"   {name:<20} -> {path}")

    print("\n>> Generated Reports:")
    for fmt, path in eval_result.report_paths.items():
        print(f"   {fmt:<20} -> {path}")

    print(f"\n{'='*60}")
    print(f"  Evaluation layer working correctly.")
    print(f"  Best model: {eval_result.best_model}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()