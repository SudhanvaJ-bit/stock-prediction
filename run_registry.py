import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_ingestion import IngestionPipeline
from src.data_processing.cleaner import DataCleaner
from src.feature_engineering import FeaturePipeline
from src.data_processing import ProcessingPipeline
from src.models import TrainingPipeline
from src.evaluation import EvaluationPipeline
from src.registry import ModelRegistry

def main():
    TICKER     = "AAPL"
    START_DATE = "2020-01-01"
    END_DATE   = "2024-12-31"
    TRAIN_LSTM = True

    print(f"\n{'='*60}")
    print(f"  Stock Price Prediction — Model Registry Layer")
    print(f"{'='*60}\n")

    # ── Steps 1-5: Full pipeline ──────────────────────
    print(">> Steps 1-5: Running full pipeline...\n")

    ingestion   = IngestionPipeline(start_date=START_DATE, end_date=END_DATE)
    saved_files = ingestion.list_available(ticker=TICKER)
    df_raw      = ingestion.load_existing(saved_files[-1]) if saved_files \
                  else ingestion.run(ticker=TICKER)[0]

    df_clean    = DataCleaner().clean(df_raw)
    df_features = FeaturePipeline().run(df_clean, ticker=TICKER)
    processed   = ProcessingPipeline().run(df_features, ticker=TICKER)

    training_result = TrainingPipeline(
        train_lr=True, train_arima=True, train_lstm=TRAIN_LSTM
    ).run(processed)

    eval_result = EvaluationPipeline().run(training_result, processed)

    # ── Step 6: Register all models ───────────────────
    print("\n>> Step 6: Registering models...\n")
    registry = ModelRegistry()
    entries  = registry.register_all(training_result, eval_result)

    # ── Step 7: Inspect registry ──────────────────────
    print("\n>> Step 7: Inspecting registry...\n")
    registry.print_registry(ticker=TICKER)

    # ── Step 8: Load best model ───────────────────────
    print("\n>> Step 8: Loading best model from registry...\n")
    try:
        model, best_entry = registry.load_best(ticker=TICKER)
        print(f"   Best model loaded successfully!")
        print(f"   Entry ID   : {best_entry.entry_id}")
        print(f"   Model Path : {best_entry.model_path}")
        print(f"   Metrics    :")
        for k, v in best_entry.metrics.items():
            if k != "Model":
                print(f"     {k:<25} : {v}")
    except Exception as e:
        print(f"   Could not load best model: {e}")

    # ── Step 9: List all entries ──────────────────────
    print(f"\n>> Step 9: All registered models for {TICKER}:\n")
    all_entries = registry.list_models(ticker=TICKER)
    for entry in all_entries:
        best_tag = " [BEST]" if entry.is_best else ""
        print(f"   {entry.entry_id}{best_tag}")
        print(f"   Registered: {entry.registered_at}")
        print()

    print(f"{'='*60}")
    print(f"  Registry layer working correctly.")
    print(f"  Total registered models: {len(all_entries)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()