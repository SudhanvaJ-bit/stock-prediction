import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_ingestion      import IngestionPipeline
from src.data_processing.cleaner import DataCleaner
from src.feature_engineering import FeaturePipeline
from src.data_processing     import ProcessingPipeline


def main():
    TICKER     = "AAPL"
    START_DATE = "2020-01-01"
    END_DATE   = "2024-12-31"

    print(f"\n{'='*55}")
    print(f"  Stock Price Prediction — Feature Engineering Layer")
    print(f"{'='*55}\n")

    # ── Step 1: Load raw data ─────────────────
    print(">> Loading raw data...\n")
    ingestion   = IngestionPipeline(start_date=START_DATE, end_date=END_DATE)
    saved_files = ingestion.list_available(ticker=TICKER)
    df_raw      = ingestion.load_existing(saved_files[-1]) if saved_files \
                  else ingestion.run(ticker=TICKER)[0]

    # ── Step 2: Clean ─────────────────────────
    print(">> Cleaning data...\n")
    cleaner  = DataCleaner()
    df_clean = cleaner.clean(df_raw)

    # ── Step 3: Feature Engineering ──────────
    print(">> Running feature engineering...\n")
    feat_pipeline = FeaturePipeline()
    df_features   = feat_pipeline.run(df_clean, ticker=TICKER)

    # ── Step 4: Show feature summary ─────────
    feature_names = feat_pipeline.get_feature_names(df_features)
    base_cols     = [c for c in df_features.columns if c not in feature_names]

    print(f"\n{'─'*55}")
    print(f"  Dataset shape   : {df_features.shape}")
    print(f"  Base columns    : {base_cols}")
    print(f"\n  Engineered features ({len(feature_names)}):")
    for i, name in enumerate(feature_names, 1):
        print(f"    {i:2}. {name}")
    print(f"{'─'*55}\n")

    # ── Step 5: Sample values ────────────────
    print(">> Sample feature values (last row):\n")
    last_row = df_features.iloc[-1]
    for col in feature_names[:10]:
        print(f"   {col:<30} : {last_row[col]:.6f}")
    print(f"   ... and {len(feature_names) - 10} more features\n")

    # ── Step 6: Feed into Processing Pipeline ─
    print(">> Feeding enriched data into processing pipeline...\n")
    proc_pipeline = ProcessingPipeline()
    data = proc_pipeline.run(df_features, ticker=TICKER)

    print(f"\n{'='*55}")
    print(f"  Feature engineering layer working correctly.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()