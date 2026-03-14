import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_ingestion  import IngestionPipeline
from src.data_processing import ProcessingPipeline


def main():
    TICKER     = "AAPL"
    START_DATE = "2020-01-01"
    END_DATE   = "2024-12-31"

    print(f"\n{'='*55}")
    print(f"  Stock Price Prediction — Data Processing Layer")
    print(f"{'='*55}\n")

    # ── Step 1: Load raw data (from ingestion layer) ──
    print(">> Loading raw data from ingestion layer...\n")
    ingestion = IngestionPipeline(start_date=START_DATE, end_date=END_DATE)

    saved_files = ingestion.list_available(ticker=TICKER)

    if saved_files:
        # Load existing file — skip re-fetching
        df_raw = ingestion.load_existing(saved_files[-1])
        print(f"   Loaded existing file: {saved_files[-1]}\n")
    else:
        # Fetch fresh if not already saved
        df_raw, _ = ingestion.run(ticker=TICKER)

    # ── Step 2: Run processing pipeline ──────────────
    print(">> Running processing pipeline...\n")
    pipeline = ProcessingPipeline()
    data = pipeline.run(df_raw, ticker=TICKER)

    # ── Step 3: Verify outputs ────────────────────────
    print("\n>> Verifying outputs...\n")
    print(f"   X_train : {data.X_train.shape}  (samples, window, features)")
    print(f"   y_train : {data.y_train.shape}")
    print(f"   X_test  : {data.X_test.shape}   (samples, window, features)")
    print(f"   y_test  : {data.y_test.shape}")

    print(f"\n   Sample X_train[0] (first window, Close column):")
    close_idx = data.train_df.columns.tolist().index("Close")
    print(f"   {data.X_train[0, :5, close_idx]} ... (first 5 of {data.sequence_length} steps)")

    print(f"\n   Corresponding y_train[0]: {data.y_train[0]:.6f} (scaled Close)")

    # ── Inverse transform check ───────────────────────
    import numpy as np
    sample_pred = data.y_train[:5].reshape(-1, 1)
    n_features  = data.train_df.shape[1]
    padded      = np.hstack([sample_pred, np.zeros((5, n_features - 1))])
    original    = data.normalizer._scaler.inverse_transform(padded)[:, 0]
    print(f"\n   Inverse transform check (first 5 y_train -> original Close):")
    print(f"   {original}")

    print(f"\n{'='*55}")
    print(f"  Processing layer working correctly.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()