import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_ingestion import IngestionPipeline


def main():
    # ── Configuration ────────────────────────
    TICKER     = "AAPL"          # Change to any valid ticker
    START_DATE = "2020-01-01"
    END_DATE   = "2024-12-31"

    print(f"\n{'='*55}")
    print(f"  Stock Price Prediction — Data Ingestion Layer")
    print(f"{'='*55}\n")

    # ── Run Pipeline ─────────────────────────
    pipeline = IngestionPipeline(
        start_date=START_DATE,
        end_date=END_DATE,
        strict_validation=True,
    )

    df, filepath = pipeline.run(ticker=TICKER)

    # ── Preview Output ───────────────────────
    print(f"\n{'─'*45}")
    print(f"  Ticker     : {TICKER}")
    print(f"  Rows       : {len(df)}")
    print(f"  Columns    : {list(df.columns)}")
    print(f"  Date Range : {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Saved To   : {filepath}")
    print(f"{'─'*45}\n")

    print("First 5 rows:")
    print(df.head())

    print("\nLast 5 rows:")
    print(df.tail())

    print(f"\n✅  Ingestion layer working correctly.")


if __name__ == "__main__":
    main()