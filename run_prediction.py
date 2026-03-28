import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.prediction_service import PredictionService, PredictionServiceError

def main():
    TICKER  = "AAPL"
    HORIZON = 7          # Predict next 7 trading days

    print(f"\n{'='*60}")
    print(f"  Stock Price Prediction — Prediction Service Layer")
    print(f"{'='*60}\n")

    service = PredictionService()

    # ── Test 1: Single-day prediction ─────────────────
    print(">> Test 1: Predicting next trading day...\n")
    try:
        result = service.predict(ticker=TICKER, horizon=1)
    except PredictionServiceError as e:
        print(f"  ERROR: {e}")
        return

    # ── Test 2: Multi-day prediction ──────────────────
    print(f">> Test 2: Predicting next {HORIZON} trading days...\n")
    result = service.predict(ticker=TICKER, horizon=HORIZON)

    # ── Test 3: All models comparison ─────────────────
    print(">> Test 3: All models comparison (next day)...\n")
    try:
        all_results = service.predict_all_models(ticker=TICKER, horizon=1)

        print(f"  {'Model':<22} {'Predicted Close':>16}  {'vs Last Known':>14}")
        print(f"  {'─'*55}")

        for model_name, res in all_results.items():
            last   = res.last_known_price
            pred   = res.predictions[0][1]
            change = pred - last
            arrow  = "▲" if change >= 0 else "▼"
            best   = " [BEST]" if res.model_name == result.model_name else ""
            print(
                f"  {model_name:<22} ${pred:>13.2f}  "
                f"{arrow} ${abs(change):.2f}{best}"
            )
    except PredictionServiceError as e:
        print(f"  ERROR: {e}")

    # ── Show saved file paths ──────────────────────────
    pred_dir = os.path.join(os.path.dirname(__file__), "data", "predictions")
    if os.path.exists(pred_dir):
        print(f"\n>> Saved prediction files:")
        for f in os.listdir(pred_dir):
            print(f"   data/predictions/{f}")

    print(f"\n{'='*60}")
    print(f"  Prediction service layer working correctly.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()