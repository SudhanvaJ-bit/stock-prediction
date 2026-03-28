import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List
from src.models.base_model import BaseModel
from src.data_processing.normalizer import DataNormalizer
from config.config import PREDICTION_HORIZON, TARGET_COLUMN
from config.logger import get_logger

logger = get_logger(__name__)

@dataclass
class PredictionResult:
    ticker:           str
    model_name:       str
    model_version:    str
    last_known_date:  pd.Timestamp
    last_known_price: float
    predictions:      List[tuple] = field(default_factory=list)
    horizon:          int = 1

    def summary(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"  Prediction Report — [{self.ticker}]",
            f"{'='*55}",
            f"  Model         : {self.model_name} ({self.model_version})",
            f"  Last Known    : {self.last_known_date.date()} | ${self.last_known_price:.2f}",
            f"  Horizon       : {self.horizon} trading day(s)",
            f"  {'─'*45}",
            f"  {'Date':<15}  {'Predicted Close':>15}  {'Change':>10}",
            f"  {'─'*45}",
        ]
        prev = self.last_known_price
        for pred_date, pred_price in self.predictions:
            change = pred_price - prev
            arrow  = "▲" if change >= 0 else "▼"
            lines.append(
                f"  {str(pred_date):<15}  ${pred_price:>13.2f}  "
                f"{arrow} ${abs(change):.2f}"
            )
            prev = pred_price
        lines.append(f"{'='*55}\n")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "ticker"           : self.ticker,
            "model_name"       : self.model_name,
            "model_version"    : self.model_version,
            "last_known_date"  : str(self.last_known_date.date()),
            "last_known_price" : round(self.last_known_price, 4),
            "predictions"      : [
                {"date": str(d), "predicted_close": round(p, 4)}
                for d, p in self.predictions
            ],
        }


class Predictor:
    def __init__(
        self,
        model:      BaseModel,
        normalizer: DataNormalizer,
        entry,                          # ModelEntry from registry
        target_column: str = TARGET_COLUMN,
    ):
        self.model         = model
        self.normalizer    = normalizer
        self.entry         = entry
        self.target_column = target_column

    # ──────────────────────────────────────────
    # Single-step Prediction
    # ──────────────────────────────────────────

    def predict_single(
        self,
        X_input:   np.ndarray,
        df_scaled: pd.DataFrame,
        last_date: pd.Timestamp,
        ticker:    str,
    ) -> PredictionResult:

        logger.info(f"[Predictor] Single-step prediction for [{ticker}]")

        # Run model inference
        scaled_pred = self.model.predict(X_input)

        # Inverse transform to real dollar value
        real_price = self._inverse(scaled_pred[0], df_scaled)

        # Get last known real price
        last_known = self._inverse(
            df_scaled[self.target_column].values[-1], df_scaled
        )

        # Next trading day
        next_date = self._next_trading_day(last_date)

        result = PredictionResult(
            ticker           = ticker,
            model_name       = self.entry.model_name,
            model_version    = self.entry.version,
            last_known_date  = last_date,
            last_known_price = float(last_known),
            predictions      = [(next_date, float(real_price))],
            horizon          = 1,
        )

        logger.info(
            f"[Predictor] Next day: {next_date} | "
            f"Predicted: ${real_price:.2f} | Last known: ${last_known:.2f}"
        )
        return result

    # ──────────────────────────────────────────
    # Multi-step Prediction
    # ──────────────────────────────────────────

    def predict_multi(
        self,
        X_input:   np.ndarray,
        df_scaled: pd.DataFrame,
        last_date: pd.Timestamp,
        ticker:    str,
        horizon:   int = PREDICTION_HORIZON,
    ) -> PredictionResult:

        logger.info(f"[Predictor] Multi-step prediction | horizon={horizon} | [{ticker}]")

        last_known = self._inverse(
            df_scaled[self.target_column].values[-1], df_scaled
        )

        current_input = X_input.copy()    # (1, seq_len, n_features)
        current_date  = last_date
        predictions   = []

        target_idx = df_scaled.columns.tolist().index(self.target_column)

        for step in range(horizon):
            # Predict next step
            scaled_pred = self.model.predict(current_input)
            real_price  = self._inverse(scaled_pred[0], df_scaled)
            next_date   = self._next_trading_day(current_date)

            predictions.append((next_date, float(real_price)))
            logger.info(f"  Step {step+1}/{horizon}: {next_date} -> ${real_price:.2f}")

            # Roll the input window: drop oldest, append new prediction
            new_row                = current_input[0, -1, :].copy()
            new_row[target_idx]    = float(scaled_pred[0])
            current_input          = np.roll(current_input, -1, axis=1)
            current_input[0, -1, :] = new_row
            current_date           = next_date

        result = PredictionResult(
            ticker           = ticker,
            model_name       = self.entry.model_name,
            model_version    = self.entry.version,
            last_known_date  = last_date,
            last_known_price = float(last_known),
            predictions      = predictions,
            horizon          = horizon,
        )
        return result

    # ──────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────

    def _inverse(self, scaled_value: float, df_scaled: pd.DataFrame) -> float:
        """Inverse transforms a single scaled Close value to real dollar price."""
        n_features = df_scaled.shape[1]
        target_idx = df_scaled.columns.tolist().index(self.target_column)

        padded = np.zeros((1, n_features))
        padded[0, target_idx] = scaled_value
        return float(self.normalizer._scaler.inverse_transform(padded)[0, target_idx])

    def _next_trading_day(self, current_date) -> date:
        """Returns the next weekday (Mon-Fri) after the given date."""
        base     = current_date.date() if hasattr(current_date, "date") else current_date
        next_day = base + timedelta(days=1)
        while next_day.weekday() >= 5:    # 5=Saturday, 6=Sunday
            next_day += timedelta(days=1)
        return next_day
