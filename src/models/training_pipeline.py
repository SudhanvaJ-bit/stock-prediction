import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List

from src.models.base_model               import BaseModel
from src.models.linear_regression_model  import LinearRegressionModel
from src.models.arima_model              import ARIMAModel
from src.models.lstm_model               import LSTMModel
from src.data_processing.processing_pipeline import ProcessedData

from config.config import MODELS_DIR
from config.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelResult:
    model: BaseModel
    predictions: np.ndarray
    metrics: Dict[str, float]


@dataclass
class TrainingResult:
    ticker: str
    results: Dict[str, ModelResult] = field(default_factory=dict)
    best_model_name: str = ""

    def add(self, name: str, result: ModelResult):
        self.results[name] = result

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  Model Training Summary — [{self.ticker}]",
            f"{'='*60}",
            f"  {'Model':<22} {'RMSE':>10} {'MAE':>10} {'MAPE':>10} {'R2':>10}",
            f"  {'-'*52}",
        ]
        for name, res in self.results.items():
            m = res.metrics
            lines.append(
                f"  {name:<22} {m['RMSE']:>10.6f} {m['MAE']:>10.6f} "
                f"{m['MAPE']:>10.4f} {m['R2']:>10.6f}"
            )
        if self.best_model_name:
            lines.append(f"\n  Best Model: {self.best_model_name} (lowest RMSE)")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


class TrainingPipeline:
    def __init__(
        self,
        train_lr:   bool = True,
        train_arima: bool = True,
        train_lstm:  bool = True,
        models_dir:  str  = MODELS_DIR,
    ):

        self.train_lr    = train_lr
        self.train_arima = train_arima
        self.train_lstm  = train_lstm
        self.models_dir  = models_dir

    def run(self, data: ProcessedData) -> TrainingResult:
        logger.info(f"=== Starting Training Pipeline | Ticker: {data.ticker} ===")

        result = TrainingResult(ticker=data.ticker)

        # ── 1. Linear Regression ─────────────
        if self.train_lr:
            result.add("LinearRegression", self._train_model(
                model   = LinearRegressionModel(),
                X_train = data.X_train,
                y_train = data.y_train,
                X_test  = data.X_test,
                y_test  = data.y_test,
                ticker  = data.ticker,
            ))

        # ── 2. ARIMA ─────────────────────────
        if self.train_arima:
            result.add("ARIMA", self._train_model(
                model   = ARIMAModel(),
                X_train = data.X_train,
                y_train = data.y_train,
                X_test  = data.X_test,
                y_test  = data.y_test,
                ticker  = data.ticker,
            ))

        # ── 3. LSTM ───────────────────────────
        if self.train_lstm:
            result.add("LSTM", self._train_model(
                model   = LSTMModel(),
                X_train = data.X_train,
                y_train = data.y_train,
                X_test  = data.X_test,
                y_test  = data.y_test,
                ticker  = data.ticker,
            ))

        # ── Pick best model by RMSE ───────────
        if result.results:
            result.best_model_name = min(
                result.results,
                key=lambda name: result.results[name].metrics["RMSE"]
            )

        logger.info(f"=== Training Complete | Best: {result.best_model_name} ===")
        print(result.summary())
        return result

    # ──────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────

    def _train_model(
        self,
        model:   BaseModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
        ticker:  str,
    ) -> ModelResult:

        logger.info(f"--- Training: {model.name} ---")

        # Train
        model.train(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Evaluate
        metrics = model.evaluate(y_test, predictions)
        logger.info(f"[{model.name}] Test metrics: {metrics}")

        # Save
        save_dir = os.path.join(self.models_dir, ticker)
        model.save(save_dir)

        return ModelResult(
            model=model,
            predictions=predictions,
            metrics=metrics,
        )