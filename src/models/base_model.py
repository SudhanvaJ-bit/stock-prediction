from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class BaseModel(ABC):
    def __init__(self, name: str):
        self.name       = name
        self.is_trained = False
        self.model      = None

    # ──────────────────────────────────────────
    # Abstract Methods — must be implemented
    # ──────────────────────────────────────────

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: str) -> str:
        pass

    @abstractmethod
    def load(self, path: str) -> "BaseModel":
        pass

    # ──────────────────────────────────────────
    # Shared Methods — available to all models
    # ──────────────────────────────────────────

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        self._assert_trained()

        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae  = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-9, y_true))) * 100
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2   = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return {
            "RMSE": round(float(rmse), 6),
            "MAE" : round(float(mae),  6),
            "MAPE": round(float(mape), 6),
            "R2"  : round(float(r2),   6),
        }

    def summary(self) -> str:
        """Returns a human-readable model summary string."""
        status = "Trained" if self.is_trained else "Not Trained"
        return f"Model: {self.name} | Status: {status}"

    def _assert_trained(self):
        """Raises an error if the model has not been trained yet."""
        if not self.is_trained:
            raise RuntimeError(
                f"Model '{self.name}' has not been trained yet. Call train() first."
            )