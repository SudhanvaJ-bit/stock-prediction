import os
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Tuple

from src.models.base_model import BaseModel
from config.config import ARIMA_ORDER
from config.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")     # Suppress ARIMA convergence warnings


class ARIMAModel(BaseModel):
    def __init__(self, order: Tuple[int, int, int] = ARIMA_ORDER):
        super().__init__(name="ARIMA")
        self.order        = order
        self._history     = None      # Full price series seen so far
        self._predictions = None      # Cached test predictions

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        logger.info(f"[{self.name}] Storing training history | n={len(y_train)} | order={self.order}")
        self._history  = list(y_train.flatten())
        self.is_trained = True
        logger.info(f"[{self.name}] Ready for walk-forward prediction.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._assert_trained()

        from statsmodels.tsa.arima.model import ARIMA

        n_steps     = X.shape[0]
        history     = self._history.copy()
        predictions = []

        logger.info(f"[{self.name}] Walk-forward prediction | steps={n_steps}")

        for i in range(n_steps):
            try:
                model_fit = ARIMA(history, order=self.order).fit()
                yhat      = model_fit.forecast(steps=1)[0]
            except Exception as e:
                logger.warning(f"[{self.name}] Step {i}: ARIMA failed ({e}). Using last value.")
                yhat = history[-1]

            predictions.append(yhat)
            history.append(yhat)       # Append prediction as next history point

            if (i + 1) % 50 == 0:
                logger.info(f"[{self.name}] Progress: {i+1}/{n_steps} steps complete")

        self._predictions = np.array(predictions)
        logger.info(f"[{self.name}] Walk-forward complete | predictions={self._predictions.shape}")
        return self._predictions

    def save(self, path: str) -> str:
        """Saves ARIMA history and config as pickle."""
        self._assert_trained()
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, f"{self.name}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump({
                "order"   : self.order,
                "history" : self._history,
            }, f)
        logger.info(f"[{self.name}] Saved -> {filepath}")
        return filepath

    def load(self, path: str) -> "ARIMAModel":
        """Loads ARIMA config and history from pickle."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.order      = data["order"]
        self._history   = data["history"]
        self.is_trained = True
        logger.info(f"[{self.name}] Loaded <- {path}")
        return self