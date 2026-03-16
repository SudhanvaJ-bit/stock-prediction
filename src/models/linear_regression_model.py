import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

from src.models.base_model import BaseModel
from config.logger import get_logger

logger = get_logger(__name__)


class LinearRegressionModel(BaseModel):
    def __init__(self, fit_intercept: bool = True):
        super().__init__(name="LinearRegression")
        self.fit_intercept = fit_intercept

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_flat = self._flatten(X_train)

        logger.info(f"[{self.name}] Training | X={X_flat.shape} | y={y_train.shape}")

        self.model = LinearRegression(fit_intercept=self.fit_intercept)
        self.model.fit(X_flat, y_train)
        self.is_trained = True

        train_preds  = self.model.predict(X_flat)
        train_metrics = self.evaluate(y_train, train_preds)
        logger.info(f"[{self.name}] Train metrics: {train_metrics}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._assert_trained()
        X_flat = self._flatten(X)
        return self.model.predict(X_flat)

    def save(self, path: str) -> str:
        """Saves the model as a pickle file."""
        self._assert_trained()
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, f"{self.name}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump({"model": self.model, "fit_intercept": self.fit_intercept}, f)
        logger.info(f"[{self.name}] Saved -> {filepath}")
        return filepath

    def load(self, path: str) -> "LinearRegressionModel":
        """Loads model from pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model         = data["model"]
        self.fit_intercept = data["fit_intercept"]
        self.is_trained    = True
        logger.info(f"[{self.name}] Loaded <- {path}")
        return self

    # ──────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """Flattens 3D sequences to 2D for sklearn compatibility."""
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X