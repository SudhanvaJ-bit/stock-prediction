import os
import numpy as np
from typing import List, Optional

from src.models.base_model import BaseModel
from config.config import (
    LSTM_UNITS,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    LSTM_PATIENCE,
    LSTM_LEARNING_RATE,
)
from config.logger import get_logger

logger = get_logger(__name__)


class LSTMModel(BaseModel):
    def __init__(
        self,
        units:          List[int] = LSTM_UNITS,
        dropout:        float     = LSTM_DROPOUT,
        epochs:         int       = LSTM_EPOCHS,
        batch_size:     int       = LSTM_BATCH_SIZE,
        patience:       int       = LSTM_PATIENCE,
        learning_rate:  float     = LSTM_LEARNING_RATE,
    ):
        super().__init__(name="LSTM")
        self.units         = units
        self.dropout       = dropout
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.patience      = patience
        self.learning_rate = learning_rate
        self.history_log   = None       # Keras training history

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.optimizers import Adam

        n_samples, seq_len, n_features = X_train.shape
        logger.info(
            f"[{self.name}] Building model | "
            f"input=({seq_len}, {n_features}) | units={self.units} | "
            f"dropout={self.dropout} | epochs={self.epochs}"
        )

        # ── Build Architecture ────────────────
        self.model = Sequential([
            Input(shape=(seq_len, n_features)),

            LSTM(self.units[0], return_sequences=True),
            Dropout(self.dropout),

            LSTM(self.units[1], return_sequences=False),
            Dropout(self.dropout),

            Dense(32, activation="relu"),
            Dense(1),
        ], name="LSTM_StockPredictor")

        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mean_squared_error",
            metrics=["mae"],
        )

        self.model.summary(print_fn=lambda x: logger.info(x))

        # ── Callbacks ────────────────────────
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        # ── Train ─────────────────────────────
        logger.info(f"[{self.name}] Starting training...")
        self.history_log = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1,
        )

        self.is_trained = True

        # Log final metrics
        final_loss = self.history_log.history["loss"][-1]
        final_val  = self.history_log.history["val_loss"][-1]
        epochs_run = len(self.history_log.history["loss"])
        logger.info(
            f"[{self.name}] Training complete | "
            f"epochs={epochs_run} | loss={final_loss:.6f} | val_loss={final_val:.6f}"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._assert_trained()
        preds = self.model.predict(X, verbose=0)
        return preds.flatten()

    def save(self, path: str) -> str:
        """Saves the Keras model in native .keras format."""
        self._assert_trained()
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, f"{self.name}.keras")
        self.model.save(filepath)
        logger.info(f"[{self.name}] Saved -> {filepath}")
        return filepath

    def load(self, path: str) -> "LSTMModel":
        """Loads a saved Keras model."""
        from tensorflow.keras.models import load_model
        self.model      = load_model(path)
        self.is_trained = True
        logger.info(f"[{self.name}] Loaded <- {path}")
        return self