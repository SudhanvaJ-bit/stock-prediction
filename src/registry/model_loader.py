import os
from src.registry.model_entry import ModelEntry
from src.models.base_model import BaseModel
from config.logger import get_logger

logger = get_logger(__name__)

class ModelLoadError(Exception):
    """Raised when a model cannot be loaded from disk."""
    pass

class ModelLoader:
    def load(self, entry: ModelEntry) -> BaseModel:
        if not os.path.exists(entry.model_path):
            raise ModelLoadError(
                f"Model file not found: '{entry.model_path}'\n"
                f"Entry: {entry.entry_id}"
            )

        logger.info(f"[ModelLoader] Loading {entry.model_name} v{entry.version} "
                    f"for [{entry.ticker}] from {entry.model_path}")

        name = entry.model_name.lower()

        if name == "linearregression":
            return self._load_linear_regression(entry.model_path)
        elif name == "arima":
            return self._load_arima(entry.model_path)
        elif name == "lstm":
            return self._load_lstm(entry.model_path)
        else:
            raise ModelLoadError(f"Unknown model type: '{entry.model_name}'")

    # ──────────────────────────────────────────
    # Private Loaders per Model Type
    # ──────────────────────────────────────────

    def _load_linear_regression(self, path: str) -> BaseModel:
        from src.models.linear_regression_model import LinearRegressionModel
        model = LinearRegressionModel()
        model.load(path)
        logger.info("[ModelLoader] LinearRegression loaded successfully.")
        return model

    def _load_arima(self, path: str) -> BaseModel:
        from src.models.arima_model import ARIMAModel
        model = ARIMAModel()
        model.load(path)
        logger.info("[ModelLoader] ARIMA loaded successfully.")
        return model

    def _load_lstm(self, path: str) -> BaseModel:
        from src.models.lstm_model import LSTMModel
        model = LSTMModel()
        model.load(path)
        logger.info("[ModelLoader] LSTM loaded successfully.")
        return model