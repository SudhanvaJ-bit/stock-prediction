import os
import pickle
from typing import Optional
from src.registry.model_registry import ModelRegistry
from src.prediction_service.data_preparator import DataPreparator
from src.prediction_service.predictor import Predictor, PredictionResult
from src.prediction_service.result_saver import ResultSaver
from src.data_processing.normalizer import DataNormalizer
from config.config import PREDICTION_HORIZON, PROCESSED_DATA_DIR
from config.logger import get_logger

logger = get_logger(__name__)

class PredictionServiceError(Exception):
    pass

class PredictionService:
    def __init__(self):
        self._registry = ModelRegistry()
        self._saver    = ResultSaver()

    def predict(
        self,
        ticker:       str,
        horizon:      int = PREDICTION_HORIZON,
        model_name:   Optional[str] = None,
        save_results: bool = True,
    ) -> PredictionResult:
        ticker = ticker.upper().strip()
        logger.info(f"=== Prediction Service | ticker={ticker} | horizon={horizon} ===")

        # ── Step 1: Load model from registry ──────────
        try:
            model, entry = self._registry.load_best(
                ticker=ticker, model_name=model_name
            )
        except Exception as e:
            raise PredictionServiceError(
                f"Could not load model for '{ticker}' from registry.\n"
                f"Make sure you have run run_registry.py first.\nError: {e}"
            ) from e

        logger.info(f"Loaded model: {entry.entry_id}")

        # ── Step 2: Load fitted scaler ─────────────────
        normalizer = self._load_scaler(ticker, entry.scaler_path)

        # ── Step 3: Prepare live data ──────────────────
        try:
            preparator = DataPreparator(normalizer)
            X_input, df_scaled, last_date = preparator.prepare(ticker)
        except Exception as e:
            raise PredictionServiceError(
                f"Failed to prepare live data for '{ticker}': {e}"
            ) from e

        # ── Step 4: Run prediction ─────────────────────
        predictor = Predictor(model, normalizer, entry)

        if horizon == 1:
            result = predictor.predict_single(X_input, df_scaled, last_date, ticker)
        else:
            result = predictor.predict_multi(X_input, df_scaled, last_date, ticker, horizon)

        # ── Step 5: Save results ───────────────────────
        if save_results:
            paths = self._saver.save(result)
            logger.info(f"Results saved: {paths}")

        logger.info(f"=== Prediction Complete | {len(result.predictions)} day(s) ahead ===")
        print(result.summary())
        return result

    def predict_all_models(
        self,
        ticker:  str,
        horizon: int = 1,
    ) -> dict:

        ticker  = ticker.upper().strip()
        results = {}
        entries = self._registry.list_models(ticker=ticker)

        if not entries:
            raise PredictionServiceError(
                f"No models registered for ticker '{ticker}'. "
                f"Run run_registry.py first."
            )

        logger.info(f"Running predictions for all {len(entries)} registered models...")

        normalizer = self._load_scaler(ticker, entries[0].scaler_path)
        preparator = DataPreparator(normalizer)
        X_input, df_scaled, last_date = preparator.prepare(ticker)

        from src.registry.model_loader import ModelLoader
        loader = ModelLoader()

        for entry in entries:
            try:
                model     = loader.load(entry)
                predictor = Predictor(model, normalizer, entry)

                if horizon == 1:
                    result = predictor.predict_single(X_input, df_scaled, last_date, ticker)
                else:
                    result = predictor.predict_multi(X_input, df_scaled, last_date, ticker, horizon)

                results[entry.model_name] = result
            except Exception as e:
                logger.warning(f"Prediction failed for {entry.model_name}: {e}")

        return results

    # ──────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────

    def _load_scaler(self, ticker: str, scaler_path: str) -> DataNormalizer:
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join(
                PROCESSED_DATA_DIR, f"{ticker}_scaler_minmax.pkl"
            )

        if not os.path.exists(scaler_path):
            raise PredictionServiceError(
                f"Scaler not found at: {scaler_path}\n"
                f"Run run_processing.py or run_registry.py first."
            )

        normalizer = DataNormalizer()
        normalizer.load(ticker)
        return normalizer