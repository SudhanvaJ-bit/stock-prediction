"""
src/registry/model_registry.py
--------------------------------
Public entry point for the Model Registry Layer.

The ModelRegistry ties together RegistryStore and ModelLoader to provide
a clean, high-level API for the rest of the system.

Responsibilities:
  - Register all models from a TrainingResult + EvaluationResult
  - Auto-version new model registrations (v1, v2, v3 ...)
  - Track which model is best per ticker
  - Load the best model on demand for the prediction service
  - List, inspect, and compare all registered models
"""

import os
import shutil
from datetime import datetime
from typing import Optional, List

from src.registry.model_entry    import ModelEntry
from src.registry.registry_store import RegistryStore
from src.registry.model_loader   import ModelLoader, ModelLoadError
from src.models.training_pipeline  import TrainingResult
from src.evaluation.evaluation_pipeline import EvaluationResult
from src.models.base_model         import BaseModel

from config.config import (
    REGISTRY_DIR,
    REGISTRY_INDEX_FILE,
    BEST_MODEL_METRIC,
    PROCESSED_DATA_DIR,
)
from config.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    High-level interface for registering, retrieving, and managing ML models.

    Usage:
        registry = ModelRegistry()

        # Register all models after training + evaluation
        registry.register_all(training_result, eval_result)

        # Load best model for prediction
        model, entry = registry.load_best(ticker="AAPL")

        # List all registered models
        registry.list_models(ticker="AAPL")
    """

    def __init__(self):
        self._store  = RegistryStore()
        self._loader = ModelLoader()

    # ──────────────────────────────────────────
    # Register
    # ──────────────────────────────────────────

    def register_all(
        self,
        training_result: TrainingResult,
        eval_result:     EvaluationResult,
    ) -> List[ModelEntry]:
        """
        Registers all models from a completed training + evaluation run.

        For each model:
          1. Determines the next version number
          2. Copies model file to versioned registry folder
          3. Creates a ModelEntry with full metadata
          4. Saves to registry index
          5. Marks the best model

        Args:
            training_result (TrainingResult)  : Output from TrainingPipeline.
            eval_result     (EvaluationResult): Output from EvaluationPipeline.

        Returns:
            List[ModelEntry]: All newly registered entries.
        """
        ticker  = training_result.ticker
        entries = []

        logger.info(f"=== Registering models for [{ticker}] ===")

        # Scaler path — shared across all models for this ticker
        scaler_path = os.path.join(
            PROCESSED_DATA_DIR, f"{ticker}_scaler_minmax.pkl"
        )

        for model_name, model_result in training_result.results.items():

            # ── Get metrics from evaluation layer ──
            metrics = {}
            if model_name in eval_result.reports:
                metrics = eval_result.reports[model_name].to_dict()

            # ── Determine next version ─────────────
            version = self._next_version(ticker, model_name)

            # ── Copy model file to registry ────────
            src_path      = model_result.model.model_path if hasattr(
                model_result.model, "model_path"
            ) else self._find_model_file(ticker, model_name)

            registry_path = self._copy_to_registry(
                src_path, ticker, model_name, version
            )

            # ── Create and store entry ─────────────
            entry = ModelEntry(
                model_name    = model_name,
                version       = version,
                ticker        = ticker,
                registered_at = datetime.now().isoformat(timespec="seconds"),
                model_path    = registry_path,
                scaler_path   = scaler_path,
                metrics       = metrics,
                is_best       = False,
                notes         = f"Auto-registered | metric={BEST_MODEL_METRIC}",
            )

            self._store.add(entry)
            entries.append(entry)

        # ── Mark best model ────────────────────────
        if eval_result.best_model and eval_result.best_model in training_result.results:
            self._store.update_best(ticker, eval_result.best_model, self._next_version(
                ticker, eval_result.best_model, peek=True
            ))

        logger.info(f"=== Registry complete | {len(entries)} models registered ===")
        self.print_registry(ticker)
        return entries

    # ──────────────────────────────────────────
    # Load
    # ──────────────────────────────────────────

    def load_best(
        self,
        ticker:     str,
        model_name: Optional[str] = None,
    ):
        """
        Loads the best registered model for a given ticker.

        Args:
            ticker     : Stock ticker (e.g. 'AAPL').
            model_name : If provided, loads best of that specific model type.

        Returns:
            Tuple[BaseModel, ModelEntry]: Loaded model and its registry entry.

        Raises:
            ModelLoadError: If no best model is found.
        """
        # If a specific model is requested, load its latest version directly
        # (no need for is_best flag — user explicitly chose this model)
        if model_name:
            all_entries = self._store.get_all(ticker=ticker.upper(), model_name=model_name)
            entry = all_entries[0] if all_entries else None
        else:
            entry = self._store.get_best(ticker=ticker.upper(), model_name=None)

        if not entry:
            raise ModelLoadError(
                f"No model found in registry for ticker='{ticker}'"
                + (f", model='{model_name}'" if model_name else "")
            )

        logger.info(f"[Registry] Loading model: {entry.entry_id}")
        model = self._loader.load(entry)
        return model, entry

    def load_by_id(self, entry_id: str):
        """
        Loads a specific model by its entry ID.

        Args:
            entry_id (str): Registry entry ID (e.g. 'AAPL__LSTM__v2')

        Returns:
            Tuple[BaseModel, ModelEntry]
        """
        entry = self._store.get(entry_id)
        if not entry:
            raise ModelLoadError(f"No registry entry found for ID: '{entry_id}'")
        model = self._loader.load(entry)
        return model, entry

    # ──────────────────────────────────────────
    # Inspect
    # ──────────────────────────────────────────

    def list_models(self, ticker: Optional[str] = None) -> List[ModelEntry]:
        """Lists all registered models, optionally filtered by ticker."""
        return self._store.get_all(ticker=ticker)

    def print_registry(self, ticker: Optional[str] = None) -> None:
        """Prints a formatted summary of all registered models."""
        entries = self._store.get_all(ticker=ticker)

        label = f"[{ticker}]" if ticker else "[All Tickers]"
        print(f"\n{'='*60}")
        print(f"  Model Registry — {label}  ({len(entries)} entries)")
        print(f"{'='*60}")

        if not entries:
            print("  No models registered yet.")
        else:
            for entry in entries:
                print(entry.summary())
                print()

        print(f"{'='*60}\n")

    # ──────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────

    def _next_version(
        self,
        ticker:     str,
        model_name: str,
        peek:       bool = False,
    ) -> str:
        """
        Returns the next version string for a ticker + model combination.
        peek=True returns the CURRENT latest version without incrementing.
        """
        existing = self._store.get_all(ticker=ticker, model_name=model_name)
        n = len(existing)
        return f"v{n}" if peek else f"v{n + 1}"

    def _find_model_file(self, ticker: str, model_name: str) -> str:
        """Fallback: finds the model file in the models/ directory."""
        from config.config import MODELS_DIR
        ext = ".keras" if model_name == "LSTM" else ".pkl"
        return os.path.join(MODELS_DIR, ticker, f"{model_name}{ext}")

    def _copy_to_registry(
        self,
        src_path:   str,
        ticker:     str,
        model_name: str,
        version:    str,
    ) -> str:
        """
        Copies a model file to a versioned location inside the registry folder.

        Destination: registry/{TICKER}/{ModelName}/{version}/{filename}
        """
        if not os.path.exists(src_path):
            logger.warning(f"[Registry] Source file not found: {src_path}. Storing path as-is.")
            return src_path

        filename = os.path.basename(src_path)
        dest_dir = os.path.join(self.registry_dir, ticker, model_name, version)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)

        shutil.copy2(src_path, dest_path)
        logger.info(f"[Registry] Copied {model_name} -> {dest_path}")
        return dest_path

    @property
    def registry_dir(self) -> str:
        return self._store.registry_dir