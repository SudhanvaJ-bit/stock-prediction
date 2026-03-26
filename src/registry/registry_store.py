import os
import json
from typing import Dict, List, Optional
from src.registry.model_entry import ModelEntry
from config.config import REGISTRY_DIR, REGISTRY_INDEX_FILE
from config.logger import get_logger

logger = get_logger(__name__)

class RegistryStore:
    def __init__(
        self,
        registry_dir:   str = REGISTRY_DIR,
        index_file:     str = REGISTRY_INDEX_FILE,
    ):
        self.registry_dir = registry_dir
        self.index_file   = index_file
        os.makedirs(self.registry_dir, exist_ok=True)
        self._index: Dict[str, dict] = self._load()

    # ──────────────────────────────────────────
    # Write Operations
    # ──────────────────────────────────────────

    def add(self, entry: ModelEntry) -> None:
        self._index[entry.entry_id] = entry.to_dict()
        self._save()
        logger.info(f"[Registry] Registered: {entry.entry_id}")

    def update_best(self, ticker: str, model_name: str, version: str) -> None:
        target_id = f"{ticker}__{model_name}__{version}"

        for entry_id, data in self._index.items():
            if data["ticker"] == ticker and data["model_name"] == model_name:
                data["is_best"] = (entry_id == target_id)

        self._save()
        logger.info(f"[Registry] Best updated -> {target_id}")

    def delete(self, entry_id: str) -> bool:
        if entry_id in self._index:
            del self._index[entry_id]
            self._save()
            logger.info(f"[Registry] Deleted entry: {entry_id}")
            return True
        logger.warning(f"[Registry] Entry not found for deletion: {entry_id}")
        return False

    # ──────────────────────────────────────────
    # Read Operations
    # ──────────────────────────────────────────

    def get(self, entry_id: str) -> Optional[ModelEntry]:
        """Retrieves a single entry by its ID."""
        data = self._index.get(entry_id)
        return ModelEntry.from_dict(data) if data else None

    def get_all(
        self,
        ticker:     Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> List[ModelEntry]:

        entries = [ModelEntry.from_dict(d) for d in self._index.values()]

        if ticker:
            entries = [e for e in entries if e.ticker == ticker.upper()]
        if model_name:
            entries = [e for e in entries if e.model_name == model_name]

        return sorted(entries, key=lambda e: e.registered_at, reverse=True)

    def get_best(
        self,
        ticker:     str,
        model_name: Optional[str] = None,
    ) -> Optional[ModelEntry]:
        entries = self.get_all(ticker=ticker, model_name=model_name)
        best    = [e for e in entries if e.is_best]
        return best[0] if best else None

    def count(self) -> int:
        """Returns total number of registered entries."""
        return len(self._index)

    # ──────────────────────────────────────────
    # Private: Persistence
    # ──────────────────────────────────────────

    def _load(self) -> Dict[str, dict]:
        """Loads the registry index from disk. Returns empty dict if not found."""
        if not os.path.exists(self.index_file):
            logger.info("[Registry] No existing index found. Starting fresh.")
            return {}
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"[Registry] Loaded index with {len(data)} entries.")
            return data
        except Exception as e:
            logger.error(f"[Registry] Failed to load index: {e}. Starting fresh.")
            return {}

    def _save(self) -> None:
        """Saves the current registry index to disk as formatted JSON."""
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self._index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[Registry] Failed to save index: {e}")