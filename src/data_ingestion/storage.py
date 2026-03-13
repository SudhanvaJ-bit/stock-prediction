import os
import pandas as pd
from datetime import datetime
from typing import Optional, List

from config.config import RAW_DATA_DIR
from config.logger import get_logger

logger = get_logger(__name__)


class StorageError(Exception):
    """Raised when a storage read/write operation fails."""
    pass


class RawDataStorage:
    def __init__(self, storage_dir: str = RAW_DATA_DIR):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def save(
        self,
        df: pd.DataFrame,
        ticker: str,
        start: str,
        end: Optional[str] = None,
    ) -> str:
        end = end or datetime.today().strftime("%Y-%m-%d")
        filename = self._build_filename(ticker, start, end)
        filepath = os.path.join(self.storage_dir, filename)

        try:
            df.to_csv(filepath, index=True)
            logger.info(f"Raw data saved → {filepath}  ({len(df)} rows)")
            return filepath
        except Exception as e:
            raise StorageError(f"Failed to save data to '{filepath}': {e}") from e

    def load(self, filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise StorageError(f"File not found: '{filepath}'")

        try:
            df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
            logger.info(f"Raw data loaded ← {filepath}  ({len(df)} rows)")
            return df
        except Exception as e:
            raise StorageError(f"Failed to load data from '{filepath}': {e}") from e

    def list_saved(self, ticker: Optional[str] = None) -> List[str]:
        all_files = [
            os.path.join(self.storage_dir, f)
            for f in os.listdir(self.storage_dir)
            if f.endswith(".csv")
        ]

        if ticker:
            all_files = [f for f in all_files if os.path.basename(f).startswith(ticker.upper())]

        logger.info(f"Found {len(all_files)} saved raw file(s)" +
                    (f" for ticker '{ticker.upper()}'" if ticker else "") + ".")
        return sorted(all_files)

    def _build_filename(self, ticker: str, start: str, end: str) -> str:
        safe_start = start.replace("/", "-")
        safe_end   = end.replace("/", "-")
        return f"{ticker.upper()}_{safe_start}_{safe_end}.csv"