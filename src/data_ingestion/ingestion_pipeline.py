import pandas as pd
from typing import Optional

from src.data_ingestion.fetcher   import StockDataFetcher, FetchError
from src.data_ingestion.validator import DataValidator
from src.data_ingestion.storage   import RawDataStorage, StorageError
from config.config import DEFAULT_START_DATE, DEFAULT_END_DATE, FETCH_INTERVAL
from config.logger import get_logger

logger = get_logger(__name__)


class IngestionPipelineError(Exception):
    """Raised when the ingestion pipeline cannot complete successfully."""
    pass


class IngestionPipeline:
    def __init__(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: Optional[str] = DEFAULT_END_DATE,
        interval: str = FETCH_INTERVAL,
        strict_validation: bool = True,
    ):
        self.start_date        = start_date
        self.end_date          = end_date
        self.interval          = interval
        self.strict_validation = strict_validation

        self._validator = DataValidator()
        self._storage   = RawDataStorage()

    def run(self, ticker: str) -> tuple:
        ticker = ticker.upper().strip()
        logger.info(f"═══ Starting Ingestion Pipeline | Ticker: {ticker} ═══")

        # ── Step 1: Fetch ────────────────────
        df = self._fetch(ticker)

        # ── Step 2: Validate ─────────────────
        self._validate(df)

        # ── Step 3: Store ────────────────────
        filepath = self._store(df, ticker)

        logger.info(f"═══ Ingestion Complete | {len(df)} rows | Saved: {filepath} ═══")
        return df, filepath

    def load_existing(self, filepath: str) -> pd.DataFrame:
        logger.info(f"Loading existing dataset: {filepath}")
        return self._storage.load(filepath)

    def list_available(self, ticker: Optional[str] = None):
        return self._storage.list_saved(ticker=ticker)
    def _fetch(self, ticker: str) -> pd.DataFrame:
        try:
            fetcher = StockDataFetcher(
                ticker=ticker,
                start_date=self.start_date,
                end_date=self.end_date,
                interval=self.interval,
            )
            return fetcher.fetch()
        except FetchError as e:
            raise IngestionPipelineError(f"[Fetch Failed] {e}") from e

    def _validate(self, df: pd.DataFrame):
        report = self._validator.validate(df)
        print(report.summary())

        if not report.is_valid and self.strict_validation:
            raise IngestionPipelineError(
                f"[Validation Failed] Pipeline halted.\n{report.summary()}"
            )

    def _store(self, df: pd.DataFrame, ticker: str) -> str:
        try:
            return self._storage.save(
                df,
                ticker=ticker,
                start=self.start_date,
                end=self.end_date,
            )
        except StorageError as e:
            raise IngestionPipelineError(f"[Storage Failed] {e}") from e