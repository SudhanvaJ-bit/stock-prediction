import yfinance as yf
import pandas as pd
from datetime import date
from typing import Optional

from config.config import (
    DEFAULT_TICKER,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    FETCH_INTERVAL,
)
from config.logger import get_logger

logger = get_logger(__name__)


class FetchError(Exception):
    pass


class StockDataFetcher:
    def __init__(
        self,
        ticker: str = DEFAULT_TICKER,
        start_date: str = DEFAULT_START_DATE,
        end_date: Optional[str] = DEFAULT_END_DATE,
        interval: str = FETCH_INTERVAL,
    ):
        self.ticker     = ticker.upper().strip()
        self.start_date = start_date
        self.end_date   = end_date or str(date.today())
        self.interval   = interval

    def fetch(self) -> pd.DataFrame:
        logger.info(
            f"Fetching [{self.ticker}] | {self.start_date} → {self.end_date} | interval={self.interval}"
        )

        try:
            raw_df = yf.download(
                tickers=self.ticker,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                auto_adjust=False,
                progress=False,
            )
        except Exception as e:
            raise FetchError(f"Yahoo Finance API call failed for '{self.ticker}': {e}") from e

        if raw_df is None or raw_df.empty:
            raise FetchError(
                f"No data returned for ticker '{self.ticker}' "
                f"between {self.start_date} and {self.end_date}."
            )

        df = self._clean_columns(raw_df)

        logger.info(f"Fetched {len(df)} rows for [{self.ticker}].")
        return df

    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Standardize column names
        df.columns = [col.strip().title().replace(" ", "_") for col in df.columns]

        # Ensure DatetimeIndex
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        # Add ticker as metadata column for traceability
        df["Ticker"] = self.ticker

        return df