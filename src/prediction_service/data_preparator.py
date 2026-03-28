import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Tuple
from src.data_ingestion.fetcher import StockDataFetcher
from src.data_processing.cleaner import DataCleaner
from src.data_processing.normalizer import DataNormalizer
from src.feature_engineering.feature_pipeline import FeaturePipeline
from src.data_processing.sequence_generator import SequenceGenerator

from config.config import SEQUENCE_LENGTH, TARGET_COLUMN
from config.logger import get_logger

logger = get_logger(__name__)


class DataPreparator:
    def __init__(
        self,
        normalizer: DataNormalizer,
        sequence_length: int = SEQUENCE_LENGTH,
        target_column: str = TARGET_COLUMN,
    ):

        self.normalizer = normalizer
        self.sequence_length = sequence_length
        self.target_column   = target_column

    def prepare(self, ticker: str) -> Tuple[np.ndarray, pd.DataFrame, pd.Timestamp]:
        ticker = ticker.upper().strip()
        logger.info(f"[DataPreparator] Preparing live data for [{ticker}]")

        # ── Step 1: Fetch recent data ──────────────────
        # Fetch extra days to account for weekends + holidays
        lookback_days = self.sequence_length * 2
        start_date    = (date.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        fetcher = StockDataFetcher(ticker=ticker, start_date=start_date)
        df_raw  = fetcher.fetch()

        if len(df_raw) < self.sequence_length:
            raise ValueError(
                f"Only {len(df_raw)} trading days available, "
                f"need at least {self.sequence_length}."
            )

        # ── Step 2: Clean ──────────────────────────────
        cleaner  = DataCleaner()
        df_clean = cleaner.clean(df_raw)

        # ── Step 3: Feature Engineering ────────────────
        feat_pipeline = FeaturePipeline()
        df_features   = feat_pipeline.run(df_clean, ticker=ticker)

        # ── Step 4: Scale using FITTED scaler ──────────
        # Only transform — never fit on live data
        df_scaled = self.normalizer.transform(df_features)

        # ── Step 5: Extract last sequence window ────────
        gen     = SequenceGenerator(self.sequence_length, self.target_column)
        X_input = gen.generate_single(df_scaled)     # shape: (1, seq_len, n_features)

        last_date = df_scaled.index[-1]

        logger.info(
            f"[DataPreparator] Prepared | X={X_input.shape} | "
            f"last_date={last_date.date()} | features={df_scaled.shape[1]}"
        )

        return X_input, df_scaled, last_date