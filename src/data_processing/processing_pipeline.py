import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple

from src.data_processing.cleaner            import DataCleaner
from src.data_processing.splitter           import TimeSeriesSplitter
from src.data_processing.normalizer         import DataNormalizer
from src.data_processing.sequence_generator import SequenceGenerator

from config.config import (
    SEQUENCE_LENGTH,
    TARGET_COLUMN,
    TRAIN_RATIO,
    NORMALIZATION_METHOD,
    MISSING_VALUE_STRATEGY,
    OUTLIER_IQR_MULTIPLIER,
    PROCESSED_DATA_DIR,
)
from config.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedData:
    X_train:        np.ndarray
    y_train:        np.ndarray
    X_test:         np.ndarray
    y_test:         np.ndarray
    train_df:       pd.DataFrame
    test_df:        pd.DataFrame
    raw_train_df:   pd.DataFrame
    raw_test_df:    pd.DataFrame
    normalizer:     DataNormalizer
    ticker:         str
    target_column:  str
    sequence_length: int

    def summary(self) -> str:
        return (
            f"\n{'='*50}\n"
            f"  Processed Data Summary — [{self.ticker}]\n"
            f"{'='*50}\n"
            f"  Target Column   : {self.target_column}\n"
            f"  Sequence Length : {self.sequence_length}\n"
            f"  X_train shape   : {self.X_train.shape}\n"
            f"  y_train shape   : {self.y_train.shape}\n"
            f"  X_test shape    : {self.X_test.shape}\n"
            f"  y_test shape    : {self.y_test.shape}\n"
            f"  Train rows      : {len(self.train_df)}\n"
            f"  Test rows       : {len(self.test_df)}\n"
            f"{'='*50}"
        )


class ProcessingPipeline:
    def __init__(
        self,
        sequence_length:    int   = SEQUENCE_LENGTH,
        target_column:      str   = TARGET_COLUMN,
        train_ratio:        float = TRAIN_RATIO,
        normalization:      str   = NORMALIZATION_METHOD,
        missing_strategy:   str   = MISSING_VALUE_STRATEGY,
        iqr_multiplier:     float = OUTLIER_IQR_MULTIPLIER,
    ):
        self.sequence_length  = sequence_length
        self.target_column    = target_column
        self.train_ratio      = train_ratio
        self.normalization    = normalization
        self.missing_strategy = missing_strategy
        self.iqr_multiplier   = iqr_multiplier

    def run(self, df_raw: pd.DataFrame, ticker: str) -> ProcessedData:
        ticker = ticker.upper().strip()
        logger.info(f"=== Starting Processing Pipeline | Ticker: {ticker} ===")

        # ── Step 1: Clean ────────────────────
        cleaner  = DataCleaner(self.missing_strategy, self.iqr_multiplier)
        df_clean = cleaner.clean(df_raw)

        # ── Step 2: Split ────────────────────
        splitter     = TimeSeriesSplitter(self.train_ratio)
        split_result = splitter.split(df_clean)

        # ── Step 3: Normalize ────────────────
        normalizer = DataNormalizer(self.normalization)
        train_scaled, test_scaled = normalizer.fit_transform(
            split_result.train, split_result.test
        )
        normalizer.save(ticker)

        # ── Step 4: Generate Sequences ───────
        gen = SequenceGenerator(self.sequence_length, self.target_column)
        X_train, y_train = gen.generate(train_scaled)
        X_test,  y_test  = gen.generate(test_scaled)

        # ── Save processed DataFrames ────────
        self._save_processed(train_scaled, ticker, "train")
        self._save_processed(test_scaled,  ticker, "test")

        data = ProcessedData(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            train_df=train_scaled,
            test_df=test_scaled,
            raw_train_df=split_result.train,
            raw_test_df=split_result.test,
            normalizer=normalizer,
            ticker=ticker,
            target_column=self.target_column,
            sequence_length=self.sequence_length,
        )

        logger.info(f"=== Processing Complete | {ticker} ===")
        print(data.summary())
        return data

    def _save_processed(self, df: pd.DataFrame, ticker: str, split: str):
        """Saves scaled train/test DataFrames to disk."""
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_{split}_scaled.csv")
        df.to_csv(path)
        logger.info(f"Saved processed {split} data -> {path}")