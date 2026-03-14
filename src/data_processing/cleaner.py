import pandas as pd
import numpy as np
from typing import List

from config.config import (
    MISSING_VALUE_STRATEGY,
    OUTLIER_IQR_MULTIPLIER,
    REQUIRED_COLUMNS,
)
from config.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    def __init__(
        self,
        missing_strategy: str = MISSING_VALUE_STRATEGY,
        iqr_multiplier: float = OUTLIER_IQR_MULTIPLIER,
    ):
        self.missing_strategy = missing_strategy
        self.iqr_multiplier   = iqr_multiplier

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starting data cleaning. Input shape: {df.shape}")

        df = df.copy()

        # Step 1: Drop non-numeric metadata columns
        df = self._drop_metadata_columns(df)

        # Step 2: Handle missing values
        df = self._handle_missing(df)

        # Step 3: Cap outliers
        df = self._handle_outliers(df)

        # Step 4: Sort chronologically
        df = df.sort_index()

        # Step 5: Strip timezone info if present
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        logger.info(f"Cleaning complete. Output shape: {df.shape}")
        return df

    def _drop_metadata_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops non-numeric columns like 'Ticker' that aren't features."""
        cols_to_drop = [
            col for col in df.columns
            if not pd.api.types.is_numeric_dtype(df[col])
        ]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Dropped non-numeric columns: {cols_to_drop}")
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the configured missing value strategy."""
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            logger.info("No missing values found.")
            return df

        logger.info(f"Found {missing_count} missing values. Strategy: '{self.missing_strategy}'")

        if self.missing_strategy == "ffill":
            df = df.ffill().bfill()           # bfill as fallback for leading NaNs
        elif self.missing_strategy == "bfill":
            df = df.bfill().ffill()
        elif self.missing_strategy == "interpolate":
            df = df.interpolate(method="time").ffill().bfill()
        elif self.missing_strategy == "drop":
            df = df.dropna()
        else:
            logger.warning(f"Unknown strategy '{self.missing_strategy}'. Defaulting to ffill.")
            df = df.ffill().bfill()

        remaining = df.isnull().sum().sum()
        logger.info(f"Missing values after cleaning: {remaining}")
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = [col for col in REQUIRED_COLUMNS if col in df.columns]
        total_capped = 0

        for col in numeric_cols:
            Q1  = df[col].quantile(0.25)
            Q3  = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - self.iqr_multiplier * IQR
            upper = Q3 + self.iqr_multiplier * IQR

            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                total_capped += outliers
                logger.info(f"  Column '{col}': {outliers} outlier(s) capped to [{lower:.2f}, {upper:.2f}]")

        if total_capped == 0:
            logger.info("No outliers detected.")

        return df