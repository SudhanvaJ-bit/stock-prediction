import os
import pandas as pd
from typing import List, Optional

from src.feature_engineering.moving_averages    import MovingAverageFeatures
from src.feature_engineering.rolling_statistics import RollingStatisticsFeatures
from src.feature_engineering.returns_volatility import ReturnsVolatilityFeatures

from config.config import (
    MA_WINDOWS,
    BOLLINGER_WINDOW,
    BOLLINGER_STD_DEV,
    RSI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    VOLATILITY_WINDOW,
    FEATURES_DATA_DIR,
)
from config.logger import get_logger

logger = get_logger(__name__)


class FeaturePipeline:
    def __init__(
        self,
        ma_windows:        List[int] = MA_WINDOWS,
        bb_window:         int       = BOLLINGER_WINDOW,
        bb_std:            float     = BOLLINGER_STD_DEV,
        rsi_period:        int       = RSI_PERIOD,
        macd_fast:         int       = MACD_FAST,
        macd_slow:         int       = MACD_SLOW,
        macd_signal:       int       = MACD_SIGNAL,
        volatility_window: int       = VOLATILITY_WINDOW,
    ):
        self._ma = MovingAverageFeatures(windows=ma_windows)
        self._rs = RollingStatisticsFeatures(
            windows=ma_windows, bb_window=bb_window, bb_std=bb_std
        )
        self._rv = ReturnsVolatilityFeatures(
            volatility_window=volatility_window,
            rsi_period=rsi_period,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
        )

    def run(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        ticker = ticker.upper().strip()
        n_cols_before = df.shape[1]

        logger.info(f"=== Starting Feature Pipeline | Ticker: {ticker} | Columns before: {n_cols_before} ===")

        # ── Step 1: Moving Averages ───────────
        df = self._ma.transform(df)

        # ── Step 2: Rolling Statistics ────────
        df = self._rs.transform(df)

        # ── Step 3: Returns & Volatility ──────
        df = self._rv.transform(df)

        # ── Drop any remaining NaN rows ───────
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with NaN after feature engineering.")

        n_cols_after = df.shape[1]
        logger.info(
            f"=== Feature Pipeline Complete | "
            f"Columns: {n_cols_before} -> {n_cols_after} "
            f"(+{n_cols_after - n_cols_before} features) | "
            f"Rows: {len(df)} ==="
        )

        # ── Save feature DataFrame ─────────────
        self._save(df, ticker)

        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Returns list of all engineered feature column names (excludes OHLCV base columns)."""
        base_cols = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]
        return [col for col in df.columns if col not in base_cols]

    def _save(self, df: pd.DataFrame, ticker: str):
        """Saves feature-enriched DataFrame to disk."""
        os.makedirs(FEATURES_DATA_DIR, exist_ok=True)
        path = os.path.join(FEATURES_DATA_DIR, f"{ticker}_features.csv")
        df.to_csv(path)
        logger.info(f"Feature data saved -> {path}")