import pandas as pd
from typing import List

from config.config import MA_WINDOWS
from config.logger import get_logger

logger = get_logger(__name__)


class MovingAverageFeatures:
    def __init__(self, windows: List[int] = MA_WINDOWS):
        self.windows = windows

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate(df)
        df = df.copy()

        for w in self.windows:
            # Simple Moving Average
            df[f"SMA_{w}"] = df["Close"].rolling(window=w, min_periods=1).mean()

            # Exponential Moving Average
            df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()

            # Price distance from SMA (momentum signal)
            df[f"Price_vs_SMA_{w}"] = (df["Close"] - df[f"SMA_{w}"]) / df[f"SMA_{w}"] * 100

        cols_added = [f"SMA_{w}" for w in self.windows] + \
                     [f"EMA_{w}" for w in self.windows] + \
                     [f"Price_vs_SMA_{w}" for w in self.windows]

        logger.info(f"Moving average features added: {cols_added}")
        return df

    def _validate(self, df: pd.DataFrame):
        if "Close" not in df.columns:
            raise ValueError("DataFrame must contain a 'Close' column.")