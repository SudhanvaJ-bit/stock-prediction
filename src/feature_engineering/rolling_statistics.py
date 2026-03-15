import pandas as pd
from typing import List

from config.config import MA_WINDOWS, BOLLINGER_WINDOW, BOLLINGER_STD_DEV
from config.logger import get_logger

logger = get_logger(__name__)


class RollingStatisticsFeatures:
    def __init__(
        self,
        windows: List[int] = MA_WINDOWS,
        bb_window: int = BOLLINGER_WINDOW,
        bb_std: float = BOLLINGER_STD_DEV,
    ):
        self.windows   = windows
        self.bb_window = bb_window
        self.bb_std    = bb_std

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate(df)
        df = df.copy()

        for w in self.windows:
            roll = df["Close"].rolling(window=w, min_periods=1)

            df[f"Rolling_Mean_{w}"]  = roll.mean()
            df[f"Rolling_Std_{w}"]   = roll.std().fillna(0)
            df[f"Rolling_Min_{w}"]   = roll.min()
            df[f"Rolling_Max_{w}"]   = roll.max()
            df[f"Rolling_Range_{w}"] = df[f"Rolling_Max_{w}"] - df[f"Rolling_Min_{w}"]

        # ── Bollinger Bands ───────────────────
        bb_roll  = df["Close"].rolling(window=self.bb_window, min_periods=1)
        bb_mean  = bb_roll.mean()
        bb_std   = bb_roll.std().fillna(0)

        df[f"BB_Upper_{self.bb_window}"]    = bb_mean + (self.bb_std * bb_std)
        df[f"BB_Lower_{self.bb_window}"]    = bb_mean - (self.bb_std * bb_std)
        df[f"BB_Width_{self.bb_window}"]    = df[f"BB_Upper_{self.bb_window}"] - \
                                               df[f"BB_Lower_{self.bb_window}"]

        # BB Position: 0 = at lower band, 1 = at upper band
        band_range = df[f"BB_Width_{self.bb_window}"].replace(0, 1e-9)
        df[f"BB_Position_{self.bb_window}"] = (
            (df["Close"] - df[f"BB_Lower_{self.bb_window}"]) / band_range
        ).clip(0, 1)

        logger.info(
            f"Rolling statistics features added | windows={self.windows} | "
            f"Bollinger window={self.bb_window}"
        )
        return df

    def _validate(self, df: pd.DataFrame):
        if "Close" not in df.columns:
            raise ValueError("DataFrame must contain a 'Close' column.")