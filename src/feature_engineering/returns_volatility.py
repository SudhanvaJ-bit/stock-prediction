import numpy as np
import pandas as pd

from config.config import (
    VOLATILITY_WINDOW,
    RSI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
)
from config.logger import get_logger

logger = get_logger(__name__)


class ReturnsVolatilityFeatures:
    def __init__(
        self,
        volatility_window: int = VOLATILITY_WINDOW,
        rsi_period:        int = RSI_PERIOD,
        macd_fast:         int = MACD_FAST,
        macd_slow:         int = MACD_SLOW,
        macd_signal:       int = MACD_SIGNAL,
    ):
        self.volatility_window = volatility_window
        self.rsi_period        = rsi_period
        self.macd_fast         = macd_fast
        self.macd_slow         = macd_slow
        self.macd_signal       = macd_signal

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate(df)
        df = df.copy()

        df = self._add_returns(df)
        df = self._add_volatility(df)
        df = self._add_intraday(df)
        df = self._add_rsi(df)
        df = self._add_macd(df)

        logger.info(
            "Returns & volatility features added: "
            "Daily_Return, Log_Return, Cumulative_Return, "
            f"Volatility_{self.volatility_window}, High_Low_Range, "
            f"Close_Open_Change, RSI_{self.rsi_period}, MACD, MACD_Signal, MACD_Histogram"
        )
        return df

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Daily, log, and cumulative returns."""
        df["Daily_Return"]      = df["Close"].pct_change().fillna(0)
        df["Log_Return"]        = np.log(df["Close"] / df["Close"].shift(1)).fillna(0)
        df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1
        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling volatility of log returns (annualised-style)."""
        df[f"Volatility_{self.volatility_window}"] = (
            df["Log_Return"]
            .rolling(window=self.volatility_window, min_periods=1)
            .std()
            .fillna(0)
        )
        return df

    def _add_intraday(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intraday price range and directional move."""
        df["High_Low_Range"]    = (df["High"] - df["Low"]) / df["Close"]
        df["Close_Open_Change"] = (df["Close"] - df["Open"]) / df["Open"]
        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        delta  = df["Close"].diff()
        gain   = delta.clip(lower=0)
        loss   = -delta.clip(upper=0)

        avg_gain = gain.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        avg_loss = loss.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()

        rs  = avg_gain / avg_loss.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))

        df[f"RSI_{self.rsi_period}"] = rsi.fillna(50)  # 50 = neutral for initial NaNs
        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = df["Close"].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=self.macd_slow, adjust=False).mean()

        df["MACD"]           = ema_fast - ema_slow
        df["MACD_Signal"]    = df["MACD"].ewm(span=self.macd_signal, adjust=False).mean()
        df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

        return df

    def _validate(self, df: pd.DataFrame):
        required = ["Open", "High", "Low", "Close"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")