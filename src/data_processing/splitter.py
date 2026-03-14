import pandas as pd
from typing import Tuple

from config.config import TRAIN_RATIO
from config.logger import get_logger

logger = get_logger(__name__)


class SplitResult:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train      = train
        self.test       = test
        self.split_date = test.index[0]
        self.train_pct  = len(train) / (len(train) + len(test)) * 100

    def summary(self) -> str:
        return (
            f"── Train/Test Split ──\n"
            f"  Train : {len(self.train)} rows  "
            f"({self.train.index[0].date()} → {self.train.index[-1].date()})"
            f"  [{self.train_pct:.1f}%]\n"
            f"  Test  : {len(self.test)} rows   "
            f"({self.test.index[0].date()} → {self.test.index[-1].date()})"
            f"  [{100 - self.train_pct:.1f}%]\n"
            f"  Split Date: {self.split_date.date()}"
        )


class TimeSeriesSplitter:
    def __init__(self, train_ratio: float = TRAIN_RATIO):
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
        self.train_ratio = train_ratio

    def split(self, df: pd.DataFrame) -> SplitResult:
        df = df.sort_index()  # Guarantee chronological order

        split_idx = int(len(df) * self.train_ratio)

        if split_idx == 0 or split_idx >= len(df):
            raise ValueError(
                f"Invalid split: train_ratio={self.train_ratio} produces "
                f"empty train or test set for {len(df)} rows."
            )

        train = df.iloc[:split_idx]
        test  = df.iloc[split_idx:]

        result = SplitResult(train, test)
        logger.info(
            f"Split complete | train={len(train)} rows | test={len(test)} rows | "
            f"split date={result.split_date.date()}"
        )
        print(result.summary())
        return result