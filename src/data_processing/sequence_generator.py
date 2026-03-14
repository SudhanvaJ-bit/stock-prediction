import numpy as np
import pandas as pd
from typing import Tuple

from config.config import SEQUENCE_LENGTH, TARGET_COLUMN
from config.logger import get_logger

logger = get_logger(__name__)


class SequenceGenerator:
    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        target_column: str = TARGET_COLUMN,
    ):
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")

        self.sequence_length = sequence_length
        self.target_column   = target_column

    def generate(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        if len(df) <= self.sequence_length:
            raise ValueError(
                f"DataFrame has {len(df)} rows but sequence_length={self.sequence_length}. "
                f"Need at least {self.sequence_length + 1} rows."
            )

        values       = df.values                                           # shape: (n_rows, n_features)
        target_idx   = df.columns.tolist().index(self.target_column)       # index of Close column

        X_list, y_list = [], []

        for i in range(self.sequence_length, len(values)):
            X_list.append(values[i - self.sequence_length : i, :])        # all features, past window
            y_list.append(values[i, target_idx])                          # target: Close at step i

        X = np.array(X_list)   # (n_samples, seq_len, n_features)
        y = np.array(y_list)   # (n_samples,)

        logger.info(
            f"Sequences generated | X={X.shape} | y={y.shape} | "
            f"window={self.sequence_length} | target='{self.target_column}'"
        )
        return X, y

    def generate_single(self, df: pd.DataFrame) -> np.ndarray:
        if len(df) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} rows for inference. Got {len(df)}."
            )

        last_window = df.values[-self.sequence_length:]                    # (seq_len, n_features)
        return last_window.reshape(1, self.sequence_length, -1)            # (1, seq_len, n_features)