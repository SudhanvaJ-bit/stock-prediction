import numpy as np
import pandas as pd
from typing import Tuple

from src.data_processing.normalizer import DataNormalizer
from config.logger import get_logger

logger = get_logger(__name__)


class InverseTransformer:
    def __init__(
        self,
        normalizer:      DataNormalizer,
        n_features:      int,
        target_col_idx:  int,
    ):
        self.normalizer     = normalizer
        self.n_features     = n_features
        self.target_col_idx = target_col_idx

    def transform(
        self,
        y_true_scaled: np.ndarray,
        y_pred_scaled: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_true_real = self._inverse(y_true_scaled)
        y_pred_real = self._inverse(y_pred_scaled)

        logger.info(
            f"Inverse transform complete | "
            f"y_true range: ${y_true_real.min():.2f} - ${y_true_real.max():.2f} | "
            f"y_pred range: ${y_pred_real.min():.2f} - ${y_pred_real.max():.2f}"
        )
        return y_true_real, y_pred_real

    def _inverse(self, values: np.ndarray) -> np.ndarray:
        values = values.flatten().reshape(-1, 1)

        # Pad with zeros for non-target columns
        padded = np.zeros((len(values), self.n_features))
        padded[:, self.target_col_idx] = values[:, 0]

        # Inverse transform the full padded array
        inversed = self.normalizer._scaler.inverse_transform(padded)

        # Return only the target column
        return inversed[:, self.target_col_idx]