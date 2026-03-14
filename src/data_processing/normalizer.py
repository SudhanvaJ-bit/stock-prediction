import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple

from config.config import NORMALIZATION_METHOD, PROCESSED_DATA_DIR
from config.logger import get_logger

logger = get_logger(__name__)


class NormalizationError(Exception):
    """Raised when scaler is used before being fitted."""
    pass


class DataNormalizer:
    def __init__(self, method: str = NORMALIZATION_METHOD):
        self.method  = method.lower()
        self._scaler = None
        self._is_fitted = False
        self._columns   = None

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        self._columns = train_df.columns.tolist()
        self._scaler  = self._build_scaler()

        # Fit ONLY on training data
        scaled_train = self._scaler.fit_transform(train_df.values)
        scaled_test  = self._scaler.transform(test_df.values)

        self._is_fitted = True
        logger.info(
            f"Normalization applied | method='{self.method}' | "
            f"train={scaled_train.shape} | test={scaled_test.shape}"
        )

        train_scaled_df = pd.DataFrame(scaled_train, index=train_df.index, columns=self._columns)
        test_scaled_df  = pd.DataFrame(scaled_test,  index=test_df.index,  columns=self._columns)

        return train_scaled_df, test_scaled_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._assert_fitted()
        scaled = self._scaler.transform(df.values)
        return pd.DataFrame(scaled, index=df.index, columns=self._columns)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        self._assert_fitted()

        # If 1D or single-column, pad to full feature width for inverse_transform
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if data.shape[1] < len(self._columns):
            n_pad = len(self._columns) - data.shape[1]
            data  = np.hstack([data, np.zeros((data.shape[0], n_pad))])

        return self._scaler.inverse_transform(data)[:, :data.shape[1]]

    def save(self, ticker: str) -> str:
        """Saves the fitted scaler to disk."""
        self._assert_fitted()
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_scaler_{self.method}.pkl")
        with open(path, "wb") as f:
            pickle.dump({"scaler": self._scaler, "columns": self._columns, "method": self.method}, f)
        logger.info(f"Scaler saved -> {path}")
        return path

    def load(self, ticker: str) -> "DataNormalizer":
        """Loads a previously saved scaler from disk."""
        path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_scaler_{self.method}.pkl")
        if not os.path.exists(path):
            raise NormalizationError(f"No saved scaler found at: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._scaler    = data["scaler"]
        self._columns   = data["columns"]
        self.method     = data["method"]
        self._is_fitted = True
        logger.info(f"Scaler loaded <- {path}")
        return self

    def _build_scaler(self):
        """Instantiates the correct scaler based on configured method."""
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        if self.method == "minmax":
            return MinMaxScaler(feature_range=(0, 1))
        elif self.method == "standard":
            return StandardScaler()
        else:
            logger.warning(f"Unknown method '{self.method}'. Defaulting to MinMaxScaler.")
            return MinMaxScaler(feature_range=(0, 1))

    def _assert_fitted(self):
        if not self._is_fitted:
            raise NormalizationError("Scaler has not been fitted yet. Call fit_transform() first.")