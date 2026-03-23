import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class MetricsReport:
    model_name:           str
    rmse:                 float
    mae:                  float
    mape:                 float
    r2:                   float
    directional_accuracy: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "Model"               : self.model_name,
            "RMSE ($)"            : round(self.rmse,                 4),
            "MAE ($)"             : round(self.mae,                  4),
            "MAPE (%)"            : round(self.mape,                 4),
            "R2"                  : round(self.r2,                   4),
            "Direction Acc (%)"   : round(self.directional_accuracy, 4),
        }

    def summary(self) -> str:
        return (
            f"  [{self.model_name}]\n"
            f"    RMSE               : ${self.rmse:.4f}\n"
            f"    MAE                : ${self.mae:.4f}\n"
            f"    MAPE               : {self.mape:.4f}%\n"
            f"    R2                 : {self.r2:.4f}\n"
            f"    Directional Acc    : {self.directional_accuracy:.2f}%\n"
        )


class MetricsCalculator:
    def compute(
        self,
        model_name: str,
        y_true:     np.ndarray,
        y_pred:     np.ndarray,
    ) -> MetricsReport:
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        rmse  = self._rmse(y_true, y_pred)
        mae   = self._mae(y_true, y_pred)
        mape  = self._mape(y_true, y_pred)
        r2    = self._r2(y_true, y_pred)
        d_acc = self._directional_accuracy(y_true, y_pred)

        return MetricsReport(
            model_name           = model_name,
            rmse                 = rmse,
            mae                  = mae,
            mape                 = mape,
            r2                   = r2,
            directional_accuracy = d_acc,
        )

    # ──────────────────────────────────────────
    # Individual Metric Functions
    # ──────────────────────────────────────────

    def _rmse(self, y_true, y_pred) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def _mae(self, y_true, y_pred) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    def _mape(self, y_true, y_pred) -> float:
        mask = y_true != 0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    def _r2(self, y_true, y_pred) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    def _directional_accuracy(self, y_true, y_pred) -> float:
        if len(y_true) < 2:
            return 0.0
        true_dir = np.diff(y_true) > 0
        pred_dir = np.diff(y_pred) > 0
        return float(np.mean(true_dir == pred_dir) * 100)