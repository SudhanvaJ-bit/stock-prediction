import os
import json
import csv
from datetime import datetime
from src.prediction_service.predictor import PredictionResult
from config.config import PREDICTIONS_DIR
from config.logger import get_logger

logger = get_logger(__name__)

class ResultSaver:
    def __init__(self, output_dir: str = PREDICTIONS_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, result: PredictionResult) -> dict:
        csv_path  = self._save_csv(result)
        json_path = self._save_json(result)
        return {"csv": csv_path, "json": json_path}

    # ──────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────

    def _save_csv(self, result: PredictionResult) -> str:
        path = os.path.join(self.output_dir, f"{result.ticker}_predictions.csv")

        rows = [
            {
                "generated_at"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker"           : result.ticker,
                "model"            : result.model_name,
                "version"          : result.model_version,
                "last_known_date"  : result.last_known_date.date(),
                "last_known_price" : result.last_known_price,
                "predicted_date"   : str(pred_date),
                "predicted_close"  : round(pred_price, 4),
            }
            for pred_date, pred_price in result.predictions
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"[ResultSaver] Predictions CSV saved -> {path}")
        return path

    def _save_json(self, result: PredictionResult) -> str:
        path = os.path.join(self.output_dir, f"{result.ticker}_predictions.json")

        payload = result.to_dict()
        payload["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(f"[ResultSaver] Predictions JSON saved -> {path}")
        return path