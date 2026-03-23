import os
import csv
from datetime import datetime
from typing import Dict, List
from src.evaluation.metrics import MetricsReport
from config.config import REPORTS_DIR
from config.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    def __init__(self, ticker: str, reports_dir: str = REPORTS_DIR):
        self.ticker      = ticker
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def save(self, reports: Dict[str, MetricsReport], best_model: str) -> Dict[str, str]:
        csv_path = self._save_csv(reports)
        txt_path = self._save_txt(reports, best_model)
        return {"csv": csv_path, "txt": txt_path}

    # ──────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────

    def _save_csv(self, reports: Dict[str, MetricsReport]) -> str:
        path = os.path.join(self.reports_dir, f"{self.ticker}_evaluation_report.csv")
        rows = [r.to_dict() for r in reports.values()]

        if not rows:
            return path

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Evaluation CSV saved -> {path}")
        return path

    def _save_txt(self, reports: Dict[str, MetricsReport], best_model: str) -> str:
        path = os.path.join(self.reports_dir, f"{self.ticker}_evaluation_report.txt")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "=" * 60,
            f"  Stock Price Prediction — Evaluation Report",
            f"  Ticker    : {self.ticker}",
            f"  Generated : {timestamp}",
            "=" * 60,
            "",
            "  METRICS ON REAL (INVERSE-TRANSFORMED) DOLLAR VALUES",
            "-" * 60,
        ]

        for name, report in reports.items():
            lines.append(report.summary())

        lines += [
            "-" * 60,
            f"  BEST MODEL : {best_model} (lowest RMSE)",
            "=" * 60,
        ]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Evaluation TXT saved -> {path}")
        return path