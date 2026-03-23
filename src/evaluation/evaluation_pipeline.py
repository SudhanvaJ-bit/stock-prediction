import numpy as np
import pandas as pd
from typing import Dict
from src.evaluation.metrics import MetricsCalculator, MetricsReport
from src.evaluation.inverse_transformer import InverseTransformer
from src.evaluation.plotter import EvaluationPlotter
from src.evaluation.report_generator import ReportGenerator
from src.models.training_pipeline import TrainingResult
from src.data_processing.processing_pipeline import ProcessedData

from config.logger import get_logger

logger = get_logger(__name__)


class EvaluationResult:
    def __init__(self, ticker: str):
        self.ticker       = ticker
        self.reports:      Dict[str, MetricsReport] = {}
        self.best_model    = ""
        self.plot_paths    = {}
        self.report_paths  = {}

    def summary(self) -> str:
        lines = [
            f"\n{'='*65}",
            f"  Evaluation Summary — [{self.ticker}] (Real Dollar Values)",
            f"{'='*65}",
            f"  {'Model':<22} {'RMSE($)':>9} {'MAE($)':>9} {'MAPE(%)':>9} {'R2':>8} {'Dir%':>8}",
            f"  {'-'*60}",
        ]
        for name, r in self.reports.items():
            marker = " <-- BEST" if name == self.best_model else ""
            lines.append(
                f"  {name:<22} {r.rmse:>9.4f} {r.mae:>9.4f} "
                f"{r.mape:>9.4f} {r.r2:>8.4f} {r.directional_accuracy:>7.2f}%"
                f"{marker}"
            )
        lines.append(f"{'='*65}")
        return "\n".join(lines)


class EvaluationPipeline:
    def run(
        self,
        training_result: TrainingResult,
        processed_data:  ProcessedData,
    ) -> EvaluationResult:
        ticker = training_result.ticker
        logger.info(f"=== Starting Evaluation Pipeline | Ticker: {ticker} ===")

        result     = EvaluationResult(ticker=ticker)
        calculator = MetricsCalculator()
        plotter    = EvaluationPlotter(ticker=ticker)
        reporter   = ReportGenerator(ticker=ticker)

        # ── Setup inverse transformer ──────────────────
        n_features     = processed_data.test_df.shape[1]
        target_col_idx = processed_data.test_df.columns.tolist().index(
            processed_data.target_column
        )
        inv = InverseTransformer(
            normalizer     = processed_data.normalizer,
            n_features     = n_features,
            target_col_idx = target_col_idx,
        )

        # ── Inverse transform ground truth ─────────────
        y_true_real, _ = inv.transform(processed_data.y_test, processed_data.y_test)

        # ── Evaluate each model ─────────────────────────
        predictions_real: Dict[str, np.ndarray] = {}

        for name, model_result in training_result.results.items():
            logger.info(f"Evaluating: {name}")

            _, y_pred_real = inv.transform(
                processed_data.y_test,
                model_result.predictions,
            )

            report = calculator.compute(name, y_true_real, y_pred_real)
            result.reports[name] = report
            predictions_real[name] = y_pred_real

            print(report.summary())

        # ── Pick best model by RMSE ─────────────────────
        result.best_model = min(result.reports, key=lambda n: result.reports[n].rmse)

        # ── Build date index for plots ──────────────────
        test_dates = processed_data.test_df.index[
            processed_data.sequence_length:
        ].to_pydatetime()

        # Align lengths (sequences remove first N rows)
        y_true_plot = y_true_real[:len(test_dates)]
        preds_plot  = {n: p[:len(test_dates)] for n, p in predictions_real.items()}

        # ── Generate plots ──────────────────────────────
        logger.info("Generating plots...")

        result.plot_paths["predictions"] = plotter.plot_predictions(
            test_dates, y_true_plot, preds_plot
        )
        result.plot_paths["metrics"] = plotter.plot_metrics(
            {name: r.to_dict() for name, r in result.reports.items()}
        )
        result.plot_paths["residuals"] = plotter.plot_residuals(
            test_dates, y_true_plot, preds_plot
        )

        # ── LSTM training history (if available) ────────
        if "LSTM" in training_result.results:
            lstm_model = training_result.results["LSTM"].model
            if hasattr(lstm_model, "history_log") and lstm_model.history_log:
                result.plot_paths["lstm_history"] = plotter.plot_lstm_history(
                    lstm_model.history_log
                )

        # ── Save reports ─────────────────────────────────
        result.report_paths = reporter.save(result.reports, result.best_model)

        logger.info(f"=== Evaluation Complete | Best Model: {result.best_model} ===")
        print(result.summary())
        return result