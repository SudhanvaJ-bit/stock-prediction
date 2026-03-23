import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional

from config.config import PLOTS_DIR
from config.logger import get_logger

logger = get_logger(__name__)

# ── Style ─────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : "#0d1117",
    "axes.facecolor"   : "#161b22",
    "axes.edgecolor"   : "#30363d",
    "axes.labelcolor"  : "#c9d1d9",
    "text.color"       : "#c9d1d9",
    "xtick.color"      : "#8b949e",
    "ytick.color"      : "#8b949e",
    "grid.color"       : "#21262d",
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.6,
    "legend.facecolor" : "#161b22",
    "legend.edgecolor" : "#30363d",
    "font.size"        : 11,
})

MODEL_COLORS = {
    "LinearRegression" : "#58a6ff",
    "ARIMA"            : "#f78166",
    "LSTM"             : "#3fb950",
    "Actual"           : "#e3b341",
}


class EvaluationPlotter:
    def __init__(self, ticker: str, plots_dir: str = PLOTS_DIR):
        self.ticker    = ticker
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    # ──────────────────────────────────────────
    # Plot 1: Predictions vs Actual
    # ──────────────────────────────────────────

    def plot_predictions(
        self,
        dates:            np.ndarray,
        y_true:           np.ndarray,
        predictions_dict: Dict[str, np.ndarray],
    ) -> str:
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(dates, y_true, label="Actual", color=MODEL_COLORS["Actual"],
                linewidth=2.0, zorder=5)

        for name, preds in predictions_dict.items():
            color = MODEL_COLORS.get(name, "#ffffff")
            ax.plot(dates, preds, label=name, color=color,
                    linewidth=1.4, alpha=0.85, linestyle="--")

        ax.set_title(f"{self.ticker} — Predicted vs Actual Close Price",
                     fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        ax.legend(loc="upper left")
        ax.grid(True)
        plt.tight_layout()

        path = os.path.join(self.plots_dir, f"{self.ticker}_predictions_vs_actual.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved -> {path}")
        return path

    # ──────────────────────────────────────────
    # Plot 2: Metrics Comparison Bar Chart
    # ──────────────────────────────────────────

    def plot_metrics(self, metrics_dict: Dict[str, Dict]) -> str:
        models  = list(metrics_dict.keys())
        metrics = ["RMSE ($)", "MAE ($)", "MAPE (%)", "R2"]
        colors  = [MODEL_COLORS.get(m, "#ffffff") for m in models]

        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        fig.suptitle(f"{self.ticker} — Model Metrics Comparison",
                     fontsize=14, fontweight="bold", y=1.02)

        for i, metric in enumerate(metrics):
            ax     = axes[i]
            values = [metrics_dict[m].get(metric, 0) for m in models]
            bars   = ax.bar(models, values, color=colors, alpha=0.85, width=0.5)

            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.02,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9, color="#c9d1d9"
                )

            ax.set_title(metric, fontsize=12, fontweight="bold")
            ax.set_ylabel(metric)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
            ax.grid(True, axis="y")

            # For R2, higher is better — add a green reference line at 1.0
            if metric == "R2":
                ax.axhline(y=1.0, color="#3fb950", linestyle=":", linewidth=1.2, alpha=0.7)

        plt.tight_layout()
        path = os.path.join(self.plots_dir, f"{self.ticker}_metrics_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved -> {path}")
        return path

    # ──────────────────────────────────────────
    # Plot 3: Residuals
    # ──────────────────────────────────────────

    def plot_residuals(
        self,
        dates:            np.ndarray,
        y_true:           np.ndarray,
        predictions_dict: Dict[str, np.ndarray],
    ) -> str:
        n_models = len(predictions_dict)
        fig, axes = plt.subplots(n_models, 1, figsize=(14, 4 * n_models), sharex=True)

        if n_models == 1:
            axes = [axes]

        for ax, (name, preds) in zip(axes, predictions_dict.items()):
            residuals = y_true - preds
            color     = MODEL_COLORS.get(name, "#ffffff")

            ax.bar(dates, residuals, color=color, alpha=0.6, width=1.0)
            ax.axhline(y=0, color="#ffffff", linewidth=0.8, linestyle="-")
            ax.set_title(f"{name} — Residuals (Actual − Predicted)", fontweight="bold")
            ax.set_ylabel("Error ($)")
            ax.grid(True)

        axes[-1].set_xlabel("Date")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        plt.tight_layout()

        path = os.path.join(self.plots_dir, f"{self.ticker}_residuals.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved -> {path}")
        return path

    # ──────────────────────────────────────────
    # Plot 4: LSTM Training History
    # ──────────────────────────────────────────

    def plot_lstm_history(self, history) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(history.history["loss"],     label="Train Loss",
                color=MODEL_COLORS["LSTM"],  linewidth=2.0)
        ax.plot(history.history["val_loss"], label="Val Loss",
                color=MODEL_COLORS["Actual"], linewidth=2.0, linestyle="--")

        ax.set_title(f"{self.ticker} — LSTM Training History",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        path = os.path.join(self.plots_dir, f"{self.ticker}_lstm_training_history.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved -> {path}")
        return path