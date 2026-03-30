"""
src/ui/charts.py
-----------------
Plotly chart builders for the Streamlit UI.
All charts follow the Bloomberg terminal fintech aesthetic.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# ── Design tokens ──────────────────────────────
BG         = "#060b18"
BG_PANEL   = "#0d1529"
GOLD       = "#f59e0b"
BLUE       = "#60a5fa"
GREEN      = "#10b981"
RED        = "#ef4444"
TEXT       = "#e2e8f0"
TEXT_DIM   = "#475569"
GRID       = "rgba(99, 130, 201, 0.08)"
BORDER     = "rgba(99, 130, 201, 0.15)"

MODEL_COLORS = {
    "LinearRegression": BLUE,
    "ARIMA"           : "#f97316",
    "LSTM"            : GREEN,
    "Actual"          : GOLD,
}

def _base_layout(title_text="", height=400):
    """Returns a clean base layout dict with no axis conflicts."""
    layout = dict(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(family="DM Mono, monospace", color=TEXT, size=11),
        margin        = dict(l=0, r=0, t=40, b=0),
        height        = height,
        legend        = dict(
            bgcolor     = "rgba(13, 21, 41, 0.8)",
            bordercolor = BORDER,
            borderwidth = 1,
            font        = dict(size=11),
        ),
        xaxis = dict(
            gridcolor = GRID,
            linecolor = BORDER,
            tickcolor = TEXT_DIM,
            tickfont  = dict(size=10),
            showgrid  = True,
        ),
        yaxis = dict(
            gridcolor = GRID,
            linecolor = BORDER,
            tickcolor = TEXT_DIM,
            tickfont  = dict(size=10),
            showgrid  = True,
        ),
    )
    if title_text:
        layout["title"] = dict(
            text = title_text,
            font = dict(size=13, color=TEXT),
            x    = 0,
        )
    return layout


def candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """OHLC candlestick chart with volume bars."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.02,
    )

    fig.add_trace(go.Candlestick(
        x     = df.index,
        open  = df["Open"],
        high  = df["High"],
        low   = df["Low"],
        close = df["Close"],
        name  = ticker,
        increasing_line_color  = GREEN,
        decreasing_line_color  = RED,
        increasing_fillcolor   = GREEN,
        decreasing_fillcolor   = RED,
        line_width = 1,
    ), row=1, col=1)

    colors = [GREEN if df["Close"].iloc[i] >= df["Open"].iloc[i]
              else RED for i in range(len(df))]

    fig.add_trace(go.Bar(
        x          = df.index,
        y          = df["Volume"],
        name       = "Volume",
        marker_color = colors,
        opacity    = 0.5,
        showlegend = False,
    ), row=2, col=1)

    fig.update_layout(**_base_layout(f"{ticker} — Price & Volume", 460))
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="Volume",      row=2, col=1, title_font=dict(size=10))
    return fig


def prediction_chart(
    dates_hist:   pd.DatetimeIndex,
    prices_hist:  np.ndarray,
    pred_dates:   list,
    pred_prices:  list,
    ticker:       str,
    model_name:   str,
    last_known:   float,
) -> go.Figure:
    """Historical prices + prediction forecast with confidence shading."""
    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x    = dates_hist,
        y    = prices_hist,
        name = "Historical",
        line = dict(color=GOLD, width=1.8),
        mode = "lines",
    ))

    # Confidence band
    uncertainty = [p * 0.025 * (i + 1) for i, p in enumerate(pred_prices)]
    upper = [p + u for p, u in zip(pred_prices, uncertainty)]
    lower = [p - u for p, u in zip(pred_prices, uncertainty)]

    fig.add_trace(go.Scatter(
        x         = [str(d) for d in pred_dates] + [str(d) for d in reversed(pred_dates)],
        y         = upper + list(reversed(lower)),
        fill      = "toself",
        fillcolor = "rgba(96, 165, 250, 0.06)",
        line      = dict(color="rgba(0,0,0,0)"),
        showlegend = False,
        name      = "Confidence",
    ))

    # Bridge + forecast line
    bridge_dates  = [dates_hist[-1]] + [str(d) for d in pred_dates]
    bridge_prices = [last_known] + pred_prices
    color = MODEL_COLORS.get(model_name, BLUE)

    fig.add_trace(go.Scatter(
        x    = bridge_dates,
        y    = bridge_prices,
        name = f"{model_name} Forecast",
        line = dict(color=color, width=2, dash="dot"),
        mode = "lines+markers",
        marker = dict(size=6, color=color, symbol="circle",
                      line=dict(color=BG, width=2)),
    ))

    fig.add_vline(
        x          = str(dates_hist[-1]),
        line_dash  = "dash",
        line_color = "rgba(245, 158, 11, 0.3)",
        line_width = 1,
    )

    fig.add_annotation(
        x         = str(dates_hist[-1]),
        y         = 1,
        yref      = "paper",
        text      = "NOW",
        showarrow = False,
        font      = dict(size=9, color=GOLD, family="DM Mono"),
        bgcolor   = "rgba(245, 158, 11, 0.1)",
        bordercolor = GOLD,
        borderwidth = 1,
        borderpad   = 3,
    )

    fig.update_layout(**_base_layout(f"{ticker} — Forecast ({model_name})", 400))
    fig.update_yaxes(title_text="Close Price (USD)", title_font=dict(size=10))
    return fig


def metrics_radar(metrics_dict: Dict[str, dict]) -> go.Figure:
    """Radar chart comparing all models across metrics."""
    categories = ["RMSE", "MAE", "MAPE", "Direction Acc"]
    fig = go.Figure()

    for model_name, metrics in metrics_dict.items():
        values = [
            metrics.get("RMSE ($)", 0),
            metrics.get("MAE ($)", 0),
            metrics.get("MAPE (%)", 0),
            metrics.get("Direction Acc (%)", 0),
        ]
        color = MODEL_COLORS.get(model_name, BLUE)
        fill_color = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.06)"

        fig.add_trace(go.Scatterpolar(
            r         = values + [values[0]],
            theta     = categories + [categories[0]],
            name      = model_name,
            line      = dict(color=color, width=2),
            fill      = "toself",
            fillcolor = fill_color,
        ))

    layout = _base_layout("Model Comparison Radar", 380)
    # Remove xaxis/yaxis from base — radar doesn't use them
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    layout["polar"] = dict(
        bgcolor     = "rgba(0,0,0,0)",
        radialaxis  = dict(visible=True, gridcolor=GRID,
                           linecolor=BORDER, tickfont=dict(size=9)),
        angularaxis = dict(gridcolor=GRID, linecolor=BORDER,
                           tickfont=dict(size=10, color=TEXT)),
    )
    fig.update_layout(**layout)
    return fig


def metrics_bar_chart(metrics_dict: Dict[str, dict], metric: str) -> go.Figure:
    """Horizontal bar chart for a single metric across all models."""
    models = list(metrics_dict.keys())
    values = [metrics_dict[m].get(metric, 0) for m in models]
    colors = [MODEL_COLORS.get(m, BLUE) for m in models]

    fig = go.Figure(go.Bar(
        x            = values,
        y            = models,
        orientation  = "h",
        marker       = dict(color=colors, opacity=0.85,
                            line=dict(color=colors, width=1)),
        text         = [f"{v:.3f}" for v in values],
        textposition = "outside",
        textfont     = dict(size=11, family="DM Mono", color=TEXT),
    ))

    layout = _base_layout(metric, 200)
    layout["margin"] = dict(l=0, r=60, t=36, b=0)
    layout["yaxis"]["showgrid"] = False
    layout["yaxis"]["tickfont"] = dict(size=11)
    fig.update_layout(**layout)
    return fig


def lstm_loss_chart(train_loss: list, val_loss: list) -> go.Figure:
    """LSTM training history loss curves."""
    epochs = list(range(1, len(train_loss) + 1))
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs, y=train_loss, name="Train Loss",
        line=dict(color=GREEN, width=2), mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=val_loss, name="Val Loss",
        line=dict(color=GOLD, width=2, dash="dot"), mode="lines",
    ))

    best_epoch = val_loss.index(min(val_loss)) + 1
    fig.add_vline(
        x=best_epoch, line_dash="dash",
        line_color="rgba(16, 185, 129, 0.4)", line_width=1,
        annotation_text=f"Best: Epoch {best_epoch}",
        annotation_font=dict(size=9, color=GREEN),
    )

    layout = _base_layout("LSTM Training History", 300)
    layout["xaxis"]["title"] = "Epoch"
    layout["yaxis"]["title"] = "Loss (MSE)"
    fig.update_layout(**layout)
    return fig


def price_sparkline(prices: np.ndarray) -> go.Figure:
    """Minimal sparkline for sidebar."""
    color = GREEN if prices[-1] >= prices[0] else RED
    fill_color = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.06)"

    fig = go.Figure(go.Scatter(
        y=prices, mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy", fillcolor=fill_color,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=60, showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig
