"""
app.py
-------
StockSense — ML Stock Price Prediction System
Streamlit UI — Layer 8

Run from project root:
    streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Page config (MUST be first Streamlit call) ──
st.set_page_config(
    page_title = "StockSense",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

from src.ui.styles     import CUSTOM_CSS
from src.ui.charts     import (
    BLUE, GREEN,
    candlestick_chart, prediction_chart,
    metrics_bar_chart, lstm_loss_chart,
    price_sparkline, metrics_radar,
)
from src.ui.components import (
    metric_card, section_header, hero_banner,
    prediction_table, sidebar_logo, live_badge,
)

# ── Inject CSS ──────────────────────────────────
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════

@st.cache_data(ttl=300)
def fetch_live_data(ticker: str, period_days: int = 365):
    """Fetch live OHLCV data with 5-min cache."""
    try:
        from src.data_ingestion.fetcher import StockDataFetcher
        start = (datetime.today() - timedelta(days=period_days)).strftime("%Y-%m-%d")
        fetcher = StockDataFetcher(ticker=ticker, start_date=start)
        df = fetcher.fetch()
        # Remove non-numeric columns
        df = df.select_dtypes(include=[np.number])
        return df
    except Exception as e:
        return None


@st.cache_resource
def get_registry():
    try:
        from src.registry.model_registry import ModelRegistry
        return ModelRegistry()
    except Exception:
        return None


def run_prediction_service(ticker: str, horizon: int, model_name: str = None):
    try:
        from src.prediction_service import PredictionService
        service = PredictionService()
        return service.predict(ticker=ticker, horizon=horizon,
                               model_name=model_name, save_results=True)
    except Exception as e:
        return str(e)


def run_all_models_prediction(ticker: str):
    try:
        from src.prediction_service import PredictionService
        service = PredictionService()
        return service.predict_all_models(ticker=ticker, horizon=1)
    except Exception as e:
        return {}


def load_evaluation_report(ticker: str) -> pd.DataFrame:
    try:
        path = os.path.join("reports", f"{ticker}_evaluation_report.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════

with st.sidebar:
    st.markdown(sidebar_logo(), unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        options=["📊  Dashboard", "🔮  Predict", "🧠  Model Analysis", "📈  Historical", "ℹ️  About"],
        label_visibility="visible",
    )

    st.markdown("---")

    # Ticker selector
    st.markdown(
        '<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.12em;'
        'text-transform:uppercase;color:#475569;margin-bottom:8px;">Ticker</div>',
        unsafe_allow_html=True,
    )
    ticker_input = st.text_input(
        "Stock Ticker", value="AAPL", max_chars=6, label_visibility="collapsed"
    ).upper().strip()

    st.markdown("---")

    # Registry status
    registry = get_registry()
    if registry:
        entries = registry.list_models(ticker=ticker_input)
        n = len(entries)
        status_color = "#10b981" if n > 0 else "#ef4444"
        st.markdown(
            f'<div style="font-size:0.7rem;color:#475569;margin-bottom:6px;">'
            f'REGISTRY STATUS</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.82rem;color:{status_color};">'
            f'{"✓" if n > 0 else "✗"} {n} model(s) registered</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.65rem;color:#334155;text-align:center;line-height:1.6;">'
        'Data via Yahoo Finance<br>Not financial advice</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════

if "Dashboard" in page:

    # Fetch data
    df = fetch_live_data(ticker_input, period_days=180)

    if df is None or df.empty:
        st.error(f"Could not fetch data for **{ticker_input}**. Check the ticker symbol.")
        st.stop()

    # ── Hero banner ────────────────────────────
    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
    change     = last_close - prev_close
    change_pct = (change / prev_close) * 100

    st.markdown(hero_banner(ticker_input, last_close, change, change_pct),
                unsafe_allow_html=True)

    # ── Key metrics row ────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    metrics_data = [
        (col1, "Open",   f"${df['Open'].iloc[-1]:.2f}",   "", ""),
        (col2, "High",   f"${df['High'].iloc[-1]:.2f}",   "", "green"),
        (col3, "Low",    f"${df['Low'].iloc[-1]:.2f}",    "", "red"),
        (col4, "Volume", f"{int(df['Volume'].iloc[-1]/1e6):.1f}M", "", "blue"),
        (col5, "52W High", f"${df['Close'].max():.2f}",   "", "gold"),
    ]

    for col, label, val, delta, color in metrics_data:
        with col:
            st.markdown(metric_card(label, val, delta, color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Candlestick chart ─────────────────────
    st.markdown(section_header("Price Chart", "OHLCV"), unsafe_allow_html=True)
    fig = candlestick_chart(df.tail(120), ticker_input)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Moving averages summary ────────────────
    st.markdown(section_header("Technical Snapshot", "LIVE"), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    close = df["Close"]

    with c1:
        sma7 = close.rolling(7).mean().iloc[-1]
        vs   = ((last_close - sma7) / sma7 * 100)
        st.markdown(metric_card("SMA 7D", f"${sma7:.2f}",
            f"{'▲' if vs>=0 else '▼'} {abs(vs):.2f}%", ""), unsafe_allow_html=True)
    with c2:
        sma21 = close.rolling(21).mean().iloc[-1]
        vs    = ((last_close - sma21) / sma21 * 100)
        st.markdown(metric_card("SMA 21D", f"${sma21:.2f}",
            f"{'▲' if vs>=0 else '▼'} {abs(vs):.2f}%", ""), unsafe_allow_html=True)
    with c3:
        # RSI
        delta_s = close.diff()
        gain  = delta_s.clip(lower=0).ewm(com=13).mean()
        loss  = (-delta_s.clip(upper=0)).ewm(com=13).mean()
        rs    = gain / loss.replace(0, 1e-9)
        rsi   = (100 - 100 / (1 + rs)).iloc[-1]
        rsi_color = "green" if rsi < 30 else "red" if rsi > 70 else ""
        st.markdown(metric_card("RSI 14", f"{rsi:.1f}",
            "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral",
            rsi_color), unsafe_allow_html=True)
    with c4:
        vol21 = close.pct_change().rolling(21).std().iloc[-1] * 100
        st.markdown(metric_card("Volatility 21D", f"{vol21:.2f}%", "", ""), unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════

elif "Predict" in page:

    st.markdown(section_header("Prediction Engine", "AI POWERED"), unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown(
            '<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.12em;'
            'text-transform:uppercase;color:#475569;margin-bottom:12px;">SETTINGS</div>',
            unsafe_allow_html=True,
        )

        horizon = st.slider("Forecast Horizon (Days)", min_value=1, max_value=30, value=7)

        registry = get_registry()
        model_options = ["Best Model (Auto)"]
        if registry:
            entries = registry.list_models(ticker=ticker_input)
            model_options += [e.model_name for e in entries]

        selected_model = st.selectbox("Model", model_options)
        model_name = None if selected_model == "Best Model (Auto)" \
                     else selected_model

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶  Run Prediction", use_container_width=True)
    with col_right:
        if run_btn:
            with st.spinner("Fetching live data and running inference..."):
                result = run_prediction_service(ticker_input, horizon, model_name)

            if isinstance(result, str):
                st.error(f"Prediction failed: {result}")
            else:
                # ── Hero result ────────────────────
                first_pred = result.predictions[0][1]
                last_known = result.last_known_price
                chg        = first_pred - last_known
                chg_pct    = chg / last_known * 100
                direction  = "▲" if chg >= 0 else "▼"
                chg_color  = "#10b981" if chg >= 0 else "#ef4444"

                st.markdown(
                    f'<div class="hero-banner" style="margin-bottom:20px;">'
                    f'<div style="font-size:0.7rem;color:#475569;letter-spacing:0.1em;margin-bottom:4px;">'
                    f'NEXT TRADING DAY — {result.predictions[0][0]}</div>'
                    f'<div style="font-family:\'DM Mono\',monospace;font-size:2.5rem;color:{chg_color};">'
                    f'${first_pred:,.2f}</div>'
                    f'<div style="font-size:0.85rem;color:{chg_color};margin-top:4px;">'
                    f'{direction} ${abs(chg):.2f} ({direction} {abs(chg_pct):.2f}%)'
                    f'<span style="color:#475569;margin-left:12px;">vs ${last_known:.2f} last known</span></div>'
                    f'<div style="margin-top:8px;font-size:0.72rem;color:#475569;">'
                    f'Model: {result.model_name} · Version: {result.model_version}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ── Fetch history for chart ────────
                df_hist = fetch_live_data(ticker_input, period_days=90)
                if df_hist is not None:
                    pred_dates  = [p[0] for p in result.predictions]
                    pred_prices = [p[1] for p in result.predictions]
                    fig = prediction_chart(
                        df_hist.index[-60:],
                        df_hist["Close"].values[-60:],
                        pred_dates, pred_prices,
                        ticker_input, result.model_name,
                        result.last_known_price,
                    )
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})

                # ── Prediction table ───────────────
                st.markdown(section_header("Full Forecast Table", f"{horizon}D"),
                            unsafe_allow_html=True)
                st.markdown(
                    prediction_table(result.predictions, result.last_known_price),
                    unsafe_allow_html=True,
                )

                # ── All models comparison ──────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(section_header("All Models — Next Day Comparison", ""),
                            unsafe_allow_html=True)
                with st.spinner("Running all models..."):
                    all_results = run_all_models_prediction(ticker_input)

                if all_results:
                    cols = st.columns(len(all_results))
                    for i, (mname, mresult) in enumerate(all_results.items()):
                        with cols[i]:
                            mpred  = mresult.predictions[0][1]
                            mchg   = mpred - mresult.last_known_price
                            mcolor = "#10b981" if mchg >= 0 else "#ef4444"
                            marrow = "▲" if mchg >= 0 else "▼"
                            st.markdown(
                                f'<div class="metric-card">'
                                f'<div class="metric-label">{mname}</div>'
                                f'<div class="metric-value" style="color:{mcolor};">${mpred:,.2f}</div>'
                                f'<div class="metric-delta {("up" if mchg>=0 else "down")}">'
                                f'{marrow} ${abs(mchg):.2f}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
        else:
            st.markdown(
                '<div style="display:flex;align-items:center;justify-content:center;'
                'height:300px;color:#334155;font-size:0.9rem;letter-spacing:0.05em;">'
                '← Configure settings and run prediction</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════
# PAGE 3 — MODEL ANALYSIS
# ══════════════════════════════════════════════

elif "Model Analysis" in page:

    st.markdown(section_header("Model Performance Analysis", "EVALUATION"), unsafe_allow_html=True)

    report_df = load_evaluation_report(ticker_input)

    if report_df is None:
        st.warning(f"No evaluation report found for **{ticker_input}**. Run `python run_evaluation.py` first.")
        st.stop()

    # Build metrics dict
    metrics_dict = {}
    best_rmse    = float("inf")
    best_model   = ""

    for _, row in report_df.iterrows():
        mname = row["Model"]
        metrics_dict[mname] = row.to_dict()
        if row.get("RMSE ($)", float("inf")) < best_rmse:
            best_rmse  = row.get("RMSE ($)", float("inf"))
            best_model = mname

    # ── Model cards ───────────────────────────
    model_types = {
        "LinearRegression": "Classical ML · Sklearn",
        "ARIMA"           : "Statistical · Walk-Forward",
        "LSTM"            : "Deep Learning · TensorFlow",
    }

    cols = st.columns(len(metrics_dict))
    for i, (mname, mmetrics) in enumerate(metrics_dict.items()):
        with cols[i]:
            is_best  = (mname == best_model)
            border   = "rgba(245,158,11,0.4)" if is_best else "rgba(99,130,201,0.15)"
            best_tag = " 🏆 BEST" if is_best else ""
            mtype    = model_types.get(mname, "ML Model")

            rows_html = ""
            for k, v in mmetrics.items():
                if k == "Model":
                    continue
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                rows_html += (
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:6px 0;border-bottom:1px solid rgba(99,130,201,0.08);">'
                    f'<span style="font-family:DM Mono,monospace;font-size:0.7rem;color:#475569;">{k}</span>'
                    f'<span style="font-family:DM Mono,monospace;font-size:0.82rem;color:#94a3b8;">{val_str}</span>'
                    f'</div>'
                )

            card_html = (
                f'<div style="background:rgba(17,28,53,0.85);border:1px solid {border};'
                f'border-radius:12px;padding:20px;">'
                f'<div style="font-size:0.95rem;font-weight:600;color:#e2e8f0;margin-bottom:4px;">{mname}{best_tag}</div>'
                f'<div style="font-family:DM Mono,monospace;font-size:0.7rem;color:#f59e0b;margin-bottom:14px;">{mtype}</div>'
                f'{rows_html}'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs for charts ───────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Metric Bars", "🕸 Radar", "📈 LSTM History"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                metrics_bar_chart(metrics_dict, "RMSE ($)"),
                use_container_width=True, config={"displayModeBar": False},
            )
            st.plotly_chart(
                metrics_bar_chart(metrics_dict, "MAPE (%)"),
                use_container_width=True, config={"displayModeBar": False},
            )
        with c2:
            st.plotly_chart(
                metrics_bar_chart(metrics_dict, "MAE ($)"),
                use_container_width=True, config={"displayModeBar": False},
            )
            st.plotly_chart(
                metrics_bar_chart(metrics_dict, "Direction Acc (%)"),
                use_container_width=True, config={"displayModeBar": False},
            )

    with tab2:
        st.plotly_chart(
            metrics_radar(metrics_dict),
            use_container_width=True, config={"displayModeBar": False},
        )
        st.markdown(
            '<div style="font-size:0.75rem;color:#475569;text-align:center;margin-top:-16px;">'
            'Note: For RMSE/MAE/MAPE lower is better. For Direction Acc higher is better.</div>',
            unsafe_allow_html=True,
        )

    with tab3:
        # Check for saved LSTM history plot
        plot_path = os.path.join("reports", "plots", f"{ticker_input}_lstm_training_history.png")
        if os.path.exists(plot_path):
            st.image(plot_path, use_container_width=True)
        else:
            st.info("LSTM training history plot not found. Run `python run_evaluation.py` to generate.")

    # ── Key insights ──────────────────────────
    st.markdown(section_header("Key Insights", "AUTO"), unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">Lowest Error (RMSE)</div>'
            f'<div class="metric-value gold">{best_model}</div>'
            '<div class="metric-delta">Best raw accuracy</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col2:
        best_dir = max(metrics_dict, key=lambda m: metrics_dict[m].get("Direction Acc (%)", 0))
        best_dir_val = metrics_dict[best_dir].get("Direction Acc (%)", 0)
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">Best Direction Acc</div>'
            f'<div class="metric-value green">{best_dir}</div>'
            f'<div class="metric-delta up">▲ {best_dir_val:.2f}%</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col3:
        worst = max(metrics_dict, key=lambda m: metrics_dict[m].get("RMSE ($)", 0))
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">Highest Error</div>'
            f'<div class="metric-value red">{worst}</div>'
            '<div class="metric-delta down">Needs improvement</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════
# PAGE 4 — HISTORICAL
# ══════════════════════════════════════════════

elif "Historical" in page:

    st.markdown(section_header("Historical Data Explorer", "OHLCV"), unsafe_allow_html=True)

    period = st.select_slider(
        "Time Period",
        options=["1M", "3M", "6M", "1Y", "2Y"],
        value="1Y",
    )

    period_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730}
    days = period_map[period]

    with st.spinner("Loading historical data..."):
        df = fetch_live_data(ticker_input, period_days=days)

    if df is None or df.empty:
        st.error(f"Could not load data for {ticker_input}")
        st.stop()

    # ── Chart ─────────────────────────────────
    fig = candlestick_chart(df, ticker_input)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Stats ─────────────────────────────────
    st.markdown(section_header("Period Statistics", period), unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    close = df["Close"]

    stats = [
        (c1, "Start Price",    f"${close.iloc[0]:.2f}",       "", ""),
        (c2, "End Price",      f"${close.iloc[-1]:.2f}",       "", ""),
        (c3, "Period High",    f"${close.max():.2f}",           "", "green"),
        (c4, "Period Low",     f"${close.min():.2f}",           "", "red"),
        (c5, "Total Return",   f"{((close.iloc[-1]/close.iloc[0])-1)*100:.2f}%",
             "▲" if close.iloc[-1] > close.iloc[0] else "▼",
             "green" if close.iloc[-1] > close.iloc[0] else "red"),
        (c6, "Avg Volume",     f"{df['Volume'].mean()/1e6:.1f}M", "", "blue"),
    ]

    for col, label, val, delta, color in stats:
        with col:
            st.markdown(metric_card(label, val, delta, color), unsafe_allow_html=True)

    # ── Raw data table ────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋  View Raw Data"):
        st.dataframe(
            df[["Open", "High", "Low", "Close", "Volume"]].tail(50).style.format({
                "Open": "${:.2f}", "High": "${:.2f}",
                "Low": "${:.2f}", "Close": "${:.2f}",
                "Volume": "{:,.0f}",
            }),
            use_container_width=True,
        )

    csv = df.to_csv().encode("utf-8")
    st.download_button(
        label     = "⬇  Download CSV",
        data      = csv,
        file_name = f"{ticker_input}_historical.csv",
        mime      = "text/csv",
    )


# ══════════════════════════════════════════════
# PAGE 5 — ABOUT
# ══════════════════════════════════════════════

elif "About" in page:

    st.markdown(section_header("About StockSense", ""), unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div style="color:#94a3b8;line-height:1.8;font-size:0.92rem;">
        <p><strong style="color:#e2e8f0;">StockSense</strong> is a modular, production-grade machine learning 
        system for stock price trend prediction built on historical Yahoo Finance data.</p>

        <p>It demonstrates end-to-end ML system design — from raw data ingestion through 
        feature engineering, model training, evaluation, versioned model registry, 
        and a real-time prediction service.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("System Architecture", "7 LAYERS"), unsafe_allow_html=True)

    layers = [
        ("Layer 1", "Data Ingestion",      "Yahoo Finance → validate → store as CSV"),
        ("Layer 2", "Data Processing",     "Clean → split → normalize → sequences"),
        ("Layer 3", "Feature Engineering", "SMA, EMA, RSI, MACD, Bollinger Bands"),
        ("Layer 4", "Model Training",      "LinearRegression · ARIMA · LSTM"),
        ("Layer 5", "Model Evaluation",    "RMSE, MAE, MAPE, R2, Direction Accuracy"),
        ("Layer 6", "Model Registry",      "Versioned JSON index · auto best-model selection"),
        ("Layer 7", "Prediction Service",  "Single-step & multi-step live forecasting"),
    ]

    for layer_id, name, desc in layers:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:16px;padding:12px 0;'
            f'border-bottom:1px solid rgba(99,130,201,0.08);">'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.72rem;'
            f'color:#f59e0b;min-width:60px;">{layer_id}</div>'
            f'<div style="font-size:0.9rem;font-weight:600;color:#e2e8f0;min-width:180px;">{name}</div>'
            f'<div style="font-size:0.82rem;color:#475569;">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("Models", "COMPARISON"), unsafe_allow_html=True)

    model_info = [
        ("Linear Regression", "Classical ML", "Baseline model. Fast and interpretable. Uses flattened 60-day sequences as input features.", BLUE),
        ("ARIMA (5,1,0)",     "Statistical",  "AutoRegressive Integrated Moving Average. Uses walk-forward validation on Close price series.", "#f97316"),
        ("LSTM",              "Deep Learning","2-layer Long Short-Term Memory network. Learns long-range temporal patterns across 44 features.", GREEN),
    ]

    cols = st.columns(3)
    for i, (mname, mtype, mdesc, mcolor) in enumerate(model_info):
        with cols[i]:
            st.markdown(
                f'<div class="metric-card" style="border-top-color:{mcolor};">'
                f'<div style="font-size:0.95rem;font-weight:600;color:#e2e8f0;margin-bottom:4px;">{mname}</div>'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.7rem;color:{mcolor};'
                f'margin-bottom:12px;">{mtype}</div>'
                f'<div style="font-size:0.82rem;color:#475569;line-height:1.6;">{mdesc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;padding:24px;color:#334155;font-size:0.75rem;'
        'letter-spacing:0.08em;text-transform:uppercase;">'
        'Built for learning · Not financial advice · Data via Yahoo Finance'
        '</div>',
        unsafe_allow_html=True,
    )
