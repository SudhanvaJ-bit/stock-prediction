"""
src/ui/components.py
---------------------
Reusable HTML component builders for the Streamlit UI.
Returns HTML strings rendered via st.markdown(unsafe_allow_html=True).
All components use fully inline styles — no CSS class dependencies.
"""

# ── Shared style tokens ──────────────────────────────────────────────────────
_BG_CARD    = "background:rgba(17,28,53,0.85)"
_BORDER     = "border:1px solid rgba(99,130,201,0.15)"
_RADIUS     = "border-radius:12px"
_FONT_MONO  = "font-family:'DM Mono',monospace"
_FONT_UI    = "font-family:'Outfit',sans-serif"


def metric_card(label: str, value: str, delta: str = "", color: str = "") -> str:
    """
    Fixed-height metric card. All cards share the same min-height so
    cards in the same row stay aligned regardless of content.
    """
    color_map = {
        "gold"  : "#f59e0b",
        "green" : "#10b981",
        "red"   : "#ef4444",
        "blue"  : "#60a5fa",
        ""      : "#e2e8f0",
    }
    val_color = color_map.get(color, "#e2e8f0")

    delta_html = ""
    if delta:
        is_up     = delta.startswith("▲") or delta.startswith("+")
        chg_color = "#10b981" if is_up else "#ef4444" if delta.startswith("▼") else "#475569"
        delta_html = (
            f'<div style="{_FONT_MONO};font-size:0.78rem;margin-top:6px;color:{chg_color};">'
            f'{delta}</div>'
        )

    top_bar = (
        'before:content-none;position:relative;overflow:hidden;'
    )

    return (
        f'<div style="{_BG_CARD};{_BORDER};{_RADIUS};padding:20px 24px;'
        f'min-height:110px;position:relative;overflow:hidden;'
        f'border-top:2px solid rgba(245,158,11,0.5);">'
        f'<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.12em;'
        f'text-transform:uppercase;color:#475569;margin-bottom:10px;">{label}</div>'
        f'<div style="{_FONT_MONO};font-size:1.7rem;font-weight:500;'
        f'color:{val_color};line-height:1;">{value}</div>'
        f'{delta_html}'
        f'</div>'
    )


def section_header(title: str, badge: str = "") -> str:
    badge_html = ""
    if badge:
        badge_html = (
            f'<span style="{_FONT_MONO};font-size:0.68rem;padding:3px 10px;'
            f'border-radius:20px;background:rgba(245,158,11,0.1);color:#f59e0b;'
            f'border:1px solid rgba(245,158,11,0.2);letter-spacing:0.08em;">{badge}</span>'
        )
    return (
        f'<div style="display:flex;align-items:center;gap:12px;'
        f'margin:28px 0 16px 0;padding-bottom:12px;'
        f'border-bottom:1px solid rgba(99,130,201,0.15);">'
        f'<span style="font-size:1.05rem;font-weight:600;color:#e2e8f0;">{title}</span>'
        f'{badge_html}'
        f'</div>'
    )


def live_badge() -> str:
    return (
        '<div style="display:inline-flex;align-items:center;gap:6px;'
        'font-family:\'DM Mono\',monospace;font-size:0.68rem;padding:4px 10px;'
        'border-radius:20px;background:rgba(16,185,129,0.1);color:#10b981;'
        'border:1px solid rgba(16,185,129,0.25);letter-spacing:0.1em;">'
        '<div style="width:6px;height:6px;border-radius:50%;background:#10b981;'
        'animation:pulse 1.8s ease-in-out infinite;"></div>'
        'LIVE DATA</div>'
    )


def hero_banner(ticker: str, price: float, change: float, change_pct: float) -> str:
    direction = "▲" if change >= 0 else "▼"
    chg_color = "#10b981" if change >= 0 else "#ef4444"
    return (
        f'<div style="{_BG_CARD};{_BORDER};border-radius:16px;padding:32px 40px;'
        f'margin-bottom:28px;position:relative;overflow:hidden;">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
        f'<div>'
        f'<div style="{_FONT_MONO};font-size:2.8rem;font-weight:500;color:#f59e0b;line-height:1;">{ticker}</div>'
        f'<div style="{_FONT_MONO};font-size:2rem;color:#e2e8f0;margin-top:4px;">${price:,.2f}</div>'
        f'<div style="font-size:0.85rem;color:{chg_color};margin-top:6px;">'
        f'{direction} ${abs(change):.2f} &nbsp;({direction} {abs(change_pct):.2f}%)'
        f'<span style="color:#475569;margin-left:12px;">vs prev close</span></div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<div style="display:inline-flex;align-items:center;gap:6px;{_FONT_MONO};font-size:0.68rem;'
        f'padding:4px 10px;border-radius:20px;background:rgba(16,185,129,0.1);color:#10b981;'
        f'border:1px solid rgba(16,185,129,0.25);">'
        f'<div style="width:6px;height:6px;border-radius:50%;background:#10b981;"></div>LIVE</div>'
        f'<div style="{_FONT_MONO};font-size:0.7rem;color:#475569;margin-top:6px;">YAHOO FINANCE</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )


def prediction_table(predictions: list, last_price: float) -> str:
    TH = (
        "padding:10px 16px;font-size:0.68rem;font-weight:600;"
        "letter-spacing:0.1em;text-transform:uppercase;color:#475569;"
        "border-bottom:1px solid rgba(99,130,201,0.15);text-align:left;"
        "font-family:'DM Mono',monospace;"
    )
    TD = (
        "padding:11px 16px;font-family:'DM Mono',monospace;"
        "font-size:0.86rem;border-bottom:1px solid rgba(99,130,201,0.06);"
    )

    rows = ""
    prev = last_price
    for pred_date, pred_price in predictions:
        change    = pred_price - prev
        direction = "▲" if change >= 0 else "▼"
        chg_color = "#10b981" if change >= 0 else "#ef4444"
        rows += (
            f'<tr>'
            f'<td style="{TD}color:#475569;">{str(pred_date)}</td>'
            f'<td style="{TD}color:#e2e8f0;font-weight:500;">${pred_price:,.2f}</td>'
            f'<td style="{TD}color:{chg_color};">{direction} ${abs(change):.2f}</td>'
            f'<td style="{TD}color:{chg_color};">{direction} {abs(change/prev*100):.2f}%</td>'
            f'</tr>'
        )
        prev = pred_price

    TABLE = (
        "width:100%;border-collapse:collapse;"
        "background:rgba(17,28,53,0.85);"
        "border:1px solid rgba(99,130,201,0.15);"
        "border-radius:10px;overflow:hidden;"
    )
    return (
        f'<table style="{TABLE}">'
        f'<thead><tr>'
        f'<th style="{TH}">Date</th>'
        f'<th style="{TH}">Predicted Close</th>'
        f'<th style="{TH}">Change ($)</th>'
        f'<th style="{TH}">Change (%)</th>'
        f'</tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'</table>'
    )


def sidebar_logo() -> str:
    return (
        '<div style="padding:8px 0 24px 0;border-bottom:1px solid rgba(99,130,201,0.15);'
        'margin-bottom:20px;">'
        '<div style="font-family:\'DM Mono\',monospace;font-size:1.1rem;font-weight:500;'
        'color:#f59e0b;letter-spacing:0.05em;">STOCKSENSE</div>'
        '<div style="font-size:0.68rem;color:#475569;margin-top:3px;'
        'letter-spacing:0.1em;text-transform:uppercase;">ML Prediction System</div>'
        '</div>'
    )
