CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root Variables ─────────────────────────── */
:root {
    --bg-primary:    #060b18;
    --bg-secondary:  #0d1529;
    --bg-panel:      #111c35;
    --bg-card:       rgba(17, 28, 53, 0.85);
    --border-subtle: rgba(99, 130, 201, 0.15);
    --border-glow:   rgba(245, 158, 11, 0.4);
    --gold:          #f59e0b;
    --gold-dim:      #d97706;
    --blue:          #60a5fa;
    --blue-bright:   #3b82f6;
    --green:         #10b981;
    --red:           #ef4444;
    --text-primary:  #e2e8f0;
    --text-secondary:#94a3b8;
    --text-dim:      #475569;
    --font-ui:       'Outfit', sans-serif;
    --font-mono:     'DM Mono', monospace;
}

/* ── Global ─────────────────────────────────── */
html, body, [class*="css"] {
    font-family: var(--font-ui) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background: 
        radial-gradient(ellipse at 20% 50%, rgba(59, 130, 246, 0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(245, 158, 11, 0.04) 0%, transparent 50%),
        var(--bg-primary);
}

/* ── Hide Streamlit Branding ─────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ─────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--gold), var(--blue), transparent);
}

/* ── Sidebar Radio Nav ───────────────────────── */
[data-testid="stSidebar"] .stRadio > label {
    color: var(--text-secondary) !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    padding: 10px 16px !important;
    margin: 3px 0 !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    color: var(--text-secondary) !important;
    font-size: 0.9rem !important;
    font-weight: 400 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    border: 1px solid transparent !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(245, 158, 11, 0.08) !important;
    color: var(--gold) !important;
    border-color: rgba(245, 158, 11, 0.2) !important;
}

/* ── Metric Cards ────────────────────────────── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--gold), transparent);
    opacity: 0.6;
}

.metric-card:hover {
    border-color: rgba(245, 158, 11, 0.3);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(245, 158, 11, 0.08);
}

.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 8px;
}

.metric-value {
    font-family: var(--font-mono);
    font-size: 1.8rem;
    font-weight: 500;
    color: var(--text-primary);
    line-height: 1;
}

.metric-value.gold { color: var(--gold); }
.metric-value.green { color: var(--green); }
.metric-value.red { color: var(--red); }
.metric-value.blue { color: var(--blue); }

.metric-delta {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    margin-top: 6px;
    color: var(--text-dim);
}

.metric-delta.up { color: var(--green); }
.metric-delta.down { color: var(--red); }

/* ── Section Headers ─────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 32px 0 20px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-subtle);
}

.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: 0.02em;
}

.section-badge {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    padding: 3px 10px;
    border-radius: 20px;
    background: rgba(245, 158, 11, 0.1);
    color: var(--gold);
    border: 1px solid rgba(245, 158, 11, 0.2);
    letter-spacing: 0.08em;
}

/* ── Prediction Table ────────────────────────── */
.pred-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 0.88rem;
}

.pred-table th {
    text-align: left;
    padding: 10px 16px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border-subtle);
}

.pred-table td {
    padding: 12px 16px;
    border-bottom: 1px solid rgba(99, 130, 201, 0.08);
    color: var(--text-secondary);
}

.pred-table tr:hover td {
    background: rgba(245, 158, 11, 0.04);
}

.pred-table td:first-child { color: var(--text-dim); }
.pred-table td.price { color: var(--text-primary); font-weight: 500; }
.pred-table td.up { color: var(--green); }
.pred-table td.down { color: var(--red); }

/* ── Model Comparison Cards ──────────────────── */
.model-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 20px;
    backdrop-filter: blur(8px);
    transition: all 0.25s ease;
}

.model-card.best {
    border-color: rgba(245, 158, 11, 0.4);
    box-shadow: 0 0 24px rgba(245, 158, 11, 0.06);
}

.model-card .model-name {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 4px;
}

.model-card .model-type {
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-bottom: 16px;
    font-family: var(--font-mono);
}

.model-card .best-badge {
    display: inline-block;
    font-size: 0.62rem;
    padding: 2px 8px;
    border-radius: 20px;
    background: rgba(245, 158, 11, 0.15);
    color: var(--gold);
    border: 1px solid rgba(245, 158, 11, 0.3);
    margin-left: 8px;
    vertical-align: middle;
    letter-spacing: 0.08em;
    font-weight: 600;
}

.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid rgba(99, 130, 201, 0.06);
}

.stat-row:last-child { border-bottom: none; }

.stat-label {
    font-size: 0.72rem;
    color: var(--text-dim);
    font-family: var(--font-mono);
}

.stat-value {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-weight: 500;
}

/* ── Hero Banner ─────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, 
        rgba(17, 28, 53, 0.9) 0%,
        rgba(13, 21, 41, 0.95) 100%
    );
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}

.hero-banner::after {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(245, 158, 11, 0.06), transparent 70%);
    pointer-events: none;
}

.hero-ticker {
    font-family: var(--font-mono);
    font-size: 3rem;
    font-weight: 500;
    color: var(--gold);
    line-height: 1;
}

.hero-price {
    font-family: var(--font-mono);
    font-size: 2rem;
    color: var(--text-primary);
    margin-top: 4px;
}

.hero-subtitle {
    font-size: 0.82rem;
    color: var(--text-dim);
    margin-top: 8px;
    letter-spacing: 0.05em;
}

/* ── Ticker Input ─────────────────────────────── */
.stTextInput input {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 1rem !important;
}

.stTextInput input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.15) !important;
}

/* ── Buttons ─────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--gold), var(--gold-dim)) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px 24px !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(245, 158, 11, 0.3) !important;
}

/* ── Selectbox & Slider ──────────────────────── */
.stSelectbox > div > div {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

.stSlider .stSlider > div {
    color: var(--gold) !important;
}

/* ── Tabs ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid var(--border-subtle) !important;
    gap: 4px !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 7px !important;
    color: var(--text-dim) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

.stTabs [aria-selected="true"] {
    background: var(--bg-secondary) !important;
    color: var(--gold) !important;
    border: 1px solid rgba(245, 158, 11, 0.2) !important;
}

/* ── Alerts & Info ───────────────────────────── */
.stInfo {
    background: rgba(96, 165, 250, 0.08) !important;
    border: 1px solid rgba(96, 165, 250, 0.2) !important;
    border-radius: 8px !important;
    color: var(--blue) !important;
}

.stSuccess {
    background: rgba(16, 185, 129, 0.08) !important;
    border: 1px solid rgba(16, 185, 129, 0.2) !important;
    border-radius: 8px !important;
}

.stWarning {
    background: rgba(245, 158, 11, 0.08) !important;
    border: 1px solid rgba(245, 158, 11, 0.2) !important;
    border-radius: 8px !important;
}

/* ── Spinner ─────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--gold) !important;
}

/* ── Divider ─────────────────────────────────── */
hr {
    border-color: var(--border-subtle) !important;
    margin: 24px 0 !important;
}

/* ── Scrollbar ───────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--text-dim); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold); }

/* ── Live Badge ──────────────────────────────── */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    padding: 4px 10px;
    border-radius: 20px;
    background: rgba(16, 185, 129, 0.1);
    color: var(--green);
    border: 1px solid rgba(16, 185, 129, 0.25);
    letter-spacing: 0.1em;
}

.live-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 1.8s ease-in-out infinite;
}
</style>
"""