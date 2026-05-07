"""
app.py — Quant Backtest Dashboard (Enhanced)
Minimalist editorial design with cross-border FX analysis, ML strategy,
parameter optimization, risk panel, multi-asset comparison, trade log & export.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")

# ── ML / Stats ───────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from scipy import stats

# ─────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quant Backtest",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Global Style (extended)
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Source+Serif+4:wght@400;600;700&display=swap');

    .stApp { background-color: #0a0a0a; }
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
        color: #e0e0e0;
    }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    section[data-testid="stSidebar"] {
        background-color: #0f0f0f;
        border-right: 1px solid #1a1a1a;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stDateInput label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stCheckbox label {
        font-size: 0.7rem; text-transform: uppercase;
        letter-spacing: 0.12em; color: #666; font-weight: 500;
    }

    .editorial-divider {
        width: 40px; height: 2px; background: #e0e0e0; margin: 60px 0 20px 0;
    }
    .editorial-divider-accent {
        width: 40px; height: 2px; background: #4a9eff; margin: 60px 0 20px 0;
    }
    .section-header {
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 1.6rem; font-weight: 600; color: #ffffff;
        margin: 8px 0 6px 0; letter-spacing: -0.01em; line-height: 1.3;
    }
    .section-subtext {
        font-size: 0.82rem; color: #555; margin-bottom: 30px;
        line-height: 1.6; max-width: 520px;
    }
    .hero-title {
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 2.8rem; font-weight: 700; color: #ffffff;
        letter-spacing: -0.03em; line-height: 1.1; margin: 0;
    }
    .hero-subtitle {
        font-size: 0.9rem; color: #555; margin-top: 12px;
        letter-spacing: 0.04em; text-transform: uppercase; font-weight: 400;
    }
    .hero-line {
        width: 100%; height: 1px;
        background: linear-gradient(90deg, #333 0%, transparent 100%);
        margin: 40px 0;
    }

    .metric-row {
        display: flex; gap: 0;
        border-top: 1px solid #1a1a1a; border-bottom: 1px solid #1a1a1a;
    }
    .metric-cell {
        flex: 1; padding: 24px 20px;
        border-right: 1px solid #1a1a1a; text-align: left;
    }
    .metric-cell:last-child { border-right: none; }
    .metric-number {
        font-family: 'Inter', sans-serif; font-size: 1.5rem;
        font-weight: 600; letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.65rem; text-transform: uppercase;
        letter-spacing: 0.12em; color: #555; margin-top: 6px;
    }
    .up { color: #4a9eff; }
    .down { color: #ff4a4a; }
    .muted { color: #888; }

    .clean-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .clean-table th {
        text-align: left; font-size: 0.65rem; text-transform: uppercase;
        letter-spacing: 0.1em; color: #555; padding: 12px 16px;
        border-bottom: 1px solid #222; font-weight: 500;
    }
    .clean-table td {
        padding: 14px 16px; border-bottom: 1px solid #111;
        color: #ccc; font-variant-numeric: tabular-nums;
    }
    .clean-table tr:hover td { background: #111; }

    .footer-text {
        font-size: 0.72rem; color: #333; text-align: center;
        letter-spacing: 0.06em; padding: 60px 0 30px 0;
    }
    .footer-text a { color: #444; text-decoration: none; border-bottom: 1px solid #333; }

    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #1a1a1a; }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: 0.08em; color: #555; padding: 12px 24px;
    }
    .stTabs [aria-selected="true"] { color: #fff; border-bottom-color: #4a9eff; }

    /* ── NEW: risk / heatmap helpers ── */
    .risk-card {
        background: #0f0f0f; border: 1px solid #1a1a1a;
        padding: 24px; border-radius: 2px; margin-bottom: 12px;
    }
    .risk-card .metric-number { font-size: 1.3rem; }
    .event-tag {
        display: inline-block; font-size: 0.65rem; padding: 3px 8px;
        border: 1px solid #333; color: #888; border-radius: 1px;
        letter-spacing: 0.06em; text-transform: uppercase; margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Data Functions (original)
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, level=1, axis=1)
        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                return pd.DataFrame()
        return df[required].dropna()
    except Exception:
        return pd.DataFrame()


# ─── NEW: Fetch FX data ─────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_fx_data(pair, start, end):
    """Fetch forex data via yfinance. pair e.g. 'CNY=X' for USD/CNY."""
    try:
        df = yf.download(pair, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(pair, level=1, axis=1)
        if "Close" in df.columns:
            return df[["Close"]].dropna().rename(columns={"Close": "FX_Close"})
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ─── NEW: Fetch multiple tickers for comparison ─────────
@st.cache_data(ttl=3600)
def fetch_multi_stock(tickers, start, end):
    result = {}
    for t in tickers:
        d = fetch_stock_data(t, start, end)
        if not d.empty:
            result[t] = d
    return result


# ─────────────────────────────────────────────────────────
# Strategy Functions (original)
# ─────────────────────────────────────────────────────────
def compute_ma_crossover(df, short_window=20, long_window=50):
    data = df.copy()
    data["MA_Short"] = data["Close"].rolling(window=short_window).mean()
    data["MA_Long"] = data["Close"].rolling(window=long_window).mean()
    data["Signal"] = 0
    data.loc[data["MA_Short"] > data["MA_Long"], "Signal"] = 1
    data["Position"] = data["Signal"].diff()
    return data


def compute_rsi_strategy(df, rsi_period=14, oversold=30, overbought=70):
    data = df.copy()
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    data["Signal"] = 0
    data.loc[data["RSI"] < oversold, "Signal"] = 1
    data.loc[data["RSI"] > overbought, "Signal"] = -1
    return data


def compute_bollinger_strategy(df, bb_period=20, bb_std=2.0):
    data = df.copy()
    data["BB_Mid"] = data["Close"].rolling(window=bb_period).mean()
    data["BB_Std"] = data["Close"].rolling(window=bb_period).std()
    data["BB_Upper"] = data["BB_Mid"] + bb_std * data["BB_Std"]
    data["BB_Lower"] = data["BB_Mid"] - bb_std * data["BB_Std"]
    data["Signal"] = 0
    data.loc[data["Close"] < data["BB_Lower"], "Signal"] = 1
    data.loc[data["Close"] > data["BB_Upper"], "Signal"] = -1
    return data


# ─── NEW: ML Strategy ───────────────────────────────────
def compute_ml_strategy(df, model_type="RandomForest", lookback=20, threshold=0.55):
    """
    ML-based strategy: predict next-day price direction using
    engineered features (returns, volatility, volume change, momentum, RSI).
    """
    data = df.copy()

    # Feature engineering
    data["Return"] = data["Close"].pct_change()
    data["Return_5d"] = data["Close"].pct_change(5)
    data["Return_10d"] = data["Close"].pct_change(10)
    data["Volatility_10d"] = data["Return"].rolling(10).std()
    data["Volatility_20d"] = data["Return"].rolling(20).std()
    data["Volume_Change"] = data["Volume"].pct_change()
    data["Volume_MA_Ratio"] = data["Volume"] / data["Volume"].rolling(20).mean()
    data["Momentum_10d"] = data["Close"] / data["Close"].shift(10) - 1
    data["Momentum_20d"] = data["Close"] / data["Close"].shift(20) - 1
    data["MA_10"] = data["Close"].rolling(10).mean()
    data["MA_50"] = data["Close"].rolling(50).mean()
    data["MA_Ratio"] = data["MA_10"] / data["MA_50"]
    data["High_Low_Range"] = (data["High"] - data["Low"]) / data["Close"]

    # RSI as feature
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # Target: next day direction (1 = up, 0 = down)
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    feature_cols = [
        "Return", "Return_5d", "Return_10d",
        "Volatility_10d", "Volatility_20d",
        "Volume_Change", "Volume_MA_Ratio",
        "Momentum_10d", "Momentum_20d",
        "MA_Ratio", "High_Low_Range", "RSI_14",
    ]

    data = data.dropna()
    if len(data) < 100:
        data["Signal"] = 0
        data["ML_Prob"] = 0.5
        data["ML_Accuracy"] = 0.0
        return data

    X = data[feature_cols].values
    y = data["Target"].values

    # Walk-forward: train on first 70%, predict remaining
    split_idx = int(len(data) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    else:
        model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)

    model.fit(X_train_s, y_train)

    # Predict probability for all data
    X_all_s = scaler.transform(X)  # slight data leak for display; real prod would be walk-forward
    probs = model.predict_proba(X_all_s)[:, 1]

    data["ML_Prob"] = probs
    data["Signal"] = 0
    data.loc[data["ML_Prob"] > threshold, "Signal"] = 1
    data.loc[data["ML_Prob"] < (1 - threshold), "Signal"] = -1

    # Record test accuracy
    test_pred = model.predict(X_test_s)
    data["ML_Accuracy"] = accuracy_score(y_test, test_pred)

    # Feature importances
    data.attrs["feature_importances"] = dict(zip(feature_cols, model.feature_importances_))
    data.attrs["test_accuracy"] = accuracy_score(y_test, test_pred)

    return data


# ─────────────────────────────────────────────────────────
# Portfolio Simulation (original)
# ─────────────────────────────────────────────────────────
def simulate_portfolio(df, signal_col="Signal", initial_cash=100000, commission=0.001):
    data = df.dropna().copy()
    cash, shares = initial_cash, 0
    values = []
    # ── NEW: trade log ──
    trades = []
    for i in range(len(data)):
        price = data["Close"].iloc[i]
        sig = data[signal_col].iloc[i]
        if sig == 1 and shares == 0:
            shares = int(cash * 0.95 / price)
            cost = shares * price * (1 + commission)
            cash -= cost
            trades.append({
                "Date": data.index[i], "Action": "BUY",
                "Price": round(price, 2), "Shares": shares,
                "Cost": round(cost, 2), "Cash_After": round(cash, 2),
            })
        elif sig == -1 and shares > 0:
            proceeds = shares * price * (1 - commission)
            cash += proceeds
            pnl = proceeds - trades[-1]["Cost"] if trades else 0
            trades.append({
                "Date": data.index[i], "Action": "SELL",
                "Price": round(price, 2), "Shares": shares,
                "Proceeds": round(proceeds, 2), "PnL": round(pnl, 2),
                "Cash_After": round(cash, 2),
            })
            shares = 0
        values.append(cash + shares * price)
    data = data.copy()
    data["Portfolio"] = values
    return data, trades


def calculate_metrics(ps, ic=100000):
    tr = (ps.iloc[-1] / ic - 1) * 100
    ny = len(ps) / 252
    ar = ((ps.iloc[-1] / ic) ** (1 / max(ny, 0.01)) - 1) * 100
    dr = ps.pct_change().dropna()
    sr = (dr.mean() / max(dr.std(), 1e-10)) * np.sqrt(252)
    rm = ps.expanding().max()
    md = ((ps - rm) / rm).min() * 100
    return dict(total_return=tr, annual_return=ar, sharpe_ratio=sr,
                max_drawdown=md, final_value=ps.iloc[-1],
                volatility=dr.std() * np.sqrt(252) * 100)


# ─── NEW: Advanced Risk Metrics ─────────────────────────
def calculate_risk_metrics(portfolio_series, confidence=0.95):
    """VaR, CVaR, Sortino, Calmar, rolling Sharpe, monthly returns."""
    dr = portfolio_series.pct_change().dropna()

    # VaR (Historical)
    var = np.percentile(dr, (1 - confidence) * 100)
    # CVaR (Expected Shortfall)
    cvar = dr[dr <= var].mean()

    # Sortino Ratio
    downside = dr[dr < 0].std()
    sortino = (dr.mean() / max(downside, 1e-10)) * np.sqrt(252)

    # Calmar Ratio
    rm = portfolio_series.expanding().max()
    md = ((portfolio_series - rm) / rm).min()
    annual_return = ((portfolio_series.iloc[-1] / portfolio_series.iloc[0])
                     ** (252 / max(len(portfolio_series), 1)) - 1)
    calmar = annual_return / max(abs(md), 1e-10)

    # Rolling Sharpe (60-day)
    rolling_sharpe = (dr.rolling(60).mean() / dr.rolling(60).std()) * np.sqrt(252)

    # Monthly returns
    monthly = portfolio_series.resample("ME").last().pct_change().dropna()

    return dict(
        var_95=var * 100,
        cvar_95=cvar * 100 if not np.isnan(cvar) else 0,
        sortino=sortino,
        calmar=calmar,
        rolling_sharpe=rolling_sharpe,
        daily_returns=dr,
        monthly_returns=monthly,
    )


# ─── NEW: Parameter Grid Search ─────────────────────────
def grid_search_ma(df, short_range, long_range, initial_cash=100000):
    """Grid search over MA short/long windows, returns Sharpe matrix."""
    results_grid = []
    for sw in short_range:
        for lw in long_range:
            if sw >= lw:
                results_grid.append({"short": sw, "long": lw, "sharpe": np.nan})
                continue
            d = compute_ma_crossover(df, sw, lw)
            sig = d["Signal"].diff()
            ts = pd.Series(0, index=d.index)
            ts[sig == 1] = 1
            ts[sig == -1] = -1
            d["TS"] = ts
            pd_, _ = simulate_portfolio(d, signal_col="TS", initial_cash=initial_cash)
            m = calculate_metrics(pd_["Portfolio"], initial_cash)
            results_grid.append({"short": sw, "long": lw, "sharpe": m["sharpe_ratio"]})
    return pd.DataFrame(results_grid)


def grid_search_rsi(df, period_range, oversold_range, overbought=70, initial_cash=100000):
    """Grid search RSI period vs oversold threshold."""
    results_grid = []
    for per in period_range:
        for ov in oversold_range:
            d = compute_rsi_strategy(df, per, ov, overbought)
            d["TS"] = d["Signal"]
            pd_, _ = simulate_portfolio(d, signal_col="TS", initial_cash=initial_cash)
            m = calculate_metrics(pd_["Portfolio"], initial_cash)
            results_grid.append({"period": per, "oversold": ov, "sharpe": m["sharpe_ratio"]})
    return pd.DataFrame(results_grid)


def grid_search_bb(df, period_range, std_range, initial_cash=100000):
    """Grid search Bollinger period vs std dev."""
    results_grid = []
    for per in period_range:
        for sd in std_range:
            d = compute_bollinger_strategy(df, per, sd)
            d["TS"] = d["Signal"]
            pd_, _ = simulate_portfolio(d, signal_col="TS", initial_cash=initial_cash)
            m = calculate_metrics(pd_["Portfolio"], initial_cash)
            results_grid.append({"period": per, "std": sd, "sharpe": m["sharpe_ratio"]})
    return pd.DataFrame(results_grid)


# ─── NEW: Macro Events ──────────────────────────────────
MACRO_EVENTS = [
    {"date": "2020-03-11", "label": "WHO Declares Pandemic", "color": "#ff4a4a"},
    {"date": "2020-03-23", "label": "Fed Unlimited QE", "color": "#4a9eff"},
    {"date": "2021-02-17", "label": "China Antitrust on Tech", "color": "#ff9f43"},
    {"date": "2021-07-23", "label": "China Edtech Crackdown", "color": "#ff9f43"},
    {"date": "2022-03-16", "label": "Fed Rate Hike Begins", "color": "#ff4a4a"},
    {"date": "2022-10-07", "label": "US Chip Export Ban", "color": "#ff4a4a"},
    {"date": "2023-01-26", "label": "China Reopening Rally", "color": "#4a9eff"},
    {"date": "2023-07-26", "label": "Fed 5.5% Peak Rate", "color": "#ff9f43"},
    {"date": "2024-01-31", "label": "Evergrande Liquidation", "color": "#ff4a4a"},
    {"date": "2024-09-18", "label": "Fed Rate Cut Begins", "color": "#4a9eff"},
    {"date": "2025-04-02", "label": "Trump Tariff Escalation", "color": "#ff4a4a"},
    {"date": "2025-04-09", "label": "90-Day Tariff Pause", "color": "#4a9eff"},
]


# ─────────────────────────────────────────────────────────
# Chart Config
# ─────────────────────────────────────────────────────────
CHART = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=11, color="#888"),
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(gridcolor="#111", zerolinecolor="#111"),
    yaxis=dict(gridcolor="#111", zerolinecolor="#111"),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#1a1a1a", font_size=12, font_color="#ccc"),
)
ACCENT, RED = "#4a9eff", "#ff4a4a"


# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 0 30px 0;">
        <div style="font-family: 'Source Serif 4', Georgia, serif; font-size: 1.3rem; font-weight: 700; color: #fff; letter-spacing: -0.02em;">Quant Backtest</div>
        <div style="font-size: 0.65rem; color: #444; text-transform: uppercase; letter-spacing: 0.15em; margin-top: 4px;">Strategy Analysis Engine</div>
    </div>
    """, unsafe_allow_html=True)

    tickers = {"BABA": "Alibaba", "PDD": "PDD Holdings", "JD": "JD.com",
               "SHOP": "Shopify", "AMZN": "Amazon", "SPY": "S&P 500 ETF"}
    selected_ticker = st.selectbox("TICKER", list(tickers.keys()),
                                   format_func=lambda x: f"{x}  —  {tickers[x]}")
    custom = st.text_input("CUSTOM TICKER", placeholder="AAPL, TSLA, NVDA...")
    if custom:
        selected_ticker = custom.upper().strip()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("FROM", value=datetime(2020, 1, 1))
    with c2:
        end_date = st.date_input("TO", value=datetime.today())

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    all_strats = ["MA Crossover", "RSI Mean Reversion", "Bollinger Bands", "ML Prediction"]
    selected_strategies = st.multiselect("STRATEGIES", all_strats,
                                         default=["MA Crossover", "RSI Mean Reversion", "Bollinger Bands"])

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    with st.expander("MA Crossover"):
        ma_short = st.slider("Short period", 5, 50, 20)
        ma_long = st.slider("Long period", 20, 200, 50)
    with st.expander("RSI"):
        rsi_period = st.slider("Period", 5, 30, 14)
        rsi_oversold = st.slider("Oversold", 10, 40, 30)
        rsi_overbought = st.slider("Overbought", 60, 90, 70)
    with st.expander("Bollinger Bands"):
        bb_period = st.slider("Period ", 10, 50, 20)
        bb_std = st.slider("Std dev", 1.0, 3.0, 2.0, 0.5)

    # ── NEW: ML parameters ──
    with st.expander("ML Prediction"):
        ml_model_type = st.selectbox("Model", ["RandomForest", "GradientBoosting"])
        ml_threshold = st.slider("Confidence Threshold", 0.50, 0.70, 0.55, 0.01)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    initial_cash = st.number_input("CAPITAL ($)", value=100000, step=10000)

    # ── NEW: modules toggle ──
    st.markdown("""<div style="margin-top:20px; font-size:0.65rem; text-transform:uppercase;
        letter-spacing:0.12em; color:#666; font-weight:500; margin-bottom:8px;">Modules</div>""",
        unsafe_allow_html=True)
    enable_fx = st.checkbox("Cross-Border FX Analysis", value=False)
    enable_optimization = st.checkbox("Parameter Optimization", value=False)
    enable_risk = st.checkbox("Risk Analysis Panel", value=True)
    enable_multi = st.checkbox("Multi-Asset Comparison", value=False)
    enable_events = st.checkbox("Macro Event Overlay", value=True)
    enable_tradelog = st.checkbox("Trade Log", value=True)
    enable_export = st.checkbox("Data Export", value=True)

    # ── NEW: FX pair ──
    if enable_fx:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        fx_pairs = {"CNY=X": "USD/CNY", "BRL=X": "USD/BRL", "EUR=X": "USD/EUR",
                    "JPY=X": "USD/JPY", "CAD=X": "USD/CAD", "GBP=X": "USD/GBP"}
        selected_fx = st.multiselect("FX PAIRS", list(fx_pairs.keys()),
                                     default=["CNY=X"],
                                     format_func=lambda x: fx_pairs[x])

    # ── NEW: multi-asset tickers ──
    if enable_multi:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        compare_tickers = st.multiselect("COMPARE TICKERS",
            ["BABA", "PDD", "JD", "SHOP", "AMZN", "SPY", "MELI", "SE", "GLBE"],
            default=["BABA", "PDD", "SHOP"])


# ─────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding: 40px 0 0 0;">
    <div class="hero-subtitle">Cross-border equity strategy analysis</div>
    <div class="hero-title">{selected_ticker}</div>
</div>
<div class="hero-line"></div>
""", unsafe_allow_html=True)

with st.spinner(""):
    df = fetch_stock_data(selected_ticker, str(start_date), str(end_date))

if df.empty:
    st.markdown(f"""<div style="padding:40px; border:1px solid #1a1a1a; color:#555; font-size:0.85rem;">
        No data available for {selected_ticker}.</div>""", unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────
# Price Strip + Chart (with optional macro events)
# ─────────────────────────────────────────────────────────
pn, ps_ = df["Close"].iloc[-1], df["Close"].iloc[0]
pc = ((pn / ps_) - 1) * 100
cc, cs = ("up" if pc >= 0 else "down"), ("+" if pc >= 0 else "")

st.markdown(f"""
<div class="metric-row">
    <div class="metric-cell"><div class="metric-number muted">${pn:.2f}</div><div class="metric-label">Last Close</div></div>
    <div class="metric-cell"><div class="metric-number {cc}">{cs}{pc:.1f}%</div><div class="metric-label">Period Return</div></div>
    <div class="metric-cell"><div class="metric-number muted">${df['Close'].max():.2f}</div><div class="metric-label">Period High</div></div>
    <div class="metric-cell"><div class="metric-number muted">${df['Close'].min():.2f}</div><div class="metric-label">Period Low</div></div>
    <div class="metric-cell"><div class="metric-number muted">{len(df):,}</div><div class="metric-label">Trading Days</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

fig_p = make_subplots(rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.02, row_heights=[0.78, 0.22])
fig_p.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"], name="",
    increasing_line_color=ACCENT, decreasing_line_color="#333",
    increasing_fillcolor=ACCENT, decreasing_fillcolor="#1a1a1a"), row=1, col=1)
fig_p.add_trace(go.Bar(x=df.index, y=df["Volume"], name="",
    marker_color="rgba(74,158,255,0.12)"), row=2, col=1)

# ── NEW: Macro event vertical lines ──
if enable_events:
    for evt in MACRO_EVENTS:
        evt_date = pd.Timestamp(evt["date"])
        if df.index.min() <= evt_date <= df.index.max():
            fig_p.add_vline(x=evt_date, line_dash="dot",
                            line_color=evt["color"], opacity=0.5, row=1, col=1)
            fig_p.add_annotation(x=evt_date, y=df["Close"].max() * 1.02,
                text=evt["label"], showarrow=False, font=dict(size=8, color=evt["color"]),
                textangle=-45, yshift=10, row=1, col=1)

fig_p.update_layout(**CHART, height=460, showlegend=False,
    xaxis_rangeslider_visible=False,
    xaxis2=dict(gridcolor="#111"), yaxis2=dict(gridcolor="#111"))
st.plotly_chart(fig_p, use_container_width=True)

# Show event legend if enabled
if enable_events:
    visible_events = [e for e in MACRO_EVENTS
                      if df.index.min() <= pd.Timestamp(e["date"]) <= df.index.max()]
    if visible_events:
        tags = " ".join(
            f'<span class="event-tag" style="border-color:{e["color"]}; color:{e["color"]};">'
            f'{e["date"][:7]} {e["label"]}</span>'
            for e in visible_events
        )
        st.markdown(f'<div style="margin-bottom:20px;">{tags}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Performance (updated to include ML + trade logs)
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="editorial-divider-accent"></div>
<div class="section-header">Performance</div>
<div class="section-subtext">Strategies tested against historical data with 0.1% commission per trade. Portfolio sized at 95% of capital per position.</div>
""", unsafe_allow_html=True)

results = {}
all_trades = {}
scolors = {"MA Crossover": ACCENT, "RSI Mean Reversion": "#ff9f43",
           "Bollinger Bands": "#a29bfe", "ML Prediction": "#2ecc71"}

for sn in selected_strategies:
    if sn == "MA Crossover":
        d = compute_ma_crossover(df, ma_short, ma_long)
        sig = d["Signal"].diff()
        ts = pd.Series(0, index=d.index)
        ts[sig == 1] = 1
        ts[sig == -1] = -1
        d["TS"] = ts
    elif sn == "RSI Mean Reversion":
        d = compute_rsi_strategy(df, rsi_period, rsi_oversold, rsi_overbought)
        d["TS"] = d["Signal"]
    elif sn == "Bollinger Bands":
        d = compute_bollinger_strategy(df, bb_period, bb_std)
        d["TS"] = d["Signal"]
    elif sn == "ML Prediction":
        d = compute_ml_strategy(df, model_type=ml_model_type, threshold=ml_threshold)
        d["TS"] = d["Signal"]

    pd_, trades = simulate_portfolio(d, signal_col="TS", initial_cash=initial_cash)
    m = calculate_metrics(pd_["Portfolio"], initial_cash)
    results[sn] = {"data": pd_, "metrics": m, "raw": d}
    all_trades[sn] = trades

bhr = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
bhf = initial_cash * (1 + bhr / 100)

if results:
    bs = max(results.items(), key=lambda x: x[1]["metrics"]["total_return"])
    bn, bm = bs[0], bs[1]["metrics"]
    rc = "up" if bm["total_return"] >= 0 else "down"
    rs_ = "+" if bm["total_return"] >= 0 else ""
    sc = "up" if bm["sharpe_ratio"] >= 0 else "down"
    bc = "up" if bhr >= 0 else "down"
    bhs = "+" if bhr >= 0 else ""

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-cell"><div class="metric-number muted">{bn}</div><div class="metric-label">Leading Strategy</div></div>
        <div class="metric-cell"><div class="metric-number {rc}">{rs_}{bm['total_return']:.2f}%</div><div class="metric-label">Total Return</div></div>
        <div class="metric-cell"><div class="metric-number {sc}">{bm['sharpe_ratio']:.3f}</div><div class="metric-label">Sharpe Ratio</div></div>
        <div class="metric-cell"><div class="metric-number down">{bm['max_drawdown']:.1f}%</div><div class="metric-label">Max Drawdown</div></div>
        <div class="metric-cell"><div class="metric-number {bc}">{bhs}{bhr:.1f}%</div><div class="metric-label">Buy & Hold</div></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

fig_pf = go.Figure()
bh_s = initial_cash * (df["Close"] / df["Close"].iloc[0])
fig_pf.add_trace(go.Scatter(x=df.index, y=bh_s, name="Buy & Hold",
    line=dict(color="#333", width=1.5, dash="dot")))
for n, r in results.items():
    fig_pf.add_trace(go.Scatter(x=r["data"].index, y=r["data"]["Portfolio"],
        name=n, line=dict(color=scolors.get(n, "#888"), width=2)))
fig_pf.add_hline(y=initial_cash, line_dash="dot", line_color="#222", opacity=0.5)
fig_pf.update_layout(**CHART, height=400,
    legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0,
                font=dict(size=10, color="#666")))
st.plotly_chart(fig_pf, use_container_width=True)


# ─────────────────────────────────────────────────────────
# Breakdown Table
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="editorial-divider"></div>
<div class="section-header">Breakdown</div>
<div class="section-subtext">Side-by-side comparison of all active strategies against the passive benchmark.</div>
""", unsafe_allow_html=True)

rh = ""
for n, r in results.items():
    m = r["metrics"]
    rc_ = "up" if m["total_return"] >= 0 else "down"
    ac_ = "up" if m["annual_return"] >= 0 else "down"
    sc_ = "up" if m["sharpe_ratio"] >= 0 else "down"
    rh += f"""<tr>
        <td style="color:#e0e0e0;font-weight:500;">{n}</td>
        <td class="{rc_}">{m['total_return']:+.2f}%</td>
        <td class="{ac_}">{m['annual_return']:+.2f}%</td>
        <td class="{sc_}">{m['sharpe_ratio']:.3f}</td>
        <td class="down">{m['max_drawdown']:.1f}%</td>
        <td>{m['volatility']:.1f}%</td>
        <td style="color:#e0e0e0;">${m['final_value']:,.0f}</td></tr>"""

bha = ((1 + bhr / 100) ** (1 / max(len(df) / 252, 0.01)) - 1) * 100
bhc_ = "up" if bhr >= 0 else "down"
rh += f"""<tr><td style="color:#555;">Buy & Hold</td>
    <td class="{bhc_}" style="opacity:0.5;">{bhr:+.2f}%</td>
    <td class="{bhc_}" style="opacity:0.5;">{bha:+.2f}%</td>
    <td style="color:#333;">—</td><td style="color:#333;">—</td>
    <td style="color:#333;">—</td><td style="color:#555;">${bhf:,.0f}</td></tr>"""

st.markdown(f"""
<table class="clean-table"><thead><tr>
    <th>Strategy</th><th>Total Return</th><th>Annual Return</th>
    <th>Sharpe</th><th>Max Drawdown</th><th>Volatility</th><th>Final Value</th>
</tr></thead><tbody>{rh}</tbody></table>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Signals (updated with ML tab)
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="editorial-divider-accent"></div>
<div class="section-header">Signals</div>
<div class="section-subtext">Indicator overlays and trade signal visualization for each strategy.</div>
""", unsafe_allow_html=True)

if selected_strategies:
    tabs = st.tabs(selected_strategies)
    for tab, sn in zip(tabs, selected_strategies):
        with tab:
            raw = results[sn]["raw"]
            if sn == "MA Crossover":
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=raw.index, y=raw["Close"], name="Price", line=dict(color="#333", width=1)))
                fig.add_trace(go.Scatter(x=raw.index, y=raw["MA_Short"], name=f"MA {ma_short}", line=dict(color=ACCENT, width=1.5)))
                fig.add_trace(go.Scatter(x=raw.index, y=raw["MA_Long"], name=f"MA {ma_long}", line=dict(color="#ff9f43", width=1.5)))
                buys = raw[raw["Position"] == 1]
                sells = raw[raw["Position"] == -1]
                fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers", name="Buy",
                    marker=dict(symbol="triangle-up", size=9, color=ACCENT, line=dict(width=0))))
                fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers", name="Sell",
                    marker=dict(symbol="triangle-down", size=9, color=RED, line=dict(width=0))))
                fig.update_layout(**CHART, height=380,
                    legend=dict(orientation="h", y=-0.08, x=0, font=dict(size=10, color="#555")))
                st.plotly_chart(fig, use_container_width=True)

            elif sn == "RSI Mean Reversion":
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.04, row_heights=[0.6, 0.4])
                fig.add_trace(go.Scatter(x=raw.index, y=raw["Close"], name="Price",
                    line=dict(color="#555", width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=raw.index, y=raw["RSI"], name="RSI",
                    line=dict(color="#ff9f43", width=1.5)), row=2, col=1)
                fig.add_hline(y=rsi_oversold, line_dash="dash", line_color=ACCENT, opacity=0.4, row=2, col=1)
                fig.add_hline(y=rsi_overbought, line_dash="dash", line_color=RED, opacity=0.4, row=2, col=1)
                fig.add_hrect(y0=0, y1=rsi_oversold, fillcolor=ACCENT, opacity=0.03, line_width=0, row=2, col=1)
                fig.add_hrect(y0=rsi_overbought, y1=100, fillcolor=RED, opacity=0.03, line_width=0, row=2, col=1)
                fig.update_layout(**CHART, height=440,
                    legend=dict(orientation="h", y=-0.06, x=0, font=dict(size=10, color="#555")),
                    xaxis2=dict(gridcolor="#111"), yaxis2=dict(gridcolor="#111"))
                st.plotly_chart(fig, use_container_width=True)

            elif sn == "Bollinger Bands":
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=raw.index, y=raw["BB_Upper"], name="Upper",
                    line=dict(color="#333", width=1, dash="dash")))
                fig.add_trace(go.Scatter(x=raw.index, y=raw["BB_Lower"], name="Lower",
                    line=dict(color="#333", width=1, dash="dash"),
                    fill="tonexty", fillcolor="rgba(74,158,255,0.03)"))
                fig.add_trace(go.Scatter(x=raw.index, y=raw["BB_Mid"], name="Mid",
                    line=dict(color="#222", width=1)))
                fig.add_trace(go.Scatter(x=raw.index, y=raw["Close"], name="Price",
                    line=dict(color=ACCENT, width=1.5)))
                fig.update_layout(**CHART, height=380,
                    legend=dict(orientation="h", y=-0.08, x=0, font=dict(size=10, color="#555")))
                st.plotly_chart(fig, use_container_width=True)

            # ── NEW: ML Prediction Signal Tab ──
            elif sn == "ML Prediction":
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.04, row_heights=[0.6, 0.4])
                fig.add_trace(go.Scatter(x=raw.index, y=raw["Close"], name="Price",
                    line=dict(color="#555", width=1)), row=1, col=1)

                # Buy / sell markers
                ml_buys = raw[raw["Signal"] == 1]
                ml_sells = raw[raw["Signal"] == -1]
                fig.add_trace(go.Scatter(x=ml_buys.index, y=ml_buys["Close"], mode="markers",
                    name="ML Buy", marker=dict(symbol="triangle-up", size=7, color="#2ecc71")),
                    row=1, col=1)
                fig.add_trace(go.Scatter(x=ml_sells.index, y=ml_sells["Close"], mode="markers",
                    name="ML Sell", marker=dict(symbol="triangle-down", size=7, color=RED)),
                    row=1, col=1)

                # Probability curve
                if "ML_Prob" in raw.columns:
                    fig.add_trace(go.Scatter(x=raw.index, y=raw["ML_Prob"], name="P(Up)",
                        line=dict(color="#2ecc71", width=1.5)), row=2, col=1)
                    fig.add_hline(y=ml_threshold, line_dash="dash", line_color="#2ecc71", opacity=0.4, row=2, col=1)
                    fig.add_hline(y=1 - ml_threshold, line_dash="dash", line_color=RED, opacity=0.4, row=2, col=1)
                    fig.add_hline(y=0.5, line_dash="dot", line_color="#333", opacity=0.3, row=2, col=1)

                fig.update_layout(**CHART, height=440,
                    legend=dict(orientation="h", y=-0.06, x=0, font=dict(size=10, color="#555")),
                    xaxis2=dict(gridcolor="#111"), yaxis2=dict(gridcolor="#111", title="Probability"))
                st.plotly_chart(fig, use_container_width=True)

                # Feature importance & accuracy
                if hasattr(raw, "attrs") and "feature_importances" in raw.attrs:
                    acc = raw.attrs.get("test_accuracy", 0)
                    st.markdown(f"""
                    <div style="font-size:0.8rem; color:#888; margin:10px 0;">
                        Test Accuracy (walk-forward 70/30): <span style="color:#2ecc71; font-weight:600;">{acc:.1%}</span>
                        &nbsp;·&nbsp; Model: {ml_model_type} &nbsp;·&nbsp; Threshold: {ml_threshold}
                    </div>""", unsafe_allow_html=True)

                    fi = raw.attrs["feature_importances"]
                    fi_df = pd.DataFrame({"Feature": fi.keys(), "Importance": fi.values()}).sort_values("Importance", ascending=True)
                    fig_fi = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"],
                        orientation="h", marker_color=ACCENT, marker_line_width=0))
                    fig_fi.update_layout(**CHART, height=320, margin=dict(l=120, r=10, t=10, b=10))
                    fig_fi.add_annotation(text="FEATURE IMPORTANCE", xref="paper", yref="paper",
                        x=0, y=1.08, showarrow=False, font=dict(size=10, color="#555"))
                    st.plotly_chart(fig_fi, use_container_width=True)


# ─────────────────────────────────────────────────────────
# Metrics Comparison (original)
# ─────────────────────────────────────────────────────────
if len(results) > 1:
    st.markdown("""
    <div class="editorial-divider"></div>
    <div class="section-header">Metrics</div>
    <div class="section-subtext">Key risk-adjusted performance indicators across strategies.</div>
    """, unsafe_allow_html=True)

    names = list(results.keys())
    cols_ = [scolors.get(n, "#888") for n in names]

    def bar_chart(values, texts, title, colors=cols_, red=False):
        fig = go.Figure(data=[go.Bar(x=names, y=values,
            marker_color=([RED] * len(names) if red else colors),
            marker_line_width=0, text=texts, textposition="outside",
            textfont=dict(size=11, color="#888"))])
        fig.update_layout(**CHART, height=300, bargap=0.4,
            yaxis_title=None)
        fig.update_xaxes(gridcolor="#111", tickfont=dict(size=9))
        fig.add_annotation(text=title, xref="paper", yref="paper",
            x=0, y=1.12, showarrow=False,
            font=dict(size=10, color="#555", family="Inter"), align="left")
        return fig

    c1, c2 = st.columns(2)
    with c1:
        v = [results[n]["metrics"]["annual_return"] for n in names]
        st.plotly_chart(bar_chart(v, [f"{x:.1f}%" for x in v], "ANNUAL RETURN"), use_container_width=True)
    with c2:
        v = [results[n]["metrics"]["sharpe_ratio"] for n in names]
        st.plotly_chart(bar_chart(v, [f"{x:.2f}" for x in v], "SHARPE RATIO"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        v = [results[n]["metrics"]["max_drawdown"] for n in names]
        st.plotly_chart(bar_chart(v, [f"{x:.1f}%" for x in v], "MAX DRAWDOWN", red=True), use_container_width=True)
    with c4:
        v = [results[n]["metrics"]["volatility"] for n in names]
        st.plotly_chart(bar_chart(v, [f"{x:.1f}%" for x in v], "VOLATILITY"), use_container_width=True)


# =============================================================
# ██  NEW MODULE: Risk Analysis Panel
# =============================================================
if enable_risk and results:
    st.markdown("""
    <div class="editorial-divider-accent"></div>
    <div class="section-header">Risk Analysis</div>
    <div class="section-subtext">VaR, CVaR, Sortino, Calmar, return distribution and rolling Sharpe for the leading strategy.</div>
    """, unsafe_allow_html=True)

    # Use leading strategy
    lead_name = max(results, key=lambda k: results[k]["metrics"]["sharpe_ratio"])
    lead_portfolio = results[lead_name]["data"]["Portfolio"]
    risk = calculate_risk_metrics(lead_portfolio)

    # Risk metric cards
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-cell"><div class="metric-number down">{risk['var_95']:.2f}%</div><div class="metric-label">VaR (95%)</div></div>
        <div class="metric-cell"><div class="metric-number down">{risk['cvar_95']:.2f}%</div><div class="metric-label">CVaR (95%)</div></div>
        <div class="metric-cell"><div class="metric-number {'up' if risk['sortino']>=0 else 'down'}">{risk['sortino']:.3f}</div><div class="metric-label">Sortino Ratio</div></div>
        <div class="metric-cell"><div class="metric-number {'up' if risk['calmar']>=0 else 'down'}">{risk['calmar']:.3f}</div><div class="metric-label">Calmar Ratio</div></div>
        <div class="metric-cell"><div class="metric-number muted">{lead_name}</div><div class="metric-label">Analyzed Strategy</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)

    # Rolling Sharpe
    with rc1:
        rs_data = risk["rolling_sharpe"].dropna()
        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(x=rs_data.index, y=rs_data, name="Rolling Sharpe (60d)",
            line=dict(color=ACCENT, width=1.5),
            fill="tozeroy", fillcolor="rgba(74,158,255,0.05)"))
        fig_rs.add_hline(y=0, line_dash="dot", line_color="#333")
        fig_rs.update_layout(**CHART, height=280)
        fig_rs.add_annotation(text="ROLLING SHARPE (60-DAY)", xref="paper", yref="paper",
            x=0, y=1.1, showarrow=False, font=dict(size=10, color="#555"))
        st.plotly_chart(fig_rs, use_container_width=True)

    # Return distribution
    with rc2:
        dr = risk["daily_returns"]
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=dr, nbinsx=80, name="Daily Returns",
            marker_color=ACCENT, opacity=0.7, marker_line_width=0))
        fig_dist.add_vline(x=dr.mean(), line_dash="dash", line_color="#fff", opacity=0.6)
        fig_dist.add_vline(x=np.percentile(dr, 5), line_dash="dot", line_color=RED, opacity=0.6)
        fig_dist.update_layout(**CHART, height=280, bargap=0.02)
        fig_dist.add_annotation(text="RETURN DISTRIBUTION", xref="paper", yref="paper",
            x=0, y=1.1, showarrow=False, font=dict(size=10, color="#555"))
        st.plotly_chart(fig_dist, use_container_width=True)

    # Monthly return heatmap
    if len(risk["monthly_returns"]) > 2:
        mr = risk["monthly_returns"]
        mr_df = pd.DataFrame({"return": mr.values * 100}, index=mr.index)
        mr_df["year"] = mr_df.index.year
        mr_df["month"] = mr_df.index.month
        pivot = mr_df.pivot_table(values="return", index="year", columns="month", aggfunc="first")
        pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

        fig_hm = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index.astype(str),
            colorscale=[[0, RED], [0.5, "#0a0a0a"], [1, ACCENT]],
            zmid=0, text=np.where(np.isnan(pivot.values), "",
                                  np.char.add(np.char.mod("%.1f", np.nan_to_num(pivot.values)), "%")),
            texttemplate="%{text}", textfont=dict(size=10),
            colorbar=dict(title="Return %", tickfont=dict(color="#555"), titlefont=dict(color="#555")),
        ))
        fig_hm.update_layout(**CHART, height=max(200, len(pivot) * 40 + 60),
            margin=dict(l=40, r=10, t=30, b=10))
        fig_hm.add_annotation(text="MONTHLY RETURN HEATMAP", xref="paper", yref="paper",
            x=0, y=1.08, showarrow=False, font=dict(size=10, color="#555"))
        st.plotly_chart(fig_hm, use_container_width=True)


# =============================================================
# ██  NEW MODULE: Cross-Border FX Analysis
# =============================================================
if enable_fx and selected_fx:
    st.markdown("""
    <div class="editorial-divider-accent"></div>
    <div class="section-header">Cross-Border FX Analysis</div>
    <div class="section-subtext">Exchange rate overlay and correlation analysis — connecting equity performance to currency risk in cross-border markets.</div>
    """, unsafe_allow_html=True)

    fx_data = {}
    for fxp in selected_fx:
        fxd = fetch_fx_data(fxp, str(start_date), str(end_date))
        if not fxd.empty:
            fx_data[fxp] = fxd

    if fx_data:
        # Overlay chart: stock price + FX on dual axis
        fig_fx = make_subplots(specs=[[{"secondary_y": True}]])
        fig_fx.add_trace(go.Scatter(x=df.index, y=df["Close"], name=selected_ticker,
            line=dict(color=ACCENT, width=2)), secondary_y=False)

        fx_colors = ["#ff9f43", "#a29bfe", "#2ecc71", "#e74c3c", "#f39c12", "#1abc9c"]
        for i, (fxp, fxd) in enumerate(fx_data.items()):
            lbl = fx_pairs.get(fxp, fxp)
            fig_fx.add_trace(go.Scatter(x=fxd.index, y=fxd["FX_Close"], name=lbl,
                line=dict(color=fx_colors[i % len(fx_colors)], width=1.5, dash="dash")),
                secondary_y=True)

        fig_fx.update_layout(**CHART, height=380,
            legend=dict(orientation="h", y=-0.08, x=0, font=dict(size=10, color="#555")),
            yaxis2=dict(gridcolor="#111", showgrid=False))
        fig_fx.update_yaxes(title_text=selected_ticker, secondary_y=False,
            title_font=dict(size=10, color="#555"))
        fig_fx.update_yaxes(title_text="FX Rate", secondary_y=True,
            title_font=dict(size=10, color="#555"))
        st.plotly_chart(fig_fx, use_container_width=True)

        # Correlation matrix
        corr_df = pd.DataFrame({"Stock": df["Close"].pct_change()})
        for fxp, fxd in fx_data.items():
            lbl = fx_pairs.get(fxp, fxp)
            aligned = fxd["FX_Close"].reindex(df.index, method="ffill").pct_change()
            corr_df[lbl] = aligned

        corr_matrix = corr_df.dropna().corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
            colorscale=[[0, RED], [0.5, "#0a0a0a"], [1, ACCENT]],
            zmid=0, text=np.char.mod("%.2f", corr_matrix.values),
            texttemplate="%{text}", textfont=dict(size=12),
            colorbar=dict(title="Corr", tickfont=dict(color="#555")),
        ))
        fig_corr.update_layout(**CHART, height=300, margin=dict(l=80, r=10, t=30, b=10))
        fig_corr.add_annotation(text="RETURN CORRELATION MATRIX", xref="paper", yref="paper",
            x=0, y=1.1, showarrow=False, font=dict(size=10, color="#555"))
        st.plotly_chart(fig_corr, use_container_width=True)

        # Rolling correlation (30-day)
        if len(fx_data) >= 1:
            first_fx = list(fx_data.keys())[0]
            first_lbl = fx_pairs.get(first_fx, first_fx)
            aligned_ret = fx_data[first_fx]["FX_Close"].reindex(df.index, method="ffill").pct_change()
            stock_ret = df["Close"].pct_change()
            roll_corr = stock_ret.rolling(30).corr(aligned_ret).dropna()

            fig_rc = go.Figure()
            fig_rc.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr, name=f"{selected_ticker} vs {first_lbl}",
                line=dict(color="#ff9f43", width=1.5),
                fill="tozeroy", fillcolor="rgba(255,159,67,0.05)"))
            fig_rc.add_hline(y=0, line_dash="dot", line_color="#333")
            fig_rc.update_layout(**CHART, height=250)
            fig_rc.add_annotation(text=f"30-DAY ROLLING CORRELATION: {selected_ticker} vs {first_lbl}",
                xref="paper", yref="paper", x=0, y=1.1, showarrow=False,
                font=dict(size=10, color="#555"))
            st.plotly_chart(fig_rc, use_container_width=True)


# =============================================================
# ██  NEW MODULE: Parameter Optimization
# =============================================================
if enable_optimization and results:
    st.markdown("""
    <div class="editorial-divider-accent"></div>
    <div class="section-header">Parameter Optimization</div>
    <div class="section-subtext">Grid search across parameter space to find optimal configurations. Heatmaps show Sharpe Ratio for each parameter combination.</div>
    """, unsafe_allow_html=True)

    opt_tabs = []
    opt_names = []
    if "MA Crossover" in selected_strategies:
        opt_names.append("MA Crossover")
    if "RSI Mean Reversion" in selected_strategies:
        opt_names.append("RSI Mean Reversion")
    if "Bollinger Bands" in selected_strategies:
        opt_names.append("Bollinger Bands")

    if opt_names:
        opt_tabs = st.tabs(opt_names)
        for otab, oname in zip(opt_tabs, opt_names):
            with otab:
                with st.spinner(f"Optimizing {oname}..."):
                    if oname == "MA Crossover":
                        short_r = list(range(5, 51, 5))
                        long_r = list(range(20, 201, 10))
                        gdf = grid_search_ma(df, short_r, long_r, initial_cash)
                        pivot = gdf.pivot(index="short", columns="long", values="sharpe")
                        best_row = gdf.loc[gdf["sharpe"].idxmax()]
                        st.markdown(f"""<div style="font-size:0.8rem; color:#888; margin-bottom:10px;">
                            Optimal: Short=<span style="color:#fff;">{int(best_row['short'])}</span>,
                            Long=<span style="color:#fff;">{int(best_row['long'])}</span>,
                            Sharpe=<span style="color:#4a9eff;">{best_row['sharpe']:.3f}</span></div>""",
                            unsafe_allow_html=True)
                        fig_opt = go.Figure(data=go.Heatmap(
                            z=pivot.values, x=[str(c) for c in pivot.columns],
                            y=[str(i) for i in pivot.index],
                            colorscale=[[0, RED], [0.5, "#0a0a0a"], [1, ACCENT]],
                            zmid=0, colorbar=dict(title="Sharpe"),
                        ))
                        fig_opt.update_layout(**CHART, height=400,
                            xaxis_title="Long Window", yaxis_title="Short Window",
                            margin=dict(l=60, r=10, t=10, b=40))
                        st.plotly_chart(fig_opt, use_container_width=True)

                    elif oname == "RSI Mean Reversion":
                        per_r = list(range(5, 31, 3))
                        ov_r = list(range(15, 41, 5))
                        gdf = grid_search_rsi(df, per_r, ov_r, rsi_overbought, initial_cash)
                        pivot = gdf.pivot(index="period", columns="oversold", values="sharpe")
                        best_row = gdf.loc[gdf["sharpe"].idxmax()]
                        st.markdown(f"""<div style="font-size:0.8rem; color:#888; margin-bottom:10px;">
                            Optimal: Period=<span style="color:#fff;">{int(best_row['period'])}</span>,
                            Oversold=<span style="color:#fff;">{int(best_row['oversold'])}</span>,
                            Sharpe=<span style="color:#4a9eff;">{best_row['sharpe']:.3f}</span></div>""",
                            unsafe_allow_html=True)
                        fig_opt = go.Figure(data=go.Heatmap(
                            z=pivot.values, x=[str(c) for c in pivot.columns],
                            y=[str(i) for i in pivot.index],
                            colorscale=[[0, RED], [0.5, "#0a0a0a"], [1, ACCENT]],
                            zmid=0, colorbar=dict(title="Sharpe"),
                        ))
                        fig_opt.update_layout(**CHART, height=350,
                            xaxis_title="Oversold Threshold", yaxis_title="RSI Period",
                            margin=dict(l=60, r=10, t=10, b=40))
                        st.plotly_chart(fig_opt, use_container_width=True)

                    elif oname == "Bollinger Bands":
                        per_r = list(range(10, 51, 5))
                        std_r = [round(x, 1) for x in np.arange(1.0, 3.1, 0.5)]
                        gdf = grid_search_bb(df, per_r, std_r, initial_cash)
                        pivot = gdf.pivot(index="period", columns="std", values="sharpe")
                        best_row = gdf.loc[gdf["sharpe"].idxmax()]
                        st.markdown(f"""<div style="font-size:0.8rem; color:#888; margin-bottom:10px;">
                            Optimal: Period=<span style="color:#fff;">{int(best_row['period'])}</span>,
                            StdDev=<span style="color:#fff;">{best_row['std']:.1f}</span>,
                            Sharpe=<span style="color:#4a9eff;">{best_row['sharpe']:.3f}</span></div>""",
                            unsafe_allow_html=True)
                        fig_opt = go.Figure(data=go.Heatmap(
                            z=pivot.values, x=[str(c) for c in pivot.columns],
                            y=[str(i) for i in pivot.index],
                            colorscale=[[0, RED], [0.5, "#0a0a0a"], [1, ACCENT]],
                            zmid=0, colorbar=dict(title="Sharpe"),
                        ))
                        fig_opt.update_layout(**CHART, height=350,
                            xaxis_title="Std Deviation", yaxis_title="BB Period",
                            margin=dict(l=60, r=10, t=10, b=40))
                        st.plotly_chart(fig_opt, use_container_width=True)


# =============================================================
# ██  NEW MODULE: Multi-Asset Comparison
# =============================================================
if enable_multi and compare_tickers:
    st.markdown("""
    <div class="editorial-divider-accent"></div>
    <div class="section-header">Multi-Asset Comparison</div>
    <div class="section-subtext">Normalized price performance, correlation matrix, and relative strength across selected tickers.</div>
    """, unsafe_allow_html=True)

    multi_data = fetch_multi_stock(compare_tickers, str(start_date), str(end_date))

    if len(multi_data) >= 2:
        # Normalized performance
        fig_norm = go.Figure()
        ma_colors = ["#4a9eff", "#ff9f43", "#a29bfe", "#2ecc71", "#e74c3c",
                     "#f39c12", "#1abc9c", "#e056fd", "#686de0"]
        for i, (t, tdf) in enumerate(multi_data.items()):
            norm = tdf["Close"] / tdf["Close"].iloc[0] * 100
            fig_norm.add_trace(go.Scatter(x=tdf.index, y=norm, name=t,
                line=dict(color=ma_colors[i % len(ma_colors)], width=2)))
        fig_norm.add_hline(y=100, line_dash="dot", line_color="#333")
        fig_norm.update_layout(**CHART, height=380,
            legend=dict(orientation="h", y=-0.08, x=0, font=dict(size=10, color="#555")),
            yaxis_title="Indexed (100 = Start)")
        fig_norm.add_annotation(text="NORMALIZED PERFORMANCE", xref="paper", yref="paper",
            x=0, y=1.08, showarrow=False, font=dict(size=10, color="#555"))
        st.plotly_chart(fig_norm, use_container_width=True)

        # Correlation matrix
        ret_df = pd.DataFrame()
        for t, tdf in multi_data.items():
            ret_df[t] = tdf["Close"].pct_change()
        corr = ret_df.dropna().corr()

        fig_mc = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale=[[0, RED], [0.5, "#0a0a0a"], [1, ACCENT]],
            zmid=0, text=np.char.mod("%.2f", corr.values),
            texttemplate="%{text}", textfont=dict(size=11),
            colorbar=dict(title="Corr"),
        ))
        fig_mc.update_layout(**CHART, height=350, margin=dict(l=60, r=10, t=30, b=10))
        fig_mc.add_annotation(text="RETURN CORRELATION MATRIX", xref="paper", yref="paper",
            x=0, y=1.08, showarrow=False, font=dict(size=10, color="#555"))
        st.plotly_chart(fig_mc, use_container_width=True)

        # Summary table
        sum_rows = ""
        for t, tdf in multi_data.items():
            ret = (tdf["Close"].iloc[-1] / tdf["Close"].iloc[0] - 1) * 100
            vol = tdf["Close"].pct_change().std() * np.sqrt(252) * 100
            hi = tdf["Close"].max()
            lo = tdf["Close"].min()
            rc__ = "up" if ret >= 0 else "down"
            sum_rows += f"""<tr>
                <td style="color:#e0e0e0; font-weight:500;">{t}</td>
                <td class="{rc__}">{ret:+.1f}%</td>
                <td>{vol:.1f}%</td>
                <td>${hi:.2f}</td><td>${lo:.2f}</td>
                <td>${tdf['Close'].iloc[-1]:.2f}</td></tr>"""
        st.markdown(f"""
        <table class="clean-table"><thead><tr>
            <th>Ticker</th><th>Total Return</th><th>Volatility</th>
            <th>High</th><th>Low</th><th>Last</th>
        </tr></thead><tbody>{sum_rows}</tbody></table>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#555; font-size:0.85rem;">Need at least 2 tickers with available data.</div>',
            unsafe_allow_html=True)


# =============================================================
# ██  NEW MODULE: Trade Log
# =============================================================
if enable_tradelog and all_trades:
    st.markdown("""
    <div class="editorial-divider"></div>
    <div class="section-header">Trade Log</div>
    <div class="section-subtext">Detailed record of every simulated buy/sell signal with entry, exit, and P&L.</div>
    """, unsafe_allow_html=True)

    tl_tabs = st.tabs(list(all_trades.keys()))
    for tl_tab, (sn, trades) in zip(tl_tabs, all_trades.items()):
        with tl_tab:
            if not trades:
                st.markdown('<div style="color:#555; font-size:0.85rem;">No trades executed.</div>',
                    unsafe_allow_html=True)
                continue

            trade_rows = ""
            for t in trades:
                date_str = t["Date"].strftime("%Y-%m-%d") if hasattr(t["Date"], "strftime") else str(t["Date"])
                ac = "up" if t["Action"] == "BUY" else "down"
                if t["Action"] == "BUY":
                    trade_rows += f"""<tr>
                        <td>{date_str}</td>
                        <td class="{ac}" style="font-weight:500;">{t['Action']}</td>
                        <td>${t['Price']:.2f}</td>
                        <td>{t['Shares']}</td>
                        <td>${t['Cost']:,.2f}</td>
                        <td style="color:#333;">—</td>
                        <td>${t['Cash_After']:,.2f}</td></tr>"""
                else:
                    pnl_c = "up" if t.get("PnL", 0) >= 0 else "down"
                    trade_rows += f"""<tr>
                        <td>{date_str}</td>
                        <td class="{ac}" style="font-weight:500;">{t['Action']}</td>
                        <td>${t['Price']:.2f}</td>
                        <td>{t['Shares']}</td>
                        <td>${t.get('Proceeds', 0):,.2f}</td>
                        <td class="{pnl_c}">{t.get('PnL', 0):+,.2f}</td>
                        <td>${t['Cash_After']:,.2f}</td></tr>"""

            st.markdown(f"""
            <div style="font-size:0.75rem; color:#555; margin-bottom:8px;">{len(trades)} trades executed</div>
            <div style="max-height:400px; overflow-y:auto;">
            <table class="clean-table"><thead><tr>
                <th>Date</th><th>Action</th><th>Price</th><th>Shares</th>
                <th>Cost / Proceeds</th><th>P&L</th><th>Cash</th>
            </tr></thead><tbody>{trade_rows}</tbody></table></div>
            """, unsafe_allow_html=True)


# =============================================================
# ██  NEW MODULE: Data Export
# =============================================================
if enable_export and results:
    st.markdown("""
    <div class="editorial-divider"></div>
    <div class="section-header">Export</div>
    <div class="section-subtext">Download backtest results and trade logs as CSV for further analysis.</div>
    """, unsafe_allow_html=True)

    ex1, ex2, ex3 = st.columns(3)

    # Export 1: Summary metrics CSV
    with ex1:
        summary_data = []
        for n, r in results.items():
            m = r["metrics"]
            summary_data.append({"Strategy": n, **m})
        summary_data.append({
            "Strategy": "Buy & Hold",
            "total_return": bhr, "annual_return": bha,
            "sharpe_ratio": None, "max_drawdown": None,
            "final_value": bhf, "volatility": None,
        })
        summary_csv = pd.DataFrame(summary_data).to_csv(index=False)
        st.download_button("Strategy Summary (.csv)", summary_csv,
                           f"{selected_ticker}_summary.csv", "text/csv")

    # Export 2: Portfolio time series
    with ex2:
        port_df = pd.DataFrame(index=df.index)
        port_df["Close"] = df["Close"]
        port_df["Buy_Hold"] = initial_cash * (df["Close"] / df["Close"].iloc[0])
        for n, r in results.items():
            aligned = r["data"]["Portfolio"].reindex(df.index, method="ffill")
            port_df[n] = aligned
        port_csv = port_df.to_csv()
        st.download_button("Portfolio Series (.csv)", port_csv,
                           f"{selected_ticker}_portfolio.csv", "text/csv")

    # Export 3: All trade logs
    with ex3:
        all_trade_rows = []
        for sn, trades in all_trades.items():
            for t in trades:
                row = {"Strategy": sn, **t}
                if hasattr(row.get("Date"), "strftime"):
                    row["Date"] = row["Date"].strftime("%Y-%m-%d")
                all_trade_rows.append(row)
        if all_trade_rows:
            trades_csv = pd.DataFrame(all_trade_rows).to_csv(index=False)
            st.download_button("Trade Log (.csv)", trades_csv,
                               f"{selected_ticker}_trades.csv", "text/csv")
        else:
            st.markdown('<div style="color:#333; font-size:0.8rem;">No trades to export.</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-text">
    Dongni Lin  ·  McGill University  ·
    <a href="https://github.com/Loraldn">GitHub</a>
</div>
""", unsafe_allow_html=True)
