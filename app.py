"""
app.py — Quant Backtest Dashboard
Minimalist editorial design. Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

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
# Theme
# ─────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "night"

THEMES = {
    "night": {
        "bg":           "#0a0a0a",
        "sidebar_bg":   "#0f0f0f",
        "border":       "#1a1a1a",
        "text":         "#e0e0e0",
        "text_strong":  "#ffffff",
        "text_mute":    "#555",
        "text_dim":     "#333",
        "text_label":   "#666",
        "hover_bg":     "#111",
        "table_border": "#222",
        "table_row":    "#111",
        "grid":         "#111",
        "hover_panel":  "#1a1a1a",
        "hover_text":   "#ccc",
        "accent":       "#4a9eff",
        "red":          "#ff4a4a",
        "candle_down":  "#333",
        "candle_dfill": "#1a1a1a",
        "vol_color":    "rgba(74,158,255,0.12)",
        "bh_line":      "#333",
        "plotly_tpl":   "plotly_dark",
    },
    "day": {
        "bg":           "#fafaf7",
        "sidebar_bg":   "#f0ede5",
        "border":       "#e2ddd0",
        "text":         "#2a2a2a",
        "text_strong":  "#0a0a0a",
        "text_mute":    "#888",
        "text_dim":     "#bbb",
        "text_label":   "#777",
        "hover_bg":     "#f0ede5",
        "table_border": "#d8d3c4",
        "table_row":    "#f0ede5",
        "grid":         "#ece8dc",
        "hover_panel":  "#ffffff",
        "hover_text":   "#2a2a2a",
        "accent":       "#d97706",   # 暖橙(蓝的反差)
        "red":          "#b91c1c",
        "candle_down":  "#cfc8b8",
        "candle_dfill": "#e8e3d4",
        "vol_color":    "rgba(217,119,6,0.18)",
        "bh_line":      "#bbb",
        "plotly_tpl":   "plotly_white",
    },
}
T = THEMES[st.session_state.theme]
# ─────────────────────────────────────────────────────────
# Global Style
# ─────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Source+Serif+4:wght@400;600;700&display=swap');

    .stApp {{ background-color: {T['bg']}; }}
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, sans-serif;
        color: {T['text']};
    }}
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    section[data-testid="stSidebar"] {{
        background-color: {T['sidebar_bg']};
        border-right: 1px solid {T['border']};
    }}
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stDateInput label,
    section[data-testid="stSidebar"] .stNumberInput label {{
        font-size: 0.7rem; text-transform: uppercase;
        letter-spacing: 0.12em; color: {T['text_label']}; font-weight: 500;
    }}

    .editorial-divider {{
        width: 40px; height: 2px; background: {T['text']}; margin: 60px 0 20px 0;
    }}
    .editorial-divider-accent {{
        width: 40px; height: 2px; background: {T['accent']}; margin: 60px 0 20px 0;
    }}
    .section-header {{
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 1.6rem; font-weight: 600; color: {T['text_strong']};
        margin: 8px 0 6px 0; letter-spacing: -0.01em; line-height: 1.3;
    }}
    .section-subtext {{
        font-size: 0.82rem; color: {T['text_mute']}; margin-bottom: 30px;
        line-height: 1.6; max-width: 520px;
    }}
    .hero-title {{
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 2.8rem; font-weight: 700; color: {T['text_strong']};
        letter-spacing: -0.03em; line-height: 1.1; margin: 0;
    }}
    .hero-subtitle {{
        font-size: 0.9rem; color: {T['text_mute']}; margin-top: 12px;
        letter-spacing: 0.04em; text-transform: uppercase; font-weight: 400;
    }}
    .hero-line {{
        width: 100%; height: 1px;
        background: linear-gradient(90deg, {T['table_border']} 0%, transparent 100%);
        margin: 40px 0;
    }}

    .metric-row {{
        display: flex; gap: 0;
        border-top: 1px solid {T['border']}; border-bottom: 1px solid {T['border']};
    }}
    .metric-cell {{
        flex: 1; padding: 24px 20px;
        border-right: 1px solid {T['border']}; text-align: left;
    }}
    .metric-cell:last-child {{ border-right: none; }}
    .metric-number {{
        font-family: 'Inter', sans-serif; font-size: 1.5rem;
        font-weight: 600; letter-spacing: -0.02em;
    }}
    .metric-label {{
        font-size: 0.65rem; text-transform: uppercase;
        letter-spacing: 0.12em; color: {T['text_mute']}; margin-top: 6px;
    }}
    .up {{ color: {T['accent']}; }}
    .down {{ color: {T['red']}; }}
    .muted {{ color: {T['text_label']}; }}

    .clean-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    .clean-table th {{
        text-align: left; font-size: 0.65rem; text-transform: uppercase;
        letter-spacing: 0.1em; color: {T['text_mute']}; padding: 12px 16px;
        border-bottom: 1px solid {T['table_border']}; font-weight: 500;
    }}
    .clean-table td {{
        padding: 14px 16px; border-bottom: 1px solid {T['table_row']};
        color: {T['text']}; font-variant-numeric: tabular-nums;
    }}
    .clean-table tr:hover td {{ background: {T['hover_bg']}; }}

    .footer-text {{
        font-size: 0.72rem; color: {T['text_dim']}; text-align: center;
        letter-spacing: 0.06em; padding: 60px 0 30px 0;
    }}
    .footer-text a {{ color: {T['text_label']}; text-decoration: none; border-bottom: 1px solid {T['text_dim']}; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {T['border']}; }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: 0.08em; color: {T['text_mute']}; padding: 12px 24px;
    }}
    .stTabs [aria-selected="true"] {{ color: {T['text_strong']}; border-bottom-color: {T['accent']}; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Data Functions
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


def simulate_portfolio(df, signal_col="Signal", initial_cash=100000, commission=0.001):
    data = df.dropna().copy()
    cash, shares = initial_cash, 0
    values = []
    for i in range(len(data)):
        price = data["Close"].iloc[i]
        sig = data[signal_col].iloc[i]
        if sig == 1 and shares == 0:
            shares = int(cash * 0.95 / price)
            cash -= shares * price * (1 + commission)
        elif sig == -1 and shares > 0:
            cash += shares * price * (1 - commission)
            shares = 0
        values.append(cash + shares * price)
    data = data.copy()
    data["Portfolio"] = values
    return data


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


CHART = dict(
    template=T["plotly_tpl"],
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=11, color=T["text_mute"]),
    xaxis=dict(gridcolor=T["grid"], zerolinecolor=T["grid"]),
    yaxis=dict(gridcolor=T["grid"], zerolinecolor=T["grid"]),
    hovermode="x unified",
    hoverlabel=dict(bgcolor=T["hover_panel"], font_size=12, font_color=T["hover_text"]),
)
ACCENT, RED = T["accent"], T["red"]


# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    # 主题切换
    is_day = st.toggle("☀  Day mode", value=(st.session_state.theme == "day"))
    new_theme = "day" if is_day else "night"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()
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
    with c1: start_date = st.date_input("FROM", value=datetime(2020, 1, 1))
    with c2: end_date = st.date_input("TO", value=datetime.today())

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    selected_strategies = st.multiselect("STRATEGIES",
        ["MA Crossover", "RSI Mean Reversion", "Bollinger Bands"],
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

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    initial_cash = st.number_input("CAPITAL ($)", value=100000, step=10000)


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
# Price Strip + Chart
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
    increasing_line_color=ACCENT, decreasing_line_color=T["candle_down"],
    increasing_fillcolor=ACCENT, decreasing_fillcolor=T["candle_dfill"]), row=1, col=1)
fig_p.add_trace(go.Bar(x=df.index, y=df["Volume"], name="",
    marker_color=T["vol_color"]), row=2, col=1)
fig_p.add_trace(go.Bar(x=df.index, y=df["Volume"], name="",
    marker_color="rgba(74,158,255,0.12)"), row=2, col=1)
fig_p.update_layout(**CHART, height=460, showlegend=False,
    xaxis_rangeslider_visible=False,
    xaxis2=dict(gridcolor=T["grid"]), yaxis2=dict(gridcolor=T["grid"]))
st.plotly_chart(fig_p, use_container_width=True)


# ─────────────────────────────────────────────────────────
# Performance
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="editorial-divider-accent"></div>
<div class="section-header">Performance</div>
<div class="section-subtext">Three strategies tested against historical data with 0.1% commission per trade. Portfolio sized at 95% of capital per position.</div>
""", unsafe_allow_html=True)

results = {}
scolors = {"MA Crossover": ACCENT, "RSI Mean Reversion": "#ff9f43", "Bollinger Bands": "#a29bfe"}

for sn in selected_strategies:
    if sn == "MA Crossover":
        d = compute_ma_crossover(df, ma_short, ma_long)
        sig = d["Signal"].diff()
        ts = pd.Series(0, index=d.index); ts[sig==1]=1; ts[sig==-1]=-1
        d["TS"] = ts
    elif sn == "RSI Mean Reversion":
        d = compute_rsi_strategy(df, rsi_period, rsi_oversold, rsi_overbought)
        d["TS"] = d["Signal"]
    elif sn == "Bollinger Bands":
        d = compute_bollinger_strategy(df, bb_period, bb_std)
        d["TS"] = d["Signal"]
    pd_ = simulate_portfolio(d, signal_col="TS", initial_cash=initial_cash)
    m = calculate_metrics(pd_["Portfolio"], initial_cash)
    results[sn] = {"data": pd_, "metrics": m, "raw": d}

bhr = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
bhf = initial_cash * (1 + bhr / 100)

if results:
    bs = max(results.items(), key=lambda x: x[1]["metrics"]["total_return"])
    bn, bm = bs[0], bs[1]["metrics"]
    rc = "up" if bm["total_return"]>=0 else "down"
    rs_ = "+" if bm["total_return"]>=0 else ""
    sc = "up" if bm["sharpe_ratio"]>=0 else "down"
    bc = "up" if bhr>=0 else "down"
    bhs = "+" if bhr>=0 else ""

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
    line=dict(color=T["bh_line"], width=1.5, dash="dot")))
...
fig_pf.add_hline(y=initial_cash, line_dash="dot", line_color=T["table_border"], opacity=0.5)
fig_pf.update_layout(**CHART, height=400,
    legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0,
                font=dict(size=10, color=T["text_label"])))
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
    rc_ = "up" if m["total_return"]>=0 else "down"
    ac_ = "up" if m["annual_return"]>=0 else "down"
    sc_ = "up" if m["sharpe_ratio"]>=0 else "down"
    rh += f"""<tr>
        <td style="color:#e0e0e0;font-weight:500;">{n}</td>
        <td class="{rc_}">{m['total_return']:+.2f}%</td>
        <td class="{ac_}">{m['annual_return']:+.2f}%</td>
        <td class="{sc_}">{m['sharpe_ratio']:.3f}</td>
        <td class="down">{m['max_drawdown']:.1f}%</td>
        <td>{m['volatility']:.1f}%</td>
        <td style="color:#e0e0e0;">${m['final_value']:,.0f}</td></tr>"""

bha = ((1+bhr/100)**(1/max(len(df)/252,0.01))-1)*100
bhc_ = "up" if bhr>=0 else "down"
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
# Signals
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
                buys = raw[raw["Position"]==1]; sells = raw[raw["Position"]==-1]
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
                    line=dict(color=T["text_dim"], width=1, dash="dash")))
                fig.add_trace(go.Scatter(x=raw.index, y=raw["BB_Lower"], name="Lower",
                    line=dict(color=T["text_dim"], width=1, dash="dash"),
                    fill="tonexty", fillcolor="rgba(74,158,255,0.03)"))
                fig.add_trace(go.Scatter(x=raw.index, y=raw["BB_Mid"], name="Mid",
                    line=dict(color=T["table_border"], width=1)))
                fig.add_trace(go.Scatter(x=raw.index, y=raw["Close"], name="Price",
                    line=dict(color=ACCENT, width=1.5)))
                fig.update_layout(**CHART, height=380,
                    legend=dict(orientation="h", y=-0.08, x=0, font=dict(size=10, color=T["text_mute"])))
                st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────
# Metrics
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
            marker_color=([RED]*len(names) if red else colors),
            marker_line_width=0, text=texts, textposition="outside",
            textfont=dict(size=11, color="#888"))])
        fig.update_layout(
            template=T["plotly_tpl"], paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=300, bargap=0.4,
            font=dict(family="Inter, sans-serif", size=11, color=T["text_mute"]),
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis=dict(gridcolor=T["grid"]), yaxis_title=None,
            hoverlabel=dict(bgcolor=T["hover_panel"], font_size=12, font_color=T["hover_text"]),
        )
        fig.update_xaxes(gridcolor=T["grid"], tickfont=dict(size=9))
        fig.add_annotation(text=title, xref="paper", yref="paper",
            x=0, y=1.12, showarrow=False,
            font=dict(size=10, color=T["text_mute"], family="Inter"), align="left")
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


# ─────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-text">
    Dongni Lin  ·  McGill University  ·
    <a href="https://github.com/Loraldn">GitHub</a>
</div>
""", unsafe_allow_html=True)
