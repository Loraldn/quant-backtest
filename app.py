"""
app.py — Streamlit Dashboard for Quant Backtest Project
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

# ─────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quant Backtest Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1a73e8, #00c9a7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f, #2a2a3d);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 4px;
    }
    .positive { color: #00c9a7; }
    .negative { color: #ff6b6b; }
    .neutral { color: #ffd93d; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, start, end):
    """Fetch OHLCV data from Yahoo Finance with caching."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        # Handle MultiIndex columns (yfinance >= 0.2.31 always returns MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, level=1, axis=1)
        # Ensure we have the required columns
        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                return pd.DataFrame()
        df = df[required].dropna()
        return df
    except Exception as e:
        st.warning(f"Error fetching data: {e}")
        return pd.DataFrame()


def compute_ma_crossover(df, short_window=20, long_window=50):
    """MA Crossover strategy signals."""
    data = df.copy()
    data["MA_Short"] = data["Close"].rolling(window=short_window).mean()
    data["MA_Long"] = data["Close"].rolling(window=long_window).mean()
    data["Signal"] = 0
    data.loc[data["MA_Short"] > data["MA_Long"], "Signal"] = 1
    data["Position"] = data["Signal"].diff()
    return data


def compute_rsi_strategy(df, rsi_period=14, oversold=30, overbought=70):
    """RSI Mean Reversion strategy signals."""
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
    """Bollinger Bands strategy signals."""
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
    """Simulate portfolio based on signals, return portfolio value series."""
    data = df.dropna().copy()
    cash = initial_cash
    shares = 0
    portfolio_values = []

    for i in range(len(data)):
        price = data["Close"].iloc[i]
        signal = data[signal_col].iloc[i]

        if signal == 1 and shares == 0:
            shares = int(cash * 0.95 / price)
            cash -= shares * price * (1 + commission)
        elif signal == -1 and shares > 0:
            cash += shares * price * (1 - commission)
            shares = 0

        portfolio_values.append(cash + shares * price)

    data = data.copy()
    data["Portfolio"] = portfolio_values
    return data


def calculate_metrics(portfolio_series, initial_cash=100000):
    """Calculate key performance metrics."""
    total_return = (portfolio_series.iloc[-1] / initial_cash - 1) * 100
    num_days = len(portfolio_series)
    num_years = num_days / 252
    annual_return = ((portfolio_series.iloc[-1] / initial_cash) ** (1 / max(num_years, 0.01)) - 1) * 100

    daily_returns = portfolio_series.pct_change().dropna()
    sharpe = (daily_returns.mean() / max(daily_returns.std(), 1e-10)) * np.sqrt(252)

    rolling_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "final_value": portfolio_series.iloc[-1],
        "volatility": daily_returns.std() * np.sqrt(252) * 100,
    }


# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # Ticker selection
    ticker_options = {
        "BABA": "Alibaba Group",
        "PDD": "PDD Holdings (Temu)",
        "JD": "JD.com",
        "SHOP": "Shopify",
        "AMZN": "Amazon",
        "SPY": "S&P 500 ETF",
    }
    selected_ticker = st.selectbox(
        "Stock Ticker",
        options=list(ticker_options.keys()),
        format_func=lambda x: f"{x} — {ticker_options[x]}",
    )

    # Or custom ticker
    custom_ticker = st.text_input("Or enter custom ticker", placeholder="e.g. AAPL, TSLA")
    if custom_ticker:
        selected_ticker = custom_ticker.upper()

    st.markdown("---")

    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())

    st.markdown("---")

    # Strategy selection
    selected_strategies = st.multiselect(
        "Strategies to Compare",
        ["MA Crossover", "RSI Mean Reversion", "Bollinger Bands"],
        default=["MA Crossover", "RSI Mean Reversion", "Bollinger Bands"],
    )

    st.markdown("---")

    # Strategy parameters
    st.markdown("### 🎛️ Parameters")

    with st.expander("MA Crossover"):
        ma_short = st.slider("Short MA Period", 5, 50, 20)
        ma_long = st.slider("Long MA Period", 20, 200, 50)

    with st.expander("RSI Mean Reversion"):
        rsi_period = st.slider("RSI Period", 5, 30, 14)
        rsi_oversold = st.slider("Oversold Threshold", 10, 40, 30)
        rsi_overbought = st.slider("Overbought Threshold", 60, 90, 70)

    with st.expander("Bollinger Bands"):
        bb_period = st.slider("BB Period", 10, 50, 20)
        bb_std = st.slider("Std Deviations", 1.0, 3.0, 2.0, 0.5)

    st.markdown("---")
    initial_cash = st.number_input("Starting Capital ($)", value=100000, step=10000)

    st.markdown("---")
    st.markdown(
        "Built by **Dongni Lin (Lora)**\n\n"
        "B.A. Statistics & Economics\n\n"
        "McGill University"
    )


# ─────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────
st.markdown('<p class="main-header">📈 Quant Backtest Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Cross-Border E-Commerce & Global Equity Strategy Analysis</p>',
    unsafe_allow_html=True,
)

# Fetch data
with st.spinner(f"Fetching {selected_ticker} data..."):
    df = fetch_stock_data(selected_ticker, start=str(start_date), end=str(end_date))

if df.empty:
    st.error(f"No data found for {selected_ticker}. Please check the ticker symbol.")
    st.stop()

# ─────────────────────────────────────────────────────────
# Section 1: Price Overview
# ─────────────────────────────────────────────────────────
st.markdown("### 📊 Price Overview")

fig_price = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.03, row_heights=[0.7, 0.3],
    subplot_titles=[f"{selected_ticker} Price", "Volume"],
)

fig_price.add_trace(
    go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color="#00c9a7", decreasing_line_color="#ff6b6b",
    ),
    row=1, col=1,
)

fig_price.add_trace(
    go.Bar(x=df.index, y=df["Volume"], name="Volume",
           marker_color="rgba(100,100,255,0.3)"),
    row=2, col=1,
)

fig_price.update_layout(
    height=500, xaxis_rangeslider_visible=False,
    template="plotly_dark", showlegend=False,
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig_price, use_container_width=True)

# ─────────────────────────────────────────────────────────
# Section 2: Run Strategies & Show Metrics
# ─────────────────────────────────────────────────────────
st.markdown("### 🏆 Strategy Performance Comparison")

strategy_results = {}
strategy_colors = {
    "MA Crossover": "#1a73e8",
    "RSI Mean Reversion": "#ff9800",
    "Bollinger Bands": "#00c9a7",
}

for strategy_name in selected_strategies:
    if strategy_name == "MA Crossover":
        data = compute_ma_crossover(df, ma_short, ma_long)
        # Convert position-based signals to buy/sell
        sig = data["Signal"].copy()
        sig_shifted = sig.diff()
        trade_signal = pd.Series(0, index=data.index)
        trade_signal[sig_shifted == 1] = 1   # buy
        trade_signal[sig_shifted == -1] = -1  # sell
        data["TradeSignal"] = trade_signal

    elif strategy_name == "RSI Mean Reversion":
        data = compute_rsi_strategy(df, rsi_period, rsi_oversold, rsi_overbought)
        data["TradeSignal"] = data["Signal"]

    elif strategy_name == "Bollinger Bands":
        data = compute_bollinger_strategy(df, bb_period, bb_std)
        data["TradeSignal"] = data["Signal"]

    portfolio_data = simulate_portfolio(data, signal_col="TradeSignal", initial_cash=initial_cash)
    metrics = calculate_metrics(portfolio_data["Portfolio"], initial_cash)
    strategy_results[strategy_name] = {
        "data": portfolio_data,
        "metrics": metrics,
        "raw": data,
    }

# Buy & Hold benchmark
buy_hold_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
buy_hold_final = initial_cash * (1 + buy_hold_return / 100)

# ── Metric Cards ──
if strategy_results:
    best_strategy = max(strategy_results.items(), key=lambda x: x[1]["metrics"]["total_return"])
    best_name = best_strategy[0]
    best_metrics = best_strategy[1]["metrics"]

    cols = st.columns(6)
    metric_items = [
        ("Best Strategy", best_name, "neutral"),
        ("Total Return", f"{best_metrics['total_return']:.2f}%",
         "positive" if best_metrics["total_return"] > 0 else "negative"),
        ("Annual Return", f"{best_metrics['annual_return']:.2f}%",
         "positive" if best_metrics["annual_return"] > 0 else "negative"),
        ("Sharpe Ratio", f"{best_metrics['sharpe_ratio']:.3f}",
         "positive" if best_metrics["sharpe_ratio"] > 0 else "negative"),
        ("Max Drawdown", f"{best_metrics['max_drawdown']:.2f}%", "negative"),
        ("Buy & Hold", f"{buy_hold_return:.2f}%",
         "positive" if buy_hold_return > 0 else "negative"),
    ]

    for col, (label, value, color_class) in zip(cols, metric_items):
        col.markdown(
            f"""<div class="metric-card">
                <div class="metric-value {color_class}">{value}</div>
                <div class="metric-label">{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────
# Section 3: Portfolio Value Chart
# ─────────────────────────────────────────────────────────
st.markdown("### 💰 Portfolio Value Over Time")

fig_portfolio = go.Figure()

# Buy & Hold line
buy_hold_series = initial_cash * (df["Close"] / df["Close"].iloc[0])
fig_portfolio.add_trace(
    go.Scatter(
        x=df.index, y=buy_hold_series,
        name=f"Buy & Hold {selected_ticker}",
        line=dict(color="#666", width=2, dash="dash"),
    )
)

# Strategy lines
for name, result in strategy_results.items():
    portfolio_data = result["data"]
    fig_portfolio.add_trace(
        go.Scatter(
            x=portfolio_data.index,
            y=portfolio_data["Portfolio"],
            name=name,
            line=dict(color=strategy_colors.get(name, "#fff"), width=2.5),
        )
    )

fig_portfolio.add_hline(
    y=initial_cash, line_dash="dot", line_color="white",
    annotation_text=f"Starting Capital: ${initial_cash:,.0f}",
    opacity=0.3,
)

fig_portfolio.update_layout(
    height=450, template="plotly_dark",
    yaxis_title="Portfolio Value ($)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig_portfolio, use_container_width=True)

# ─────────────────────────────────────────────────────────
# Section 4: Comparison Table
# ─────────────────────────────────────────────────────────
st.markdown("### 📋 Detailed Comparison")

table_data = []
for name, result in strategy_results.items():
    m = result["metrics"]
    table_data.append({
        "Strategy": name,
        "Final Value": f"${m['final_value']:,.2f}",
        "Total Return": f"{m['total_return']:.2f}%",
        "Annual Return": f"{m['annual_return']:.2f}%",
        "Sharpe Ratio": f"{m['sharpe_ratio']:.3f}",
        "Max Drawdown": f"{m['max_drawdown']:.2f}%",
        "Volatility": f"{m['volatility']:.2f}%",
    })

# Add Buy & Hold row
table_data.append({
    "Strategy": f"Buy & Hold {selected_ticker}",
    "Final Value": f"${buy_hold_final:,.2f}",
    "Total Return": f"{buy_hold_return:.2f}%",
    "Annual Return": f"{((1 + buy_hold_return/100) ** (1 / max((len(df)/252), 0.01)) - 1) * 100:.2f}%",
    "Sharpe Ratio": "—",
    "Max Drawdown": "—",
    "Volatility": "—",
})

st.dataframe(
    pd.DataFrame(table_data).set_index("Strategy"),
    use_container_width=True,
)

# ─────────────────────────────────────────────────────────
# Section 5: Strategy Indicator Charts
# ─────────────────────────────────────────────────────────
st.markdown("### 🔍 Strategy Indicators")

tab_names = [name for name in selected_strategies]
if tab_names:
    tabs = st.tabs(tab_names)

    for tab, strategy_name in zip(tabs, selected_strategies):
        with tab:
            result = strategy_results[strategy_name]
            raw = result["raw"]

            if strategy_name == "MA Crossover":
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=raw.index, y=raw["Close"], name="Close", line=dict(color="#888", width=1)))
                fig.add_trace(go.Scatter(x=raw.index, y=raw["MA_Short"], name=f"MA {ma_short}", line=dict(color="#1a73e8", width=2)))
                fig.add_trace(go.Scatter(x=raw.index, y=raw["MA_Long"], name=f"MA {ma_long}", line=dict(color="#ff9800", width=2)))

                # Buy/Sell markers
                buys = raw[raw["Position"] == 1]
                sells = raw[raw["Position"] == -1]
                fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers", name="Buy",
                                         marker=dict(symbol="triangle-up", size=12, color="#00c9a7")))
                fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers", name="Sell",
                                         marker=dict(symbol="triangle-down", size=12, color="#ff6b6b")))

                fig.update_layout(height=400, template="plotly_dark", title="MA Crossover Signals",
                                  margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)

            elif strategy_name == "RSI Mean Reversion":
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                    row_heights=[0.6, 0.4], subplot_titles=["Price", "RSI"])
                fig.add_trace(go.Scatter(x=raw.index, y=raw["Close"], name="Close", line=dict(color="#888")), row=1, col=1)
                fig.add_trace(go.Scatter(x=raw.index, y=raw["RSI"], name="RSI", line=dict(color="#ff9800", width=2)), row=2, col=1)
                fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="#00c9a7", row=2, col=1)
                fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="#ff6b6b", row=2, col=1)
                fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)

            elif strategy_name == "Bollinger Bands":
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=raw.index, y=raw["Close"], name="Close", line=dict(color="#888", width=1)))
                fig.add_trace(go.Scatter(x=raw.index, y=raw["BB_Upper"], name="Upper Band",
                                         line=dict(color="#ff6b6b", width=1, dash="dash")))
                fig.add_trace(go.Scatter(x=raw.index, y=raw["BB_Mid"], name="Middle Band",
                                         line=dict(color="#ffd93d", width=1)))
                fig.add_trace(go.Scatter(x=raw.index, y=raw["BB_Lower"], name="Lower Band",
                                         line=dict(color="#00c9a7", width=1, dash="dash"),
                                         fill="tonexty", fillcolor="rgba(0,201,167,0.05)"))
                fig.update_layout(height=400, template="plotly_dark", title="Bollinger Bands",
                                  margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────
# Section 6: Metrics Dashboard
# ─────────────────────────────────────────────────────────
st.markdown("### 📊 Performance Metrics Dashboard")

if len(strategy_results) > 1:
    col1, col2 = st.columns(2)

    names = list(strategy_results.keys())
    colors = [strategy_colors.get(n, "#fff") for n in names]

    with col1:
        # Annual Return bar chart
        fig_bar1 = go.Figure(data=[
            go.Bar(
                x=names,
                y=[strategy_results[n]["metrics"]["annual_return"] for n in names],
                marker_color=colors,
                text=[f"{strategy_results[n]['metrics']['annual_return']:.1f}%" for n in names],
                textposition="outside",
            )
        ])
        fig_bar1.update_layout(
            title="Annual Return (%)", height=350, template="plotly_dark",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_bar1, use_container_width=True)

    with col2:
        # Sharpe Ratio bar chart
        fig_bar2 = go.Figure(data=[
            go.Bar(
                x=names,
                y=[strategy_results[n]["metrics"]["sharpe_ratio"] for n in names],
                marker_color=colors,
                text=[f"{strategy_results[n]['metrics']['sharpe_ratio']:.2f}" for n in names],
                textposition="outside",
            )
        ])
        fig_bar2.update_layout(
            title="Sharpe Ratio", height=350, template="plotly_dark",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_bar2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Max Drawdown bar chart
        fig_bar3 = go.Figure(data=[
            go.Bar(
                x=names,
                y=[strategy_results[n]["metrics"]["max_drawdown"] for n in names],
                marker_color=["#ff6b6b"] * len(names),
                text=[f"{strategy_results[n]['metrics']['max_drawdown']:.1f}%" for n in names],
                textposition="outside",
            )
        ])
        fig_bar3.update_layout(
            title="Max Drawdown (%)", height=350, template="plotly_dark",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_bar3, use_container_width=True)

    with col4:
        # Volatility bar chart
        fig_bar4 = go.Figure(data=[
            go.Bar(
                x=names,
                y=[strategy_results[n]["metrics"]["volatility"] for n in names],
                marker_color=colors,
                text=[f"{strategy_results[n]['metrics']['volatility']:.1f}%" for n in names],
                textposition="outside",
            )
        ])
        fig_bar4.update_layout(
            title="Annualized Volatility (%)", height=350, template="plotly_dark",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_bar4, use_container_width=True)


# ─────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666; font-size:0.85rem;'>"
    "Built by Dongni Lin (Lora) · McGill University · B.A. Statistics & Economics · "
    "<a href='https://github.com/Loraldn' style='color:#1a73e8;'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
