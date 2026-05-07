# Quantitative Trading Strategy Backtesting System

A Streamlit-based backtesting dashboard for evaluating systematic trading strategies on cross-border e-commerce stocks and global equity benchmarks. Three classical strategies, one interactive dashboard, no day-trading regrets.

Built with Python, Streamlit, Plotly, and yfinance. Designed to be the kind of tool you'd actually open twice.

---

## What This Project Is

Most backtesting tutorials stop at "here's a Sharpe ratio, congratulations." This one goes a step further: it's a fully interactive dashboard where you can pick a ticker, tune strategy parameters with sliders, switch between day and night themes, and watch three strategies fight it out against a passive buy-and-hold benchmark — all in real time.

The motivation is practical. While interning at Alibaba's 1688 cross-border division, I spent a lot of time staring at e-commerce performance dashboards and wondering whether the same companies looked equally compelling from a public-markets angle. This project is the answer to that curiosity, formalized into something I can hand to a recruiter.

---

## What It Does

The dashboard runs four things end-to-end:

**Data acquisition.** Pulls historical OHLCV data from Yahoo Finance for any ticker, with sensible caching so you're not hammering the API every time you nudge a slider.

**Strategy execution.** Three strategies, each with adjustable parameters in the sidebar. No black-box magic — the logic is short enough to read in one sitting.

**Portfolio simulation.** A vectorized simulator that respects transaction costs (0.1% commission per trade by default), sizes positions at 95% of available capital, and tracks the full equity curve. Not institutional-grade, but honest.

**Performance analytics.** Total return, annualized return, Sharpe ratio, max drawdown, volatility, final portfolio value — calculated for each strategy and shown side-by-side against buy-and-hold.

---

## Strategies

| Strategy | Logic | Works Best When |
|---|---|---|
| MA Crossover | Buy when short MA crosses above long MA; sell on the reverse cross | Markets are trending and committed about it |
| RSI Mean Reversion | Buy when RSI dips below 30 (oversold); sell when it pushes above 70 (overbought) | Price is range-bound and snaps back to its mean |
| Bollinger Bands | Buy at the lower band; sell at the upper band | Volatility is the dominant feature, not direction |

All three are textbook strategies on purpose. The point of the project isn't to invent alpha — it's to demonstrate that I can build the infrastructure to evaluate alpha when I find it.

---

## Tech Stack

- **Python 3.10+** — core language
- **Streamlit** — interactive dashboard frontend
- **Plotly** — candlestick charts, equity curves, signal overlays
- **yfinance** — historical market data
- **pandas / NumPy** — data manipulation and indicator calculations
- **Custom CSS** — because default Streamlit looks like a 2019 internal tool, and I have opinions

---

## Features Worth Mentioning

**Day / Night theme toggle.** The dashboard ships with a dark editorial theme by default and a light theme as the contrast counterpart. Toggle lives in the sidebar. The light theme uses warm amber as the accent color — the complement of the dark theme's electric blue — so the visual identity stays coherent either way.

**Three strategies, one chart.** All active strategies plot on the same equity curve alongside the passive benchmark, so you can see at a glance which approach actually paid for its own commissions.

**Tunable parameters.** Every strategy exposes its knobs — MA windows, RSI thresholds, Bollinger period and standard deviation multiplier — through sidebar sliders. Change a parameter, the entire dashboard recomputes.

**Signal visualization.** A dedicated tab for each strategy plots the underlying indicator (moving averages, RSI bands, Bollinger envelope) with buy/sell markers overlaid on price. Useful for sanity-checking that the strategy is actually doing what you think it is.

**Risk-adjusted comparison.** Beyond raw returns, the metrics section visualizes annualized return, Sharpe ratio, max drawdown, and volatility side-by-side. A strategy that returned 40% with a 60% drawdown is not the same animal as one that returned 30% with a 15% drawdown, and the dashboard makes that obvious.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Loraldn/quant-backtest.git
cd quant-backtest

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. Pick a ticker from the sidebar (BABA, PDD, JD, SHOP, AMZN, SPY are pre-loaded; any Yahoo Finance ticker works as a custom entry), set a date range, and the rest is point-and-click.

---

## Project Structure
- [GitHub](https://github.com/YOUR_USERNAME)
