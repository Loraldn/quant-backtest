# 📈 Quantitative Trading Strategy Backtesting System

A Python-based backtesting framework for evaluating quantitative trading strategies on cross-border e-commerce and global equity markets. Built with `backtrader`, `yfinance`, and `pandas`.

## 🎯 Project Overview

This project implements a complete quantitative trading pipeline:

1. **Data Acquisition** — Fetch historical OHLCV data via Yahoo Finance API
2. **Strategy Development** — Implement classic and custom trading strategies (MA Crossover, RSI Mean Reversion, Bollinger Bands)
3. **Backtesting Engine** — Run strategies against historical data with realistic transaction costs
4. **Performance Analytics** — Calculate Sharpe Ratio, Max Drawdown, Annual Return, Win Rate
5. **Visualization** — Generate interactive charts comparing strategy vs. benchmark

## 📊 Strategies Implemented

| Strategy | Logic | Best For |
|---|---|---|
| **MA Crossover** | Buy when short MA crosses above long MA; sell on reverse | Trend-following markets |
| **RSI Mean Reversion** | Buy when RSI < 30 (oversold); sell when RSI > 70 (overbought) | Range-bound markets |
| **Bollinger Bands** | Buy at lower band; sell at upper band | Volatility-driven markets |

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/quant-backtest.git
cd quant-backtest

# Install dependencies
pip install -r requirements.txt

# Run the backtest
python main.py

# Run with custom parameters
python main.py --ticker BABA --strategy ma_crossover --start 2020-01-01 --end 2025-12-31
```

## 📁 Project Structure

```
quant-backtest/
├── main.py                  # Entry point — run backtests from CLI
├── requirements.txt         # Python dependencies
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py      # Yahoo Finance data acquisition
│   ├── strategies.py        # Trading strategy implementations
│   ├── backtest_engine.py   # Backtrader wrapper & execution
│   └── performance.py       # Performance metrics & visualization
├── notebooks/
│   └── strategy_analysis.py # Full analysis script with visualizations
├── output/                  # Generated charts and reports
└── data/                    # Cached market data
```

## 🛠️ Tech Stack

- **Python 3.10+**
- **backtrader** — Backtesting framework
- **yfinance** — Market data API
- **pandas / numpy** — Data manipulation
- **matplotlib / seaborn** — Visualization
- **tabulate** — Performance report formatting

## 📈 Sample Results

```
=== Backtest Results: MA Crossover on BABA (2020-01-01 to 2025-12-31) ===
Starting Portfolio Value:  $100,000.00
Final Portfolio Value:     $XXX,XXX.XX
Total Return:              XX.XX%
Annual Return:             XX.XX%
Sharpe Ratio:              X.XX
Max Drawdown:              XX.XX%
Total Trades:              XX
Win Rate:                  XX.XX%
```

## 🌏 Market Focus

This project focuses on **cross-border e-commerce and global equities**, including:
- **BABA** (Alibaba Group) — China's largest e-commerce platform
- **PDD** (PDD Holdings / Temu) — Fast-growing cross-border retailer
- **JD** (JD.com) — China's leading direct sales e-commerce company
- **SHOP** (Shopify) — Global e-commerce infrastructure
- **SPY** (S&P 500 ETF) — Benchmark index

## 📝 License

MIT License

## 👤 Author

**Dongni Lin (Lora)**
- B.A. Statistics & Economics, McGill University
- [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
- [GitHub](https://github.com/YOUR_USERNAME)
