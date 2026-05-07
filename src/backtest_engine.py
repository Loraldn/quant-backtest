"""
backtest_engine.py — Run backtests using backtrader.
"""

import backtrader as bt
import pandas as pd
from src.data_fetcher import fetch_data
from src.strategies import get_strategy


def run_backtest(
    ticker: str,
    strategy_name: str,
    start: str = "2020-01-01",
    end: str = None,
    cash: float = 100_000.0,
    commission: float = 0.001,
    strategy_params: dict = None,
) -> dict:
    """
    Run a backtest for a given ticker and strategy.

    Args:
        ticker: Stock symbol
        strategy_name: Name of strategy ('ma_crossover', 'rsi', 'bollinger')
        start: Backtest start date
        end: Backtest end date
        cash: Starting capital (default: $100,000)
        commission: Commission per trade (default: 0.1%)
        strategy_params: Optional dict of strategy-specific parameters

    Returns:
        Dict with backtest results and metadata
    """
    # 1. Fetch data
    print(f"\n{'='*60}")
    print(f"Running backtest: {strategy_name.upper()} on {ticker}")
    print(f"{'='*60}")
    df = fetch_data(ticker, start=start, end=end)

    # 2. Set up Cerebro engine
    cerebro = bt.Cerebro()

    # Add data feed
    data_feed = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data_feed)

    # Add strategy
    strategy_cls = get_strategy(strategy_name)
    if strategy_params:
        cerebro.addstrategy(strategy_cls, **strategy_params)
    else:
        cerebro.addstrategy(strategy_cls)

    # Set broker parameters
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    # Size: invest 95% of portfolio per trade
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.04)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # 3. Run backtest
    starting_value = cerebro.broker.getvalue()
    results = cerebro.run()
    ending_value = cerebro.broker.getvalue()

    # 4. Extract results
    strat = results[0]

    # Sharpe Ratio
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe_analysis.get("sharperatio", None)

    # Drawdown
    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_drawdown = dd_analysis.max.drawdown

    # Returns
    ret_analysis = strat.analyzers.returns.get_analysis()
    total_return = ret_analysis.get("rtot", 0) * 100

    # Trade stats
    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get("total", {}).get("total", 0)
    won_trades = trade_analysis.get("won", {}).get("total", 0)
    lost_trades = trade_analysis.get("lost", {}).get("total", 0)
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

    # Annual return (approximate)
    num_days = len(df)
    num_years = num_days / 252
    annual_return = ((ending_value / starting_value) ** (1 / num_years) - 1) * 100 if num_years > 0 else 0

    result = {
        "ticker": ticker,
        "strategy": strategy_name,
        "start_date": df.index[0].strftime("%Y-%m-%d"),
        "end_date": df.index[-1].strftime("%Y-%m-%d"),
        "starting_value": starting_value,
        "ending_value": ending_value,
        "total_return_pct": total_return,
        "annual_return_pct": annual_return,
        "sharpe_ratio": round(sharpe_ratio, 4) if sharpe_ratio else None,
        "max_drawdown_pct": round(max_drawdown, 2),
        "total_trades": total_trades,
        "won_trades": won_trades,
        "lost_trades": lost_trades,
        "win_rate_pct": round(win_rate, 2),
        "cerebro": cerebro,  # Keep for plotting
    }

    return result


def run_comparison(
    ticker: str,
    strategies: list = None,
    start: str = "2020-01-01",
    end: str = None,
    cash: float = 100_000.0,
) -> list:
    """
    Run multiple strategies on the same ticker for comparison.

    Args:
        ticker: Stock symbol
        strategies: List of strategy names (default: all)
        start: Start date
        end: End date
        cash: Starting capital

    Returns:
        List of result dicts
    """
    if strategies is None:
        strategies = ["ma_crossover", "rsi", "bollinger"]

    results = []
    for strategy_name in strategies:
        try:
            result = run_backtest(
                ticker=ticker,
                strategy_name=strategy_name,
                start=start,
                end=end,
                cash=cash,
            )
            results.append(result)
        except Exception as e:
            print(f"  Error running {strategy_name}: {e}")

    return results
