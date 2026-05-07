"""
performance.py — Performance metrics display and visualization.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from tabulate import tabulate
from src.data_fetcher import fetch_data


def print_results(result: dict):
    """Print a formatted summary of backtest results."""
    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS: {result['strategy'].upper()} on {result['ticker']}")
    print(f"  Period: {result['start_date']} to {result['end_date']}")
    print(f"{'='*60}")

    rows = [
        ["Starting Value", f"${result['starting_value']:,.2f}"],
        ["Ending Value", f"${result['ending_value']:,.2f}"],
        ["Total Return", f"{result['total_return_pct']:.2f}%"],
        ["Annual Return", f"{result['annual_return_pct']:.2f}%"],
        ["Sharpe Ratio", f"{result['sharpe_ratio']}" if result['sharpe_ratio'] else "N/A"],
        ["Max Drawdown", f"{result['max_drawdown_pct']:.2f}%"],
        ["Total Trades", result['total_trades']],
        ["Win Rate", f"{result['win_rate_pct']:.1f}% ({result['won_trades']}W / {result['lost_trades']}L)"],
    ]

    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="simple"))
    print()


def print_comparison(results: list):
    """Print a comparison table of multiple strategy results."""
    if not results:
        print("No results to compare.")
        return

    ticker = results[0]["ticker"]
    print(f"\n{'='*70}")
    print(f"  STRATEGY COMPARISON on {ticker}")
    print(f"  Period: {results[0]['start_date']} to {results[0]['end_date']}")
    print(f"{'='*70}")

    headers = ["Strategy", "Return", "Annual", "Sharpe", "Max DD", "Trades", "Win Rate"]
    rows = []
    for r in results:
        rows.append([
            r["strategy"],
            f"{r['total_return_pct']:.2f}%",
            f"{r['annual_return_pct']:.2f}%",
            f"{r['sharpe_ratio']}" if r["sharpe_ratio"] else "N/A",
            f"{r['max_drawdown_pct']:.2f}%",
            r["total_trades"],
            f"{r['win_rate_pct']:.1f}%",
        ])

    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print()


def plot_strategy_result(result: dict, save_path: str = None):
    """
    Plot backtest results: price chart with buy/sell signals overlay.
    Uses backtrader's built-in plotting saved to file.
    """
    cerebro = result["cerebro"]
    fig = cerebro.plot(style="candle", barup="green", bardown="red", volume=False)[0][0]
    fig.set_size_inches(14, 8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Chart saved to {save_path}")
    plt.close(fig)


def plot_comparison_chart(
    results: list,
    benchmark_ticker: str = "SPY",
    start: str = "2020-01-01",
    end: str = None,
    save_path: str = None,
):
    """
    Plot portfolio value curves for all strategies + buy & hold benchmark.
    """
    if not results:
        return

    ticker = results[0]["ticker"]
    start_date = results[0]["start_date"]
    end_date = results[0]["end_date"]

    # Fetch benchmark data for buy & hold comparison
    try:
        bench_df = fetch_data(ticker, start=start_date, end=end_date)
        bench_return = (bench_df["Close"] / bench_df["Close"].iloc[0]) * 100_000
    except Exception:
        bench_return = None

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

    # ---- Top: Performance comparison ----
    ax1 = axes[0]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, r in enumerate(results):
        ax1.axhline(
            y=r["ending_value"],
            color=colors[i % len(colors)],
            linestyle="--",
            alpha=0.3,
        )
        ax1.annotate(
            f"{r['strategy']}: ${r['ending_value']:,.0f}",
            xy=(1.01, r["ending_value"]),
            xycoords=("axes fraction", "data"),
            fontsize=9,
            color=colors[i % len(colors)],
            va="center",
        )

    if bench_return is not None:
        ax1.plot(
            bench_return.index,
            bench_return.values,
            label=f"Buy & Hold {ticker}",
            color="#9E9E9E",
            linewidth=1.5,
            alpha=0.7,
        )

    ax1.axhline(y=100_000, color="black", linestyle=":", alpha=0.3, label="Starting Capital")
    ax1.set_title(f"Strategy Performance Comparison — {ticker}", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # ---- Bottom: Summary metrics bar chart ----
    ax2 = axes[1]
    strategy_names = [r["strategy"] for r in results]
    returns = [r["total_return_pct"] for r in results]
    bar_colors = [colors[i % len(colors)] for i in range(len(results))]

    bars = ax2.bar(strategy_names, returns, color=bar_colors, alpha=0.8, edgecolor="white")

    for bar, ret in zip(bars, returns):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{ret:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax2.set_ylabel("Total Return (%)")
    ax2.set_title("Total Return by Strategy", fontsize=12)
    ax2.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Comparison chart saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_metrics_dashboard(results: list, save_path: str = None):
    """
    Create a 2x2 dashboard comparing key metrics across strategies.
    """
    if not results:
        return

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    names = [r["strategy"] for r in results]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    # 1. Annual Return
    ax = axes[0, 0]
    vals = [r["annual_return_pct"] for r in results]
    bars = ax.bar(names, vals, color=colors[: len(names)], alpha=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_title("Annual Return (%)")
    ax.axhline(y=0, color="black", linewidth=0.5)

    # 2. Sharpe Ratio
    ax = axes[0, 1]
    vals = [r["sharpe_ratio"] if r["sharpe_ratio"] else 0 for r in results]
    bars = ax.bar(names, vals, color=colors[: len(names)], alpha=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Sharpe Ratio")
    ax.axhline(y=0, color="black", linewidth=0.5)

    # 3. Max Drawdown
    ax = axes[1, 0]
    vals = [-r["max_drawdown_pct"] for r in results]
    bars = ax.bar(names, vals, color=["#F44336"] * len(names), alpha=0.7)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.1f}%", ha="center", va="top", fontsize=10)
    ax.set_title("Max Drawdown (%)")

    # 4. Win Rate
    ax = axes[1, 1]
    vals = [r["win_rate_pct"] for r in results]
    bars = ax.bar(names, vals, color=colors[: len(names)], alpha=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.0f}%", ha="center", va="bottom", fontsize=10)
    ax.set_title("Win Rate (%)")
    ax.set_ylim(0, 100)

    ticker = results[0]["ticker"]
    fig.suptitle(f"Strategy Metrics Dashboard — {ticker}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Dashboard saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
