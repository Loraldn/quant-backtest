"""
strategy_analysis.py — Full analysis script.

Run this to generate a complete analysis report with visualizations.
This can also be converted to a Jupyter notebook using jupytext.

Usage:
    python notebooks/strategy_analysis.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetcher import fetch_data, DEFAULT_TICKERS
from src.backtest_engine import run_backtest, run_comparison
from src.performance import (
    print_results,
    print_comparison,
    plot_comparison_chart,
    plot_metrics_dashboard,
)


def main():
    os.makedirs("output", exist_ok=True)

    # =========================================================================
    # Part 1: Data Exploration
    # =========================================================================
    print("\n" + "=" * 60)
    print("  PART 1: Data Exploration")
    print("=" * 60)

    ticker = "BABA"
    df = fetch_data(ticker, start="2020-01-01")
    print(f"\n{ticker} Data Overview:")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Trading days: {len(df)}")
    print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    print(f"  Current price: ${df['Close'].iloc[-1]:.2f}")
    print(f"  Avg daily volume: {df['Volume'].mean():,.0f}")

    # =========================================================================
    # Part 2: Single Strategy Backtest
    # =========================================================================
    print("\n" + "=" * 60)
    print("  PART 2: MA Crossover Strategy on BABA")
    print("=" * 60)

    result = run_backtest(
        ticker="BABA",
        strategy_name="ma_crossover",
        start="2020-01-01",
        cash=100_000,
    )
    print_results(result)

    # =========================================================================
    # Part 3: Strategy Comparison
    # =========================================================================
    print("\n" + "=" * 60)
    print("  PART 3: All Strategies Comparison on BABA")
    print("=" * 60)

    results = run_comparison(ticker="BABA", start="2020-01-01")
    print_comparison(results)

    plot_comparison_chart(
        results,
        save_path="output/BABA_strategy_comparison.png",
    )

    plot_metrics_dashboard(
        results,
        save_path="output/BABA_metrics_dashboard.png",
    )

    # =========================================================================
    # Part 4: Cross-Border E-Commerce Stock Analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("  PART 4: Cross-Border E-Commerce Stocks")
    print("=" * 60)

    tickers = ["BABA", "PDD", "JD", "SHOP"]
    best_strategies = {}

    for t in tickers:
        t_results = run_comparison(ticker=t, start="2020-01-01")
        if t_results:
            print_comparison(t_results)
            best = max(t_results, key=lambda r: r["total_return_pct"])
            best_strategies[t] = best

            plot_metrics_dashboard(
                t_results,
                save_path=f"output/{t}_dashboard.png",
            )

    # =========================================================================
    # Part 5: Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print("\n  Best strategy per stock:\n")
    for t, best in best_strategies.items():
        sharpe = best["sharpe_ratio"] if best["sharpe_ratio"] else "N/A"
        print(f"    {t:6s} → {best['strategy']:15s} | "
              f"Return: {best['total_return_pct']:>8.2f}% | "
              f"Sharpe: {str(sharpe):>6s} | "
              f"Max DD: {best['max_drawdown_pct']:>6.2f}%")

    print(f"\n  Charts saved to output/ directory.")
    print(f"  Generated files:")
    for f in sorted(os.listdir("output")):
        if f.endswith(".png"):
            print(f"    - output/{f}")
    print()


if __name__ == "__main__":
    main()
