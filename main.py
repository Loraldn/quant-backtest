"""
main.py — Entry point for the Quant Backtest project.

Usage:
    # Run single strategy
    python main.py --ticker BABA --strategy ma_crossover

    # Compare all strategies on one ticker
    python main.py --ticker BABA --compare

    # Run across multiple cross-border e-commerce stocks
    python main.py --multi

    # Custom date range
    python main.py --ticker SHOP --strategy rsi --start 2021-01-01 --end 2025-01-01
"""

import argparse
import os
from src.backtest_engine import run_backtest, run_comparison
from src.performance import (
    print_results,
    print_comparison,
    plot_comparison_chart,
    plot_metrics_dashboard,
)
from src.strategies import list_strategies


def parse_args():
    parser = argparse.ArgumentParser(description="Quantitative Trading Strategy Backtester")
    parser.add_argument("--ticker", type=str, default="BABA", help="Stock ticker symbol")
    parser.add_argument("--strategy", type=str, default="ma_crossover",
                        choices=list_strategies(), help="Strategy to run")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--cash", type=float, default=100_000, help="Starting capital")
    parser.add_argument("--compare", action="store_true", help="Compare all strategies")
    parser.add_argument("--multi", action="store_true",
                        help="Run best strategy across cross-border e-commerce stocks")
    return parser.parse_args()


def run_single(args):
    """Run a single strategy on one ticker."""
    result = run_backtest(
        ticker=args.ticker,
        strategy_name=args.strategy,
        start=args.start,
        end=args.end,
        cash=args.cash,
    )
    print_results(result)
    return result


def run_compare(args):
    """Compare all strategies on one ticker."""
    results = run_comparison(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        cash=args.cash,
    )

    # Print individual results
    for r in results:
        print_results(r)

    # Print comparison table
    print_comparison(results)

    # Generate charts
    os.makedirs("output", exist_ok=True)

    plot_comparison_chart(
        results,
        save_path=f"output/{args.ticker}_comparison.png",
    )

    plot_metrics_dashboard(
        results,
        save_path=f"output/{args.ticker}_dashboard.png",
    )

    return results


def run_multi(args):
    """Run comparison across multiple cross-border e-commerce stocks."""
    tickers = ["BABA", "PDD", "JD", "SHOP"]
    all_results = {}

    for ticker in tickers:
        print(f"\n{'#'*60}")
        print(f"  ANALYZING: {ticker}")
        print(f"{'#'*60}")

        results = run_comparison(
            ticker=ticker,
            start=args.start,
            end=args.end,
            cash=args.cash,
        )

        if results:
            all_results[ticker] = results
            print_comparison(results)

            os.makedirs("output", exist_ok=True)
            plot_metrics_dashboard(
                results,
                save_path=f"output/{ticker}_dashboard.png",
            )

    # Summary: best strategy per ticker
    print(f"\n{'='*60}")
    print("  SUMMARY: Best Strategy per Ticker")
    print(f"{'='*60}")
    for ticker, results in all_results.items():
        best = max(results, key=lambda r: r["total_return_pct"])
        print(f"  {ticker}: {best['strategy']} (Return: {best['total_return_pct']:.2f}%, "
              f"Sharpe: {best['sharpe_ratio']})")
    print()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  QUANT BACKTEST — Trading Strategy Analysis")
    print("=" * 60)

    if args.multi:
        run_multi(args)
    elif args.compare:
        run_compare(args)
    else:
        run_single(args)

    print("Done! Check the output/ folder for charts.\n")


if __name__ == "__main__":
    main()
