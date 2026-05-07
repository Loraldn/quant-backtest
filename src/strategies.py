"""
strategies.py — Trading strategy implementations using backtrader.

Strategies:
    1. MACrossover  — Dual moving average crossover (trend-following)
    2. RSIMeanReversion — RSI-based mean reversion
    3. BollingerBands — Bollinger Bands breakout/reversion
"""

import backtrader as bt


# =============================================================================
# Strategy 1: Moving Average Crossover
# =============================================================================
class MACrossover(bt.Strategy):
    """
    Classic dual moving average crossover strategy.

    Buy signal:  Short MA crosses ABOVE Long MA (Golden Cross)
    Sell signal: Short MA crosses BELOW Long MA (Death Cross)

    Parameters:
        short_period: Short moving average window (default: 20)
        long_period:  Long moving average window (default: 50)
    """

    params = (
        ("short_period", 20),
        ("long_period", 50),
    )

    def __init__(self):
        self.ma_short = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.short_period
        )
        self.ma_long = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.long_period
        )
        self.crossover = bt.indicators.CrossOver(self.ma_short, self.ma_long)

        # Track orders
        self.order = None
        self.trade_count = 0

    def next(self):
        if self.order:
            return

        if not self.position:
            # Not in market — look for buy signal
            if self.crossover > 0:
                self.order = self.buy()
                self.trade_count += 1
        else:
            # In market — look for sell signal
            if self.crossover < 0:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None


# =============================================================================
# Strategy 2: RSI Mean Reversion
# =============================================================================
class RSIMeanReversion(bt.Strategy):
    """
    RSI-based mean reversion strategy.

    Buy signal:  RSI drops below oversold threshold (default: 30)
    Sell signal: RSI rises above overbought threshold (default: 70)

    Parameters:
        rsi_period:  RSI calculation window (default: 14)
        oversold:    Buy threshold (default: 30)
        overbought:  Sell threshold (default: 70)
    """

    params = (
        ("rsi_period", 14),
        ("oversold", 30),
        ("overbought", 70),
    )

    def __init__(self):
        self.rsi = bt.indicators.RelativeStrengthIndex(
            self.data.close, period=self.params.rsi_period
        )
        self.order = None
        self.trade_count = 0

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.rsi < self.params.oversold:
                self.order = self.buy()
                self.trade_count += 1
        else:
            if self.rsi > self.params.overbought:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None


# =============================================================================
# Strategy 3: Bollinger Bands
# =============================================================================
class BollingerBands(bt.Strategy):
    """
    Bollinger Bands mean reversion strategy.

    Buy signal:  Price touches or drops below lower band
    Sell signal: Price touches or rises above upper band

    Parameters:
        bb_period:  Bollinger Bands period (default: 20)
        bb_dev:     Number of standard deviations (default: 2.0)
    """

    params = (
        ("bb_period", 20),
        ("bb_dev", 2.0),
    )

    def __init__(self):
        self.bband = bt.indicators.BollingerBands(
            self.data.close,
            period=self.params.bb_period,
            devfactor=self.params.bb_dev,
        )
        self.order = None
        self.trade_count = 0

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.data.close[0] < self.bband.lines.bot[0]:
                self.order = self.buy()
                self.trade_count += 1
        else:
            if self.data.close[0] > self.bband.lines.top[0]:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None


# =============================================================================
# Strategy Registry — easy lookup by name
# =============================================================================
STRATEGY_MAP = {
    "ma_crossover": MACrossover,
    "rsi": RSIMeanReversion,
    "bollinger": BollingerBands,
}


def get_strategy(name: str) -> bt.Strategy:
    """Get strategy class by name."""
    if name not in STRATEGY_MAP:
        available = ", ".join(STRATEGY_MAP.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return STRATEGY_MAP[name]


def list_strategies() -> list:
    """List all available strategy names."""
    return list(STRATEGY_MAP.keys())
