"""
Backtest script for VWAP Trend-Trading strategy on BTC/USDT minute data.
"""

import os
import time
from typing import List, Tuple

import ccxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
START_DATE = "2019-01-01T00:00:00Z"
END_DATE = "2025-07-01T00:00:00Z"
DATA_FILE = "btc_usdt_1m.csv"
COMMISSION = 0.0004  # 0.04% per side


def fetch_ohlcv(symbol: str, start: str, end: str, file_path: str) -> pd.DataFrame:
    """Fetch OHLCV data from Binance or load from cache."""
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df["time(ms)"] = pd.to_datetime(df["time(ms)"], unit="ms", utc=True)
        return df

    exchange = ccxt.binance({"enableRateLimit": True})
    start_ms = exchange.parse8601(start)
    end_ms = exchange.parse8601(end)
    since = start_ms
    limit = 1000
    all_candles = []
    while since < end_ms:
        try:
            candles = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=limit)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            sleep_time = exchange.rateLimit / 1000 * 2
            print(f"{e}; sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
            continue
        if not candles:
            break
        since = candles[-1][0] + 60_000
        for candle in candles:
            if candle[0] >= end_ms:
                break
            all_candles.append(candle)
        time.sleep(exchange.rateLimit / 1000)
    df = pd.DataFrame(all_candles, columns=["time(ms)", "open", "high", "low", "close", "volume"])
    df.to_csv(file_path, index=False)
    df["time(ms)"] = pd.to_datetime(df["time(ms)"], unit="ms", utc=True)
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute running session VWAP for each day."""
    df = df.copy()
    df["price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["date"] = df["time(ms)"].dt.floor("D")
    df["pv"] = df["price"] * df["volume"]
    df["cum_pv"] = df.groupby("date")["pv"].cumsum()
    df["cum_vol"] = df.groupby("date")["volume"].cumsum()
    df["vwap"] = df["cum_pv"] / df["cum_vol"]
    return df.drop(columns=["price", "pv", "cum_pv", "cum_vol"])


def trade_return(entry: float, exit: float, direction: int, commission: float) -> float:
    """Calculate net return for a trade."""
    gross = direction * (exit - entry) / entry
    return gross - 2 * commission


def backtest(df: pd.DataFrame, commission: float) -> Tuple[pd.DataFrame, List[float]]:
    """Run VWAP trend-trading backtest."""
    df = df.copy()
    df["date"] = df["time(ms)"].dt.date
    equity = 1.0
    daily_equity = []
    trades: List[float] = []

    for date, day in df.groupby("date"):
        day = day.reset_index(drop=True)
        if len(day) < 2:
            continue
        entry_price = day.loc[1, "close"]
        direction = 1 if entry_price > day.loc[1, "vwap"] else -1
        for i in range(2, len(day)):
            price = day.loc[i, "close"]
            vwap = day.loc[i, "vwap"]
            if direction == 1 and price < vwap:
                r = trade_return(entry_price, price, direction, commission)
                equity *= 1 + r
                trades.append(r)
                direction = -1
                entry_price = price
            elif direction == -1 and price > vwap:
                r = trade_return(entry_price, price, direction, commission)
                equity *= 1 + r
                trades.append(r)
                direction = 1
                entry_price = price
        final_price = day.iloc[-1]["close"]
        r = trade_return(entry_price, final_price, direction, commission)
        equity *= 1 + r
        trades.append(r)
        daily_equity.append({"date": pd.Timestamp(date), "equity": equity})

    equity_df = pd.DataFrame(daily_equity)
    return equity_df, trades


def compute_metrics(equity: pd.DataFrame, trades: List[float]) -> pd.DataFrame:
    """Calculate performance metrics."""
    total_ret = equity["equity"].iloc[-1] - 1
    years = (equity["date"].iloc[-1] - equity["date"].iloc[0]).days / 365.25
    cagr = equity["equity"].iloc[-1] ** (1 / years) - 1
    daily_returns = equity["equity"].pct_change().dropna()
    vol = daily_returns.std() * np.sqrt(365)
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365)
    running_max = equity["equity"].cummax()
    max_dd = (equity["equity"] / running_max - 1).min()
    trades_arr = np.array(trades)
    hit_ratio = np.mean(trades_arr > 0)
    gains = trades_arr[trades_arr > 0]
    losses = trades_arr[trades_arr < 0]
    avg_gain = gains.mean() if len(gains) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    gain_loss_ratio = abs(avg_gain / avg_loss) if avg_loss != 0 else np.nan
    num_trades = len(trades)
    trades_per_day = num_trades / len(equity)

    metrics = pd.DataFrame(
        {
            "Metric": [
                "Total Return",
                "CAGR",
                "Annual Volatility",
                "Sharpe Ratio",
                "Max Drawdown",
                "Hit Ratio",
                "Average Gain",
                "Average Loss",
                "Gain/Loss Ratio",
                "# Trades",
                "Trades/Day",
            ],
            "Value": [
                total_ret,
                cagr,
                vol,
                sharpe,
                max_dd,
                hit_ratio,
                avg_gain,
                avg_loss,
                gain_loss_ratio,
                num_trades,
                trades_per_day,
            ],
        }
    )
    return metrics


def main() -> None:
    """Main execution flow."""
    df = fetch_ohlcv(SYMBOL, START_DATE, END_DATE, DATA_FILE)
    df = add_vwap(df)
    equity, trades = backtest(df, COMMISSION)
    metrics = compute_metrics(equity, trades)

    equity.to_csv("btc_vwap_trend.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(equity["date"], equity["equity"], label="Equity")
    plt.title("VWAP Trend-Trading Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("equity_curve.png")

    print(metrics.to_markdown(index=False, floatfmt=".4f"))
    plt.show()


if __name__ == "__main__":
    main()
