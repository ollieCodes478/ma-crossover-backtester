"""
Moving-average crossover backtest on the iShares Core FTSE 100 UCITS ETF
(London: ISF, Yahoo Finance: ISF.L).

Strategy
    Hold the ETF while the short simple moving average (SMA) is above the long SMA,
    otherwise sit in cash. A signal is read from day t's close and the position is
    held from that same close (the shift(1) below), so a day's return is never
    earned using information from that day.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.ticker import PercentFormatter

TICKER = "ISF.L"          # iShares Core FTSE 100 UCITS ETF GBP (Dist)
SHORT_WINDOW = 20         # about one trading month
LONG_WINDOW = 50          # about two and a half trading months
COST_PER_TRADE = 0.0005   # 0.05% (5 bps) of the portfolio each time we switch in or out
TRADING_DAYS = 252


# --------------------------------------------------------------------------- data
def download_prices(ticker: str) -> pd.Series:
    """Daily split- and dividend-adjusted closes from Yahoo Finance."""
    # Recent yfinance versions default to auto_adjust=True: 'Close' is already adjusted
    # and there is no 'Adj Close' column. Setting it explicitly keeps this stable.
    raw = yf.download(ticker, period="max", interval="1d", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(
            f"Yahoo returned no data for {ticker}. Check the ticker and your connection, "
            "or upgrade the library: pip install -U yfinance"
        )
    if isinstance(raw.columns, pd.MultiIndex):  # recent versions: (field, ticker) columns
        raw.columns = raw.columns.get_level_values(0)

    prices = raw["Close"].dropna().astype(float)
    prices = prices[prices > 0].rename("adj_close")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "date"
    return prices


def load_prices(ticker: str, data_dir: Path, offline: bool = False) -> pd.Series:
    """Download fresh data (and save a snapshot), or reuse the saved snapshot."""
    path = data_dir / f"{ticker}.csv"
    if offline:
        if not path.exists():
            raise FileNotFoundError(f"No saved snapshot at {path}. Run once without --offline first.")
        return pd.read_csv(path, index_col=0, parse_dates=True)["adj_close"]

    prices = download_prices(ticker)
    data_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, header=True)
    return prices


def flag_suspect_moves(prices: pd.Series, threshold: float = 0.15) -> None:
    """Yahoo occasionally serves bad ticks (e.g. pence/pounds mix-ups on .L tickers)."""
    moves = prices.pct_change().dropna()
    bad = moves[moves.abs() > threshold]
    if not bad.empty:
        print(f"WARNING: {len(bad)} daily moves larger than {threshold:.0%} - possible bad data, "
              "so treat the results below with care:")
        print(bad.map("{:+.1%}".format).to_string())
        print()


# ----------------------------------------------------------------------- backtest
def run_backtest(prices: pd.Series, short_window: int, long_window: int, cost: float) -> pd.DataFrame:
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")
    if len(prices) < long_window + 2:
        raise ValueError(f"Need at least {long_window + 2} prices, got {len(prices)}")

    df = prices.to_frame("adj_close")
    df["ma_short"] = df["adj_close"].rolling(short_window).mean()
    df["ma_long"] = df["adj_close"].rolling(long_window).mean()

    # 1 while the short MA is above the long MA, else 0. Comparisons with NaN are False,
    # so the warm-up days (before the long MA exists) are 0 = cash.
    df["signal"] = (df["ma_short"] > df["ma_long"]).astype(int)
    # Today's position comes from yesterday's signal: no look-ahead bias.
    df["position"] = df["signal"].shift(1).fillna(0).astype(int)

    df["asset_return"] = df["adj_close"].pct_change()
    df["strategy_return"] = df["position"] * df["asset_return"]  # cash earns 0

    # A trade happens whenever the position changes (0 -> 1 buy, 1 -> 0 sell).
    df["trade"] = df["position"].diff().abs().fillna(0)
    df["cost"] = df["trade"] * cost
    df["strategy_return_net"] = df["strategy_return"] - df["cost"]

    # Score all three series over the same window: from the first day the long MA exists.
    # (Dropping one more row makes buy & hold start from that day's close, too.)
    start = df["ma_long"].first_valid_index()
    df = df.loc[start:].iloc[1:].copy()

    df["cum_asset"] = (1 + df["asset_return"]).cumprod()
    df["cum_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["cum_strategy_net"] = (1 + df["strategy_return_net"]).cumprod()
    return df


# ------------------------------------------------------------------------ metrics
def drawdown_series(returns: pd.Series) -> pd.Series:
    growth = (1 + returns).cumprod()
    peak = growth.cummax().clip(lower=1.0)  # the starting value of 1 counts as a peak
    return growth / peak - 1


def performance_metrics(
    returns: pd.Series, position: pd.Series | None = None, n_trades: float | None = None
) -> dict:
    returns = returns.dropna()
    years = len(returns) / TRADING_DAYS
    total = (1 + returns).prod() - 1
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    return {
        "total_return": total,
        "annualised_return": (1 + total) ** (1 / years) - 1,
        "annualised_vol": vol,
        # Sharpe with a risk-free rate of 0: annualised mean daily return / annualised vol
        "sharpe": returns.mean() * TRADING_DAYS / vol if vol > 0 else np.nan,
        "max_drawdown": drawdown_series(returns).min(),
        "time_in_market": 1.0 if position is None else position.mean(),
        "trades": np.nan if n_trades is None else n_trades,
    }


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    n_trades = df["trade"].sum()
    rows = {
        "Buy & hold": performance_metrics(df["asset_return"]),
        "MA crossover (gross)": performance_metrics(df["strategy_return"], df["position"], n_trades),
        "MA crossover (net)": performance_metrics(df["strategy_return_net"], df["position"], n_trades),
    }
    return pd.DataFrame(rows).T


def format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    pct = "{:.2%}".format
    out = pd.DataFrame(index=summary.index)
    out["Total return"] = summary["total_return"].map(pct)
    out["Annualised return"] = summary["annualised_return"].map(pct)
    out["Annualised vol"] = summary["annualised_vol"].map(pct)
    out["Sharpe (rf=0)"] = summary["sharpe"].map("{:.2f}".format)
    out["Max drawdown"] = summary["max_drawdown"].map(pct)
    out["Time invested"] = summary["time_in_market"].map("{:.0%}".format)
    out["Trades"] = summary["trades"].map(lambda x: "-" if pd.isna(x) else f"{x:.0f}")
    return out


def to_markdown(table: pd.DataFrame) -> str:
    """Tiny markdown-table writer so no extra dependency (tabulate) is needed."""
    cols = ["Strategy", *table.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join([name, *row]) + " |" for name, row in zip(table.index, table.to_numpy())]
    return "\n".join(lines)


# -------------------------------------------------------------------------- plots
def plot_price_and_ma(df: pd.DataFrame, ticker: str, short_window: int, long_window: int, years: int = 3):
    """Zoomed to the last few years so the crossovers are actually visible."""
    recent = df.iloc[-years * TRADING_DAYS:]
    change = recent["signal"].diff()
    buys, sells = recent[change == 1], recent[change == -1]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(recent.index, recent["adj_close"], label="Adjusted close", alpha=0.7)
    ax.plot(recent.index, recent["ma_short"], label=f"{short_window}-day MA", linewidth=2)
    ax.plot(recent.index, recent["ma_long"], label=f"{long_window}-day MA", linewidth=2)
    ax.scatter(buys.index, buys["adj_close"], marker="^", color="green", s=80, zorder=3, label="Buy signal")
    ax.scatter(sells.index, sells["adj_close"], marker="v", color="red", s=80, zorder=3, label="Sell signal")
    ax.set(
        title=f"{ticker}: price and moving averages (last {years} years)",
        xlabel="Date",
        ylabel="Price (dividend-adjusted)",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_equity(df: pd.DataFrame, ticker: str, short_window: int, long_window: int):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["cum_asset"], label="Buy & hold", alpha=0.8)
    ax.plot(df.index, df["cum_strategy"], label="MA crossover (gross)", alpha=0.7)
    ax.plot(df.index, df["cum_strategy_net"], label="MA crossover (net of costs)", linewidth=2)
    ax.set(
        title=f"{ticker}: buy & hold vs {short_window}/{long_window} MA crossover",
        xlabel="Date",
        ylabel="Growth of 1 invested",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_drawdowns(df: pd.DataFrame, ticker: str):
    series = {"Buy & hold": df["asset_return"], "MA crossover (net)": df["strategy_return_net"]}
    fig, axes = plt.subplots(len(series), 1, figsize=(12, 7), sharex=True, sharey=True)
    for ax, (name, returns) in zip(axes, series.items()):
        dd = drawdown_series(returns)
        ax.fill_between(dd.index, dd, 0, color="red", alpha=0.3)
        ax.plot(dd.index, dd, color="red", linewidth=1)
        ax.set_title(f"{name}: max drawdown {dd.min():.1%}")
        ax.set_ylabel("Drawdown")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Date")
    fig.suptitle(f"{ticker} drawdowns")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- parameter sweep
def parameter_sweep(
    prices: pd.Series, shorts: list[int], longs: list[int], cost: float
) -> tuple[pd.DataFrame, dict]:
    """Backtest every valid (short, long) pair and score them all on one common window.

    Each run_backtest() call starts scoring from its own long-MA warm-up, so a 200-day
    window would otherwise be judged over fewer years than a 50-day one. Cutting every
    run to the start date of the slowest pair keeps the comparison like for like.
    Returns (one row of net-of-cost metrics per pair, buy & hold metrics for the same window).
    """
    pairs = [(s, l) for s in shorts for l in longs if s < l]
    if not pairs:
        raise ValueError("No valid combinations: each short window must be smaller than the long window")
    max_long = max(l for _, l in pairs)
    if len(prices) < max_long + 2:
        raise ValueError(f"Need at least {max_long + 2} prices for a {max_long}-day window, got {len(prices)}")

    start = prices.index[max_long]  # first scored day of the slowest pair
    rows = []
    for s, l in pairs:
        d = run_backtest(prices, s, l, cost).loc[start:]
        m = performance_metrics(d["strategy_return_net"], d["position"], d["trade"].sum())
        rows.append({"short": s, "long": l, **m})

    baseline = performance_metrics(prices.pct_change().loc[start:])
    return pd.DataFrame(rows), baseline


def plot_sweep_heatmap(results: pd.DataFrame, column: str, label: str, ticker: str, fmt: str = "{:.2f}"):
    """Heatmap of one metric: short window on the y axis, long window on the x axis.
    Green is better for every supported metric (higher Sharpe, higher return, drawdown closer to 0)."""
    grid = results.pivot(index="short", columns="long", values=column)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(grid.to_numpy(dtype=float), origin="lower", aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(grid.columns)), labels=grid.columns)
    ax.set_yticks(range(len(grid.index)), labels=grid.index)

    if grid.size <= 150:  # only print numbers in the cells while they stay readable
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid.iat[i, j]
                if pd.notna(value):
                    ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=8, color="black")

    best = results.loc[results[column].idxmax()]
    ax.scatter(list(grid.columns).index(best["long"]), list(grid.index).index(best["short"]),
               marker="*", s=350, color="white", edgecolor="black", zorder=3, label="Best in sample")
    ax.set(title=f"{ticker}: {label} by SMA window pair (net of costs)",
           xlabel="Long SMA (trading days)", ylabel="Short SMA (trading days)")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Moving-average crossover backtest (default: iShares FTSE 100, ISF.L)")
    p.add_argument("--ticker", default=TICKER)
    p.add_argument("--short", type=int, default=SHORT_WINDOW, help="short SMA window in trading days")
    p.add_argument("--long", type=int, default=LONG_WINDOW, help="long SMA window in trading days")
    p.add_argument("--cost", type=float, default=COST_PER_TRADE, help="cost per trade as a fraction (0.0005 = 5 bps)")
    p.add_argument("--datadir", type=Path, default=Path("data"), help="where the price snapshot is saved")
    p.add_argument("--outdir", type=Path, default=Path("results"), help="where charts and tables are saved")
    p.add_argument("--offline", action="store_true", help="use the saved snapshot instead of downloading")
    p.add_argument("--show", action="store_true", help="open the charts in a window")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    prices = load_prices(args.ticker, args.datadir, args.offline)
    flag_suspect_moves(prices)
    print(f"{args.ticker}: {len(prices):,} daily prices, {prices.index[0]:%d %b %Y} to {prices.index[-1]:%d %b %Y}")

    df = run_backtest(prices, args.short, args.long, args.cost)
    print(
        f"Backtest window: {df.index[0]:%d %b %Y} to {df.index[-1]:%d %b %Y} "
        f"({args.short}/{args.long}-day SMA, {args.cost:.2%} per trade)\n"
    )

    table = format_summary(summarise(df))
    print(table.to_string())
    print()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.outdir / "backtest_daily.csv")
    (args.outdir / "metrics.md").write_text(to_markdown(table) + "\n", encoding="utf-8")
    figures = {
        "price_and_ma": plot_price_and_ma(df, args.ticker, args.short, args.long),
        "equity_curves": plot_equity(df, args.ticker, args.short, args.long),
        "drawdowns": plot_drawdowns(df, args.ticker),
    }
    for name, fig in figures.items():
        fig.savefig(args.outdir / f"{name}.png", dpi=150)
    print(f"Saved metrics.md, backtest_daily.csv and {len(figures)} charts to {args.outdir}/")

    if args.show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
