"""
Streamlit front end for backtest_ma.py.

Run with:   streamlit run app.py
(keep this file in the same folder as backtest_ma.py)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

import backtest_ma as bt

plt.style.use("dark_background")
plt.rcParams.update({"figure.facecolor": "#0e1117", "axes.facecolor": "#0e1117"})

DATA_DIR = Path("data")

# Yahoo Finance ticker -> description. Edit freely to add more ETFs.
ETFS = {
    "VOO": "Vanguard S&P 500 ETF (USD)",
    "SPY": "SPDR S&P 500 ETF Trust (USD)",
    "IVV": "iShares Core S&P 500 ETF (USD)",
    "VTI": "Vanguard Total Stock Market ETF (USD)",
    "QQQ": "Invesco QQQ Trust, Nasdaq-100 (USD)",
    "VUKE.L": "Vanguard FTSE 100 UCITS ETF, distributing (London, GBP)",
    "ISF.L": "iShares Core FTSE 100 UCITS ETF (London, GBP), the original script's default",
}

st.set_page_config(page_title="MA crossover backtester", layout="wide")


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading prices...")
def get_prices(ticker: str, offline: bool):
    return bt.load_prices(ticker, DATA_DIR, offline)


@st.cache_data(ttl=6 * 3600, show_spinner="Running parameter sweep...")
def get_sweep(ticker: str, offline: bool, shorts: tuple, longs: tuple, cost: float):
    return bt.parameter_sweep(get_prices(ticker, offline), list(shorts), list(longs), cost)


# label -> (column in the sweep results, number format for the heatmap cells)
SWEEP_METRICS = {
    "Sharpe (rf=0)": ("sharpe", "{:.2f}"),
    "Annualised return": ("annualised_return", "{:.0%}"),
    "Max drawdown": ("max_drawdown", "{:.0%}"),
}
TOP_TABLE_FORMATS = {
    "annualised_return": "{:.2%}", "annualised_vol": "{:.2%}", "sharpe": "{:.2f}",
    "max_drawdown": "{:.1%}", "time_in_market": "{:.0%}", "trades": "{:.0f}",
}


def show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Strategy settings")
    short = st.number_input("Short SMA (trading days)", min_value=2, max_value=250, value=bt.SHORT_WINDOW)
    long = st.number_input("Long SMA (trading days)", min_value=3, max_value=500, value=bt.LONG_WINDOW)
    cost_bps = st.number_input("Cost per trade (bps)", min_value=0.0, max_value=100.0,
                               value=bt.COST_PER_TRADE * 10_000, step=1.0)
    years = st.slider("Years shown on the price chart", 1, 10, 3)
    offline = st.checkbox("Use saved snapshot (no download)", value=False)
    if st.button("Clear cached data"):
        st.cache_data.clear()

if short >= long:
    st.error("The short SMA must be smaller than the long SMA.")
    st.stop()

# --------------------------------------------------------------------- main
st.title("Moving-average crossover backtest")

ticker = st.radio("ETF", list(ETFS), horizontal=True)
st.caption(ETFS[ticker])

try:
    prices = get_prices(ticker, offline)
    df = bt.run_backtest(prices, int(short), int(long), cost_bps / 10_000)
except Exception as exc:  # network problems, missing snapshot, too little history...
    st.error(f"Could not run the backtest for {ticker}: {exc}")
    st.stop()

moves = prices.pct_change().dropna()
bad = moves[moves.abs() > 0.15]
if not bad.empty:
    with st.expander(f"Warning: {len(bad)} daily moves larger than 15%, possible bad data"):
        st.dataframe(bad.map("{:+.1%}".format).rename("move"))

st.caption(
    f"{len(prices):,} daily prices ({prices.index[0]:%d %b %Y} to {prices.index[-1]:%d %b %Y}). "
    f"Backtest window: {df.index[0]:%d %b %Y} to {df.index[-1]:%d %b %Y}."
)

summary = bt.summarise(df)
net, bh = summary.loc["MA crossover (net)"], summary.loc["Buy & hold"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Annualised return (net)", f"{net.annualised_return:.2%}",
          f"{(net.annualised_return - bh.annualised_return) * 100:+.2f} pp vs buy & hold")
c2.metric("Sharpe (rf=0)", f"{net.sharpe:.2f}", f"{net.sharpe - bh.sharpe:+.2f} vs buy & hold")
c3.metric("Max drawdown", f"{net.max_drawdown:.1%}",
          f"{(net.max_drawdown - bh.max_drawdown) * 100:+.1f} pp vs buy & hold")
c4.metric("Trades", f"{net.trades:.0f}", f"{net.time_in_market:.0%} of days invested", delta_color="off")

st.dataframe(bt.format_summary(summary), use_container_width=True)

tab_price, tab_equity, tab_dd, tab_sweep, tab_data = st.tabs(
    ["Price & moving averages", "Equity curves", "Drawdowns", "Parameter sweep", "Daily data"]
)
with tab_price:
    show(bt.plot_price_and_ma(df, ticker, int(short), int(long), years))
with tab_equity:
    show(bt.plot_equity(df, ticker, int(short), int(long)))
with tab_dd:
    show(bt.plot_drawdowns(df, ticker))
with tab_sweep:
    st.caption(
        "Backtests every short/long SMA pair in the ranges below, net of costs, all scored over the "
        "same dates. The star marks the best pair in sample. Look for broad green areas: a single "
        "bright cell surrounded by red is probably overfitting."
    )
    c1, c2, c3 = st.columns(3)
    s_lo, s_hi = c1.slider("Short SMA range (step 5)", 5, 100, (5, 50), step=5)
    l_lo, l_hi = c2.slider("Long SMA range (step 10)", 20, 300, (20, 200), step=10)
    metric_label = c3.selectbox("Colour by", list(SWEEP_METRICS))
    column, fmt = SWEEP_METRICS[metric_label]

    shorts, longs = tuple(range(s_lo, s_hi + 1, 5)), tuple(range(l_lo, l_hi + 1, 10))
    n_combos = sum(s < l for s in shorts for l in longs)
    if n_combos == 0:
        st.warning("No valid pairs: every short window must be smaller than the long window.")
    elif n_combos > 800:
        st.warning(f"{n_combos} combinations is too many. Narrow the ranges to 800 or fewer.")
    else:
        try:
            results, baseline = get_sweep(ticker, offline, shorts, longs, cost_bps / 10_000)
        except Exception as exc:
            st.error(f"Sweep failed for {ticker}: {exc}")
        else:
            show(bt.plot_sweep_heatmap(results, column, metric_label, ticker, fmt))
            beat = (results[column] > baseline[column]).mean()
            st.caption(
                f"Buy & hold over the same window: {fmt.format(baseline[column])}. "
                f"{beat:.0%} of the {len(results)} pairs beat it on {metric_label.lower()}."
            )
            cols = ["short", "long", "annualised_return", "annualised_vol", "sharpe",
                    "max_drawdown", "time_in_market", "trades"]
            top = results.nlargest(10, column)[cols].copy()
            for c, f in TOP_TABLE_FORMATS.items():
                top[c] = top[c].map(f.format)
            st.subheader("Top 10 pairs")
            st.dataframe(top, hide_index=True, use_container_width=True)

with tab_data:
    st.dataframe(df.iloc[::-1], use_container_width=True)
    st.download_button("Download CSV", df.to_csv().encode("utf-8"), f"{ticker}_backtest_daily.csv", "text/csv")
