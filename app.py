# Market Mind = Streamlit Dashboard (with Live Market Mood Gauge)
# Run: python -m streamlit run app.py

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from datetime import timedelta

st.set_page_config(page_title="Market Mind", layout="wide")

# ---------------------- Sidebar Controls ----------------------
with st.sidebar:
    st.title("⚙️ Controls")

    period = st.selectbox("Data Period", ["6mo", "1y", "2y", "5y"], index=1)
    lookback_days = st.slider("Chart Lookback (days)", 90, 730, 180, step=10)

    st.markdown("**Signal Windows**")
    MOM_WIN = st.slider("Momentum window (days)", 5, 40, 10)
    VOL_WIN = st.slider("Volatility window (days)", 5, 40, 10)
    CORR_WIN = st.slider("SPX–VIX corr window (days)", 10, 60, 20)
    BREADTH_WIN = st.slider("Breadth lookback (days)", 3, 20, 5)

    st.markdown("**Score Weights**")
    w_mom = st.slider("Weight: Momentum", 0.0, 1.0, 0.35, 0.05)
    w_vol_spx = st.slider("Weight: SPX Vol (inverse)", 0.0, 1.0, 0.20, 0.05)
    w_vol_vix = st.slider("Weight: VIX Vol (inverse)", 0.0, 1.0, 0.10, 0.05)
    w_corr = st.slider("Weight: −Corr(SPX,VIX)", 0.0, 1.0, 0.20, 0.05)
    w_breadth = st.slider("Weight: Breadth", 0.0, 1.0, 0.15, 0.05)

    SCORE_NEUTRAL_BAND = st.slider("Neutral band (±)", 0.00, 0.50, 0.15, 0.01)
    min_run_days = st.slider("Merge runs shorter than (days)", 1, 10, 4)
    score_smooth_span = st.slider("Score smoothing EWMA span", 1, 20, 5)

    st.markdown("---")
    st.caption("Tip: widen the neutral band (≈0.15–0.20) to reduce regime flicker.")

# ---------------------- Constants ----------------------
TICKERS = ["^GSPC", "^IXIC", "^DJI", "^VIX"]
SECTOR_ETFS = ["XLK","XLY","XLE","XLF","XLV","XLI","XLB","XLU","XLRE","XLC"]
DPI = 140

# ---------------------- Data Layer ----------------------
@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_close_prices(tickers, period) -> pd.DataFrame:
    df = yf.download(tickers, period=period, auto_adjust=True,
                     group_by="ticker", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        close = df.xs("Close", level=1, axis=1)
    else:
        if "Close" in df.columns:
            close = df[["Close"]].copy()
            close.columns = [tickers[0]]
        elif "Adj Close" in df.columns:
            close = df[["Adj Close"]].copy()
            close.columns = [tickers[0]]
        else:
            raise ValueError("Could not find Close/Adj Close columns.")
    keep = [t for t in tickers if t in close.columns]
    return close[keep].dropna(how="all")


def rolling_signals(close: pd.DataFrame) -> pd.DataFrame:
    ret = close.pct_change()
    mom = ret["^GSPC"].rolling(MOM_WIN).mean()
    vol_spx = ret["^GSPC"].rolling(VOL_WIN).std()
    vol_vix = ret["^VIX"].rolling(VOL_WIN).std()
    corr = ret["^GSPC"].rolling(CORR_WIN).corr(ret["^VIX"])
    perf = (close / close.shift(BREADTH_WIN) - 1)
    idx_perf = perf[["^GSPC", "^IXIC", "^DJI"]]
    breadth = (idx_perf > 0).mean(axis=1)
    return pd.DataFrame({
        "mom": mom,
        "vol_spx": vol_spx,
        "vol_vix": vol_vix,
        "corr_spx_vix": corr,
        "breadth": breadth
    })


def to_score(components: pd.DataFrame, weights: dict) -> pd.Series:
    df = components.copy()

    def rz(s):
        # robust rolling z over 90d window
        m = s.rolling(90, min_periods=20).mean()
        sd = s.rolling(90, min_periods=20).std()
        sd = sd.where((sd != 0) & np.isfinite(sd), np.nan)
        return (s - m) / sd

    z_mom   = rz(df["mom"])               # +
    z_vol_s = -rz(df["vol_spx"])          # inverse
    z_vol_v = -rz(df["vol_vix"])          # inverse
    z_corr  = rz(-df["corr_spx_vix"])     # more negative corr → better
    z_br    = rz(df["breadth"])           # +

    raw = (weights["mom"] * z_mom +
           weights["vol_spx"] * z_vol_s +
           weights["vol_vix"] * z_vol_v +
           weights["corr"] * z_corr +
           weights["breadth"] * z_br)

    return np.tanh(raw)  # squash to [-1, +1]


def label_regime(score: pd.Series, neutral_band=0.15) -> pd.Series:
    lab = pd.Series(index=score.index, dtype="object")
    lab[score >  neutral_band]  = "Risk-On"
    lab[score < -neutral_band]  = "Risk-Off"
    lab[(score >= -neutral_band) & (score <= neutral_band)] = "Neutral"
    return lab


def backtest_direction(regime: pd.Series, spx_close: pd.Series) -> dict:
    ret1 = spx_close.pct_change().shift(-1)
    mask_on, mask_off = regime=="Risk-On", regime=="Risk-Off"
    hit_on  = (ret1[mask_on]  > 0).mean() if mask_on.any() else np.nan
    hit_off = (ret1[mask_off] < 0).mean() if mask_off.any() else np.nan
    avg_on  = ret1[mask_on].mean()  if mask_on.any() else np.nan
    avg_off = ret1[mask_off].mean() if mask_off.any() else np.nan
    overall_hits = pd.concat([(ret1[mask_on] > 0), (ret1[mask_off] < 0)]).mean() \
        if (mask_on.any() or mask_off.any()) else np.nan
    return {
        "obs_on": int(mask_on.sum()),
        "obs_off": int(mask_off.sum()),
        "hit_rate_on": float(hit_on) if pd.notna(hit_on) else np.nan,
        "hit_rate_off": float(hit_off) if pd.notna(hit_off) else np.nan,
        "overall_hit_rate": float(overall_hits) if pd.notna(overall_hits) else np.nan,
        "avg_next_day_on": float(avg_on) if pd.notna(avg_on) else np.nan,
        "avg_next_day_off": float(avg_off) if pd.notna(avg_off) else np.nan,
    }


def strategy_equity_curve(regime: pd.Series, spx_close: pd.Series, cash_rate_daily: float = 0.0):
    spx_ret = spx_close.pct_change().fillna(0.0)
    reidx = spx_ret.index.intersection(regime.index)
    spx_ret = spx_ret.reindex(reidx)
    reg = regime.reindex(reidx)

    pos = (reg == "Risk-On").astype(float)
    strat_ret = pos * spx_ret + (1 - pos) * cash_rate_daily

    equity = (1 + strat_ret).cumprod().rename("Regime Strategy")
    spy_equity = (1 + spx_ret).cumprod().rename("SPY Buy & Hold")

    ar = strat_ret.mean() * 252
    vol = strat_ret.std() * np.sqrt(252)
    sharpe = ar / vol if vol > 0 else np.nan
    dd = (equity / equity.cummax() - 1).min()

    return equity, spy_equity, {
        "Ann. Return": ar, "Ann. Vol": vol, "Sharpe": sharpe, "Max Drawdown": dd
    }


def _merge_short_runs(regime: pd.Series, min_days: int) -> pd.Series:
    r = regime.copy()
    start = 0
    while start < len(r):
        end = start
        while end < len(r) and r.iloc[end] == r.iloc[start]:
            end += 1
        run_len = end - start
        if run_len < min_days and 0 < start < len(r)-run_len:
            r.iloc[start:end] = r.iloc[start-1]
        start = end
    return r


@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_sector_close(period="6mo"):
    df = yf.download(SECTOR_ETFS, period=period, auto_adjust=True, group_by="ticker", progress=False)
    close = df.xs("Close", level=1, axis=1) if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    return close.dropna(how="all")

# ---------------------- Plotting Helpers ----------------------
def plot_regime(spx: pd.Series, regime: pd.Series, score: pd.Series,
                lookback_days=180, min_run_days=4, score_smooth_span=5):
    end_date = spx.index[-1]
    start_date = end_date - timedelta(days=lookback_days)
    spx = spx.loc[spx.index >= start_date]
    regime = regime.reindex(spx.index, method="pad")
    score = score.reindex(spx.index).ewm(span=score_smooth_span, adjust=False).mean()

    regime = _merge_short_runs(regime, min_run_days)

    plt.rcParams.update({
        "figure.dpi": DPI,
        "axes.facecolor": "white",
        "axes.edgecolor": "#333",
        "axes.labelcolor": "#111",
        "text.color": "#111",
        "axes.titlesize": 18,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "grid.color": "#e6e6e6",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "figure.figsize": (11.5, 5.3)
    })

    fig, ax = plt.subplots()
    ax.grid(True, axis="y")

    # background bands
    curr, band_start = None, None
    for d, r in regime.items():
        if curr is None:
            curr, band_start = r, d
        elif r != curr:
            color = "#d1f2d7" if curr=="Risk-On" else ("#f9d6d6" if curr=="Risk-Off" else "#efefef")
            ax.axvspan(band_start, d, color=color, alpha=0.6, linewidth=0)
            curr, band_start = r, d
    if curr is not None:
        color = "#d1f2d7" if curr=="Risk-On" else ("#f9d6d6" if curr=="Risk-Off" else "#efefef")
        ax.axvspan(band_start, spx.index[-1], color=color, alpha=0.6, linewidth=0)

    ax.plot(spx.index, spx.values, linewidth=2.2, color="#0b62ff", label="S&P 500")

    ax2 = ax.twinx()
    ax2.plot(score.index, score.values, linestyle="--", linewidth=1.8, color="#3b3b3b", label="Market Score")
    ax2.axhline(0, color="#b0b0b0", linestyle=":", linewidth=1)
    ax2.set_ylim(-1.05, 1.05)

    latest_regime = regime.iloc[-1]
    ax.text(0.01, 0.96, f"Now: {latest_regime}", transform=ax.transAxes,
            fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"))

    ax.set_title("S&P 500 — Market Score & Regime Bands")
    ax.set_xlabel("Date"); ax.set_ylabel("Index Level"); ax2.set_ylabel("Score")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax.legend(h1+h2, l1+l2, loc="upper left", frameon=True, framealpha=1, edgecolor="#dddddd")
    for txt in leg.get_texts():
        txt.set_fontsize(11)

    fig.tight_layout()
    return fig


def plot_equity(equity: pd.Series, spy_equity: pd.Series):
    plt.figure(figsize=(11.5, 4.8), dpi=140)
    plt.plot(equity.index, equity.values, label="Regime Strategy", linewidth=2.0, color="#0b62ff")
    plt.plot(spy_equity.index, spy_equity.values, label="SPY Buy & Hold", linewidth=1.6, color="#7f8c8d", linestyle="--")
    plt.grid(True, axis="y", color="#e9e9e9")
    plt.title("Regime Strategy vs Buy & Hold")
    plt.ylabel("Equity (starts at 1.0)")
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def plot_sector_bar(zscores: pd.Series):
    plt.figure(figsize=(11.5, 4.8), dpi=140)
    order = list(zscores.index)
    vals = zscores.values
    colors = ["#1f78b4" if v >= 0 else "#e74c3c" for v in vals]
    plt.bar(order, vals, color=colors)
    plt.axhline(0, color="#b0b0b0", linestyle=":", linewidth=1)
    plt.title("Sector Momentum (20d, z-score)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return plt.gcf()

# ---------------------- Live Mood Gauge ----------------------
def mood_gauge(score_value: float, neutral_band: float = 0.15):
    """
    Render a clean horizontal gauge showing Risk-On % based on score in [-1, 1].
    """
    # map score [-1..+1] -> [0..100] risk-on percentage
    pct = float((score_value + 1) / 2) * 100.0
    label = "Risk-On" if score_value > neutral_band else ("Risk-Off" if score_value < -neutral_band else "Neutral")
    color = "#2ecc71" if label == "Risk-On" else ("#e74c3c" if label == "Risk-Off" else "#f1c40f")

    st.markdown(
        f"""
        <div style="width:100%;margin:8px 0 18px 0;">
          <div style="font-size:18px;font-weight:700;margin-bottom:6px;">
            Current Market Mood: <span style="color:{color}">{label}</span>
            <span style="font-size:14px;color:#666;font-weight:600;">&nbsp;({pct:.0f}% Risk-On)</span>
          </div>
          <div style="position:relative;width:100%;height:22px;background:#eee;border-radius:12px;overflow:hidden;">
            <div style="position:absolute;left:0;top:0;height:100%;width:{pct:.1f}%;background:{color};opacity:0.85;"></div>
            <div style="position:absolute;left:8px;top:1px;font-size:12px;color:#555;">0%</div>
            <div style="position:absolute;right:8px;top:1px;font-size:12px;color:#555;">100%</div>
            <div style="position:absolute;left:50%;top:0;height:100%;width:1px;background:#bbb;opacity:0.8;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------- App Logic ----------------------
st.title("📈 Market Mind")

# Load data and compute
close = fetch_close_prices(TICKERS, period)
components = rolling_signals(close).dropna()
weights = {"mom": w_mom, "vol_spx": w_vol_spx, "vol_vix": w_vol_vix, "corr": w_corr, "breadth": w_breadth}
score = to_score(components, weights).dropna()
regime = label_regime(score, SCORE_NEUTRAL_BAND).reindex(score.index)

# === Live Mood Gauge ===
mood_gauge(float(score.iloc[-1]), neutral_band=SCORE_NEUTRAL_BAND)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Regime Chart", "Strategy", "Sectors", "Data/Export"])

with tab1:
    fig = plot_regime(
        close["^GSPC"].reindex(score.index),
        regime,
        score,
        lookback_days=lookback_days,
        min_run_days=min_run_days,
        score_smooth_span=score_smooth_span
    )
    st.pyplot(fig, clear_figure=True)

with tab2:
    equity, spy_equity, stats = strategy_equity_curve(regime, close["^GSPC"].reindex(score.index))
    colA, colB = st.columns([2,1])
    with colA:
        st.pyplot(plot_equity(equity, spy_equity), clear_figure=True)
    with colB:
        st.markdown("### Strategy Stats")
        nice = {k: (f"{v:.4f}" if np.isfinite(v) else "n/a") for k, v in stats.items()}
        st.table(pd.DataFrame(nice, index=["value"]).T)
    st.caption("Rule: Hold SPY in Risk-On, cash otherwise (no fees/slippage).")

with tab3:
    sec_close = fetch_sector_close(period="6mo")
    ret = sec_close.pct_change()
    mom_win = 20
    mom = ret.rolling(mom_win).mean().iloc[-1]

    # Cross-sectional z-score (safe std handling)
    std = float(mom.std())
    if not np.isfinite(std) or std == 0:
        z = (mom - mom.mean()) * 0
    else:
        z = (mom - mom.mean()) / std

    perf_20d = sec_close.iloc[-1] / sec_close.iloc[-mom_win] - 1
    breadth_pct = (perf_20d > 0).mean()

    st.pyplot(plot_sector_bar(z.sort_values(ascending=False)), clear_figure=True)
    st.write(f"**Sector breadth (share up over {mom_win}d):** {breadth_pct*100:.1f}%")

with tab4:
    out = pd.concat([
        close["^GSPC"].reindex(score.index).rename("SPX_Close"),
        components.rename(columns={
            "mom":"mom_10d", "vol_spx":"vol_spx_10d", "vol_vix":"vol_vix_10d", "corr_spx_vix":"corr_spx_vix_20d",
            "breadth":"breadth_5d"
        }),
        score.rename("market_score"),
        regime.rename("regime")
    ], axis=1)
    st.dataframe(out.tail(30))
    csv = out.to_csv(index=True).encode()
    st.download_button("Download CSV", data=csv, file_name="market_mind_export.csv", mime="text/csv")

st.caption("Data source: Yahoo Finance via yfinance. Educational/analytical use only; not investment advice.")


