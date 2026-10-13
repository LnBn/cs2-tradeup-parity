"""Segment price indices, event-study windows and co-movement for the pilot.

Conventions
-----------
* Prices are daily Steam Community Market prices (hand-collected; see README),
  in the collecting account's wallet currency. Daily prices mix floats, patterns
  and stickers, so every item-level statistic below uses medians of daily log
  prices rather than means, and only log changes within an item.
* Event: the "Trade Up Contract" patch was posted 2025-10-22 23:00 UTC, so
  day 0 is 2025-10-23 (first full UTC day) and 2025-10-22 is excluded from the
  baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_DAY = pd.Timestamp("2025-10-23")
SOUVENIR_TRADEUP_DAY = pd.Timestamp("2026-05-22")    # second supply change (see README)
BASELINE = (-31, -2)                                  # 2025-09-22 .. 2025-10-21

# (label, first day, last day) relative to EVENT_DAY, inclusive
WINDOWS = [
    ("pre [-90,-32]", -90, -32),     # placebo: should be ~0 if no pre-trend
    ("[0,2]", 0, 2),
    ("[3,9]", 3, 9),
    ("[10,30]", 10, 30),
    ("[31,90]", 31, 90),
    ("[91,180]", 91, 180),
    ("[181,350]", 181, 350),
]


def add_event_time(panel: pd.DataFrame, price_col: str = "avg_price_usd") -> pd.DataFrame:
    out = panel.copy()
    out["log_p"] = np.log(out[price_col])
    out["t"] = (out["day"] - EVENT_DAY).dt.days
    return out


def baseline_levels(panel: pd.DataFrame, min_days: int = 15) -> pd.Series:
    """Median baseline log price per item (items with too few baseline days dropped)."""
    lo, hi = BASELINE
    base = panel[(panel.t >= lo) & (panel.t <= hi)].groupby("market_hash_name")["log_p"]
    n = base.size()
    return base.median()[n >= min_days]


def window_changes(panel: pd.DataFrame, base: pd.Series, min_obs: int = 2) -> pd.DataFrame:
    """Item x window table of (median log price in window) - (baseline level)."""
    cols = {}
    for label, lo, hi in WINDOWS:
        w = panel[(panel.t >= lo) & (panel.t <= hi)].groupby("market_hash_name")["log_p"]
        med = w.median()[w.size() >= min_obs]
        cols[label] = med - base
    return pd.DataFrame(cols).loc[base.index]


def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = 4000, seed: int = 0):
    """Statistic and percentile 95% CI from resampling items with replacement."""
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    boots = stat(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return stat(x), *np.percentile(boots, [2.5, 97.5])


def bootstrap_diff_ci(x: np.ndarray, y: np.ndarray, stat=np.mean, n_boot: int = 4000,
                      seed: int = 0):
    """Difference in a statistic (x - y), resampling each group independently."""
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    rng = np.random.default_rng(seed)
    bx = stat(rng.choice(x, size=(n_boot, len(x))), axis=1)
    by = stat(rng.choice(y, size=(n_boot, len(y))), axis=1)
    return stat(x) - stat(y), *np.percentile(bx - by, [2.5, 97.5])


def segment_table(changes: pd.DataFrame, segment: pd.Series, control: str) -> pd.DataFrame:
    """Per segment and window: n, mean and median log change with item-bootstrap 95% CIs,
    and the difference in medians vs the control segment."""
    rows = []
    seg = segment.loc[changes.index]
    for s in seg.unique():
        for label in changes.columns:
            x = changes.loc[seg == s, label].to_numpy()
            y = changes.loc[seg == control, label].to_numpy()
            m, lo, hi = bootstrap_ci(x, np.mean)
            md, mdlo, mdhi = bootstrap_ci(x, np.median)
            d, dlo, dhi = bootstrap_diff_ci(x, y, np.median)
            rows.append({"segment": s, "window": label, "n": int((~np.isnan(x)).sum()),
                         "mean": m, "mean_lo": lo, "mean_hi": hi,
                         "median": md, "median_lo": mdlo, "median_hi": mdhi,
                         "median_diff_vs_control": d, "diff_lo": dlo, "diff_hi": dhi})
    return pd.DataFrame(rows)


def segment_index(panel: pd.DataFrame, base: pd.Series, segment: pd.Series) -> pd.DataFrame:
    """Daily segment index: cross-item median of log(p_it) - baseline_i. Columns = segments."""
    p = panel[panel.market_hash_name.isin(base.index)].copy()
    p["rel"] = p["log_p"] - p["market_hash_name"].map(base)
    p["segment"] = p["market_hash_name"].map(segment)
    return p.pivot_table(index="day", columns="segment", values="rel", aggfunc="median")


def weekly_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Week x item matrix of weekly log returns (weekly price = median daily log price)."""
    p = panel.assign(week=panel["day"].dt.to_period("W-SUN").dt.start_time)
    wk = p.pivot_table(index="week", columns="market_hash_name", values="log_p", aggfunc="median")
    return wk.diff()


def block_correlations(returns: pd.DataFrame, segment: pd.Series, order: list[str],
                       min_obs: int = 20) -> pd.DataFrame:
    """Mean pairwise correlation within/between segments (diagonal excludes self-pairs)."""
    keep = returns.columns[returns.notna().sum() >= min_obs]
    c = returns[keep].corr(min_periods=min_obs)
    seg = segment.loc[keep]
    out = pd.DataFrame(index=order, columns=order, dtype=float)
    for a in order:
        for b in order:
            block = c.loc[seg.index[seg == a], seg.index[seg == b]].to_numpy()
            if a == b:
                block = block[~np.eye(len(block), dtype=bool)]
            out.loc[a, b] = np.nanmean(block) if block.size else np.nan
    return out


def moving_block_bootstrap_corr_diff(ret_pre: pd.DataFrame, ret_post: pd.DataFrame,
                                     segment: pd.Series, a: str, b: str,
                                     block: int = 4, n_boot: int = 500, seed: int = 0):
    """CI for change (post - pre) in mean a-b correlation, resampling blocks of weeks."""
    rng = np.random.default_rng(seed)

    def resample(r):
        n = len(r)
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        return r.iloc[idx].reset_index(drop=True)

    stats = []
    for _ in range(n_boot):
        pre = block_correlations(resample(ret_pre), segment, [a, b], min_obs=15).loc[a, b]
        post = block_correlations(resample(ret_post), segment, [a, b], min_obs=15).loc[a, b]
        stats.append(post - pre)
    return np.nanpercentile(stats, [2.5, 97.5])
