"""Robustness of the Covert -> knife parity results (Tests 1 and 2 of scripts/06).

One change at a time, starting from the scripts/06 baseline and then from the preferred
specification (Doppler phases priced from their own sales, weighted by 90-day sale
counts):
  * phase pricing    : listing-quantity-weighted asks (06) / sale-count weights / P1-4 only
  * price reference  : 7-, 30- or 90-day Skinport median sale
  * vanilla knives   : in the pools (as in the schema) or removed
  * liquidity screen : >= 1, 5 or 20 Skinport sales in 30 days
  * fee on output    : 0%, 8% (Skinport), 15% (Steam)
For the baseline and preferred specifications it also reports bootstrap CIs (items for
item statistics and the Test-2 slope; cases for cheapest-per-case statistics) and a
leave-one-case-out check.

Outputs: results/parity/robustness_*.csv, figures/parity/robustness.png
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import parity as P  # noqa: E402
import parity_inputs as PI  # noqa: E402
from http_cache import PROJECT_ROOT  # noqa: E402

RES, FIG = PROJECT_ROOT / "results" / "parity", PROJECT_ROOT / "figures" / "parity"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6})
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"

inp = PI.load()
SHIFT = PI.index_shift(inp, ["Knife", "Gloves"])
EVENT_ITEMS = inp.sample.index[inp.sample.segment == "Covert (gold pool)"]
pools_no_vanilla = P.drop_vanilla(inp.pools)
_price_cache = {}


def prices_for(phase: str, window: str) -> pd.Series:
    key = (phase, window)
    if key not in _price_cache:
        _price_cache[key] = (P.price_table(inp.sales, inp.items) if phase == "v06"
                             else P.price_table_v2(inp.sales, inp.items, window, phase))
    return _price_cache[key]


def run(phase="sales", window="30", vanilla=True, min_vol=5, fee=0.0, drop_case=None):
    """Return (Test-1 table, Test-2 event table) for one specification."""
    prices = prices_for(phase, window)
    pools = inp.pools if vanilla else pools_no_vanilla
    xs = P.covert_cross_section(prices, inp.vol30, inp.schema_df, pools, min_vol=min_vol, fee=fee)
    p_now = {n: P.parity_for(n, inp.schema_df, pools, prices)["parity"] for n in EVENT_ITEMS}
    ev = P.event_gap(EVENT_ITEMS, p_now, prices, inp.base, inp.now, inp.changes, SHIFT, fee=fee)
    if drop_case is not None:
        xs = xs[~xs.cases.str.contains(drop_case, regex=False)]
        ev_cases = {n: P.parity_for(n, inp.schema_df, pools, prices)["cases"] for n in ev.market_hash_name}
        ev = ev[[drop_case not in ev_cases[n] for n in ev.market_hash_name]]
    return xs, ev


def headline(xs: pd.DataFrame, ev: pd.DataFrame) -> dict:
    ns, st = xs[~xs.stattrak], xs[xs.stattrak]
    a, b = P.ratio_summary(ns), P.ratio_summary(st)
    slope, se, _ = P.cheapest_slope(ns)
    m = P.gap_models(ev).set_index("model")
    return {"nonST_items": a["n_items"], "nonST_median_ratio": a["median_ratio"],
            "nonST_share_below_0.85": a["share_below_0.85"],
            "nonST_cheapest_median": a["cheapest_median_ratio"],
            "nonST_cheapest_slope": slope, "nonST_cheapest_slope_se": se,
            "ST_items": b["n_items"], "ST_share_below_0.85": b["share_below_0.85"],
            "ST_cheapest_median": b["cheapest_median_ratio"],
            "event_n": int(m.loc["change ~ max(0, gap)  [parity]", "n"]),
            "event_slope": m.loc["change ~ max(0, gap)  [parity]", "slope"],
            "event_slope_se": m.loc["change ~ max(0, gap)  [parity]", "se"],
            "event_placebo_slope": m.loc["placebo: pre-window change ~ max(0, gap)", "slope"],
            "event_n_below_floor": int((ev.gap > 0).sum())}


# ---- 1. One change at a time ---------------------------------------------------------------
BASE06 = dict(phase="v06")
PREF = dict(phase="sales")
VARIANTS = [
    ("scripts/06 baseline", BASE06),
    ("06 + no vanilla knives", {**BASE06, "vanilla": False}),
    ("preferred: phases by sale counts", PREF),
    ("pref + phases 1-4 only (gems ignored)", {**PREF, "phase": "core"}),
    ("pref + 7-day medians", {**PREF, "window": "7"}),
    ("pref + 90-day medians", {**PREF, "window": "90"}),
    ("pref + no vanilla knives", {**PREF, "vanilla": False}),
    ("pref + liquidity >= 1 sale", {**PREF, "min_vol": 1}),
    ("pref + liquidity >= 20 sales", {**PREF, "min_vol": 20}),
    ("pref + 8% fee", {**PREF, "fee": 0.08}),
    ("pref + 15% fee", {**PREF, "fee": 0.15}),
]
rob = pd.DataFrame([{"spec": label, **headline(*run(**kw))} for label, kw in VARIANTS])
rob.to_csv(RES / "robustness_variants.csv", index=False, float_format="%.3f")
pd.set_option("display.width", 260)
print("Headline numbers under one-at-a-time changes:")
print(rob.round(3).to_string(index=False))

# ---- 2. Bootstrap CIs (baseline and preferred) -------------------------------------------


def cheapest(d):
    return d.loc[d.groupby("cases").log_ratio.idxmin()]


def slope_of(c):
    return stats.linregress(np.log(c.parity), np.log(c.price)).slope if len(c) > 2 else np.nan


def event_slope(e):
    lift = np.maximum(e.gap, 0)
    return stats.linregress(lift, e.change).slope if lift.nunique() > 2 else np.nan


ci_rows = []
for label, kw in [("scripts/06 baseline", BASE06), ("preferred", PREF)]:
    xs, ev = run(**kw)
    ev = ev.dropna(subset=["change"])
    for st in (False, True):
        d = xs[xs.stattrak == st]
        c = cheapest(d)
        tag = "ST" if st else "nonST"
        for stat, fn, data, point in [
                ("median_ratio", lambda x: np.exp(x.log_ratio.median()), d, np.exp(d.log_ratio.median())),
                ("share_below_0.85", lambda x: (x.log_ratio < np.log(0.85)).mean(), d,
                 (d.log_ratio < np.log(0.85)).mean()),
                ("cheapest_median_ratio", lambda x: np.exp(x.log_ratio.median()), c, np.exp(c.log_ratio.median())),
                ("cheapest_slope", slope_of, c, slope_of(c))]:
            lo, hi = PI.bootstrap(fn, data)
            ci_rows.append({"spec": label, "group": tag, "statistic": stat, "estimate": point,
                            "ci_lo": lo, "ci_hi": hi, "resampled": "cases" if data is c else "items"})
    lo, hi = PI.bootstrap(event_slope, ev)
    ci_rows.append({"spec": label, "group": "event (Steam)", "statistic": "slope on max(0,gap)",
                    "estimate": event_slope(ev), "ci_lo": lo, "ci_hi": hi, "resampled": "items"})
cis = pd.DataFrame(ci_rows)
cis.to_csv(RES / "robustness_bootstrap.csv", index=False, float_format="%.3f")
print("\nBootstrap 95% CIs (2,000 draws):")
print(cis.round(3).to_string(index=False))

# ---- 3. Leave one case out (preferred) -------------------------------------------------------
xs_pref, _ = run(**PREF)
cases = sorted({c for cs in xs_pref.cases for c in cs.split("; ")})
loo = pd.DataFrame([{"dropped_case": c, **headline(*run(**PREF, drop_case=c))} for c in cases])
loo.to_csv(RES / "robustness_leave_one_case_out.csv", index=False, float_format="%.3f")
cols = ["nonST_cheapest_median", "nonST_cheapest_slope", "ST_share_below_0.85", "event_slope"]
rng = pd.DataFrame({"min": loo[cols].min(), "max": loo[cols].max(),
                    "case at min": [loo.loc[loo[c].idxmin(), "dropped_case"] for c in cols],
                    "case at max": [loo.loc[loo[c].idxmax(), "dropped_case"] for c in cols]})
rng.to_csv(RES / "robustness_leave_one_case_out_range.csv", float_format="%.3f")
print(f"\nLeave-one-case-out ranges over {len(cases)} cases (preferred spec):")
print(rng.round(3).to_string())

# ---- Figure: how much each change moves the headline numbers -------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
y = np.arange(len(rob))[::-1]
for ax, col, title, ref in [
        (axes[0], "nonST_cheapest_median", "Cheapest non-ST Covert per case,\nmedian price / P*", 1.0),
        (axes[1], "ST_share_below_0.85", "StatTrak Coverts below 0.85 P*\n(share of items)", 0.0),
        (axes[2], "event_slope", "Event slope on max(0, gap)\n(Steam, +/- 1 se)", 1.0)]:
    colors = [ORANGE if s.startswith(("scripts/06", "06 +")) else BLUE for s in rob.spec]
    xerr = rob["event_slope_se"] if col == "event_slope" else None
    ax.errorbar(rob[col], y, xerr=xerr, fmt="none", ecolor=GREY, lw=1)
    ax.scatter(rob[col], y, c=colors, s=36, zorder=3, edgecolor="white", lw=1)
    ax.axvline(ref, color=GREY, lw=0.8, ls="--")
    ax.set_title(title, loc="left")
axes[0].set_yticks(y, rob.spec)
fig.suptitle("Robustness (orange: scripts/06 pricing; blue: phase-fixed pricing)", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(FIG / "robustness.png", dpi=150)
