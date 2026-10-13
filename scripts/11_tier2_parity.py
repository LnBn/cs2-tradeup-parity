"""Tier 2: the parity tests on the expanded hand-collected Steam data.

1. Expanded event test (Covert -> knife). Every eligible Covert with a Steam history
   (pilot sample + tier-2 list), not just the 23 in the pilot.
   P* during days 31-90 is carried back from today's Skinport P* in two ways:
     global : one knife+glove index for every case (as in scripts/06 and 10)
     case   : a case-specific pool index, built from the knives/gloves of that case's
              pool that were hand-collected (results/collection_lists/tier2.csv "pool")
2. Cascade event test (10-for-1). Classified (-> Covert) and Restricted (-> Classified)
   items with Steam histories, P* carried back with the output tier's Steam index.
3. Daily P*(t) for the cheapest-relative-to-P* Covert of each case (case pool index), to
   ask whether deviations from the floor after the event are short-lived (AR(1) on the
   daily log deviation) and how often the price sits below 0.9 P*.

Preferred price specification from scripts/10: Skinport 30-day median sales, Doppler
phases priced from their own sales and weighted by 90-day sale counts.
Data: pinned Skinport snapshot, ByMykel schema, the hand-collected Steam panel.
No network access.
Outputs: results/parity_tier2/*.csv, figures/parity_tier2/*.png
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import parity as P  # noqa: E402
import parity_inputs as PI  # noqa: E402
from http_cache import PROJECT_ROOT  # noqa: E402

RES, FIG = PROJECT_ROOT / "results" / "parity_tier2", PROJECT_ROOT / "figures" / "parity_tier2"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6})
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"

# Excluded from every Steam calculation here:
EXCLUDE = {
    # sat at or near Steam's per-item price cap (>= GBP 1,250) for >= 5 pre-event days
    "★ Butterfly Knife | Tiger Tooth (Factory New)", "★ Karambit | Tiger Tooth (Factory New)",
    "★ M9 Bayonet", "★ Sport Gloves | Vice (Field-Tested)", "★ Talon Knife | Fade (Factory New)",
}
WIN = (31, 90)            # post-event window (days after 23 Oct 2025)

inp = PI.load()
steam = inp.steam[~inp.steam.market_hash_name.isin(EXCLUDE)]
base, changes, now = inp.base.drop(EXCLUDE, errors="ignore"), inp.changes, inp.now
prices = P.price_table_v2(inp.sales, inp.items, window="30", phase_weight="sales")
tier2 = pd.read_csv(PROJECT_ROOT / "results/collection_lists/tier2.csv")
collected = set(steam.market_hash_name)
rarity = inp.schema_df.set_index("skin")["rarity"]


def skin_of(name: str) -> str:
    return P.split_market_hash_name(name)[0]


def group_shift(names) -> float:
    """Median over items of (log level in days 31-90) - (log level over the last 30 days)."""
    names = pd.Index(names).intersection(base.index).intersection(now.index)
    win = changes.loc[names, "[31,90]"] + base.loc[names]
    return float((win - now.loc[names]).median()), len(names)


def ols_ci(ev: pd.DataFrame, n_boot: int = 2000, cluster: str | None = None):
    """Slope of change on max(0, gap) with a bootstrap 95% CI (items, or whole cases)."""
    def slope(d):
        x = np.maximum(d.gap.to_numpy(), 0)
        y = d.change.to_numpy()
        m = ~np.isnan(y)
        return np.polyfit(x[m], y[m], 1)[0] if m.sum() > 2 and x[m].std() > 0 else np.nan
    return slope(ev), *PI.bootstrap(slope, ev, n_boot=n_boot, cluster=cluster)


# ---- Pool items and case-specific pool indices ------------------------------------
gold = [n for n in collected if n.startswith("★")]
pool_items = {}                                     # case -> collected pool items
for case, pool in inp.pools.items():
    skins = set(pool.skin)
    pool_items[case] = [n for n in gold if skin_of(n) in skins]
case_shift = {c: group_shift(v) for c, v in pool_items.items()}
GLOBAL_SHIFT, n_gold = group_shift(gold)
cs = pd.DataFrame([{"case": c, "pool_items_collected": n, "case_shift": s}
                   for c, (s, n) in case_shift.items()]).sort_values("case")
cs.to_csv(RES / "case_pool_shifts.csv", index=False, float_format="%.4f")
print(f"global knife/glove shift (days 31-90 vs now): {GLOBAL_SHIFT:+.3f} from {n_gold} items")
print("collected pool items per case:", cs.pool_items_collected.describe()[["min", "50%", "max"]].to_dict())

# ---- 1. Expanded Covert event test ----------------------------------------------------
coverts = sorted(n for n in collected
                 if not n.startswith("★") and rarity.get(skin_of(n)) == "Covert"
                 and not P.split_market_hash_name(n)[2])
p_now = {}
case_of = {}
for n in coverts:
    r = P.parity_for(n, inp.schema_df, inp.pools, prices)
    if np.isfinite(r["parity"]) and r["pool_priced"] >= 0.8 * r["pool_size"]:
        p_now[n] = r["parity"]
        case_of[n] = r["cases"].split("; ")[0]
eligible = list(p_now)

ev_global = P.event_gap(eligible, p_now, prices, base, now, changes, GLOBAL_SHIFT)
rows = []
for n in eligible:
    s, k = case_shift.get(case_of[n], (np.nan, 0))
    if k >= 2:                                      # need >= 2 collected pool items
        rows.append(P.event_gap([n], p_now, prices, base, now, changes, s))
ev_case = pd.concat(rows, ignore_index=True)
for ev in (ev_global, ev_case):
    ev["case"] = ev.market_hash_name.map(case_of)

summary = []
for label, ev in [("global index", ev_global), ("case-specific index", ev_case)]:
    m = P.gap_models(ev).assign(spec=label)
    s, lo, hi = ols_ci(ev)
    s_c, lo_c, hi_c = ols_ci(ev, cluster="case")
    m.loc[m.model.str.contains(r"\[parity\]"), ["boot_lo", "boot_hi", "case_boot_lo", "case_boot_hi"]] = [lo, hi, lo_c, hi_c]
    below = ev[ev.gap > 0]
    post_ratio = np.exp(below.change - below.gap)
    m["n_below_floor_pre"] = len(below)
    m["post_ratio_median_below"] = post_ratio.median()
    m["post_ratio_iqr_below"] = "{:.2f}-{:.2f}".format(*post_ratio.quantile([.25, .75]))
    summary.append(m)
summary = pd.concat(summary)
summary.to_csv(RES / "covert_event_models.csv", index=False, float_format="%.3f")
ev_global.to_csv(RES / "covert_event_gap_global.csv", index=False, float_format="%.4f")
ev_case.to_csv(RES / "covert_event_gap_case.csv", index=False, float_format="%.4f")
cols = ["spec", "model", "slope", "se", "boot_lo", "boot_hi", "case_boot_lo", "case_boot_hi", "r2", "n"]
print("\n1. Covert -> knife event test, days 31-90 (Steam):")
print(summary[cols].round(3).to_string(index=False))
print(summary.groupby("spec")[["n_below_floor_pre", "post_ratio_median_below", "post_ratio_iqr_below"]].first())

# ---- 2. Cascade event test ------------------------------------------------------------
casc = []
for tier, out_tier in [("Classified", "Covert"), ("Restricted", "Classified")]:
    names = sorted(n for n in collected if rarity.get(skin_of(n)) == tier
                   and not P.split_market_hash_name(n)[2])
    out_names = [n for n in collected if rarity.get(skin_of(n)) == out_tier and not n.startswith("★")]
    shift, k = group_shift(out_names)
    pn = {n: P.tier_parity(n, inp.coll_df, prices)["parity"] for n in names}
    ev = P.event_gap(names, pn, prices, base, now, changes, shift)
    m = P.gap_models(ev).assign(tier=f"{tier} -> {out_tier}", output_index_items=k,
                                output_shift=shift)
    s, lo, hi = ols_ci(ev)
    m.loc[m.model.str.contains(r"\[parity\]"), ["boot_lo", "boot_hi"]] = [lo, hi]
    m["n_below_floor_pre"] = int((ev.gap > 0).sum())
    casc.append(m)
    ev.assign(tier=tier).to_csv(RES / f"cascade_event_gap_{tier.lower()}.csv", index=False, float_format="%.4f")
casc = pd.concat(casc)
casc.to_csv(RES / "cascade_event_models.csv", index=False, float_format="%.3f")
print("\n2. Cascade event test (Steam, days 31-90):")
print(casc[["tier", "model", "slope", "se", "boot_lo", "boot_hi", "r2", "n", "n_below_floor_pre"]]
      .round(3).to_string(index=False))

# ---- 3. Daily P*(t): how tightly does the floor hold after the event? ----------------------
def daily_level(names):
    """Daily median over items of (log price - item's last-30-day level)."""
    p = steam[steam.market_hash_name.isin(pd.Index(names).intersection(now.index))]
    return (p.log_p - p.market_hash_name.map(now)).groupby(p.day).median()


dyn = []
paths = {}
for case in sorted(set(case_of.values())):
    if case_shift.get(case, (np.nan, 0))[1] < 2:
        continue
    pool_idx = daily_level(pool_items[case]).rolling(7, min_periods=4, center=True).median()
    members = [n for n in eligible if case_of[n] == case and n in now.index and n in prices.index]
    if not members:
        continue
    # the member that sits closest to its floor today
    ratio_now = {n: prices[n] / p_now[n] for n in members}
    n = min(ratio_now, key=ratio_now.get)
    item = steam[steam.market_hash_name == n].set_index("day").log_p
    fx = np.log(prices[n]) - now[n]                       # Steam log units -> Skinport USD
    dev = (item + fx - (np.log(p_now[n]) + pool_idx.reindex(item.index))).dropna()
    post = dev.loc["2025-11-01":]
    if len(post) < 60:
        continue
    pre = dev.loc[:"2025-10-21"]
    x, y = post.iloc[:-1].to_numpy(), post.iloc[1:].to_numpy()
    phi = np.polyfit(x - x.mean(), y - y.mean(), 1)[0]
    dyn.append({"case": case, "item": n, "ratio_now": ratio_now[n], "days_post": len(post),
                "median_dev_post": np.exp(post.median()), "share_days_below_0.9": (post < np.log(0.9)).mean(),
                "ar1_phi": phi, "half_life_days": np.log(0.5) / np.log(phi) if 0 < phi < 1 else np.nan,
                "median_dev_pre": np.exp(pre.loc["2025-06-01":].median()) if len(pre) else np.nan,
                "share_days_below_0.9_pre": (pre.loc["2025-06-01":] < np.log(0.9)).mean() if len(pre) else np.nan})
    paths[n] = dev
dyn = pd.DataFrame(dyn).sort_values("ratio_now")
dyn.to_csv(RES / "floor_dynamics.csv", index=False, float_format="%.3f")
print("\n3. Cheapest Covert per case vs daily P*(t), Nov 2025 onward:")
print(f"cases: {len(dyn)}; median price/P*: {dyn.median_dev_post.median():.2f}; "
      f"median share of days below 0.9 P*: {dyn['share_days_below_0.9'].median():.3f}; "
      f"median AR(1) phi: {dyn.ar1_phi.median():.2f} (half-life {dyn.half_life_days.median():.1f} days)")
print(f"same items, Jun-Oct 2025 (before the trade-up existed), vs the counterfactual P*: "
      f"median price/P* {dyn.median_dev_pre.median():.2f}; "
      f"share of days below 0.9 P*: {dyn['share_days_below_0.9_pre'].median():.2f}")

# ---- Figures --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
ev = ev_case
x = np.linspace(min(ev.gap.min(), -1), ev.gap.max() + 0.2, 100)
ax.plot(x, np.maximum(x, 0), color=GREY, lw=1, label="prediction: max(0, gap)")
ax.scatter(ev.gap, ev.change, s=30, color=BLUE, edgecolor="white", lw=0.8, zorder=3,
           label=f"eligible Coverts, days 31-90 (n={ev.change.notna().sum()})")
ax.scatter(ev.gap, ev.placebo_change, s=16, color=ORANGE, alpha=0.6, zorder=2,
           label="same items, pre-event placebo window")
ax.axhline(0, color=GREY, lw=0.6)
ax.set_xlabel("gap = log(P* after event / price before event)  [case-specific pool index]")
ax.set_ylabel("log price change vs pre-event baseline")
ax.set_title("Lift to the trade-up floor, all eligible Coverts (Steam)", loc="left")
ax.legend(frameon=False, fontsize=7.5)
ax = axes[1]
for n, dev in paths.items():
    d = dev.loc["2025-06-01":].rolling(7, min_periods=4, center=True).median()
    ax.plot(d.index, np.exp(d), lw=0.8, alpha=0.5)
ax.axhline(1, color=GREY, lw=1)
ax.axvline(pd.Timestamp("2025-10-23"), color=GREY, lw=1, ls="--")
ax.set_yscale("log")
ax.set_ylabel("price / P*(t)  (7-day median)")
ax.set_title(f"Cheapest Covert per case vs its daily floor (n={len(paths)} cases)", loc="left")
fig.tight_layout()
fig.savefig(FIG / "tier2_event_and_dynamics.png", dpi=150)
