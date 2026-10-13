"""Segment price indices, event study and co-movement around 22 Oct 2025 (Steam).

Usage  : 03_event_study.py [--source steam]     (Steam is the only source)
Inputs : data/processed/panel_steam.parquet (from 05_parse_steam_manual.py),
         results/sample.csv
Outputs: results/steam/*.csv and figures/steam/*.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analysis as A  # noqa: E402
from catalog import SEGMENTS  # noqa: E402
from http_cache import PROJECT_ROOT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["steam"], default="steam",
                help="price source (only the hand-collected Steam panel is supported)")
ap.parse_args()
PANEL, PRICE_COL, LABEL = "panel_steam.parquet", "price", "Steam Market, hand-collected"
RES, FIG = PROJECT_ROOT / "results" / "steam", PROJECT_ROOT / "figures" / "steam"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
CONTROL = "Restricted (case)"
# Fixed categorical order (validated reference palette, light mode).
COLORS = dict(zip(SEGMENTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6})

raw = pd.read_parquet(PROJECT_ROOT / "data/processed" / PANEL)
CURRENCY = raw["currency"].iloc[0] if "currency" in raw else "USD"
sample = pd.read_csv(PROJECT_ROOT / "results" / "sample.csv").set_index("market_hash_name")
panel = A.add_event_time(raw[raw.market_hash_name.isin(sample.index)], PRICE_COL)
segment = sample["segment"]

# ---- 1. Event-study windows -------------------------------------------------
base = A.baseline_levels(panel)
kept = segment.loc[base.index]
print("items with adequate baseline:", kept.value_counts().reindex(SEGMENTS).to_dict())
changes = A.window_changes(panel, base)
table = A.segment_table(changes, segment, control=CONTROL)
for stat in ["mean", "median"]:
    table[f"{stat}_pct"] = 100 * np.expm1(table[stat])
table.to_csv(RES / "event_windows.csv", index=False, float_format="%.4f")
changes.join(segment).to_csv(RES / "item_window_changes.csv", float_format="%.4f")

for stat in ["median", "mean"]:
    pivot = table.pivot(index="segment", columns="window", values=f"{stat}_pct")
    print(f"\n{stat.title()} item % change vs baseline (2025-09-22..10-21):")
    print(pivot.loc[SEGMENTS, [w[0] for w in A.WINDOWS]].round(1).to_string())

# ---- 2. Segment indices --------------------------------------------------------
idx = A.segment_index(panel, base, segment)[SEGMENTS]
idx.to_csv(RES / "segment_index_daily.csv", float_format="%.4f")
smooth = idx.rolling(7, min_periods=4, center=True).median()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.7, 1]})
ax = axes[0]
span = smooth.loc["2025-06-01":]
for s in SEGMENTS:
    ax.plot(span.index, 100 * np.expm1(span[s]), color=COLORS[s], lw=2, label=s)
for d, lab in [(A.EVENT_DAY, "Covert->knife trade-ups\n22 Oct 2025"),
               (A.SOUVENIR_TRADEUP_DAY, "Souvenir inputs\n22 May 2026")]:
    ax.axvline(d, color="#52514e", lw=1, ls="--")
    ax.text(d, ax.get_ylim()[1], " " + lab, fontsize=7.5, va="top", color="#52514e")
ax.set_ylabel("% vs. 30-day pre-event baseline (7-day rolling median)")
ax.set_title(f"Segment price indices ({LABEL})", loc="left")
ax.legend(fontsize=7.5, loc="lower left", frameon=False)

ax = axes[1]
zoom = idx.loc[A.EVENT_DAY - pd.Timedelta(days=21): A.EVENT_DAY + pd.Timedelta(days=30)]
for s in SEGMENTS:
    ax.plot((zoom.index - A.EVENT_DAY).days, 100 * np.expm1(zoom[s]), color=COLORS[s], lw=1.6)
ax.axvline(0, color="#52514e", lw=1, ls="--")
ax.set_xlabel("days relative to 23 Oct 2025 (day 0)")
ax.set_title("Zoom: daily index, -21..+30 days", loc="left")
fig.tight_layout()
fig.savefig(FIG / "segment_indices.png", dpi=150)

# ---- 3. Window chart with bootstrap CIs ----------------------------------------
wins = [w[0] for w in A.WINDOWS]
fig, ax = plt.subplots(figsize=(10, 4))
width = 0.13
for k, s in enumerate(SEGMENTS):
    t = table[table.segment == s].set_index("window").loc[wins]
    x = np.arange(len(wins)) + (k - 2.5) * width
    y = 100 * np.expm1(t["median"])
    lo = 100 * np.expm1(t["median_lo"])
    hi = 100 * np.expm1(t["median_hi"])
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", ms=5, color=COLORS[s], lw=1.4,
                capsize=0, label=f"{s} (n={int(t['n'].max())})")
ax.axhline(0, color="#52514e", lw=0.8)
ax.set_xticks(np.arange(len(wins)), [f"{w}" for w in wins])
ax.set_xlabel("event window (days relative to 23 Oct 2025)")
ax.set_ylabel("median item % change vs. baseline\n(95% item-bootstrap CI)")
ax.set_title("Event-study windows by segment", loc="left")
ax.legend(fontsize=7.5, ncol=3, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "event_windows.png", dpi=150)

# ---- 3b. Heterogeneity: response vs pre-event price level -----------------------
# Trade-up parity: five Coverts buy one draw from the collection's knife/glove pool,
# so Coverts priced far below (pool value / 5) should be bid up hardest.
het = pd.DataFrame({"log_base_price": base, "change": changes["[31,90]"],
                    "segment": segment.loc[base.index]}).dropna()
fig, ax = plt.subplots(figsize=(6.5, 4))
het_rows = []
for s in ["Covert (gold pool)", "Classified (case)", "Restricted (case)"]:
    h = het[het.segment == s]
    ax.scatter(np.exp(h.log_base_price), 100 * np.expm1(h.change), s=36, color=COLORS[s],
               edgecolor="white", linewidth=1, label=s, zorder=3)
    rho = h[["log_base_price", "change"]].corr(method="spearman").iloc[0, 1]
    het_rows.append({"segment": s, "n": len(h), "spearman_rho": rho})
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=100)
ax.axhline(0, color="#52514e", lw=0.8)
ax.set_xlabel(f"pre-event baseline price, {CURRENCY} (log scale)")
ax.set_ylabel("% change, days 31-90 vs baseline (symlog)")
ax.set_title("Covert response vs. pre-event price (days 31-90)", loc="left")
ax.legend(fontsize=7.5, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "covert_heterogeneity.png", dpi=150)
het_tab = pd.DataFrame(het_rows)
het_tab.to_csv(RES / "heterogeneity_spearman.csv", index=False, float_format="%.3f")
print("\nSpearman(baseline log price, change days 31-90):")
print(het_tab.round(3).to_string(index=False))

# ---- 4. How rare was the move? Weekly index changes vs. pre-event history -------
sub = idx.loc["2025-04-14":]
wk = sub.groupby(sub.index.to_period("W-SUN").start_time).median()  # Monday-start weeks
pre = wk.loc[: "2025-10-12"].diff().dropna()
# Event (Thu 23 Oct) falls in the week of Mon 20 Oct: compare the week after it with the
# week before it, i.e. a two-week change that straddles the event week.
jump = wk.loc[pd.Timestamp("2025-10-27")] - wk.loc[pd.Timestamp("2025-10-13")]
z = jump / (pre.std() * np.sqrt(2))
rare = pd.DataFrame({"two_week_log_change": jump, "pre_weekly_sd": pre.std(),
                     "z_score": z, "n_pre_weeks": pre.count()}).loc[SEGMENTS]
rare.to_csv(RES / "event_z_scores.csv", float_format="%.3f")
print("\nTwo-week index change around the event, in pre-event weekly-sd units:")
print(rare.round(3).to_string())

# ---- 5. Co-movement before vs after ---------------------------------------------
ret = A.weekly_returns(panel[panel.market_hash_name.isin(base.index)])
ret_pre = ret.loc["2025-04-21":"2025-10-13"]     # 26 weeks before the event week
ret_post = ret.loc["2025-11-03":"2026-04-27"]    # 26 weeks after, skipping event weeks
blk_pre = A.block_correlations(ret_pre, segment, SEGMENTS)
blk_post = A.block_correlations(ret_post, segment, SEGMENTS)
blk_pre.to_csv(RES / "corr_blocks_pre.csv", float_format="%.3f")
blk_post.to_csv(RES / "corr_blocks_post.csv", float_format="%.3f")
pairs = [("Knife", "Covert (gold pool)"), ("Gloves", "Covert (gold pool)"),
         ("Knife", "Covert (no gold pool)"), ("Covert (gold pool)", "Covert (gold pool)"),
         ("Knife", "Knife"), ("Knife", "Restricted (case)")]
rows = []
for a, b in pairs:
    lo, hi = A.moving_block_bootstrap_corr_diff(ret_pre, ret_post, segment, a, b)
    rows.append({"pair": f"{a} ~ {b}", "pre": blk_pre.loc[a, b], "post": blk_post.loc[a, b],
                 "change": blk_post.loc[a, b] - blk_pre.loc[a, b], "ci_lo": lo, "ci_hi": hi})
comove = pd.DataFrame(rows)
comove.to_csv(RES / "comovement_changes.csv", index=False, float_format="%.3f")
print("\nMean pairwise weekly-return correlation, pre vs post (block-bootstrap 95% CI of change):")
print(comove.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
short = ["Knife", "Gloves", "Cov. gold", "Cov. no-gold", "Classified", "Restricted"]
for ax, mat, title, cmap, lim in [
        (axes[0], blk_pre, "Pre (Apr-Oct 2025)", "Blues", (0, 0.6)),
        (axes[1], blk_post, "Post (Nov 2025-Apr 2026)", "Blues", (0, 0.6)),
        (axes[2], blk_post - blk_pre, "Change (post - pre)", "RdBu_r", (-0.4, 0.4))]:
    im = ax.imshow(mat.to_numpy(float), cmap=cmap, vmin=lim[0], vmax=lim[1])
    ax.set_xticks(range(6), short, rotation=45, ha="right")
    ax.set_yticks(range(6), short)
    ax.grid(False)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="#0b0b0b")
    ax.set_title(title, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Mean pairwise correlation of weekly log returns, by segment block", x=0.01,
             ha="left")
fig.tight_layout()
fig.savefig(FIG / "comovement_blocks.png", dpi=150)
