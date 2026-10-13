"""Parity cascade: does the knife floor propagate down the standard 10-for-1 trade-ups?

A standard contract takes 10 inputs of one rarity and returns one item of the next
rarity from the inputs' collections, uniform over that collection's next-tier skins
(single-collection contract), with the normalised-float rule and StatTrak in = StatTrak
out. So for an input of tier k:
    P*_k = (1 - fee) * E[value of a tier-(k+1) output] / 10,
with outputs priced at today's Skinport market prices (Covert prices already embed the
knife floor). Classified inputs are priced against Coverts, Restricted inputs against
Classifieds.

Test 1 (cross-section, Skinport 13 Oct 2026): price/P* by tier, collection type and
  StatTrak. Placebo: inputs one tier lower priced against the same outputs (a tier
  they cannot reach), which should sit far below "parity".
Test 2 (event, hand-collected Steam): the Classified/Restricted items in the pilot
  sample, change in days 31-90 vs max(0, gap), with P* carried back using the output
  tier's index. n is small (about 20).

Outputs: results/parity/cascade_*.csv, figures/parity/cascade_*.png
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
from catalog import split_market_hash_name  # noqa: E402
from http_cache import PROJECT_ROOT  # noqa: E402

RES, FIG = PROJECT_ROOT / "results" / "parity", PROJECT_ROOT / "figures" / "parity"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
MIN_VOL_30D = 5
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"

inp = PI.load()
prices = P.price_table_v2(inp.sales, inp.items, window="30", phase_weight="sales")
ctype = P.collection_types(inp.coll_df, inp.pools)
rarity = inp.coll_df.set_index("skin")["rarity"]

# ---- Test 1: cross-section ---------------------------------------------------------------
TESTS = [  # (label, input tier, output tier, is_placebo)
    ("Classified -> Covert", "Classified", "Covert", False),
    ("Restricted -> Classified", "Restricted", "Classified", False),
    ("placebo: Restricted priced vs Covert", "Restricted", "Covert", True),
    ("placebo: Mil-Spec priced vs Classified", "Mil-Spec Grade", "Classified", True),
]
rows = []
for name in prices.index:
    skin, wear, st, sv = split_market_hash_name(name)
    if sv or wear is None or skin.startswith("★") or inp.vol30.get(name, 0) < MIN_VOL_30D:
        continue
    for label, tier_in, tier_out, placebo in TESTS:
        if rarity.get(skin) != tier_in:
            continue
        r = P.tier_parity(name, inp.coll_df, prices, target_rarity=tier_out)
        if not np.isfinite(r["parity"]) or r["pool_priced"] < 0.8 * r["pool_size"]:
            continue
        r.update(test=label, placebo=placebo, price=prices[name], vol_30d=inp.vol30[name],
                 collection_type=ctype.get(r["collection"], "no gold pool"))
        rows.append(r)
xs = pd.DataFrame(rows)
xs["log_ratio"] = np.log(xs.price / xs.parity)
xs.to_csv(RES / "cascade_cross_section.csv", index=False, float_format="%.4f")

summary = []
for (label, ctyp, st), d in xs.groupby(["test", "collection_type", "stattrak"]):
    s = P.ratio_summary(d, "collection")
    slope, se, n = P.cheapest_slope(d, "collection")
    lo, hi = PI.bootstrap(lambda x: np.exp(x.log_ratio.median()), d)
    summary.append({"test": label, "collection_type": ctyp, "stattrak": st, **s,
                    "median_ratio_ci": f"{lo:.2f}-{hi:.2f}", "cheapest_slope": slope,
                    "cheapest_slope_se": se})
summary = pd.DataFrame(summary).sort_values(["test", "collection_type", "stattrak"])
summary.to_csv(RES / "cascade_summary.csv", index=False, float_format="%.3f")
pd.set_option("display.width", 250)
print("Test 1 - cascade cross-section (Skinport, P* with no fee, phase-fixed prices):")
print(summary.drop(columns=["cheapest_iqr"]).round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
bins = np.linspace(-6, 3, 61)
for ax, (real, plac, title) in zip(axes, [
        ("Classified -> Covert", "placebo: Restricted priced vs Covert", "Classified inputs"),
        ("Restricted -> Classified", "placebo: Mil-Spec priced vs Classified", "Restricted inputs")]):
    for lab, color, alpha, sel in [
            (f"{real}, case collections", BLUE, 0.85, (xs.test == real) & (xs.collection_type == "case")),
            (f"{real}, other collections", AQUA, 0.7, (xs.test == real) & (xs.collection_type != "case")),
            (plac, ORANGE, 0.55, xs.test == plac)]:
        d = xs[sel]
        ax.hist(d.log_ratio.clip(-6, 3), bins=bins, color=color, alpha=alpha, label=f"{lab} (n={len(d)})")
    ax.axvline(0, color=GREY, lw=1)
    ax.axvline(np.log(0.85), color=GREY, lw=0.8, ls="--")
    ax.set_xlabel("log(price / P*)   (solid: P*, dashed: 0.85 P*)")
    ax.set_ylabel("liquid items")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "cascade_cross_section.png", dpi=150)

# ---- Test 2: event, Classified/Restricted in the pilot sample (Steam) --------------------
EVENT_SETS = [("Classified (case)", ["Covert (gold pool)"]),
              ("Restricted (case)", ["Classified (case)"])]
ev_all, models = [], []
for seg, out_segs in EVENT_SETS:
    names = inp.sample.index[inp.sample.segment == seg]
    p_now = {n: P.tier_parity(n, inp.coll_df, prices)["parity"] for n in names}
    shift = PI.index_shift(inp, out_segs)
    ev = P.event_gap(names, p_now, prices, inp.base, inp.now, inp.changes, shift)
    ev["segment"] = seg
    ev_all.append(ev)
    m = P.gap_models(ev)
    m.insert(0, "segment", seg)
    m["output_index_shift"] = shift
    models.append(m)
ev_all = pd.concat(ev_all, ignore_index=True)
pooled = P.gap_models(ev_all)
pooled.insert(0, "segment", "pooled")
models = pd.concat(models + [pooled], ignore_index=True)
ev_all.to_csv(RES / "cascade_event_gap.csv", index=False, float_format="%.4f")
models.to_csv(RES / "cascade_event_models.csv", index=False, float_format="%.3f")
print("\nTest 2 - cascade event (Steam, days 31-90), P* carried back with the output tier's index:")
print(models.round(3).to_string(index=False))
print("\nItems:")
print(ev_all.assign(ratio_pre=np.exp(-ev_all.gap))[["segment", "market_hash_name", "gap", "change",
      "placebo_change", "ratio_pre"]].round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(6.5, 4.2))
x = np.linspace(min(ev_all.gap.min(), -1), max(ev_all.gap.max(), 0.5) + 0.2, 100)
ax.plot(x, np.maximum(x, 0), color=GREY, lw=1, label="prediction: max(0, gap)")
for seg, color in [("Classified (case)", BLUE), ("Restricted (case)", AQUA)]:
    e = ev_all[ev_all.segment == seg]
    ax.scatter(e.gap, e.change, s=36, color=color, edgecolor="white", lw=1, zorder=3,
               label=f"{seg}, days 31-90 (n={e.change.notna().sum()})")
ax.axhline(0, color=GREY, lw=0.6)
ax.set_xlabel("gap = log(P* after event / price before event)")
ax.set_ylabel("log price change vs pre-event baseline")
ax.set_title("Cascade: lift to the 10-for-1 floor (Steam)", loc="left")
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig(FIG / "cascade_event_gap.png", dpi=150)
