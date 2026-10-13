"""Trade-up parity: compute P* for every eligible Covert and test whether it acts as a floor.

Test 1 (cross-section, Skinport now): if arbitrage binds, no liquid Covert trades much
  below P*, and the cheapest Covert of each case sits close to it. Placebo: Classified
  skins of the same cases, priced against the same pools, which they cannot trade into.
Test 2 (event, hand-collected Steam history, pilot sample): "lift to the floor". An item's
  post-event change should be about max(0, gap), where gap = log(P*_post / P_pre). The
  rival explanation is that cheap items simply rose most (pre-event price level). Placebo:
  the pre-event window [-90,-32] should show no relation to the gap.

Inputs : data/raw/bymykel/skins.json, the pinned Skinport snapshot (src/parity_inputs.py),
         data/processed/panel_steam.parquet, results/sample.csv
Outputs: results/parity/*.csv, figures/parity/*.png
"""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analysis as A  # noqa: E402
import parity as P  # noqa: E402
import parity_inputs as PI  # noqa: E402
from catalog import split_market_hash_name  # noqa: E402
from http_cache import PROJECT_ROOT  # noqa: E402

RES, FIG = PROJECT_ROOT / "results" / "parity", PROJECT_ROOT / "figures" / "parity"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
MIN_VOL_30D = 5                 # liquidity screen for the cross-section
FEES = [0.0, 0.08, 0.15]        # none / Skinport-like / Steam-like seller fee on the output
CENSORED = {"★ Butterfly Knife | Tiger Tooth (Factory New)",   # hit Steam's price cap
            "★ Sport Gloves | Vice (Field-Tested)"}            # pre-event (see README)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6})
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"

schema_df = P.schema_frame(json.loads((PROJECT_ROOT / "data/raw/bymykel/skins.json").read_text()))
pools = P.gold_pools(schema_df)
snap = PI.SNAPSHOT        # pinned 13 Oct 2026 snapshot, shared with scripts 07-11
sales = pd.read_parquet(snap)
items = pd.read_parquet(str(snap).replace("_sales", "_items"))
prices = P.price_table(sales, items)
vol30 = sales.groupby("market_hash_name")["last_30_days.volume"].sum()
print(f"Skinport snapshot {snap.parent.name}/{snap.name}: {len(prices)} priced names, {len(pools)} case pools")

# ---- Test 1: cross-section --------------------------------------------------------
rarity = schema_df.set_index("skin")["rarity"]
has_pool = schema_df.set_index("skin")["cases"].map(lambda cs: any(c in pools for c in cs))
rows = []
for name in prices.index:
    skin, wear, st, sv = split_market_hash_name(name)
    if sv or wear is None or skin.startswith("★") or skin not in rarity.index:
        continue
    if rarity[skin] not in ("Covert", "Classified") or not has_pool[skin]:
        continue
    if vol30.get(name, 0) < MIN_VOL_30D:
        continue
    r = P.parity_for(name, schema_df, pools, prices)
    if rarity[skin] == "Classified":
        # Placebo: price Classified skins against the same pool as if they were inputs.
        row = schema_df[schema_df.skin == skin].iloc[0]
        n = P.normalised_mid(wear, row.fmin, row.fmax)
        cases = [c for c in row.cases if c in pools]
        if n is None or not cases:
            continue
        pool = pd.concat([pools[c] for c in cases]).drop_duplicates("skin")
        ev, priced, size = P.expected_output(pool, n, st, prices)
        r.update(parity=ev / 5, pool_priced=priced, pool_size=size, cases="; ".join(cases))
    r.update(rarity=rarity[skin], price=prices[name], vol_30d=vol30.get(name, 0))
    rows.append(r)
xs = pd.DataFrame(rows).dropna(subset=["parity"])
xs = xs[xs.pool_priced >= 0.8 * xs.pool_size]          # need most of the pool priced
xs["log_ratio"] = np.log(xs.price / xs.parity)
xs.to_csv(RES / "cross_section.csv", index=False, float_format="%.4f")

cov = xs[xs.rarity == "Covert"]
cls = xs[xs.rarity == "Classified"]
summary = []
for label, d in [("Covert (eligible)", cov), ("Classified (placebo)", cls)]:
    for st in (False, True):
        e = d[d.stattrak == st]
        if e.empty:
            continue
        cheapest = e.loc[e.groupby("cases").log_ratio.idxmin()]   # cheapest relative to P*, per case
        summary.append({
            "group": label, "stattrak": st, "n_items": len(e), "n_cases": len(cheapest),
            "median_ratio": np.exp(e.log_ratio.median()),
            "share_below_0.85": (e.log_ratio < np.log(0.85)).mean(),
            "cheapest_per_case_median_ratio": np.exp(cheapest.log_ratio.median()),
            "cheapest_per_case_iqr": "{:.2f}-{:.2f}".format(*np.exp(cheapest.log_ratio.quantile([.25, .75]))),
            "share_cases_cheapest_in_0.85-1.25": cheapest.log_ratio.between(np.log(0.85), np.log(1.25)).mean(),
        })
summary = pd.DataFrame(summary)
summary.to_csv(RES / "cross_section_summary.csv", index=False, float_format="%.3f")
print("\nTest 1 - price / parity (P* with no fee):")
print(summary.round(3).to_string(index=False))

# Cheapest-per-case: does price track P* with slope 1 and zero offset (log-log)?
cheap = cov[~cov.stattrak].loc[cov[~cov.stattrak].groupby("cases").log_ratio.idxmin()]
fit = stats.linregress(np.log(cheap.parity), np.log(cheap.price))
print(f"\nCheapest non-StatTrak Covert per case vs P* (n={len(cheap)} cases): "
      f"slope={fit.slope:.2f} (se {fit.stderr:.2f}), intercept={fit.intercept:.2f}, r={fit.rvalue:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
bins = np.linspace(-3, 4, 57)
ax.hist(cov.log_ratio, bins=bins, color=BLUE, alpha=0.8, label=f"Covert inputs (n={len(cov)})")
ax.hist(cls.log_ratio, bins=bins, color=ORANGE, alpha=0.6, label=f"Classified placebo (n={len(cls)})")
for fee in FEES:
    ax.axvline(np.log(1 - fee), color=GREY, lw=0.8, ls="--" if fee else "-")
ax.set_xlabel("log(price / P*)   (lines: P* after 0%, 8%, 15% fee)")
ax.set_ylabel("liquid items")
ax.set_title("Price relative to trade-up parity, Skinport", loc="left")
ax.legend(frameon=False, fontsize=8)
ax = axes[1]
ax.scatter(cov.parity, cov.price, s=8, color=BLUE, alpha=0.25, label="all eligible Coverts")
ax.scatter(cheap.parity, cheap.price, s=26, color=BLUE, edgecolor="white", lw=0.8,
           label="cheapest vs P* in each case")
lim = np.array([cov.parity.min() * 0.7, cov.parity.max() * 1.4])
ax.plot(lim, lim, color=GREY, lw=1, label="price = P*")
ax.fill_between(lim, lim * 0.85, lim, color=GREY, alpha=0.12, label="0-15% fee band")
ax.set(xscale="log", yscale="log", xlabel="parity P*, USD", ylabel="Skinport price, USD")
ax.set_title("Eligible Coverts vs parity (non-StatTrak + StatTrak)", loc="left")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "cross_section.png", dpi=150)

# ---- Test 2: event response vs gap to the floor (Steam history) ------------------------
sample = pd.read_csv(PROJECT_ROOT / "results/sample.csv").set_index("market_hash_name")
steam = pd.read_parquet(PROJECT_ROOT / "data/processed/panel_steam.parquet")
steam = A.add_event_time(steam[~steam.market_hash_name.isin(CENSORED)], "price")
base = A.baseline_levels(steam)
changes = A.window_changes(steam, base)
now = steam[steam.day > steam.day.max() - pd.Timedelta(days=30)].groupby("market_hash_name").log_p.median()

# Knife+glove index (median item log change vs baseline) to carry today's P* back in time.
gold_items = sample.index[sample.segment.isin(["Knife", "Gloves"])].intersection(base.index)
idx_win = changes.loc[gold_items, "[31,90]"].median()
idx_now = (now.loc[gold_items] - base.loc[gold_items]).median()

ev_rows = []
for name in sample.index[sample.segment == "Covert (gold pool)"]:
    if name not in base.index or name not in prices.index:
        continue
    r = P.parity_for(name, schema_df, pools, prices)
    if np.isnan(r["parity"]):
        continue
    fx = prices[name] / np.exp(now[name])              # Skinport USD per Steam GBP, this item
    p_pre = np.exp(base[name]) * fx                   # pre-event price, Skinport-equivalent
    for fee in FEES:
        p_star = (1 - fee) * r["parity"] * np.exp(idx_win - idx_now)   # P* during days 31-90
        ev_rows.append({"market_hash_name": name, "fee": fee, "p_pre": p_pre, "p_star": p_star,
                        "gap": np.log(p_star / p_pre), "change": changes.loc[name, "[31,90]"],
                        "placebo_change": changes.loc[name, "pre [-90,-32]"],
                        "log_p_pre": np.log(p_pre)})
ev = pd.DataFrame(ev_rows)
ev.to_csv(RES / "event_gap.csv", index=False, float_format="%.4f")


def ols(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    f = stats.linregress(x[m], y[m])
    return {"slope": f.slope, "se": f.stderr, "intercept": f.intercept, "r2": f.rvalue ** 2, "n": int(m.sum())}


models = []
for fee in FEES:
    e = ev[ev.fee == fee]
    lift = np.maximum(e.gap.to_numpy(), 0)
    models += [{"fee": fee, "model": "change ~ max(0, gap)  [parity]", **ols(lift, e.change.to_numpy())},
               {"fee": fee, "model": "placebo: pre-window change ~ max(0, gap)",
                **ols(lift, e.placebo_change.to_numpy())}]
e0 = ev[ev.fee == 0]
models.append({"fee": np.nan, "model": "change ~ log pre-event price  [rival]",
               **ols(e0.log_p_pre.to_numpy(), e0.change.to_numpy())})
both = e0.dropna(subset=["change"])
X = np.column_stack([np.ones(len(both)), np.maximum(both.gap, 0), both.log_p_pre])
beta, *_ = np.linalg.lstsq(X, both.change.to_numpy(), rcond=None)
resid = both.change.to_numpy() - X @ beta
cov_b = resid @ resid / (len(both) - 3) * np.linalg.inv(X.T @ X)
models.append({"fee": 0.0, "model": "joint: change ~ max(0,gap) + log price",
               "slope": beta[1], "se": np.sqrt(cov_b[1, 1]), "intercept": beta[0],
               "r2": 1 - resid.var() / both.change.var(), "n": len(both),
               "log_price_coef": beta[2], "log_price_se": np.sqrt(cov_b[2, 2])})
models = pd.DataFrame(models)
models.to_csv(RES / "event_models.csv", index=False, float_format="%.3f")
print("\nTest 2 - log change, days 31-90 vs baseline (Steam), eligible Coverts in the sample:")
print(models.round(3).to_string(index=False))
print(f"(knife+glove index: days 31-90 {100*np.expm1(idx_win):.1f}%, last 30 days "
      f"{100*np.expm1(idx_now):.1f}% vs baseline)")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
x = np.linspace(min(e0.gap.min(), -1), e0.gap.max() + 0.2, 100)
ax.plot(x, np.maximum(x, 0), color=GREY, lw=1, label="prediction: max(0, gap)")
ax.scatter(e0.gap, e0.change, s=36, color=BLUE, edgecolor="white", lw=1, zorder=3,
           label="eligible Coverts, days 31-90")
ax.scatter(e0.gap, e0.placebo_change, s=22, color=ORANGE, alpha=0.7, zorder=2,
           label="same items, pre-event placebo window")
ax.axhline(0, color=GREY, lw=0.6)
ax.set_xlabel("gap = log(P* after event / price before event)")
ax.set_ylabel("log price change vs pre-event baseline")
ax.set_title("Lift to the trade-up floor (Steam, P* with no fee)", loc="left")
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig(FIG / "event_gap.png", dpi=150)
