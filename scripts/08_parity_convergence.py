"""How fast did below-floor Coverts converge to the new trade-up floor? (Steam, daily)

For each eligible Covert in the pilot sample whose pre-event price was below its
post-event parity P* (gap > 0):
  * P*_i(t), in Steam currency, is today's Skinport P* moved with the daily knife+glove
    index (Steam, centred 5-day median, smoothed separately on each side of the event) and converted with the item's own Skinport/Steam
    price ratio over the last 30 days (as in scripts/06).
  * deficit d_i(t) = log P*_i(t) - log p_i(t). The initial gap d0_i is the Test-2 gap:
    P* during days 31-90 relative to the pre-event baseline price (items with d0 > 0 are
    "below the floor").
  * days to 0.9 P*: the first day t >= 0 that begins three consecutive trading days with
    p >= 0.9 P*.
  * half-life (items at least 20% below P* only): fit d_i(t) = d_inf + (d0_i - d_inf) exp(-lambda t) on days 0-60; half-life =
    ln 2 / lambda. Pooled: median fraction of the gap closed, 1 - d_i(t)/d0_i, fitted with
    c_inf (1 - exp(-lambda t)), with an item-bootstrap CI.
Daily data cannot resolve anything faster than about one day; such cases are reported
as "< 1 day".

Outputs: results/parity/convergence_*.csv, figures/parity/convergence.png
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import parity as P  # noqa: E402
import parity_inputs as PI  # noqa: E402
from http_cache import PROJECT_ROOT  # noqa: E402

RES, FIG = PROJECT_ROOT / "results" / "parity", PROJECT_ROOT / "figures" / "parity"
FIT_DAYS = 60
LAMBDA_MAX = 5.0          # upper bound on the decay rate (half-life >= 0.14 day)
MATERIAL_GAP = np.log(1 / 0.8)   # half-lives only for items >= 20% below P*: smaller gaps
                                 # are within day-to-day noise and give meaningless fits
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6})
BLUE, GREY = "#2a78d6", "#52514e"

inp = PI.load()
prices = P.price_table_v2(inp.sales, inp.items, window="30", phase_weight="sales")

# Daily knife+glove index relative to its level over the last 30 days. Smoothed with a
# centred 5-day median, separately before and after the event so the crash is not blurred.
raw = PI.daily_index(inp, ["Knife", "Gloves"])
split = raw.index >= PI.A.EVENT_DAY
gold = pd.concat([raw[~split].rolling(5, center=True, min_periods=2).median(),
                  raw[split].rolling(5, center=True, min_periods=2).median()])
gold_now = gold[gold.index > gold.index.max() - pd.Timedelta(days=30)].median()
gold_shift = (gold - gold_now).rename("shift")

shift_31_90 = PI.index_shift(inp, ["Knife", "Gloves"])
covs = inp.sample.index[inp.sample.segment == "Covert (gold pool)"]
covs = [n for n in covs if n in inp.base.index and n in inp.now.index and n in prices.index]
rows, paths = [], []
for name in covs:
    p_star_now = P.parity_for(name, inp.schema_df, inp.pools, prices)["parity"]
    if not np.isfinite(p_star_now):
        continue
    fx = prices[name] / np.exp(inp.now[name])
    s = inp.steam[(inp.steam.market_hash_name == name) & inp.steam.t.between(-30, 120)]
    s = s.set_index("day").join(gold_shift, how="left")
    s["log_pstar"] = np.log(p_star_now / fx) + s["shift"]
    s["d"] = s.log_pstar - s.log_p
    # Initial gap as in Test 2: P* during days 31-90 (Skinport P* carried back with the
    # knife+glove index) relative to the pre-event baseline price.
    d0 = np.log(p_star_now / fx) + shift_31_90 - inp.base[name]
    if not np.isfinite(d0) or d0 <= 0:
        continue                                  # was not below the new floor
    post = s[s.t >= 0]
    above = (post.log_p >= np.log(0.9) + post.log_pstar).to_numpy()
    run = np.convolve(above.astype(int), np.ones(3, int), "valid") == 3
    t_hit = int(post.t.iloc[np.argmax(run)]) if run.any() else np.nan

    fit_win = post[post.t <= FIT_DAYS]
    try:
        (d_inf, lam), _ = curve_fit(lambda t, a, l: a + (d0 - a) * np.exp(-l * t),
                                    fit_win.t, fit_win.d, p0=[0.0, 0.3],
                                    bounds=([-2.0, 1e-3], [2.0, LAMBDA_MAX]))
    except RuntimeError:
        d_inf, lam = np.nan, np.nan
    if d0 < MATERIAL_GAP:
        d_inf, lam = np.nan, np.nan
    rows.append({"market_hash_name": name, "pre_ratio_to_pstar": np.exp(-d0),
                 "days_to_0.9_pstar": t_hit, "lambda": lam,
                 "half_life_days": np.log(2) / lam, "at_bound": lam >= LAMBDA_MAX * 0.999,
                 "d_inf": d_inf, "settled_ratio": np.exp(-d_inf)})
    paths.append(post.assign(name=name, closed=1 - post.d / d0, material=d0 >= MATERIAL_GAP)
                 [["t", "name", "d", "closed", "material"]])

conv = pd.DataFrame(rows).sort_values("pre_ratio_to_pstar")
conv.to_csv(RES / "convergence_items.csv", index=False, float_format="%.3f")
paths = pd.concat(paths)
pd.set_option("display.width", 220)
print(f"Below-floor eligible Coverts: {len(conv)} of {len(covs)}")
print(conv.round(3).to_string(index=False))


def pooled_fit(pth: pd.DataFrame) -> tuple[float, float]:
    med = pth[pth.t <= FIT_DAYS].groupby("t").closed.median()
    (c_inf, lam), _ = curve_fit(lambda t, c, l: c * (1 - np.exp(-l * t)), med.index, med.values,
                                p0=[1.0, 0.3], bounds=([0, 1e-3], [3, LAMBDA_MAX]))
    return c_inf, lam


mat = paths[paths.material]
c_inf, lam = pooled_fit(mat)
rng = np.random.default_rng(0)
names = mat.name.unique()
boot = []
for _ in range(500):
    pick = rng.choice(names, len(names))
    boot.append(np.log(2) / pooled_fit(pd.concat([mat[mat.name == n].assign(name=f"{n}#{j}")
                                                  for j, n in enumerate(pick)]))[1])
lo, hi = np.percentile(boot, [2.5, 97.5])
pooled = pd.DataFrame([{
    "n_items": len(conv), "median_days_to_0.9_pstar": conv["days_to_0.9_pstar"].median(),
    "share_within_3_days": (conv["days_to_0.9_pstar"] <= 3).mean(),
    "share_never_in_120_days": conv["days_to_0.9_pstar"].isna().mean(),
    "n_material_gap": int(len(names)),
    "median_item_half_life_material": conv.half_life_days.median(),
    "pooled_half_life": np.log(2) / lam, "pooled_half_life_ci": f"{lo:.2f}-{hi:.2f}",
    "pooled_share_closed_asymptote": c_inf}])
pooled.to_csv(RES / "convergence_summary.csv", index=False, float_format="%.3f")
print("\nPooled:")
print(pooled.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
for n, g in paths.groupby("name"):
    ax.plot(g.t, -g.d, color=BLUE, alpha=0.3, lw=1)
med = paths.groupby("t").d.median()
ax.plot(med.index, -med.values, color=BLUE, lw=2.2, label="median item")
ax.axhline(0, color=GREY, lw=1)
ax.axhline(np.log(0.9), color=GREY, lw=0.8, ls="--")
ax.set_xlim(-2, 120)
ax.set_xlabel("days since 23 Oct 2025")
ax.set_ylabel("log(price / P*)   (dashed: 0.9 P*)")
ax.set_title(f"Below-floor Coverts converging to P* (n={len(conv)})", loc="left")
ax.legend(frameon=False, fontsize=8)
ax = axes[1]
mc = mat[mat.t <= 90].groupby("t").closed.median()
ax.plot(mc.index, mc.values, "o", ms=3, color=BLUE,
        label=f"median share of gap closed (items >= 20% below P*, n={len(names)})")
tt = np.linspace(0, 90, 200)
ax.plot(tt, c_inf * (1 - np.exp(-lam * tt)), color=GREY, lw=1.2,
        label=f"fit: half-life {np.log(2)/lam:.1f} d (95% CI {lo:.1f}-{hi:.1f})")
ax.axhline(1, color=GREY, lw=0.6)
ax.set_xlabel("days since 23 Oct 2025")
ax.set_ylabel("share of initial gap closed")
ax.set_title("Pooled convergence", loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "convergence.png", dpi=150)
