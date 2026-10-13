"""Why do about 22% of liquid StatTrak Coverts trade below 0.85 P* in scripts/06?

Candidate explanations, checked in turn:
  1. Where the violations sit (by case).
  2. Pricing of Doppler-style knives: scripts/06 prices multi-phase names from the
     *listing-quantity-weighted median ask*. In thin StatTrak markets a few unsold gem
     listings (Ruby, Sapphire, ...) can dominate that average.
  3. Thin or ask-only StatTrak knife prices (Skinport 30/90-day sale volumes of the pool).
  4. The StatTrak rule itself: StatTrak Coverts give StatTrak knives only (no gloves), so
     glove-only cases have no StatTrak route (see README sources).
Outputs: results/parity/stattrak_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import parity as P  # noqa: E402
import parity_inputs as PI  # noqa: E402
from http_cache import PROJECT_ROOT  # noqa: E402

RES = PROJECT_ROOT / "results" / "parity"
inp = PI.load()
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 60)

price_sets = {
    "v06: phases by listing-qty-weighted ask": P.price_table(inp.sales, inp.items),
    "phases by 90-day sale counts (preferred)": P.price_table_v2(inp.sales, inp.items, "30", "sales"),
    "phases 1-4 equal, gems ignored": P.price_table_v2(inp.sales, inp.items, "30", "core"),
}

# ---- 1-2. Where are the violations, and what drives P* there? --------------------------
v06 = P.covert_cross_section(price_sets["v06: phases by listing-qty-weighted ask"],
                             inp.vol30, inp.schema_df, inp.pools)
st = v06[v06.stattrak].copy()
st["below"] = st.log_ratio < np.log(0.85)
by_case = (st.groupby("cases").agg(n=("below", "size"), n_below=("below", "sum"),
                                   median_ratio=("log_ratio", lambda x: np.exp(x.median())))
           .sort_values("median_ratio"))
by_case.to_csv(RES / "stattrak_by_case.csv", float_format="%.3f")
print("StatTrak Coverts below 0.85 P* (v06 prices), by case:")
print(by_case[by_case.n_below > 0].round(2).to_string())

# EV contribution of each pool item for a typical violating input.
probe = "StatTrak™ M4A1-S | Hyper Beast (Field-Tested)"
skin_row = inp.schema_df[inp.schema_df.skin == "M4A1-S | Hyper Beast"].iloc[0]
n = P.normalised_mid("Field-Tested", skin_row.fmin, skin_row.fmax)
pool = inp.pools["Chroma 2 Case"]
pool = pool[~pool.is_glove]
s30 = inp.sales.groupby("market_hash_name")["last_30_days.volume"].sum()
s90 = inp.sales.groupby("market_hash_name")["last_90_days.volume"].sum()
rows = []
for r in pool.itertuples():
    name = P.output_name(r.skin, P.wear_of(r.fmin + n * (r.fmax - r.fmin)), True)
    rows.append({"output": name, **{k: v.get(name, np.nan) for k, v in price_sets.items()},
                 "sales_30d": s30.get(name, 0), "sales_90d": s90.get(name, 0)})
decomp = pd.DataFrame(rows)
decomp.to_csv(RES / "stattrak_pool_decomposition_chroma2.csv", index=False, float_format="%.2f")
cols = list(price_sets)
print(f"\nPool of {probe} (Chroma 2 Case, StatTrak knives): P* = mean / 5")
print(decomp.sort_values(cols[0], ascending=False).head(8).round(0).to_string(index=False))
print("P* under each pricing:", {k: round(decomp[k].mean() / 5, 1) for k in cols},
      f"| item price {price_sets[cols[1]].get(probe):.2f}")

# Phase rows behind the worst offender.
worst = "★ StatTrak™ Karambit | Doppler (Factory New)"
ph = inp.items[inp.items.market_hash_name == worst][["version", "min_price", "median_price", "quantity"]]
ph = ph.merge(inp.sales[inp.sales.market_hash_name == worst][
    ["version", "last_30_days.median", "last_90_days.median", "last_90_days.volume"]],
    on="version", how="outer")
ph.to_csv(RES / "stattrak_worst_phase_rows.csv", index=False, float_format="%.2f")
print(f"\n{worst}: Skinport phase rows (asks and sales)")
print(ph.to_string(index=False))

# ---- 3. Thin markets: StatTrak vs normal knife outputs ----------------------------------
gold = inp.schema_df[inp.schema_df.gold & ~inp.schema_df.is_glove]
liq = []
for stt in (False, True):
    names = [P.output_name(g.skin, w, stt) for g in gold.itertuples()
             for w in P.WEAR_BOUNDS if g.fmin < P.WEAR_BOUNDS[w][1] and g.fmax > P.WEAR_BOUNDS[w][0]]
    v30 = pd.Series([s30.get(x, np.nan) for x in names])
    v90 = pd.Series([s90.get(x, np.nan) for x in names])
    liq.append({"stattrak": stt, "knife names": len(names), "listed or sold on Skinport": v90.notna().mean(),
                "median 30-day sales": v30.median(), "share with 0 sales in 90 days": (v90.fillna(0) == 0).mean()})
liq = pd.DataFrame(liq)
liq.to_csv(RES / "stattrak_knife_liquidity.csv", index=False, float_format="%.3f")
print("\nKnife output liquidity on Skinport:")
print(liq.round(3).to_string(index=False))

# ---- Re-run the StatTrak summary under each pricing and output-liquidity screen ---------
rows = []
for label, prices in price_sets.items():
    for min_out_sales in (0, 3):
        pr = prices
        if min_out_sales:
            thin = [x for x in pr.index if x.startswith("★") and s90.get(x, 0) < min_out_sales]
            pr = pr.drop(thin)          # treat thin knife outputs as unpriced
        xs = P.covert_cross_section(pr, inp.vol30, inp.schema_df, inp.pools)
        for stt in (False, True):
            d = xs[xs.stattrak == stt]
            rows.append({"pricing": label, "knife outputs need >= N sales in 90d": min_out_sales,
                         "stattrak": stt, **P.ratio_summary(d)})
rerun = pd.DataFrame(rows).drop(columns=["cheapest_iqr"])
rerun.to_csv(RES / "stattrak_rerun.csv", index=False, float_format="%.3f")
print("\nTest 1 under alternative knife pricing:")
print(rerun.round(3).to_string(index=False))

# ---- 4. The rule: StatTrak Coverts from glove-only cases have no route -------------------
glove_only = [c for c, p in inp.pools.items() if p.is_glove.all()]
covert = set(inp.schema_df.skin[(inp.schema_df.rarity == "Covert") & ~inp.schema_df.gold])
st_glove = [x for x in price_sets[cols[1]].index
            if x.startswith("StatTrak™") and inp.vol30.get(x, 0) >= 5
            and P.split_market_hash_name(x)[0] in covert
            and any(c in glove_only for c in P.parity_for(x, inp.schema_df, inp.pools,
                                                         price_sets[cols[1]])["cases"].split("; "))]
print(f"\nGlove-only cases: {len(glove_only)}; liquid StatTrak Coverts in them (no trade-up "
      f"route under the published rule, so excluded from Test 1): {len(st_glove)}")
