"""Trade-up parity for Covert -> knife/glove contracts (since the 22 Oct 2025 update).

Model (from Valve's patch note plus third-party guides; see README)
-------------------------------------------------------------------
* Five Covert inputs give one "rare special item" (knife or gloves). Each input picks its
  own collection's case pool with weight 1/5, and the output is uniform over the items
  in that pool. StatTrak inputs give StatTrak knives only (gloves have no StatTrak).
* Output float = out_min + n * (out_max - out_min), where n is the mean *normalised*
  input float, (f - in_min) / (in_max - in_min).

Parity price of one input, for a contract of five copies of the same skin and wear:
    P* = (1 - fee) * E[output value] / 5
If inputs were cheaper than P*, buying five and trading up would pay on average. So
arbitrage should put a floor near P* under the cheapest Coverts of each collection.

Known simplifications: vanilla knives are priced only if the schema lists them;
Doppler-style phases are priced as a quantity-weighted mean over Skinport's phase rows;
each input wear is represented by the midpoint of its float interval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from catalog import split_market_hash_name

WEAR_BOUNDS = {"Factory New": (0.00, 0.07), "Minimal Wear": (0.07, 0.15),
               "Field-Tested": (0.15, 0.38), "Well-Worn": (0.38, 0.45),
               "Battle-Scarred": (0.45, 1.00)}


def wear_of(f: float) -> str:
    for wear, (lo, hi) in WEAR_BOUNDS.items():
        if lo <= f < hi:
            return wear
    return "Battle-Scarred"


def normalised_mid(wear: str, fmin: float, fmax: float) -> float | None:
    """Midpoint of the wear interval clipped to the skin's float range, normalised to [0, 1]."""
    lo, hi = WEAR_BOUNDS[wear]
    lo, hi = max(lo, fmin), min(hi, fmax)
    if hi <= lo or fmax <= fmin:
        return None
    return ((lo + hi) / 2 - fmin) / (fmax - fmin)


def schema_frame(schema: list[dict]) -> pd.DataFrame:
    """One row per skin with rarity, float range, case(s) and whether it is a gold item."""
    rows = []
    for s in schema:
        rows.append({"skin": s["name"], "rarity": s["rarity"]["name"],
                     "fmin": s.get("min_float") or 0.0, "fmax": s.get("max_float") or 1.0,
                     "gold": s["name"].startswith("★"),
                     "is_glove": "Gloves" in s["name"] or "Wraps" in s["name"],
                     "souvenir_only": bool(s.get("souvenir")) and not s.get("crates"),
                     "cases": tuple(sorted(c["name"] for c in s.get("crates") or []))})
    return pd.DataFrame(rows).drop_duplicates("skin")   # Doppler phases share one name


def gold_pools(schema_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """case name -> gold items (knives/gloves) that the case contains."""
    gold = schema_df[schema_df.gold].explode("cases").dropna(subset=["cases"])
    return {case: g.drop(columns="cases") for case, g in gold.groupby("cases")}


def price_table(sales: pd.DataFrame, items: pd.DataFrame) -> pd.Series:
    """market_hash_name -> reference price (USD).

    Skinport 30-day median sale, falling back to the 90-day median, then the current
    median ask. Names with several rows (Doppler phases) are averaged with weights equal to
    the listing quantity, a rough proxy for how often each phase occurs."""
    s = sales.set_index("market_hash_name")
    ref = s["last_30_days.median"].fillna(s["last_90_days.median"])
    ask = items.groupby("market_hash_name").apply(
        lambda g: np.average(g.median_price.fillna(g.min_price), weights=g.quantity.clip(lower=1))
        if g.median_price.notna().any() or g.min_price.notna().any() else np.nan,
        include_groups=False)
    multi = items.market_hash_name.value_counts()
    multi = multi[multi > 1].index
    ref = ref.groupby(level=0).mean()
    ref.loc[ref.index.intersection(multi)] = ask.loc[ref.index.intersection(multi)]
    return ref.combine_first(ask).dropna()


def output_name(skin: str, wear: str | None, stattrak: bool) -> str:
    knife, _, finish = skin.partition(" | ")
    prefix = "★ StatTrak™ " if stattrak else "★ "
    knife = knife.replace("★ ", "")
    if finish in ("", "Vanilla"):
        return f"{prefix}{knife}"            # vanilla knives carry no wear
    return f"{prefix}{knife} | {finish} ({wear})"


def expected_output(pool: pd.DataFrame, n: float, stattrak: bool,
                    prices: pd.Series) -> tuple[float, int, int]:
    """E[value] of the trade-up output for normalised input float n.
    Returns (expected value, priced pool items, total pool items)."""
    if stattrak:
        pool = pool[~pool.is_glove]
    vals = []
    for row in pool.itertuples():
        f_out = row.fmin + n * (row.fmax - row.fmin)
        name = output_name(row.skin, wear_of(f_out), stattrak)
        if name in prices.index:
            vals.append(prices[name])
    return (float(np.mean(vals)) if vals else np.nan), len(vals), len(pool)


def parity_for(market_hash_name: str, schema_df: pd.DataFrame, pools: dict,
               prices: pd.Series, fee: float = 0.0) -> dict:
    """Parity price and diagnostics for one Covert input item."""
    skin, wear, stattrak, souvenir = split_market_hash_name(market_hash_name)
    out = {"market_hash_name": market_hash_name, "skin": skin, "wear": wear,
           "stattrak": stattrak, "parity": np.nan, "pool_priced": 0, "pool_size": 0,
           "cases": ""}
    row = schema_df[schema_df.skin == skin]
    if row.empty or wear is None or souvenir:
        return out
    row = row.iloc[0]
    cases = [c for c in row.cases if c in pools]
    out["cases"] = "; ".join(cases)
    if row.rarity != "Covert" or row.gold or not cases:
        return out
    n = normalised_mid(wear, row.fmin, row.fmax)
    if n is None:
        return out
    pool = pd.concat([pools[c] for c in cases]).drop_duplicates("skin")
    ev, priced, size = expected_output(pool, n, stattrak, prices)
    out.update(parity=(1 - fee) * ev / 5, pool_priced=priced, pool_size=size, n_norm=n)
    return out


# =====================================================================================
# Extensions (scripts/07+): robust phase pricing, the standard-trade-up cascade, and
# reusable versions of the two tests in scripts/06. Nothing above this line changed, so
# scripts/06 reproduces its original output.
# =====================================================================================
RARITY_ORDER = ["Consumer Grade", "Industrial Grade", "Mil-Spec Grade",
                "Restricted", "Classified", "Covert"]
WINDOW_FALLBACK = {"7": ["7", "30", "90"], "30": ["30", "90"], "90": ["90"]}


def price_table_v2(sales: pd.DataFrame, items: pd.DataFrame, window: str = "30",
                   phase_weight: str = "sales") -> pd.Series:
    """market_hash_name -> reference price (USD), with explicit choices.

    window       : "7", "30" or "90"-day Skinport median sale, falling back to longer
                   windows and finally the current median ask.
    phase_weight : how Doppler/Gamma Doppler phase rows (same name, different
                   ``version``) are combined:
                   "sales"     - each phase's own median sale, weighted by its 90-day sale
                                 count (an empirical proxy for phase frequency);
                   "quantity"  - median ask weighted by listing quantity (close to scripts/06
                                 does; slow-selling gem listings get overweighted);
                   "core"      - equal weights on Phases 1-4 only (gems ignored; a lower
                                 bound).
    """
    cols = [f"last_{w}_days.median" for w in WINDOW_FALLBACK[window]]
    s = sales.copy()
    s["ref"] = s[cols].bfill(axis=1).iloc[:, 0]
    ask = items.assign(ask=items.median_price.fillna(items.min_price))
    s = s.merge(ask[["market_hash_name", "version", "ask", "quantity"]],
                on=["market_hash_name", "version"], how="outer")
    s["ref"] = s["ref"].fillna(s["ask"])
    s["w_sales"] = s["last_90_days.volume"].fillna(0)
    s["w_qty"] = s["quantity"].fillna(0).clip(lower=1)
    s = s.dropna(subset=["ref"])

    def combine(g: pd.DataFrame) -> float:
        if len(g) == 1:
            return float(g.ref.iloc[0])
        if phase_weight == "quantity":
            a = g.dropna(subset=["ask"])
            return float(np.average(a.ask, weights=a.w_qty)) if len(a) else float(g.ref.mean())
        if phase_weight == "core":
            core = g[g.version.isin(["Phase 1", "Phase 2", "Phase 3", "Phase 4"])]
            return float(core.ref.mean()) if len(core) else float(g.ref.mean())
        w = g.w_sales.to_numpy()
        return float(np.average(g.ref, weights=w)) if w.sum() > 0 else float(g.ref.mean())

    return s.groupby("market_hash_name").apply(combine, include_groups=False).dropna()


def drop_vanilla(pools: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Pools without vanilla knives (names with no ' | finish'), for sensitivity checks."""
    return {c: p[p.skin.str.contains(r" \| ", regex=True)] for c, p in pools.items()}


def collection_frame(schema: list[dict]) -> pd.DataFrame:
    """One row per (non-gold) skin with its collection, rarity, float range and cases."""
    rows = []
    for s in schema:
        if s["name"].startswith("★") or not s.get("collections"):
            continue
        rows.append({"skin": s["name"], "rarity": s["rarity"]["name"],
                     "collection": s["collections"][0]["name"],
                     "fmin": s.get("min_float") or 0.0, "fmax": s.get("max_float") or 1.0,
                     "cases": tuple(sorted(c["name"] for c in s.get("crates") or []))})
    return pd.DataFrame(rows).drop_duplicates("skin")


def collection_types(coll_df: pd.DataFrame, pools: dict) -> pd.Series:
    """collection -> "case" if any of its skins drops from a case with a knife/glove pool,
    else "no gold pool" (map, armory, operation and other collections)."""
    has_gold = coll_df.groupby("collection")["cases"].apply(
        lambda cs: any(c in pools for t in cs for c in t))
    return has_gold.map({True: "case", False: "no gold pool"})


def tier_parity(market_hash_name: str, coll_df: pd.DataFrame, prices: pd.Series,
                target_rarity: str | None = None, n_inputs: int = 10,
                fee: float = 0.0) -> dict:
    """Parity of one input in a standard 10-for-1 trade-up within its own collection.

    P* = (1 - fee) * E[value of a uniform draw from the collection's next-tier skins] / 10,
    output float from the normalised-float rule, output StatTrak iff the input is.
    ``target_rarity`` overrides the output tier (used for the one-tier-lower placebo,
    which prices an input against a tier it cannot actually reach)."""
    skin, wear, stattrak, souvenir = split_market_hash_name(market_hash_name)
    out = {"market_hash_name": market_hash_name, "skin": skin, "wear": wear,
           "stattrak": stattrak, "parity": np.nan, "pool_priced": 0, "pool_size": 0}
    row = coll_df[coll_df.skin == skin]
    if row.empty or wear is None or souvenir:
        return out
    row = row.iloc[0]
    out.update(collection=row.collection, rarity=row.rarity)
    k = RARITY_ORDER.index(row.rarity) if row.rarity in RARITY_ORDER else None
    if target_rarity is None:
        if k is None or k + 1 >= len(RARITY_ORDER):
            return out
        target_rarity = RARITY_ORDER[k + 1]
    n = normalised_mid(wear, row.fmin, row.fmax)
    outputs = coll_df[(coll_df.collection == row.collection) & (coll_df.rarity == target_rarity)]
    if n is None or outputs.empty:
        return out
    vals = []
    for o in outputs.itertuples():
        f_out = o.fmin + n * (o.fmax - o.fmin)
        name = f"{'StatTrak™ ' if stattrak else ''}{o.skin} ({wear_of(f_out)})"
        if name in prices.index:
            vals.append(prices[name])
    out.update(parity=(1 - fee) * np.mean(vals) / n_inputs if vals else np.nan,
               pool_priced=len(vals), pool_size=len(outputs), n_norm=n,
               output_rarity=target_rarity)
    return out


def ratio_summary(d: pd.DataFrame, group_col: str = "cases") -> dict:
    """Test-1 summary statistics for one group of items with columns log_ratio, group_col."""
    cheapest = d.loc[d.groupby(group_col).log_ratio.idxmin()]
    return {"n_items": len(d), "n_groups": len(cheapest),
            "median_ratio": np.exp(d.log_ratio.median()),
            "share_below_0.85": (d.log_ratio < np.log(0.85)).mean(),
            "cheapest_median_ratio": np.exp(cheapest.log_ratio.median()),
            "cheapest_iqr": "{:.2f}-{:.2f}".format(*np.exp(cheapest.log_ratio.quantile([.25, .75]))),
            "share_cheapest_in_0.85-1.25":
                cheapest.log_ratio.between(np.log(0.85), np.log(1.25)).mean()}


def cheapest_slope(d: pd.DataFrame, group_col: str = "cases") -> tuple[float, float, int]:
    """log(price) on log(P*) for the cheapest-relative-to-P* item per group: slope, se, n."""
    from scipy import stats
    cheapest = d.loc[d.groupby(group_col).log_ratio.idxmin()]
    if len(cheapest) < 3:
        return np.nan, np.nan, len(cheapest)
    f = stats.linregress(np.log(cheapest.parity), np.log(cheapest.price))
    return f.slope, f.stderr, len(cheapest)


def covert_cross_section(prices: pd.Series, vol30: pd.Series, schema_df: pd.DataFrame,
                         pools: dict, min_vol: int = 5, fee: float = 0.0) -> pd.DataFrame:
    """Test 1 rows for eligible Covert inputs (same filters as scripts/06)."""
    rarity = schema_df.set_index("skin")["rarity"]
    rows = []
    for name in prices.index:
        skin, wear, st, sv = split_market_hash_name(name)
        if sv or wear is None or skin.startswith("★") or rarity.get(skin) != "Covert":
            continue
        if vol30.get(name, 0) < min_vol:
            continue
        r = parity_for(name, schema_df, pools, prices, fee=fee)
        if np.isnan(r["parity"]) or r["pool_priced"] < 0.8 * r["pool_size"]:
            continue
        r.update(price=prices[name], vol_30d=vol30.get(name, 0))
        rows.append(r)
    xs = pd.DataFrame(rows)
    xs["log_ratio"] = np.log(xs.price / xs.parity)
    return xs


def event_gap(names, parity_now: dict, prices: pd.Series, base: pd.Series, now: pd.Series,
              changes: pd.DataFrame, idx_shift: float, fee: float = 0.0) -> pd.DataFrame:
    """Test-2 rows: gap to the post-event floor and the realised change (as in scripts/06).

    parity_now : name -> P* today (Skinport USD, no fee)
    base, now  : Steam log price at the pre-event baseline and over the last 30 days
    idx_shift  : log level of the output tier's index in days 31-90 minus its level today,
                 used to carry today's P* back to the post-event window.
    """
    rows = []
    for name in names:
        if name not in base.index or name not in prices.index or name not in now.index:
            continue
        p_star_now = parity_now.get(name, np.nan)
        if not np.isfinite(p_star_now):
            continue
        fx = prices[name] / np.exp(now[name])          # Skinport USD per Steam unit
        p_pre = np.exp(base[name]) * fx
        p_star = (1 - fee) * p_star_now * np.exp(idx_shift)
        rows.append({"market_hash_name": name, "fee": fee, "p_pre": p_pre, "p_star": p_star,
                     "gap": np.log(p_star / p_pre), "change": changes.loc[name, "[31,90]"],
                     "placebo_change": changes.loc[name, "pre [-90,-32]"],
                     "log_p_pre": np.log(p_pre)})
    return pd.DataFrame(rows)


def gap_models(ev: pd.DataFrame) -> pd.DataFrame:
    """Parity, placebo, rival and joint regressions for one event_gap table (one fee)."""
    from scipy import stats

    def ols(x, y):
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < 3:
            return {"slope": np.nan, "se": np.nan, "intercept": np.nan, "r2": np.nan, "n": int(m.sum())}
        f = stats.linregress(x[m], y[m])
        return {"slope": f.slope, "se": f.stderr, "intercept": f.intercept,
                "r2": f.rvalue ** 2, "n": int(m.sum())}

    lift = np.maximum(ev.gap.to_numpy(), 0)
    rows = [{"model": "change ~ max(0, gap)  [parity]", **ols(lift, ev.change.to_numpy())},
            {"model": "placebo: pre-window change ~ max(0, gap)",
             **ols(lift, ev.placebo_change.to_numpy())},
            {"model": "change ~ log pre-event price  [rival]",
             **ols(ev.log_p_pre.to_numpy(), ev.change.to_numpy())}]
    both = ev.dropna(subset=["change"])
    if len(both) > 4:
        X = np.column_stack([np.ones(len(both)), np.maximum(both.gap, 0), both.log_p_pre])
        beta, *_ = np.linalg.lstsq(X, both.change.to_numpy(), rcond=None)
        resid = both.change.to_numpy() - X @ beta
        cov_b = resid @ resid / (len(both) - 3) * np.linalg.inv(X.T @ X)
        rows.append({"model": "joint: change ~ max(0,gap) + log price", "slope": beta[1],
                     "se": np.sqrt(cov_b[1, 1]), "intercept": beta[0],
                     "r2": 1 - resid.var() / both.change.var(), "n": len(both),
                     "log_price_coef": beta[2], "log_price_se": np.sqrt(cov_b[2, 2])})
    return pd.DataFrame(rows)
