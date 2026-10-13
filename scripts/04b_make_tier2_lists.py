"""Build the Tier-2 hand-collection list: results/collection_lists/tier2.csv.

Priorities (collect in this order; the page interleaves within each priority):
  P1 covert  - every eligible Covert skin (case collection with a knife/glove pool) not yet
               collected, in its most-traded non-StatTrak wear. Completes the event test.
  P1 pool    - the POOL_TOP most-traded non-StatTrak knives/gloves of each case pool, so
               that P*(t) can be built from case-specific historical pool indices.
  P2 classified, P3 restricted - most-traded wear of each skin in case collections, for an
               event version of the parity-cascade test (10 inputs -> next tier).
Items already in data/processed/panel_steam.parquet are skipped.
Prices/volumes: latest Skinport snapshot (documented public API). Schema: ByMykel (MIT).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import parity as P  # noqa: E402
from catalog import split_market_hash_name  # noqa: E402
from http_cache import PROJECT_ROOT  # noqa: E402

POOL_TOP = 4
MIN_VOL_90D = 15
CAPS = {"classified": 60, "restricted": 40}     # keep the manual workload bounded

schema_df = P.schema_frame(json.loads((PROJECT_ROOT / "data/raw/bymykel/skins.json").read_text()))
pools = P.gold_pools(schema_df)
snap = sorted((PROJECT_ROOT / "data/snapshots").glob("*/*_skinport_sales.parquet"))[-1]
sales = pd.read_parquet(snap)
have = set(pd.read_parquet(PROJECT_ROOT / "data/processed/panel_steam.parquet").market_hash_name)

sp = sales[["market_hash_name", "last_90_days.volume"]].rename(columns={"last_90_days.volume": "vol_90d"})
parts = pd.DataFrame([split_market_hash_name(n) for n in sp.market_hash_name],
                     columns=["skin", "wear", "stattrak", "souvenir"], index=sp.index)
sp = pd.concat([sp, parts], axis=1).merge(schema_df[["skin", "rarity", "gold", "cases"]], on="skin")
sp["cases"] = sp.cases.map(lambda cs: tuple(c for c in cs if c in pools))
sp = sp[~sp.stattrak & ~sp.souvenir & (sp.cases.map(len) > 0) & (sp.vol_90d >= MIN_VOL_90D)]


def most_traded_wear(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.wear.notna()].sort_values("vol_90d", ascending=False).drop_duplicates("skin")


rows = []
cov = most_traded_wear(sp[(sp.rarity == "Covert") & ~sp.gold])
rows += [{"market_hash_name": r.market_hash_name, "group": "covert", "priority": 1,
          "case": r.cases[0]} for r in cov.itertuples()]

gold = sp[sp.gold].explode("cases").rename(columns={"cases": "case"})
top = gold.sort_values("vol_90d", ascending=False).groupby("case").head(POOL_TOP)
rows += [{"market_hash_name": r.market_hash_name, "group": "pool", "priority": 1, "case": r.case}
         for r in top.itertuples()]

for prio, (group, rarity) in enumerate([("classified", "Classified"), ("restricted", "Restricted")], 2):
    d = most_traded_wear(sp[sp.rarity == rarity])
    d = d.assign(case=d.cases.str[0]).groupby("case").head(2)          # spread across cases
    d = d.sort_values("vol_90d", ascending=False).head(CAPS[group])
    rows += [{"market_hash_name": r.market_hash_name, "group": group, "priority": prio,
              "case": r.case} for r in d.itertuples()]

lst = pd.DataFrame(rows).drop_duplicates("market_hash_name")     # a knife may sit in many pools
lst = lst[~lst.market_hash_name.isin(have)]
out = PROJECT_ROOT / "results/collection_lists/tier2.csv"
out.parent.mkdir(parents=True, exist_ok=True)
lst.to_csv(out, index=False)
print(f"wrote {out}: {len(lst)} items")
print(lst.groupby(["priority", "group"]).size().to_string())
print("cases covered by the pool subset (incl. items already collected):",
      top.case.nunique(), "of", len(pools))
