"""Classify items into market segments and pick a liquid pilot sample.

Segments are defined relative to the 22 Oct 2025 trade-up change:

* ``Knife`` / ``Gloves``     - the "gold" items whose supply the update increased.
* ``Covert (gold pool)``     - Covert weapon skins from a case that contains
                               knives/gloves, i.e. eligible trade-up inputs.
* ``Covert (no gold pool)``  - Covert skins from map/armory/collection drops with
                               no knife/glove pool: same tier, not usable as
                               inputs (our within-tier control).
* ``Classified (case)``      - one tier below eligible Coverts (inputs to the
                               Classified -> Covert trade-up): tests spillover.
* ``Restricted (case)``      - two tiers below: distal control.
"""
from __future__ import annotations

import re

import pandas as pd

WEAR_RE = re.compile(r"\s\((Factory New|Minimal Wear|Field-Tested|Well-Worn|Battle-Scarred)\)$")

SEGMENTS = ["Knife", "Gloves", "Covert (gold pool)", "Covert (no gold pool)",
            "Classified (case)", "Restricted (case)"]


def schema_table(schema: list[dict]) -> pd.DataFrame:
    """One row per skin name with its segment (or None if out of scope)."""
    gold_crates = {c["name"] for x in schema
                   if x.get("category") and x["category"]["name"] in ("Knives", "Gloves")
                   for c in x.get("crates", [])}
    rows = []
    for x in schema:
        cat = x["category"]["name"] if x.get("category") else None
        rarity = x["rarity"]["name"]
        crates = {c["name"] for c in x.get("crates", [])}
        in_gold_case = bool(crates & gold_crates)
        if cat == "Knives":
            seg = "Knife"
        elif cat == "Gloves":
            seg = "Gloves"
        elif rarity == "Covert":
            seg = "Covert (gold pool)" if in_gold_case else "Covert (no gold pool)"
        elif rarity == "Classified" and in_gold_case:
            seg = "Classified (case)"
        elif rarity == "Restricted" and in_gold_case:
            seg = "Restricted (case)"
        else:
            seg = None
        rows.append({
            "skin": x["name"], "segment": seg, "category": cat, "rarity": rarity,
            "weapon": x["weapon"]["name"] if x.get("weapon") else None,
            "collection": "; ".join(c["name"] for c in x.get("collections", [])),
            "min_float": x.get("min_float"), "max_float": x.get("max_float"),
        })
    # Doppler/Gamma Doppler phases appear as several rows with the same name.
    return pd.DataFrame(rows).drop_duplicates("skin")


def split_market_hash_name(name: str) -> tuple[str, str | None, bool, bool]:
    """'StatTrak™ AK-47 | Redline (Field-Tested)' -> ('AK-47 | Redline', 'Field-Tested', True, False)."""
    stattrak = "StatTrak™ " in name
    souvenir = name.startswith("Souvenir ")
    base = name.replace("StatTrak™ ", "").replace("Souvenir ", "")
    m = WEAR_RE.search(base)
    wear = m.group(1) if m else None
    if m:
        base = base[: m.start()]
    return base, wear, stattrak, souvenir


def select_sample(schema_df: pd.DataFrame, skinport_sales: list[dict],
                  n_per_segment: dict[str, int]) -> pd.DataFrame:
    """Pick the most-traded (Skinport 90-day volume) plain variant of each skin,
    then the top-N skins per segment. Returns one row per market_hash_name."""
    recs = []
    for s in skinport_sales:
        base, wear, st, sv = split_market_hash_name(s["market_hash_name"])
        if st or sv:
            continue
        recs.append({"market_hash_name": s["market_hash_name"], "skin": base, "wear": wear,
                     "vol_90d": s["last_90_days"]["volume"] or 0,
                     "median_30d": s["last_30_days"]["median"]})
    sp = pd.DataFrame(recs)
    # Skinport writes vanilla knives as "★ Karambit"; the schema calls them "★ Karambit | Vanilla".
    schema_df = schema_df.assign(skin=schema_df["skin"].str.replace(" | Vanilla", "", regex=False))
    df = sp.merge(schema_df, on="skin", how="inner").dropna(subset=["segment"])
    best_variant = df.sort_values("vol_90d", ascending=False).drop_duplicates("skin")
    picks = [best_variant[best_variant.segment == seg].head(n)
             for seg, n in n_per_segment.items()]
    return pd.concat(picks, ignore_index=True)
