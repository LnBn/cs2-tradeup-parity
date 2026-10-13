"""Shared, read-only inputs for the parity scripts (06-11).

Only local files are read: a Skinport snapshot written by scripts/collect_snapshot.py,
the cached ByMykel item schema (MIT) and the hand-collected Steam panel written by
scripts/05_parse_steam_manual.py. Nothing here touches the network.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import analysis as A
import parity as P
from http_cache import PROJECT_ROOT

# The published results use the Skinport snapshot taken at 17:09 UTC on 13 Oct 2026.
# It is pinned so that a newer snapshot does not silently change the results. Skinport
# serves only current data, so a fresh clone cannot re-download it: set the environment
# variable CS2_SKINPORT_SNAPSHOT to a *_skinport_sales.parquet file, or the latest
# snapshot under data/snapshots/ is used (results will then differ from the published ones).
PINNED_SNAPSHOT = PROJECT_ROOT / "data/snapshots/2026-10-13/1709_skinport_sales.parquet"


def _resolve_snapshot() -> Path:
    if os.environ.get("CS2_SKINPORT_SNAPSHOT"):
        return Path(os.environ["CS2_SKINPORT_SNAPSHOT"]).resolve()
    if PINNED_SNAPSHOT.exists():
        return PINNED_SNAPSHOT
    found = sorted((PROJECT_ROOT / "data/snapshots").glob("*/*_skinport_sales.parquet"))
    if not found:
        sys.exit("No Skinport snapshot found: run scripts/collect_snapshot.py first.")
    print(f"note: pinned snapshot {PINNED_SNAPSHOT.relative_to(PROJECT_ROOT)} not found; "
          f"using {found[-1].relative_to(PROJECT_ROOT)}", file=sys.stderr)
    return found[-1]


SNAPSHOT = _resolve_snapshot()
STEAM_PANEL = PROJECT_ROOT / "data/processed/panel_steam.parquet"
CENSORED = {"★ Butterfly Knife | Tiger Tooth (Factory New)",   # hit Steam's price cap
            "★ Sport Gloves | Vice (Field-Tested)"}            # pre-event (see README)


@dataclass
class Inputs:
    schema_df: pd.DataFrame
    coll_df: pd.DataFrame
    pools: dict
    sales: pd.DataFrame
    items: pd.DataFrame
    vol30: pd.Series
    sample: pd.DataFrame
    steam: pd.DataFrame = field(repr=False)
    base: pd.Series = field(repr=False)
    changes: pd.DataFrame = field(repr=False)
    now: pd.Series = field(repr=False)


def load() -> Inputs:
    schema = json.loads((PROJECT_ROOT / "data/raw/bymykel/skins.json").read_text())
    schema_df = P.schema_frame(schema)
    sales = pd.read_parquet(SNAPSHOT)
    items = pd.read_parquet(str(SNAPSHOT).replace("_sales", "_items"))
    sample = pd.read_csv(PROJECT_ROOT / "results/sample.csv").set_index("market_hash_name")
    steam = pd.read_parquet(STEAM_PANEL)
    steam = A.add_event_time(steam[~steam.market_hash_name.isin(CENSORED)], "price")
    base = A.baseline_levels(steam)
    changes = A.window_changes(steam, base)
    now = (steam[steam.day > steam.day.max() - pd.Timedelta(days=30)]
           .groupby("market_hash_name").log_p.median())
    return Inputs(schema_df=schema_df, coll_df=P.collection_frame(schema),
                  pools=P.gold_pools(schema_df), sales=sales, items=items,
                  vol30=sales.groupby("market_hash_name")["last_30_days.volume"].sum(),
                  sample=sample, steam=steam, base=base, changes=changes, now=now)


def index_shift(inp: Inputs, segments: list[str]) -> float:
    """Median item log level of a segment in days 31-90 minus its level over the last 30
    days (both vs baseline), used to carry today's P* back to the post-event window."""
    names = inp.sample.index[inp.sample.segment.isin(segments)].intersection(inp.base.index)
    names = names.intersection(inp.now.index)
    win = inp.changes.loc[names, "[31,90]"].median()
    now = (inp.now.loc[names] - inp.base.loc[names]).median()
    return float(win - now)


def daily_index(inp: Inputs, segments: list[str]) -> pd.Series:
    """Daily median (log price - baseline) across the sample items of the given segments."""
    names = inp.sample.index[inp.sample.segment.isin(segments)].intersection(inp.base.index)
    p = inp.steam[inp.steam.market_hash_name.isin(names)]
    rel = p.log_p - p.market_hash_name.map(inp.base)
    return rel.groupby(p.day).median()


def bootstrap(stat_fn, data: pd.DataFrame, n_boot: int = 2000, seed: int = 0,
              cluster: str | None = None) -> tuple[float, float]:
    """Percentile 95% CI of stat_fn(resampled data); resamples rows, or whole clusters."""
    rng = np.random.default_rng(seed)
    out = []
    if cluster is None:
        for _ in range(n_boot):
            out.append(stat_fn(data.iloc[rng.integers(0, len(data), len(data))]))
    else:
        groups = [g for _, g in data.groupby(cluster)]
        for _ in range(n_boot):
            pick = rng.integers(0, len(groups), len(groups))
            # relabel clusters so repeated draws count as separate groups
            out.append(stat_fn(pd.concat([groups[k].assign(**{cluster: f"{j}"})
                                          for j, k in enumerate(pick)])))
    return tuple(np.nanpercentile(out, [2.5, 97.5]))
