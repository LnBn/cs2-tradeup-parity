"""Write a local page for collecting Steam price histories BY HAND.

Why by hand: Steam's `pricehistory` endpoint needs a logged-in session, and the Steam
Subscriber Agreement (s. 4.C) forbids scripts/bots interacting with Steam. So no code
here touches Steam. The page lists items with links; the user opens each link in their
own logged-in browser, copies the JSON shown, and pastes it into the page. The page
bundles everything into one JSON file to download, which 05_parse_steam_manual.py
turns into a panel.

Usage:
  04_make_steam_helper.py                       pilot sample -> tools/steam_collect.html
  04_make_steam_helper.py --list results/collection_lists/tier2.csv --name tier2
                                                -> tools/steam_collect_tier2.html
A list CSV needs `market_hash_name`; optional `priority`, `group`, `case` columns.
Items are ordered by priority, then interleaved (round-robin) across segments or cases,
so a partial collection is still balanced. Each page keeps its own browser storage.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from catalog import SEGMENTS  # noqa: E402
from http_cache import PROJECT_ROOT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--list", type=Path, help="CSV of items (default: results/sample.csv)")
ap.add_argument("--name", default="", help="list name, used in the page title and file name")
args = ap.parse_args()


def round_robin(df: pd.DataFrame, key: str, order=None) -> list[dict]:
    keys = order or list(dict.fromkeys(df[key]))
    queues = {k: df[df[key] == k].to_dict("records") for k in keys}
    out = []
    while any(queues.values()):
        for k in keys:
            if queues[k]:
                out.append(queues[k].pop(0))
    return out


if args.list is None:
    # Pilot sample: interleave segments, most liquid (Skinport 90-day volume) first. The shipped
    # sample.csv is already in that order (its price and volume columns are not published);
    # a freshly reselected sample still carries vol_90d and is sorted here.
    df = pd.read_csv(PROJECT_ROOT / "results" / "sample.csv")
    if "vol_90d" in df:
        df = df.sort_values("vol_90d", ascending=False)
    items = [{"name": r["market_hash_name"], "segment": r["segment"]}
             for r in round_robin(df, "segment", SEGMENTS)]
else:
    df = pd.read_csv(args.list)
    for col, default in [("priority", 1), ("group", ""), ("case", "")]:
        if col not in df:
            df[col] = default
    df = df.fillna({"group": "", "case": ""})
    items = []
    for prio, d in df.groupby("priority", sort=True):
        for r in round_robin(d, "case"):
            label = " · ".join(x for x in [f"P{prio}", r["group"], r["case"]] if x)
            items.append({"name": r["market_hash_name"], "segment": label})

suffix = f"_{args.name}" if args.name else ""
template = (PROJECT_ROOT / "tools" / "steam_collect.template.html").read_text()
html = (template.replace("/*__ITEMS__*/[]", json.dumps(items, ensure_ascii=False))
        .replace('/*__STORE_KEY__*/"cs2-steam-manual-v1"',
                 json.dumps("cs2-steam-manual-v1" + suffix))
        .replace('/*__LIST_NAME__*/""', json.dumps(args.name)))
out = PROJECT_ROOT / "tools" / f"steam_collect{suffix}.html"
out.write_text(html)
print(f"wrote {out} with {len(items)} items")
