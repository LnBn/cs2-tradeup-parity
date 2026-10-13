"""Parse hand-collected Steam price histories into a daily panel.

Input : data/manual_steam/*.json, bundles downloaded from tools/steam_collect.html
        (or pass bundle paths as arguments). Later captures of an item win.
Output: data/processed/panel_steam.parquet, with columns
        market_hash_name, day, price, n_sales, currency

Steam's `prices` rows are [date string, median price, volume as a string], e.g.
["Oct 13 2025 01: +0", 152.31, "17"]. Dates are UTC. Older history is one row per day;
the most recent month is one row per hour, so rows are aggregated to days: the daily
price is the volume-weighted mean of the row medians and n_sales is the summed volume.
Prices are in the collecting account's wallet currency (price_prefix/suffix), which is
fine for the event study because it only uses log changes within an item.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from http_cache import PROJECT_ROOT  # noqa: E402

paths = [Path(p) for p in sys.argv[1:]] or sorted((PROJECT_ROOT / "data/manual_steam").glob("*.json"))
if not paths:
    sys.exit("No bundles found: put the downloaded file(s) in data/manual_steam/")

# Merge bundles; for an item collected more than once, keep the latest capture.
records = {}
for path in paths:
    bundle = json.loads(path.read_text())
    if bundle.get("format") != "steam_pricehistory_manual_v1":
        sys.exit(f"{path}: not a steam_collect.html bundle")
    for name, rec in bundle["items"].items():
        if name not in records or rec["captured_at"] > records[name]["captured_at"]:
            records[name] = rec

frames = []
for name, rec in records.items():
    rows = pd.DataFrame(rec["prices"], columns=["date", "price", "volume"])
    # "Oct 13 2025 01: +0" -> drop the "+0" UTC offset, parse "%b %d %Y %H:"
    rows["ts"] = pd.to_datetime(rows["date"].str.replace(r"\s*\+0$", "", regex=True),
                                format="%b %d %Y %H:")
    rows["volume"] = pd.to_numeric(rows["volume"], errors="coerce").fillna(0)
    rows["day"] = rows["ts"].dt.floor("D")
    rows["pv"] = rows["price"] * rows["volume"]
    daily = rows.groupby("day").agg(pv=("pv", "sum"), n_sales=("volume", "sum"),
                                    plain=("price", "mean"))
    # Volume-weighted mean; fall back to the plain mean on (rare) zero-volume days.
    daily["price"] = np.where(daily.n_sales > 0, daily.pv / daily.n_sales.where(daily.n_sales > 0),
                              daily.plain)
    daily = daily.reset_index()[["day", "price", "n_sales"]]
    daily.insert(0, "market_hash_name", name)
    daily["currency"] = (rec.get("price_prefix", "") + rec.get("price_suffix", "")).strip()
    frames.append(daily)

panel = pd.concat(frames, ignore_index=True)
panel["n_sales"] = panel["n_sales"].astype(int)
out = PROJECT_ROOT / "data/processed/panel_steam.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
panel.to_parquet(out, index=False)

sample = pd.read_csv(PROJECT_ROOT / "results" / "sample.csv").set_index("market_hash_name")
have = sample.index.intersection(panel.market_hash_name.unique())
print(f"{len(records)} items from {len(paths)} bundle(s) -> {out} ({len(panel)} item-days)")
print("currency:", panel.currency.unique().tolist())
print(f"pilot sample: {len(have)}/{len(sample)}; per segment:",
      sample.loc[have, "segment"].value_counts().to_dict())
for lst_path in sorted((PROJECT_ROOT / "results/collection_lists").glob("*.csv")):
    lst = pd.read_csv(lst_path)
    lst["have"] = lst.market_hash_name.isin(records)
    by = lst.groupby(["priority", "group"]).have.agg(["sum", "size"])
    print(f"{lst_path.stem}: {int(lst.have.sum())}/{len(lst)} collected;",
          {f"P{p} {g}": f"{r['sum']}/{r['size']}" for (p, g), r in by.iterrows()})
