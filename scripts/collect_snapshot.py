"""Take one timestamped snapshot of Skinport market data. NOT scheduled by default.

What it saves (under data/snapshots/<UTC date>/<HHMM>_<source>.parquet):
  * skinport_items   - every CS2 item: min/median/mean/max ask and *listing quantity*
                       (a supply proxy that no free source provides historically)
  * skinport_sales   - rolling 24h/7d/30d/90d sale aggregates
It also caches the ByMykel item schema in data/raw/bymykel/skins.json if it is missing
(the parity scripts read it from there).

Only Skinport's documented public API is used (2 requests; the limit is 8 per 5 min).
Steam is never contacted by code: its Subscriber Agreement (s. 4.C) forbids scripts.
Run at most a few times a day.

Example cron line (runs at 06:05 and 18:05 UTC; adjust the path and interpreter):
  5 6,18 * * *  cd /path/to/cs2-tradeup-parity && python scripts/collect_snapshot.py >> data/collector.log 2>&1
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from http_cache import PROJECT_ROOT  # noqa: E402
import sources  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()

    now = datetime.now(timezone.utc)
    out_dir = PROJECT_ROOT / "data" / "snapshots" / now.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%H%M")

    items = pd.DataFrame(sources.skinport_items(refresh=True))
    items["snapshot_utc"] = now
    items.to_parquet(out_dir / f"{stamp}_skinport_items.parquet", index=False)

    sales = pd.json_normalize(sources.skinport_sales_history(refresh=True))
    sales["snapshot_utc"] = now
    sales.to_parquet(out_dir / f"{stamp}_skinport_sales.parquet", index=False)
    print(f"{now:%Y-%m-%d %H:%M} skinport: {len(items)} items, {len(sales)} sales rows "
          f"-> {out_dir.relative_to(PROJECT_ROOT)}/{stamp}_*")

    schema = sources.item_schema()      # cached: fetched only if not already on disk
    print(f"item schema: {len(schema)} entries (data/raw/bymykel/skins.json)")


if __name__ == "__main__":
    main()
