"""Select the liquid pilot sample: results/sample.csv.

For each skin in the six segments (see src/catalog.py), take the non-StatTrak,
non-Souvenir wear variant with the highest Skinport 90-day sale volume, then the
top N skins per segment.

The sample used for every result in this repository was selected on 13 Oct 2026 and is
frozen in results/sample.csv, so by default this script only reports it. Pass
--reselect to draw a fresh sample from the current Skinport data (2 Skinport API
requests plus the ByMykel schema); this overwrites results/sample.csv and will not
reproduce the published results.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from http_cache import PROJECT_ROOT  # noqa: E402
import catalog  # noqa: E402
import sources  # noqa: E402

N_PER_SEGMENT = {
    "Knife": 25, "Gloves": 15,
    "Covert (gold pool)": 25, "Covert (no gold pool)": 10,
    "Classified (case)": 12, "Restricted (case)": 12,
}

sample_path = PROJECT_ROOT / "results" / "sample.csv"
if sample_path.exists() and "--reselect" not in sys.argv:
    sample = pd.read_csv(sample_path)
    print(f"frozen sample {sample_path.relative_to(PROJECT_ROOT)} (pass --reselect to redraw)")
else:
    schema = catalog.schema_table(sources.item_schema())
    sample = catalog.select_sample(schema, sources.skinport_sales_history(), N_PER_SEGMENT)
    sample_path.parent.mkdir(exist_ok=True)
    sample.to_csv(sample_path, index=False)
    print(f"wrote {sample_path.relative_to(PROJECT_ROOT)}")
print(sample["segment"].value_counts().reindex(list(N_PER_SEGMENT)).to_string())
