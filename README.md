# Trade-up parity in the CS2 skin market

**How a game update created a no-arbitrage price floor, and how fast the market found it.**

## Summary

Counter-Strike 2 has a "Trade Up Contract": players hand in several weapon skins of one rarity and receive one skin of the next rarity. On 22 October 2025 Valve extended it so that **five Covert (red) skins can be exchanged for one knife or pair of gloves** from the case collection of one of the inputs. Knives and gloves are the most valuable items in the game. Overnight, every eligible Covert skin became a claim on one fifth of a lottery over its case's knife and glove pool.

That implies a simple no-arbitrage floor. If a Covert sells for less than a fifth of the expected value of the knife or glove it can be traded into (after fees), buying five and trading up is profitable on average, so its price should not stay below

> **P\* = (1 − fee) · E[value of the output] / 5**

This project tests that prediction with public market data and asks four questions:

1. **Do prices respect the floor today?** Yes. No eligible (non-StatTrak) Covert trades clearly below it (below 0.85·P\*), and the cheapest Covert in each case sits close to it. A placebo tier that cannot be traded into knives sits far below its would-be floor.
2. **Did the update cause it?** Items that were priced below their new floor before the update were lifted onto it, by almost exactly the size of the gap (slope 0.94 on 23 items, 0.89 on 77). Placebo windows before the update show nothing.
3. **How fast did the market adjust?** Within about a day: most below-floor items reached 90% of their floor on the first trading day. Since then, deviations from the floor die out with a half-life of about four days.
4. **Did the floor spread?** Yes. Because Covert prices now carry the knife floor, the older 10-for-1 trade-ups pass it one tier further down: the cheapest Classified skin per collection sits at about its implied floor.

The same update made knives and gloves much more plentiful: their prices fell by about half within three days and did not recover.

Developed with AI assistance (Claude Code).

## The model

The trade-up rules come from Valve's patch note ([Steam announcement](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/1814309641424632)) plus community guides ([SteamDB](https://steamdb.com/en/articles/cs2-trade-up-contract-guide), [CSDelta](https://csdelta.com/tutorials/profitable-cs2-trade-up-guide), [Refrag](https://refrag.gg/blog/cs2-skins-update-trade-ups)). `src/parity.py` holds the model.

* **Five-for-one.** Each input picks its own case's knife/glove pool with probability 1/5, and the output is equally likely to be any item in that pool. Its wear follows from the inputs' wear. Outputs are priced at their recent median sale price on Skinport.
* **Ten-for-one cascade.** The older contracts take ten skins of one rarity and return one skin of the next rarity from the same collection, which gives floors one and two tiers further down.
* **StatTrak.** StatTrak inputs give StatTrak knives only. Some StatTrak knives trade so rarely that their listings contain stale asks, in one case more than 30 times the item's recent sale prices; pricing each knife variant from its own sales removes the apparent violations.
* **Day 0** is 23 October 2025, the first full trading day after the patch.

## Results

**The floor binds in the cross-section** (`scripts/06_parity_test.py`; Skinport, 13 October 2026). For 285 eligible non-StatTrak Coverts, none trades below 0.85·P\*, and the cheapest Covert per case sits at a median of 1.13·P\*. The Classified skins of the same cases, which cannot be traded into knives directly, sit at a median of 0.19 of the same floor.

![Cross-section](figures/parity/cross_section.png)

**The update lifted prices onto the floor** (Steam price histories, days 31–90 after the update against a pre-update baseline). The prediction is that an item's price change equals the gap between its old price and its new floor when it was below it, and zero otherwise. The fitted slope is 0.94 (standard error 0.05) on 23 items and 0.89 on an expanded set of 77, with R² of about 0.93. The same model fitted to a window before the update gives a slope of about zero. A rival explanation, that cheap items simply rose more, fits worse.

![Event gap](figures/parity/event_gap.png)

**Adjustment was almost immediate** (`scripts/08_parity_convergence.py`). Of the 13 Coverts below their floor before the update, 11 reached 90% of it within three days, most on day 0. Tracking the Covert closest to its floor in each of 38 cases (`scripts/11_tier2_parity.py`), prices have sat at a median of 1.2 times the floor since the update, and deviations die out with a half-life of about four days. Before the update the same items sat at about a fifth of their (not yet existing) floor.

![Convergence](figures/parity/convergence.png)

**The floor passes down a tier** (`scripts/07_parity_cascade.py`, `scripts/11_tier2_parity.py`). The cheapest Classified skin per collection sits at a median of 1.02 times its cascade floor, and placebo tiers sit far below theirs. The ten-for-one rule existed before the update, so this cross-section shows that the floor binds, not that the update caused it; the event test on Classified skins (58 items) shows the same lift, with slope 0.78.

**The results are robust** (`scripts/10_parity_robustness.py`) to the pricing window, the fee, the minimum number of sales, removing plain knives, and leaving out any one case.

![Robustness](figures/parity/robustness.png)

## Limitations

* **Small event samples:** 23 items in the main event test and 77 in the expanded one. The test two tiers down has too few below-floor items to say anything.
* **The floor is backcast.** It is computed from October 2026 prices and carried back to the event with a knife and glove price index, so noise in that index pulls the slopes below 1.
* **Rules partly from third parties.** Valve does not publish every detail (for example, the odds of each knife finish), so some come from community guides.
* **Steam prices** are in non-withdrawable wallet funds, carry a fee of about 15% and are capped per item. All event statistics are changes within an item, so the currency and fee cancel.
* **Selection and confounders.** The sample was chosen by liquidity after the event. There is no clean untreated control inside the game, because every tier is linked to the treated tiers by trade-ups. New cases, tournaments and a later rule change (May 2026) may also move prices.

## Data

The repository ships **no raw data and no per-item prices**: `data/` is git-ignored, and so are the intermediate result files that list individual items' prices or sales volumes (see `.gitignore`). The scripts regenerate them from your own data. What is shipped are summary tables, model fits, aggregate price indices and figures.

| source | used for | how it is collected |
|---|---|---|
| [Skinport API](https://docs.skinport.com/) | current prices and sales volumes | `scripts/collect_snapshot.py` (documented public API, no key, 2 requests) |
| Steam Community Market price history | daily price histories for the event tests | **by hand**, in your own logged-in browser, using the helper pages in `tools/`. The Steam Subscriber Agreement (s. 4.C) forbids scripts and bots, so no code here contacts Steam |
| [ByMykel/CSGO-API](https://github.com/ByMykel/CSGO-API) item schema (MIT) | rarity, collections, cases, wear ranges, knife/glove pools | cached by `collect_snapshot.py` |

CSFloat's terms do not permit automated access, so it is not used.

**Collecting the Steam histories by hand.**
1. Open `tools/steam_collect.html` (the 99-item pilot sample) and `tools/steam_collect_tier2.html` (178 further items) in a browser where you are logged in to Steam. To rebuild the pages, run `scripts/04_make_steam_helper.py` (and `--list results/collection_lists/tier2.csv --name tier2`).
2. For each item, open the link, copy the JSON that Steam shows and paste it into the page. The page saves progress in browser storage.
3. Click **Download data** and move the downloaded JSON bundle(s) into `data/manual_steam/`. `05_parse_steam_manual.py` merges all bundles, and a later capture of an item replaces an earlier one. Use this to fix a bad paste.

**Exact reproduction.** The published numbers use the Skinport snapshot taken at 17:09 UTC on 13 Oct 2026. Its path is pinned in `src/parity_inputs.py` as `data/snapshots/2026-10-13/1709_skinport_sales.parquet` (with a matching `_items` file), and the Steam histories were collected on the same day. Skinport serves only current data, so a fresh clone cannot re-download that snapshot. If the pinned file is absent, scripts 06–11 use the latest snapshot under `data/snapshots/`, or the file named by the environment variable `CS2_SKINPORT_SNAPSHOT`. The pipeline then runs end to end, but the cross-sectional numbers will reflect the new date. The frozen sample (`results/sample.csv`, without its price and volume columns) and the Tier 2 list (`results/collection_lists/tier2.csv`) are shipped, so the same items are collected.

## Reproduce

```bash
git clone <this repo> cs2-tradeup-parity && cd cs2-tradeup-parity
conda env create -f environment.yml && conda activate cs2-tradeup-parity

# 0. Data
python scripts/collect_snapshot.py        # Skinport snapshot -> data/snapshots/<date>/, plus the item schema
#    then collect the Steam histories by hand (see "Data") into data/manual_steam/

# 1. Steam panel and segment event study
python scripts/05_parse_steam_manual.py   # data/manual_steam/*.json -> data/processed/panel_steam.parquet
python scripts/03_event_study.py          # -> results/steam/, figures/steam/

# 2. Trade-up parity (local files only, no network)
python scripts/06_parity_test.py          # Tests 1-2, Covert -> knife/glove   -> results/parity/, figures/parity/
python scripts/07_parity_cascade.py       # 10-for-1 cascade + placebos
python scripts/08_parity_convergence.py   # days to 0.9 P*, half-life
python scripts/09_stattrak_diagnosis.py   # the StatTrak pricing artefact
python scripts/10_parity_robustness.py    # one-at-a-time robustness, bootstrap, leave-one-case-out (~1 min)
python scripts/11_tier2_parity.py         # Tier 2: expanded event test, cascade event, daily floor dynamics
```

Optional: `scripts/02_select_sample.py` reports the frozen sample, and `--reselect` redraws it from current Skinport volumes. `scripts/04b_make_tier2_lists.py` rebuilds the Tier 2 collection list; it skips items already in the Steam panel, so it will not reproduce the shipped list once those items are collected.

With the original 13 Oct 2026 snapshot and Steam bundles in `data/`, scripts 05, 03 and 06–11 reproduce every CSV in `results/` (including the unshipped per-item files) and every figure byte for byte.

## Layout

```
src/http_cache.py      cached, throttled GET (JSON only; responses treated as untrusted data)
src/sources.py         Skinport API and ByMykel schema wrappers
src/catalog.py         segment definitions and liquid-sample selection
src/analysis.py        event windows, segment indices, bootstrap CIs, co-movement blocks
src/parity.py          trade-up model: pools, P*, cascade, phase-robust prices
src/parity_inputs.py   shared local inputs for 06-11 (pinned snapshot, Steam panel)
scripts/               02 sample, 03 event study, 04/04b/05 Steam by hand, 06-11 parity,
                       collect_snapshot (Skinport)
tools/                 hand-collection helper pages (template + generated pages)
results/               summary tables (per-item price files are regenerated, not shipped); figures/ PNGs
data/                  snapshots, schema cache, Steam bundles, panel (git-ignored, not shipped)
```
