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

## The model

The trade-up rules come from Valve's patch note ([Steam announcement](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/1814309641424632)) plus community guides ([SteamDB](https://steamdb.com/en/articles/cs2-trade-up-contract-guide), [CSDelta](https://csdelta.com/tutorials/profitable-cs2-trade-up-guide), [Refrag](https://refrag.gg/blog/cs2-skins-update-trade-ups)). `src/parity.py` holds the model.

* **Five-for-one.** Each input picks its own case's knife/glove pool with probability 1/5, and the output is equally likely to be any item in that pool. Its wear follows from the inputs' wear. Outputs are priced at their recent median sale price on Skinport.
* **Ten-for-one cascade.** The older contracts take ten skins of one rarity and return one skin of the next rarity from the same collection, which gives floors one and two tiers further down.
* **StatTrak.** StatTrak inputs give StatTrak knives only. Some StatTrak knives trade so rarely that their listings contain stale asks, in one case more than 30 times the item's recent sale prices; pricing each knife variant from its own sales removes the apparent violations.
* **Day 0** is 23 October 2025, the first full trading day after the patch.

## Data

Prices come from Skinport's documented public API and from Steam Community Market price histories collected **by hand** (Steam's terms forbid scripts and bots). CSFloat is not used. The repository ships summary tables, model fits and figures, but no raw data and no per-item prices.

## Results

**Prices respect the floor.** Today no eligible Covert trades clearly below its floor, and the cheapest Covert in each case sits close to it. Skins one tier down, which cannot be traded into knives directly, sit far below their would-be floor.

**The update caused it.** Skins priced below their new floor before the update were lifted onto it, by almost exactly the size of the gap (slope about 0.9 to 1). The same test on a window before the update shows nothing.

![Price change after the update against the gap to the new floor](figures/parity/event_gap.png)

**The market adjusted within about a day.** Most below-floor skins reached 90% of their floor on the first trading day after the update, and deviations from the floor since then die out within a few days.

**The floor spread down a tier** through the older ten-for-one trade-ups: the cheapest skin of the next tier down also sits at about its implied floor.

---

AI assistance was used in this writeup.
