"""Thin wrappers around the free, automatable data sources used here.

Only two sources are fetched by code:

* Skinport's documented public API (no key; 8 requests per 5 minutes).
* The ByMykel/CSGO-API item schema (static JSON on GitHub, MIT licence).

Steam price histories are *not* fetched by code: the Steam Subscriber Agreement
(s. 4.C) forbids scripts and bots, so they are collected by hand (see README).

All calls go through ``http_cache.cached_get_json`` so each URL is fetched at
most once unless ``refresh=True``.
"""
from __future__ import annotations

from http_cache import cached_get_json

# --- Skinport (documented public API, no key; 8 requests / 5 min) -----------
SKINPORT = "https://api.skinport.com/v1"


def skinport_items(currency: str = "USD", refresh: bool = False):
    """Current listing summary (min/median/mean/max ask, quantity) per item."""
    return cached_get_json("skinport", f"items_{currency}", f"{SKINPORT}/items",
                           {"app_id": 730, "currency": currency},
                           min_interval=40, refresh=refresh)


def skinport_sales_history(currency: str = "USD", refresh: bool = False):
    """Aggregated sales (min/max/avg/median/volume) over 24h/7d/30d/90d windows."""
    return cached_get_json("skinport", f"sales_history_{currency}", f"{SKINPORT}/sales/history",
                           {"app_id": 730, "currency": currency},
                           min_interval=40, refresh=refresh)


# --- Static item schema (ByMykel/CSGO-API, MIT licence) ----------------------
def item_schema(refresh: bool = False):
    """Every CS2 skin with rarity, category, collection, crates and float range."""
    return cached_get_json(
        "bymykel", "skins",
        "https://raw.githubusercontent.com/ByMykel/CSGO-API/main/public/api/en/skins.json",
        refresh=refresh)
