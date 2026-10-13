"""Polite, cached HTTP GET for public market-data endpoints.

Every response is written to data/raw/<source>/<name>.json together with a
small metadata sidecar (URL, timestamp, status, headers we care about).
Re-running a script re-uses the cache unless ``refresh=True``, so we hit
third-party servers as rarely as possible.

Responses are treated as untrusted data: we only ever ``json.loads`` them.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Skinport serves its public API brotli-compressed only; `brotli` must be installed
# for requests to decode it transparently.
DEFAULT_HEADERS = {
    "Accept-Encoding": "br, gzip",
    "User-Agent": "cs2-skin-market-research/0.1 (academic side project)",
}

_last_call: dict[str, float] = {}


def _throttle(source: str, min_interval: float) -> None:
    """Sleep so that consecutive calls to one source are >= min_interval s apart."""
    wait = _last_call.get(source, 0.0) + min_interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call[source] = time.monotonic()


def cached_get_json(source: str, name: str, url: str, params: dict | None = None,
                    min_interval: float = 1.0, refresh: bool = False,
                    timeout: float = 60.0):
    """Return parsed JSON for ``url``; cached under data/raw/<source>/<name>.json.

    Returns ``None`` (and records the status in the sidecar) on non-200 replies,
    so callers can log access failures as findings rather than crash.
    """
    out_dir = RAW_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)
    body_path = out_dir / f"{name}.json"
    meta_path = out_dir / f"{name}.meta.json"

    if body_path.exists() and not refresh:
        return json.loads(body_path.read_text())

    _throttle(source, min_interval)
    resp = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
    meta = {
        "url": resp.url,
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": resp.status_code,
        "headers": {k: v for k, v in resp.headers.items()
                    if k.lower().startswith(("x-ratelimit", "retry-after", "content-type"))},
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    if resp.status_code != 200:
        meta["body_head"] = resp.text[:300]
        meta_path.write_text(json.dumps(meta, indent=2))
        return None
    data = resp.json()
    body_path.write_text(json.dumps(data))
    return data
