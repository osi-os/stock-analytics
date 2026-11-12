"""Small client for Massive (formerly Polygon.io) option data.

The API key is not handled here: the cloud environment adds it as an
Authorization header for api.massive.com. To run elsewhere, set MASSIVE_API_KEY
and it is sent the same way.

The free Options Basic plan allows 5 requests a minute, so calls are spaced out
(`MIN_INTERVAL`) and every response is cached on disk, which lets a long run
resume after a restart without spending requests again.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

BASE = "https://api.massive.com"
MIN_INTERVAL = 12.5  # seconds between calls: 5 per minute on the free plan
CACHE_DIR = Path(os.environ.get("MASSIVE_CACHE", "data/massive"))

_last_call = 0.0


def _get(path: str, params: dict | None = None, cache: bool = True) -> dict:
    global _last_call
    url = f"{BASE}{path}" + (f"?{urlencode(sorted((params or {}).items()))}" if params else "")
    key = hashlib.sha1(url.encode()).hexdigest()
    cfile = CACHE_DIR / f"{key}.json"
    if cache and cfile.exists():
        return json.loads(cfile.read_text())

    headers = {}
    if os.environ.get("MASSIVE_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['MASSIVE_API_KEY']}"
    for attempt in range(6):
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 429:
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        data = r.json()
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cfile.write_text(json.dumps(data))
        return data
    raise RuntimeError(f"rate-limited too many times: {path}")


def contracts(underlying: str, kind: str, as_of: date, near: float,
              band: float = 0.08, max_days: int = 10) -> pd.DataFrame:
    """Contracts listed on `as_of` for the nearest expiries, strikes within +-band of `near`.

    Returns columns ticker, expiration_date, strike_price, contract_type.
    """
    params = {
        "underlying_ticker": underlying,
        "contract_type": kind,
        "as_of": as_of.isoformat(),
        "expiration_date.gte": as_of.isoformat(),
        "expiration_date.lte": (as_of + timedelta(days=max_days)).isoformat(),
        "strike_price.gte": round(near * (1 - band), 2),
        "strike_price.lte": round(near * (1 + band), 2),
        "sort": "expiration_date",
        "order": "asc",
        "limit": 1000,
    }
    data = _get("/v3/reference/options/contracts", params)
    df = pd.DataFrame(data.get("results", []))
    if df.empty:
        return df
    df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.date
    return df[["ticker", "expiration_date", "strike_price", "contract_type"]]


def minute_bars(option_ticker: str, day: date) -> pd.DataFrame:
    """One day of 1-minute bars for an option contract (trades only, no quotes).

    Index: bar start, tz-aware America/New_York. Columns open, high, low, close,
    volume, vwap, trades. Minutes with no trade have no row.
    """
    d = day.isoformat()
    data = _get(f"/v2/aggs/ticker/{option_ticker}/range/1/minute/{d}/{d}",
                {"adjusted": "true", "sort": "asc", "limit": 50000})
    df = pd.DataFrame(data.get("results", []))
    if df.empty:
        return df
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close",
                            "v": "volume", "vw": "vwap", "n": "trades"})
    return df[["open", "high", "low", "close", "volume", "vwap", "trades"]]
