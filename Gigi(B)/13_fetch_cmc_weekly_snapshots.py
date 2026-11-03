"""Download CoinMarketCap weekly historical snapshots (top 100) via public data-api.

Endpoint (no paid key; still sleep & respect rate limits):
  https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listings/historical?date=YYYY-MM-DD&limit=100&start=1

Pilot first (--pilot), then full Sundays 2018-01-07 .. 2026-09-27.

Output: data/raw/cmc_snapshots/YYYYMMDD.csv
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "cmc_snapshots"
API = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listings/historical"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
SLEEP_S = 2.5


def sundays(start: date, end: date) -> list[date]:
    d = start
    while d.weekday() != 6:
        d += timedelta(days=1)
    out = []
    while d <= end:
        out.append(d)
        d += timedelta(days=7)
    return out


def fetch_snapshot(d: date, limit: int = 100) -> pd.DataFrame:
    r = requests.get(
        API,
        params={"date": d.isoformat(), "limit": limit, "start": 1},
        headers={"User-Agent": UA, "Accept": "application/json"},
        timeout=60,
    )
    r.raise_for_status()
    payload = r.json()
    data = payload.get("data") or []
    rows = []
    for item in data:
        quotes = item.get("quotes") or []
        q = quotes[0] if quotes else {}
        rows.append(
            {
                "snapshot_date": d.isoformat(),
                "rank": item.get("cmcRank"),
                "cmc_id": item.get("id"),
                "name": item.get("name"),
                "symbol": item.get("symbol"),
                "slug": item.get("slug"),
                "market_cap": q.get("marketCap"),
                "price": q.get("price"),
                "circulating_supply": item.get("circulatingSupply"),
                "total_supply": item.get("totalSupply"),
                "volume_24h": q.get("volume24h"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--start", default="2018-01-07")
    ap.add_argument("--end", default="2026-09-27")
    ap.add_argument("--sleep", type=float, default=SLEEP_S)
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    dates = sundays(start, end)
    if args.pilot:
        want = {
            date(2018, 1, 7),
            date(2021, 5, 9),
            date(2022, 5, 8),
            date(2022, 11, 6),
            date(2025, 1, 5),
        }
        dates = [d for d in dates if d in want] or dates[:5]
        print(f"PILOT {len(dates)} Sundays: {[x.isoformat() for x in dates]}")
    else:
        print(f"FULL {len(dates)} Sundays {start} .. {end}")

    ok, bad = [], []
    for i, d in enumerate(dates, 1):
        ymd = d.strftime("%Y%m%d")
        dest = OUT / f"{ymd}.csv"
        if dest.exists() and dest.stat().st_size > 200:
            print(f"[{i}/{len(dates)}] skip {ymd}")
            ok.append(ymd)
            continue
        print(f"[{i}/{len(dates)}] fetch {ymd} ...")
        try:
            df = fetch_snapshot(d, limit=args.limit)
            if df.empty:
                raise RuntimeError("empty")
            df.to_csv(dest, index=False)
            # quick check for problem coins on 2021-05-09
            syms = set(df["symbol"].astype(str).str.upper())
            hits = [s for s in ("EOS", "MATIC", "FTM", "MKR", "LUNA", "FTT") if s in syms]
            print(f"  -> {len(df)} rows; hits={hits}; top3={list(df['symbol'].head(3))}")
            ok.append(ymd)
        except Exception as e:
            print(f"  FAIL {ymd}: {e}")
            bad.append(ymd)
        time.sleep(args.sleep)

    print(f"\ndone ok={len(ok)} bad={len(bad)} -> {OUT}")
    if bad:
        print("failed:", ", ".join(bad))


if __name__ == "__main__":
    main()
