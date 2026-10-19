"""
Day 1 (2026-09-22): Full-period daily OHLCV from Binance public API.

Sample window: 2018-01-01 -> 2026-09-30 (UTC)
- Paginates klines (max 1000 per request)
- Writes raw per-coin CSV under data/raw/ohlcv/
- Builds long panel data/processed/ohlcv_panel_v0.csv
- Does NOT invent market_cap (that is Day 3 / 9-24)
- Does NOT apply universe filters beyond using the pilot symbol list
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fyp_data.binance_public import PILOT_SYMBOLS  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "ohlcv"
PROCESSED = ROOT / "data" / "processed"
BASE = "https://api.binance.com/api/v3/klines"

START = datetime(2018, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 9, 30, 23, 59, 59, tzinfo=timezone.utc)
SLEEP_S = 0.35
LIMIT = 1000


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_symbol(symbol: str) -> pd.DataFrame:
    """Paginate daily klines for one Binance symbol."""
    start_ms = ms(START)
    end_ms = ms(END)
    rows: list[list] = []
    while start_ms < end_ms:
        r = requests.get(
            BASE,
            params={
                "symbol": symbol,
                "interval": "1d",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": LIMIT,
            },
            timeout=60,
        )
        if r.status_code == 429:
            time.sleep(5)
            continue
        if r.status_code == 400:
            # invalid symbol / no pair
            print(f"  skip {symbol}: HTTP 400 {r.text[:120]}")
            break
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_start = last_open + 24 * 60 * 60 * 1000
        if next_start <= start_ms:
            break
        start_ms = next_start
        time.sleep(SLEEP_S)
        if len(batch) < LIMIT:
            break
    if not rows:
        return pd.DataFrame()
    retrieved = now_iso()
    out = []
    for row in rows:
        out.append(
            {
                "date": pd.to_datetime(row[0], unit="ms", utc=True).date().isoformat(),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[7]),  # quote volume USDT
                "retrieval_timestamp": retrieved,
            }
        )
    df = pd.DataFrame(out).drop_duplicates(subset=["date"]).sort_values("date")
    return df.reset_index(drop=True)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    frames = []
    for symbol, coin_id, _sym in PILOT_SYMBOLS:
        print(f"fetching {symbol} ({coin_id}) ...")
        df = fetch_symbol(symbol)
        if df.empty:
            print(f"  no rows for {symbol}")
            continue
        raw_path = RAW_DIR / f"{coin_id}_1d.csv"
        df.to_csv(raw_path, index=False)
        print(f"  raw -> {raw_path}  rows={len(df)}  {df['date'].min()} .. {df['date'].max()}")
        part = df.copy()
        part.insert(1, "coin_id", coin_id)
        frames.append(part)

    if not frames:
        raise SystemExit("No data downloaded. Check network / Binance access.")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["date", "coin_id"]).drop_duplicates(["date", "coin_id"])
    out = PROCESSED / "ohlcv_panel_v0.csv"
    panel.to_csv(out, index=False)
    print("\nDONE")
    print(f"panel -> {out}")
    print(f"coins={panel['coin_id'].nunique()} rows={len(panel)}")
    print(f"date range: {panel['date'].min()} .. {panel['date'].max()}")
    print(f"duplicates={panel.duplicated(['date','coin_id']).sum()}")


if __name__ == "__main__":
    main()
