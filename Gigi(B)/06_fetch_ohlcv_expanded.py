"""
Day 9/23: Expand OHLCV coverage to candidate supersets (report-aligned exclusions only).

- Sample window: 2018-01-01 -> 2026-09-30
- Skips pairs Binance rejects (HTTP 400)
- Writes raw + data/processed/ohlcv_panel_v1.csv
- Does not build final universe (that needs historical mcap on 9/24-9/25)
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

from fyp_data.symbol_universe_candidates import CANDIDATE_SYMBOLS  # noqa: E402

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
    start_ms, end_ms = ms(START), ms(END)
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
            print(f"  skip {symbol}: not available on Binance spot")
            return pd.DataFrame()
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
    out = [
        {
            "date": pd.to_datetime(row[0], unit="ms", utc=True).date().isoformat(),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[7]),
            "retrieval_timestamp": retrieved,
        }
        for row in rows
    ]
    return (
        pd.DataFrame(out)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    frames = []
    ok, skipped = [], []
    for symbol, coin_id, _ in CANDIDATE_SYMBOLS:
        print(f"fetching {symbol} ({coin_id}) ...")
        df = fetch_symbol(symbol)
        if df.empty:
            skipped.append(symbol)
            continue
        path = RAW_DIR / f"{coin_id}_1d.csv"
        df.to_csv(path, index=False)
        print(f"  -> {path.name} rows={len(df)} {df['date'].min()} .. {df['date'].max()}")
        part = df.copy()
        part.insert(1, "coin_id", coin_id)
        frames.append(part)
        ok.append(coin_id)

    if not frames:
        raise SystemExit("No data downloaded.")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["date", "coin_id"]).drop_duplicates(["date", "coin_id"])
    out = PROCESSED / "ohlcv_panel_v1.csv"
    panel.to_csv(out, index=False)

    summary = PROCESSED / "ohlcv_expand_summary_v1.csv"
    pd.DataFrame({"coin_id": ok}).to_csv(summary, index=False)

    print("\nDONE")
    print(f"panel -> {out}")
    print(f"coins_ok={len(ok)} skipped={len(skipped)} rows={len(panel)}")
    print(f"date range: {panel['date'].min()} .. {panel['date'].max()}")
    print(f"duplicates={int(panel.duplicated(['date','coin_id']).sum())}")
    if skipped:
        print("skipped:", ", ".join(skipped))


if __name__ == "__main__":
    main()
