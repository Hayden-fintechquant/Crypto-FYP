"""
Step 1 (Week 1): Fetch a SMALL pilot panel so you can see real files.

Tries Binance public API first (usually works without a key).
Falls back to a bundled offline sample if the network is blocked/rate-limited.

IMPORTANT:
- Pilot list is for learning the pipeline only.
- Final FYP universe must use historical market-cap ranks (script 02).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fyp_data.binance_public import PILOT_SYMBOLS, daily_klines, now_iso  # noqa: E402
from fyp_data.paths import PROCESSED, RAW, SAMPLE  # noqa: E402


def fetch_from_binance(limit: int = 90) -> pd.DataFrame:
    retrieved = now_iso()
    frames = []
    meta_rows = []
    for i, (symbol, coin_id, sym) in enumerate(PILOT_SYMBOLS, start=1):
        print(f"  fetching {symbol} ...")
        kl = daily_klines(symbol, limit=limit)
        meta_rows.append(
            {"coin_id": coin_id, "symbol": sym, "binance_symbol": symbol, "pilot_rank": i}
        )
        for row in kl:
            # Binance kline: open time, open, high, low, close, volume, close time, quote volume, ...
            frames.append(
                {
                    "date": pd.to_datetime(row[0], unit="ms", utc=True).date().isoformat(),
                    "coin_id": coin_id,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[7]),  # quote volume in USDT
                    "market_cap": pd.NA,  # Binance spot has no mcap; fill later via aggregator
                    "retrieval_timestamp": retrieved,
                }
            )
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(RAW / "pilot_coin_list.csv", index=False)
    return pd.DataFrame(frames)


def write_offline_sample() -> pd.DataFrame:
    """Tiny deterministic sample so learning never depends on the network."""
    print("  network unavailable — writing bundled offline sample")
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    rows = []
    retrieved = now_iso()
    for coin, px0 in [("bitcoin", 40000), ("ethereum", 2200), ("solana", 100)]:
        px = px0
        for d in dates:
            px = px * (1.0 + ((hash((coin, str(d.date()))) % 7) - 3) / 1000.0)
            rows.append(
                {
                    "date": d.date().isoformat(),
                    "coin_id": coin,
                    "open": px * 0.99,
                    "high": px * 1.01,
                    "low": px * 0.98,
                    "close": px,
                    "volume": 1e8,
                    "market_cap": px * 1e6,
                    "retrieval_timestamp": retrieved,
                }
            )
    return pd.DataFrame(rows)


def main(limit: int = 90) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    SAMPLE.mkdir(parents=True, exist_ok=True)

    print("1) Building pilot panel...")
    try:
        panel = fetch_from_binance(limit=limit)
        # proxy market_cap with close * constant so script 02 can rank (TEACHING ONLY)
        scale = {"bitcoin": 19e6, "ethereum": 120e6, "binancecoin": 150e6, "solana": 400e6,
                 "ripple": 45e9, "cardano": 35e9, "dogecoin": 140e9, "avalanche-2": 400e6}
        panel["market_cap"] = panel.apply(
            lambda r: float(r["close"]) * scale.get(r["coin_id"], 1e9), axis=1
        )
        print("   note: market_cap here is a PLACEHOLDER (price × assumed supply) for ranking practice.")
        print("   later replace with real market-cap history from an aggregator.")
    except Exception as e:
        print(f"   Binance failed: {e}")
        panel = write_offline_sample()

    panel = panel.sort_values(["date", "coin_id"]).reset_index(drop=True)
    out_csv = PROCESSED / "pilot_panel.csv"
    out_sample = SAMPLE / "pilot_panel_head.csv"
    panel.to_csv(out_csv, index=False)
    panel.head(50).to_csv(out_sample, index=False)
    print(f"2) Done. Rows={len(panel)} coins={panel['coin_id'].nunique()}")
    print(f"   full:   {out_csv}")
    print(f"   sample: {out_sample}")
    print("\nNext: python scripts/02_build_universe_skeleton.py")


if __name__ == "__main__":
    main()
