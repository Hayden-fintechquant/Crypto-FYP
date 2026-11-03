"""v0.1: add volume_usd + avg_volume_28d to panel (brief problem 8.3).

Binance kline field we stored as `volume` is already quote volume in USDT
(see 05_fetch_ohlcv_full.py: row[7]). Treat USDT ≈ USD → volume_usd = volume.
Do NOT multiply by close again.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    path = PROCESSED / "price_mcap_panel_v0_1.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["coin_id", "date"])

    vol = pd.to_numeric(df["volume"], errors="coerce")
    # Already USDT quote volume from Binance
    df["volume_usd"] = vol
    df["avg_volume_28d"] = (
        df.groupby("coin_id")["volume_usd"].transform(
            lambda s: s.rolling(28, min_periods=1).mean()
        )
    )

    df.to_csv(path, index=False)
    print(f"updated {path}")
    print("note: volume is already Binance quoteVolume (USDT); volume_usd = volume")
    print(
        df[["volume_usd", "avg_volume_28d"]]
        .describe()
        .loc[["count", "mean", "50%", "max"]]
        .to_string()
    )
    b = df[df["coin_id"] == "bitcoin"].tail(3)[
        ["date", "close", "volume", "volume_usd", "avg_volume_28d"]
    ]
    print("\nBTC last 3:")
    print(b.to_string(index=False))
    print("done")


if __name__ == "__main__":
    main()
