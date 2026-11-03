"""v0.1: flag stale prices (>=3 identical closes) and zero volume — flag only, never delete."""
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
    df = df.sort_values(["coin_id", "date"]).reset_index(drop=True)

    close = pd.to_numeric(df["close"], errors="coerce")
    vol = pd.to_numeric(df["volume_usd"] if "volume_usd" in df.columns else df["volume"], errors="coerce")

    same_as_prev = df.groupby("coin_id")["close"].transform(lambda s: s.eq(s.shift(1)))
    # run length of identical closes ending at this row
    # group consecutive True runs within coin
    grp = (~same_as_prev.fillna(False)).groupby(df["coin_id"]).cumsum()
    run = same_as_prev.fillna(False).groupby([df["coin_id"], grp]).cumsum()
    # stale if this close equals previous AND run length >= 2 means 3+ identical days
    # day1: first of run run=0/False; day2 same run=1; day3 same run=2 → flag when run>=2
    df["stale_price_flag"] = run.fillna(0).astype(int) >= 2
    df["zero_volume_flag"] = vol.fillna(0) <= 0

    df.to_csv(path, index=False)
    print(f"updated {path}")
    print(f"stale_price_flag True: {int(df['stale_price_flag'].sum()):,}")
    print(f"zero_volume_flag True: {int(df['zero_volume_flag'].sum()):,}")
    top = (
        df[df["stale_price_flag"]]
        .groupby("coin_id")
        .size()
        .sort_values(ascending=False)
        .head(8)
    )
    print("top stale coins:")
    print(top.to_string())
    print("done")


if __name__ == "__main__":
    main()
