"""Merge terra-luna + ftx-token into price_mcap_panel_v1.csv; update coin_id_map."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    base_path = PROCESSED / "price_mcap_panel_v0_1.csv"
    crash_path = PROCESSED / "crash_coins_panel_v1.csv"
    if not base_path.exists():
        raise SystemExit(f"Missing {base_path}")
    if not crash_path.exists():
        raise SystemExit(f"Missing {crash_path}")

    base = pd.read_csv(base_path)
    crash = pd.read_csv(crash_path)
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    crash["date"] = pd.to_datetime(crash["date"]).dt.normalize()

    # Align crash columns to base schema (fill extras)
    for col in base.columns:
        if col not in crash.columns:
            crash[col] = pd.NA
    # Keep crash-only useful cols too
    extra = [c for c in crash.columns if c not in base.columns]
    cols = list(base.columns) + extra
    crash_aligned = crash.reindex(columns=cols)
    base_aligned = base.reindex(columns=cols)

    # Drop if already present (re-run safe)
    base_aligned = base_aligned[~base_aligned["coin_id"].isin(["terra-luna", "ftx-token"])]

    out = pd.concat([base_aligned, crash_aligned], ignore_index=True)
    out = out.sort_values(["coin_id", "date"]).drop_duplicates(["coin_id", "date"], keep="last")
    out_path = PROCESSED / "price_mcap_panel_v1.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(f"rows={len(out):,} coins={out['coin_id'].nunique()}")
    print(
        out[out["coin_id"].isin(["terra-luna", "ftx-token"])]
        .groupby("coin_id")
        .agg(n=("date", "size"), start=("date", "min"), end=("date", "max"))
        .to_string()
    )

    # Update coin_id_map
    map_path = PROCESSED / "coin_id_map.csv"
    if map_path.exists():
        m = pd.read_csv(map_path)
    else:
        m = pd.DataFrame(columns=["coin_id", "symbol", "successor_id", "successor_symbol", "migration_date", "first_trade_date", "last_trade_date", "end_reason", "notes"])

    add = pd.DataFrame(
        [
            {
                "coin_id": "terra-luna",
                "symbol": "LUNA",
                "successor_id": "terra-luna-v2",
                "successor_symbol": "LUNA",
                "migration_date": "2022-05-28",
                "first_trade_date": "2020-08-21",
                "last_trade_date": "2022-05-13",
                "end_reason": "collapsed",
                "notes": "Classic LUNA through May 2022 collapse; Binance LUNAUSDT after this is v2 — excluded from this series",
            },
            {
                "coin_id": "ftx-token",
                "symbol": "FTT",
                "successor_id": "",
                "successor_symbol": "",
                "migration_date": "",
                "first_trade_date": "2019-12-20",
                "last_trade_date": "2026-09-29",
                "end_reason": "collapsed",
                "notes": "FTX Nov 2022 collapse; thin post-crash tape may still trade — crisis window uses Nov 2022",
            },
        ]
    )
    m = m[~m["coin_id"].isin(["terra-luna", "ftx-token"])]
    m = pd.concat([m, add], ignore_index=True).sort_values("coin_id")
    m.to_csv(map_path, index=False)
    print(f"updated {map_path}")
    print(m[m["end_reason"] == "collapsed"][["coin_id", "symbol", "last_trade_date", "end_reason"]].to_string(index=False))
    print("done")


if __name__ == "__main__":
    main()
