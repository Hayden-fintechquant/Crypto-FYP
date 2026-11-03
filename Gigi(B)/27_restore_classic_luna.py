"""Restore classic terra-luna (through 2022-05-13) overwritten by batch LUNC/v2 fetch.

- Removes panel rows coin_id==terra-luna
- Re-inserts from crash_coins_panel_v1.csv (classic only)
- Keeps any LUNC under coin_id=='terra-luna' post-crash out of classic id;
  if those rows were stored as terra-luna, they are dropped here (still in CMC universe as LUNC symbol).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    panel_path = PROCESSED / "price_mcap_panel_v1.csv"
    crash_path = PROCESSED / "crash_coins_panel_v1.csv"
    panel = pd.read_csv(panel_path, low_memory=False)
    crash = pd.read_csv(crash_path)
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    crash["date"] = pd.to_datetime(crash["date"]).astype("datetime64[ns]")

    before = int((panel["coin_id"] == "terra-luna").sum())
    panel = panel[panel["coin_id"] != "terra-luna"].copy()
    classic = crash[crash["coin_id"] == "terra-luna"].copy()
    print(f"removed terra-luna rows={before}; restoring classic rows={len(classic)}")

    for c in panel.columns:
        if c not in classic.columns:
            classic[c] = pd.NA
    classic = classic.reindex(columns=panel.columns)
    out = pd.concat([panel, classic], ignore_index=True)
    out = out.sort_values(["coin_id", "date"]).drop_duplicates(["coin_id", "date"], keep="last")
    out.to_csv(panel_path, index=False)

    s = out[out["coin_id"] == "terra-luna"].sort_values("date")
    print(f"terra-luna now {len(s)} rows {s['date'].min().date()} .. {s['date'].max().date()}")
    w = s[(s["date"] >= "2022-05-09") & (s["date"] <= "2022-05-13")][["date", "close"]]
    print(w.to_string(index=False))
    print("done")


if __name__ == "__main__":
    main()
