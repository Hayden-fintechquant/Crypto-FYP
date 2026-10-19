"""
Step 2: Survivorship-aware universe — SKELETON + teaching logic.

Your FYP rule: on each rebalance date, take top 30-50 by market cap *as of that date*.
This script shows the algorithm on the pilot panel (limited history/coins).
Replace the data source later with a full historical mcap archive.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SAMPLE = ROOT / "data" / "sample"


def build_membership(panel: pd.DataFrame, top_n: int = 5, every_days: int = 7) -> pd.DataFrame:
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    dates = sorted(panel["date"].unique())
    # rebalance calendar: every_days
    reb = dates[::every_days]
    rows = []
    for d in reb:
        day = panel[panel["date"] == d].dropna(subset=["market_cap"])
        day = day.sort_values("market_cap", ascending=False).head(top_n)
        for rank, (_, r) in enumerate(day.iterrows(), start=1):
            rows.append(
                {
                    "rebalance_date": d.date().isoformat(),
                    "coin_id": r["coin_id"],
                    "rank_by_mcap": rank,
                    "market_cap": r["market_cap"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    src = PROCESSED / "pilot_panel.csv"
    if not src.exists():
        raise SystemExit("Run scripts/01_fetch_pilot_panel.py first.")
    panel = pd.read_csv(src)
    mem = build_membership(panel, top_n=5, every_days=7)
    out = PROCESSED / "pilot_universe_membership.csv"
    mem.to_csv(out, index=False)
    print(f"Wrote {out} ({len(mem)} rows)")
    print("This is a TOY membership file — full FYP needs historical ranks for delisted coins too.")
    print(mem.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
