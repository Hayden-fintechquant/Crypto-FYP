"""Day 9/25: survivorship-aware universe membership from market_cap.

Report rules only:
- On each rebalance date, rank coins by THAT date's market_cap
- Keep top N (default 40, within report's 30-50)
- Exclude stablecoins and wrapped assets (base assets)
- Membership is built from the candidate panel you downloaded
  (honest note: global all-crypto top50 would need every coin's mcap;
   here we rank within the 50-coin candidate set already in the panel)

Input: data/processed/price_mcap_panel_v0.csv
Output:
  data/processed/universe_membership_v0.csv
  data/processed/universe_membership_summary_v0.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# Report exclusions (base assets). USDT pairs are quote, not rows.
STABLE_OR_WRAP = {
    "tether",
    "usd-coin",
    "binance-usd",
    "dai",
    "first-digital-usd",
    "true-usd",
    "wrapped-bitcoin",
    "wrapped-steth",
    "weth",
    "staked-ether",
}


def build_membership(
    panel: pd.DataFrame,
    top_n: int = 40,
    every_days: int = 7,
) -> pd.DataFrame:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[~df["coin_id"].isin(STABLE_OR_WRAP)]
    df = df.dropna(subset=["market_cap"])
    df = df[df["market_cap"] > 0]

    all_dates = sorted(df["date"].unique())
    reb_dates = all_dates[::every_days]
    rows = []
    for d in reb_dates:
        day = df[df["date"] == d].sort_values("market_cap", ascending=False)
        # if fewer coins have data that day, take all available up to top_n
        day = day.head(top_n)
        for rank, (_, r) in enumerate(day.iterrows(), start=1):
            rows.append(
                {
                    "rebalance_date": d.date().isoformat(),
                    "coin_id": r["coin_id"],
                    "rank_by_mcap": rank,
                    "market_cap": float(r["market_cap"]),
                    "market_cap_source": r.get("market_cap_source", ""),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=40, help="30-50 per report; default 40")
    parser.add_argument("--every-days", type=int, default=7, help="rebalance spacing in days")
    args = parser.parse_args()
    if args.top_n < 30 or args.top_n > 50:
        raise SystemExit("--top-n must be between 30 and 50 (report range)")

    src = PROCESSED / "price_mcap_panel_v0.csv"
    if not src.exists():
        raise SystemExit(f"Missing {src}. Finish 9/24 first.")

    panel = pd.read_csv(src)
    mem = build_membership(panel, top_n=args.top_n, every_days=args.every_days)
    out = PROCESSED / "universe_membership_v0.csv"
    mem.to_csv(out, index=False)

    summary = (
        mem.groupby("rebalance_date")
        .agg(
            n_coins=("coin_id", "nunique"),
            top1=("coin_id", "first"),
            mcap_top1=("market_cap", "first"),
        )
        .reset_index()
    )
    # fix top1: actually first after sort is rank1 — already sorted when built
    summary_path = PROCESSED / "universe_membership_summary_v0.csv"
    # better top1 via filter rank==1
    top1 = mem[mem["rank_by_mcap"] == 1][["rebalance_date", "coin_id", "market_cap"]].rename(
        columns={"coin_id": "top1_coin", "market_cap": "top1_mcap"}
    )
    counts = mem.groupby("rebalance_date")["coin_id"].nunique().rename("n_coins").reset_index()
    summary = counts.merge(top1, on="rebalance_date")
    summary.to_csv(summary_path, index=False)

    print(f"wrote {out} rows={len(mem)}")
    print(f"wrote {summary_path} rebalances={len(summary)}")
    print(f"top_n={args.top_n} every_days={args.every_days}")
    print(f"date range: {mem['rebalance_date'].min()} .. {mem['rebalance_date'].max()}")
    print("sample (first 12 rows):")
    print(mem.head(12).to_string(index=False))
    print("\nNOTE: ranked within your downloaded candidate set (not every coin on earth).")
    print("Early dates may use approx market_cap (see market_cap_source).")


if __name__ == "__main__":
    main()
