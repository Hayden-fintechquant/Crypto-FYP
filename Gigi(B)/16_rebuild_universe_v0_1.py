"""v0.1 follow-up: flag pre-CMC-top100 zero mcaps + rebuild universe from panel_v0_1.

Remaining fantom/matic zeros are dates BEFORE those coins entered CMC top-100
snapshots (no point-in-time supply). We do NOT invent look-ahead supply.
Flag them; rebuild membership so EOS/FTM/MATIC/MKR can enter when mcap>0.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

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

FOUR = ["eos", "fantom", "matic-network", "maker"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--every-days", type=int, default=7)
    ap.add_argument("--end", default="2026-09-29", help="inclusive sample end (UTC date)")
    args = ap.parse_args()
    if args.top_n < 30 or args.top_n > 50:
        raise SystemExit("--top-n must be 30-50")

    src = PROCESSED / "price_mcap_panel_v0_1.csv"
    if not src.exists():
        raise SystemExit(f"Missing {src}. Run script 15 first.")

    df = pd.read_csv(src)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    end = pd.Timestamp(args.end)
    before = len(df)
    df = df[df["date"] <= end].copy()
    print(f"trim date<={args.end}: {before:,} -> {len(df):,} rows")

    # Honest flag: approx zeros with no CMC coverage
    mcap = pd.to_numeric(df["market_cap"], errors="coerce")
    zero_approx = (df["market_cap_source"].astype(str) == "approx") & (mcap.fillna(0) <= 0)
    n_flag = int(zero_approx.sum())
    df.loc[zero_approx, "market_cap"] = pd.NA
    df.loc[zero_approx, "market_cap_source"] = "missing_pre_cmc_top100"
    df["mcap_bad_flag"] = df["market_cap"].notna() & (pd.to_numeric(df["market_cap"], errors="coerce") <= 0)
    print(f"flagged {n_flag} pre-CMC-top100 zeros -> market_cap=NA, source=missing_pre_cmc_top100")

    out_panel = PROCESSED / "price_mcap_panel_v0_1.csv"
    df.to_csv(out_panel, index=False)
    print(f"updated {out_panel}")

    # Universe from panel (candidate = current 50), exclude stables, top N
    u = df[~df["coin_id"].isin(STABLE_OR_WRAP)].copy()
    u["market_cap"] = pd.to_numeric(u["market_cap"], errors="coerce")
    u = u.dropna(subset=["market_cap"])
    u = u[u["market_cap"] > 0]
    all_dates = sorted(u["date"].unique())
    reb = all_dates[:: args.every_days]
    rows = []
    for d in reb:
        day = u[u["date"] == d].sort_values("market_cap", ascending=False).head(args.top_n)
        for rank, (_, r) in enumerate(day.iterrows(), start=1):
            rows.append(
                {
                    "rebalance_date": pd.Timestamp(d).date().isoformat(),
                    "coin_id": r["coin_id"],
                    "rank_by_mcap": rank,
                    "market_cap": float(r["market_cap"]),
                    "market_cap_source": r.get("market_cap_source", ""),
                }
            )
    mem = pd.DataFrame(rows)
    mem_path = PROCESSED / "universe_membership_v0_1.csv"
    mem.to_csv(mem_path, index=False)

    ever = mem.groupby("coin_id").agg(
        n_weeks=("rebalance_date", "nunique"),
        best_rank=("rank_by_mcap", "min"),
        first=("rebalance_date", "min"),
        last=("rebalance_date", "max"),
    )
    print(f"universe top-{args.top_n}: {len(mem):,} rows, ever-members={ever.shape[0]} -> {mem_path}")
    print("\n=== four coins in panel universe ===")
    for cid in FOUR:
        if cid in ever.index:
            r = ever.loc[cid]
            print(
                f"{cid}: weeks={int(r.n_weeks)} best_rank={int(r.best_rank)} "
                f"{r.first} .. {r.last}"
            )
        else:
            print(f"{cid}: NOT in universe")

    # source mix after flag
    print("\n=== market_cap_source mix ===")
    print(df["market_cap_source"].value_counts(dropna=False).head(10).to_string())
    bad = int(df["mcap_bad_flag"].fillna(False).sum())
    print(f"mcap_bad_flag remaining: {bad}")
    print("done")


if __name__ == "__main__":
    main()
