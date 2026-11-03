"""Build candidate pool + weekly universe ranks from CMC weekly snapshots.

Report rules (Student A modification brief, problem 3):
- Start from all weekly top-100 CMC snapshots
- Exclude stablecoins & wrapped assets (programmatic rules; log UST etc.)
- Keep LUNA / FTT (not stables)
- Candidate pool = union of coins that ever appeared in top-100 after exclusion
- Weekly universe = re-rank remaining by that Sunday's mcap, take top N (default 40)
- Rebalance date = next Monday (Sunday snapshot decides Monday membership)

Inputs:  data/raw/cmc_snapshots/YYYYMMDD.csv
Outputs:
  data/processed/cmc_snapshots_clean_v0_1.csv
  data/processed/candidate_pool_v0_1.csv
  data/processed/universe_membership_cmc_v0_1.csv
  data/processed/universe_membership_cmc_summary_v0_1.csv
  data/processed/cmc_exclusion_log.csv
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "cmc_snapshots"
PROCESSED = ROOT / "data" / "processed"

# CMC symbol (upper) — stables & wrapped. Expand carefully; do NOT put LUNA/FTT here.
STABLE_SYMBOLS = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDD", "USDP", "GUSD",
    "FRAX", "LUSD", "USDE", "PYUSD", "EURC", "EURS", "EUROC", "AGEUR",
    "UST", "USTC", "USDN", "USDK", "HUSD", "SUSD", "CUSD", "OUSD",
    "MIM", "FEI", "TRIBE", "RSV", "PAX", "USDX", "USDJ", "DOLA",
    "CRVUSD", "GHO", "USDY", "USD0", "RLUSD", "AUSD",
}
WRAP_SYMBOLS = {
    "WBTC", "WETH", "STETH", "WSTETH", "CBETH", "RETH", "WEETH",
    "RENBTC", "HBTC", "TBTC", "SBTC", "BTCB", "BTC.B",
    "WBNB", "WAVAX", "WMATIC", "WFTM", "WSOL",
}

# CMC slug (lower) extras when symbol alone is ambiguous
STABLE_OR_WRAP_SLUGS = {
    "tether", "usd-coin", "binance-usd", "dai", "true-usd", "first-digital-usd",
    "terrausd", "terrausd-wormhole", "terraclassicusd", "wrapped-bitcoin",
    "weth", "steth", "lido-staked-ether", "wrapped-steth", "coinbase-wrapped-btc",
    "binance-wrapped-btc", "wrapped-bnb", "pax-dollar", "gemini-dollar",
    "frax", "liquity-usd", "ethena-usde", "paypal-usd", "euro-coin",
    "first-digital-usd", "usd-coin-bridged", "tether-gold",  # XAUT sometimes listed; keep out of wrap if gold — skip
}


def is_excluded(symbol: str, slug: str) -> tuple[bool, str]:
    sym = (symbol or "").strip().upper()
    sl = (slug or "").strip().lower()
    if sym in STABLE_SYMBOLS or sl in STABLE_OR_WRAP_SLUGS:
        reason = "stablecoin"
        if sym in {"UST", "USTC"} or "terra" in sl and "usd" in sl:
            reason = "stablecoin_ust"
        return True, reason
    if sym in WRAP_SYMBOLS or sl.startswith("wrapped-") or sl.startswith("weth"):
        return True, "wrapped"
    if "wrapped" in sl and "bitcoin" in sl:
        return True, "wrapped"
    return False, ""


def load_all_snapshots() -> pd.DataFrame:
    files = sorted(RAW.glob("*.csv"))
    if not files:
        raise SystemExit(f"No snapshots in {RAW}. Run script 13 first.")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if df.empty:
            continue
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df["snapshot_date"] = pd.to_datetime(all_df["snapshot_date"])
    all_df["symbol"] = all_df["symbol"].astype(str)
    all_df["slug"] = all_df.get("slug", pd.Series([""] * len(all_df))).astype(str)
    return all_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=40, help="universe size 30-50; default 40")
    args = ap.parse_args()
    if args.top_n < 30 or args.top_n > 50:
        raise SystemExit("--top-n must be between 30 and 50")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    raw = load_all_snapshots()
    print(f"loaded {len(raw):,} rows from {raw['snapshot_date'].nunique()} Sundays")

    flags = raw.apply(lambda r: is_excluded(r["symbol"], r["slug"]), axis=1)
    raw["excluded"] = [f[0] for f in flags]
    raw["exclude_reason"] = [f[1] for f in flags]

    excl = raw[raw["excluded"]].copy()
    excl_log = (
        excl.groupby(["symbol", "slug", "exclude_reason"], dropna=False)
        .agg(n_rows=("rank", "size"), first_seen=("snapshot_date", "min"), last_seen=("snapshot_date", "max"))
        .reset_index()
        .sort_values(["exclude_reason", "symbol"])
    )
    excl_path = PROCESSED / "cmc_exclusion_log.csv"
    excl_log.to_csv(excl_path, index=False)
    ust_n = int((excl_log["exclude_reason"] == "stablecoin_ust").sum())
    print(f"excluded {len(excl):,} rows; unique excluded tickers={excl_log['symbol'].nunique()}; UST-family tickers={ust_n}")
    print(f"  -> {excl_path}")

    clean = raw[~raw["excluded"]].drop(columns=["excluded", "exclude_reason"]).copy()
    clean_path = PROCESSED / "cmc_snapshots_clean_v0_1.csv"
    clean.to_csv(clean_path, index=False)
    print(f"clean snapshots: {len(clean):,} rows -> {clean_path}")

    # Candidate pool = union ever in top-100 after exclusion
    pool = (
        clean.groupby(["cmc_id", "symbol", "slug", "name"], dropna=False)
        .agg(
            first_seen=("snapshot_date", "min"),
            last_seen=("snapshot_date", "max"),
            n_snapshots=("snapshot_date", "nunique"),
            best_rank=("rank", "min"),
            max_mcap=("market_cap", "max"),
        )
        .reset_index()
        .sort_values(["best_rank", "symbol"])
    )
    pool_path = PROCESSED / "candidate_pool_v0_1.csv"
    pool.to_csv(pool_path, index=False)
    syms = set(pool["symbol"].astype(str).str.upper())
    must = [s for s in ("LUNA", "FTT", "EOS", "MATIC", "FTM", "MKR") if s in syms]
    miss = [s for s in ("LUNA", "FTT", "EOS", "MATIC", "FTM", "MKR") if s not in syms]
    print(f"candidate pool: {len(pool)} coins -> {pool_path}")
    print(f"  must-have present: {must}")
    if miss:
        print(f"  WARNING missing: {miss}")

    # Weekly universe: re-rank clean by mcap, top N; rebalance = next Monday
    rows = []
    for d, day in clean.groupby("snapshot_date"):
        day = day.dropna(subset=["market_cap"])
        day = day[day["market_cap"] > 0].sort_values("market_cap", ascending=False)
        day = day.drop_duplicates(subset=["cmc_id"], keep="first").head(args.top_n)
        reb = (pd.Timestamp(d) + timedelta(days=1)).date().isoformat()
        snap = pd.Timestamp(d).date().isoformat()
        for rank, (_, r) in enumerate(day.iterrows(), start=1):
            rows.append(
                {
                    "snapshot_date": snap,
                    "rebalance_date": reb,
                    "coin_id": r.get("slug") or str(r.get("cmc_id")),
                    "cmc_id": r.get("cmc_id"),
                    "symbol": r.get("symbol"),
                    "name": r.get("name"),
                    "rank_by_mcap": rank,
                    "market_cap": float(r["market_cap"]),
                    "price": r.get("price"),
                    "circulating_supply": r.get("circulating_supply"),
                    "mcap_source": "cmc_snapshot",
                    "is_member": True,
                }
            )
    mem = pd.DataFrame(rows)
    mem_path = PROCESSED / "universe_membership_cmc_v0_1.csv"
    mem.to_csv(mem_path, index=False)

    # turnover + n_members
    summary_rows = []
    prev = set()
    for reb, g in mem.groupby("rebalance_date"):
        cur = set(g["symbol"].astype(str).str.upper())
        added = sorted(cur - prev)
        dropped = sorted(prev - cur)
        summary_rows.append(
            {
                "rebalance_date": reb,
                "n_members": len(cur),
                "n_added": len(added),
                "n_dropped": len(dropped),
                "added": ",".join(added[:12]) + ("…" if len(added) > 12 else ""),
                "dropped": ",".join(dropped[:12]) + ("…" if len(dropped) > 12 else ""),
                "top1": g.loc[g["rank_by_mcap"].idxmin(), "symbol"] if len(g) else "",
            }
        )
        prev = cur
    summary = pd.DataFrame(summary_rows)
    sum_path = PROCESSED / "universe_membership_cmc_summary_v0_1.csv"
    summary.to_csv(sum_path, index=False)

    mean_add = summary["n_added"].mean() if len(summary) else 0
    print(f"universe top-{args.top_n}: {len(mem):,} membership rows -> {mem_path}")
    print(f"  weeks={len(summary)}; mean weekly adds={mean_add:.2f} (v0 was ~0.09)")
    print(f"  -> {sum_path}")

    # Quick crisis-era check
    for label, start, end in (
        ("LUNA crash window", "2022-04-01", "2022-05-31"),
        ("FTX window", "2022-10-01", "2022-11-30"),
    ):
        w = mem[(mem["rebalance_date"] >= start) & (mem["rebalance_date"] <= end)]
        hit = sorted(set(w["symbol"].astype(str).str.upper()) & {"LUNA", "FTT", "LUNC"})
        print(f"  {label}: members include {hit or '(none yet)'}")

    print("done")


if __name__ == "__main__":
    main()
