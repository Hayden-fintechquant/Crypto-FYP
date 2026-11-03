"""Fill high-priority missing coins using CMC weekly snapshot prices.

For coins with no Binance USDT: use Sunday CMC price/mcap, forward-fill close
to daily between Sundays (price_source=cmc_weekly_ffill). Volume left NA.

Fills ALL rows still listed in missing_coins.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SAMPLE_END = pd.Timestamp("2026-09-29")


def expand_weekly_to_daily(week: pd.DataFrame) -> pd.DataFrame:
    week = week.sort_values("snapshot_date").drop_duplicates("snapshot_date", keep="last")
    if week.empty:
        return pd.DataFrame()
    start = week["snapshot_date"].min()
    end = min(week["snapshot_date"].max() + pd.Timedelta(days=6), SAMPLE_END)
    idx = pd.date_range(start, end, freq="D")
    w = week.set_index("snapshot_date")[["price", "market_cap", "circulating_supply", "total_supply", "symbol", "slug"]]
    daily = w.reindex(idx)
    # ffill from Sunday
    daily[["price", "market_cap", "circulating_supply", "total_supply", "symbol", "slug"]] = daily[
        ["price", "market_cap", "circulating_supply", "total_supply", "symbol", "slug"]
    ].ffill()
    daily = daily.dropna(subset=["price"])
    daily = daily.reset_index().rename(columns={"index": "date", "price": "close", "slug": "cmc_slug"})
    on_sun = daily["date"].isin(set(week["snapshot_date"]))
    daily["open"] = daily["close"]
    daily["high"] = daily["close"]
    daily["low"] = daily["close"]
    daily["volume"] = np.nan
    daily["volume_usd"] = np.nan
    daily["market_cap_source"] = np.where(on_sun, "cmc_snapshot", "cmc_weekly_ffill")
    daily["price_source"] = np.where(on_sun, "cmc_snapshot", "cmc_weekly_ffill")
    daily["supply_source"] = "cmc_snapshot_ffill"
    return daily


def main() -> None:
    miss_path = PROCESSED / "missing_coins.csv"
    panel_path = PROCESSED / "price_mcap_panel_v1.csv"
    cmc_path = PROCESSED / "cmc_snapshots_clean_v0_1.csv"
    missing = pd.read_csv(miss_path)
    panel = pd.read_csv(panel_path, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    cmc = pd.read_csv(cmc_path)
    cmc["snapshot_date"] = pd.to_datetime(cmc["snapshot_date"]).astype("datetime64[ns]")

    pri = missing.sort_values(["best_rank", "n_weeks"], ascending=[True, False]).copy()
    print(f"all remaining missing to fill from CMC weekly: {len(pri)}")

    frames = []
    done_slugs = set()
    for _, r in pri.iterrows():
        slug = str(r["cmc_slug"])
        if slug in done_slugs:
            continue
        done_slugs.add(slug)
        sub = cmc[cmc["slug"].astype(str) == slug].copy()
        if sub.empty:
            print(f"  no CMC rows for {slug}")
            continue
        daily = expand_weekly_to_daily(sub)
        if daily.empty:
            continue
        daily["coin_id"] = slug
        frames.append(daily)
        print(f"  {r['symbol']} / {slug}: {len(daily)} days {daily['date'].min().date()} .. {daily['date'].max().date()}")

    if not frames:
        print("nothing filled")
        print("done")
        return

    new = pd.concat(frames, ignore_index=True)
    for c in panel.columns:
        if c not in new.columns:
            new[c] = pd.NA
    new = new.reindex(columns=list(dict.fromkeys(list(panel.columns) + list(new.columns))))
    panel = panel.reindex(columns=new.columns)

    ids = set(new["coin_id"].unique())
    panel = panel[~panel["coin_id"].isin(ids)]
    out = pd.concat([panel, new], ignore_index=True)
    out = out.sort_values(["coin_id", "date"]).drop_duplicates(["coin_id", "date"], keep="last")
    out.to_csv(panel_path, index=False)
    print(f"updated {panel_path}: coins={out['coin_id'].nunique()} rows={len(out):,} added={len(ids)}")

    # refresh missing: drop filled slugs
    left = missing[~missing["cmc_slug"].astype(str).isin(ids)].copy()
    left.to_csv(miss_path, index=False)
    print(f"remaining missing: {len(left)} -> {miss_path}")
    print("done")


if __name__ == "__main__":
    main()
