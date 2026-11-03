"""Refill Monero prices after Binance delist using CoinGecko Demo (last ~365d).

Older XMR rows already in panel; this patches dates after last Binance day.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
PROCESSED = ROOT / "data" / "processed"
API_KEY = os.getenv("COINGECKO_API_KEY", "").strip()
BASE = "https://api.coingecko.com/api/v3"


def main() -> None:
    if not API_KEY:
        raise SystemExit("Need COINGECKO_API_KEY in .env")
    panel_path = PROCESSED / "price_mcap_panel_v1.csv"
    df = pd.read_csv(panel_path)
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    xmr = df[df["coin_id"] == "monero"].copy()
    last_binance = xmr["date"].max()
    print(f"panel XMR last date: {last_binance.date()} rows={len(xmr)}")

    url = f"{BASE}/coins/monero/market_chart"
    r = requests.get(
        url,
        params={"vs_currency": "usd", "days": "365", "x_cg_demo_api_key": API_KEY},
        headers={"Accept": "application/json", "x-cg-demo-api-key": API_KEY},
        timeout=120,
    )
    r.raise_for_status()
    time.sleep(1.2)
    data = r.json()
    prices = {pd.to_datetime(t, unit="ms").normalize(): float(p) for t, p in data.get("prices") or []}
    mcaps = {pd.to_datetime(t, unit="ms").normalize(): float(m) for t, m in data.get("market_caps") or [] if m is not None}
    vols = {pd.to_datetime(t, unit="ms").normalize(): float(v) for t, v in data.get("total_volumes") or []}

    # build daily from day after last_binance through sample end
    start = (last_binance + pd.Timedelta(days=1)).normalize()
    end = pd.Timestamp("2026-09-29")
    days = pd.date_range(start, end, freq="D")
    rows = []
    for d in days:
        d = pd.Timestamp(d).as_unit("ns")
        if d not in prices:
            continue
        rows.append(
            {
                "date": d,
                "coin_id": "monero",
                "open": prices[d],
                "high": prices[d],
                "low": prices[d],
                "close": prices[d],
                "volume": vols.get(d, pd.NA),
                "volume_usd": vols.get(d, pd.NA),
                "market_cap": mcaps.get(d, pd.NA),
                "market_cap_source": "coingecko_api" if d in mcaps else "missing_cmc",
                "price_source": "coingecko",
            }
        )
    if not rows:
        print("no new CG rows (maybe already covered / key limit)")
        print("done")
        return
    add = pd.DataFrame(rows)
    print(f"adding {len(add)} XMR rows {add['date'].min().date()} .. {add['date'].max().date()}")
    for c in df.columns:
        if c not in add.columns:
            add[c] = pd.NA
    add = add.reindex(columns=df.columns)
    out = pd.concat([df, add], ignore_index=True)
    out = out.sort_values(["coin_id", "date"]).drop_duplicates(["coin_id", "date"], keep="last")
    out.to_csv(panel_path, index=False)

    # update coin_id_map end_reason for monero
    mp = PROCESSED / "coin_id_map.csv"
    m = pd.read_csv(mp)
    m.loc[m["coin_id"] == "monero", "end_reason"] = "still_trading"
    m.loc[m["coin_id"] == "monero", "last_trade_date"] = str(out.loc[out["coin_id"] == "monero", "date"].max().date())
    m.loc[m["coin_id"] == "monero", "notes"] = "Binance truncated 2024-02-20; refilled via CoinGecko after that"
    m.to_csv(mp, index=False)
    print(f"updated {panel_path} and coin_id_map")
    print("done")


if __name__ == "__main__":
    main()
