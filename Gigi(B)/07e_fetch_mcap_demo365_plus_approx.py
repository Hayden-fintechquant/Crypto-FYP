"""Day 9/24 workaround: CoinGecko Demo only allows ~365d historical market_cap.

Strategy (honest for midterm, no paid plan):
1) Pull last 365 days TRUE market_cap via market_chart(days=365)
2) Pull latest circulating_supply via /coins/markets
3) For older dates: market_cap_approx = close * latest_circulating_supply
4) Final column market_cap prefers true values; fills approx where missing
5) market_cap_source = api | approx

Report to A/C: pre-365d mcap is approximation until paid historical access.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

RAW_DIR = ROOT / "data" / "raw" / "mcap"
PROCESSED = ROOT / "data" / "processed"
BASE = "https://api.coingecko.com/api/v3"
API_KEY = os.getenv("COINGECKO_API_KEY", "").strip()
SLEEP_S = float(os.getenv("COINGECKO_SLEEP_S", "1.2"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def auth_params():
    return {"x_cg_demo_api_key": API_KEY}


def auth_headers():
    return {"Accept": "application/json", "x-cg-demo-api-key": API_KEY}


def get_json(path: str, params: dict) -> dict | list:
    url = f"{BASE}{path}"
    p = {**params, **auth_params()}
    for attempt in range(6):
        r = requests.get(url, params=p, headers=auth_headers(), timeout=120)
        if r.status_code == 429:
            wait = SLEEP_S * (attempt + 3)
            print(f"  429 — sleep {wait:.0f}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        time.sleep(SLEEP_S)
        return r.json()
    raise RuntimeError(f"failed GET {path}")


def load_coin_ids() -> list[str]:
    for name in ("ohlcv_panel_v1.csv", "ohlcv_panel_v0.csv"):
        path = PROCESSED / name
        if path.exists():
            ids = sorted(pd.read_csv(path, usecols=["coin_id"])["coin_id"].dropna().unique())
            print(f"coin list from {path.name}: {len(ids)} coins")
            return list(ids)
    raise SystemExit("No ohlcv panel found.")


def fetch_true_mcap_365(coin_id: str) -> pd.DataFrame:
    chart = get_json(
        f"/coins/{coin_id}/market_chart",
        {"vs_currency": "usd", "days": "365"},
    )
    rows = []
    retrieved = now_iso()
    for t, m in chart.get("market_caps") or []:
        if m is None:
            continue
        rows.append(
            {
                "date": pd.to_datetime(t, unit="ms", utc=True).date().isoformat(),
                "coin_id": coin_id,
                "market_cap_api": float(m),
                "retrieval_timestamp": retrieved,
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )


def fetch_latest_supply(coin_ids: list[str]) -> pd.DataFrame:
    """Batch latest circulating_supply + market_cap from /coins/markets."""
    out = []
    # API allows ids comma-separated; keep batches small
    for i in range(0, len(coin_ids), 20):
        batch = coin_ids[i : i + 20]
        data = get_json(
            "/coins/markets",
            {
                "vs_currency": "usd",
                "ids": ",".join(batch),
                "order": "market_cap_desc",
                "per_page": 50,
                "page": 1,
                "sparkline": "false",
            },
        )
        for row in data:
            out.append(
                {
                    "coin_id": row["id"],
                    "circulating_supply_latest": row.get("circulating_supply"),
                    "market_cap_latest": row.get("market_cap"),
                }
            )
    return pd.DataFrame(out)


def main() -> None:
    if not API_KEY:
        raise SystemExit("Missing COINGECKO_API_KEY in .env")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    ping = get_json("/ping", {})
    print("ping ok", ping)

    coin_ids = load_coin_ids()
    print("fetching latest circulating_supply ...")
    supply = fetch_latest_supply(coin_ids)
    supply.to_csv(PROCESSED / "latest_supply_snapshot.csv", index=False)

    frames, ok, bad = [], [], []
    for i, cid in enumerate(coin_ids, 1):
        print(f"[{i}/{len(coin_ids)}] true mcap last 365d: {cid} ...")
        try:
            df = fetch_true_mcap_365(cid)
        except Exception as e:
            print(f"  fail {cid}: {e}")
            bad.append(cid)
            continue
        if df.empty:
            bad.append(cid)
            continue
        path = RAW_DIR / f"{cid}_mcap_365d.csv"
        df.to_csv(path, index=False)
        print(f"  -> {path.name} rows={len(df)}")
        frames.append(df)
        ok.append(cid)

    mcap365 = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not mcap365.empty:
        mcap365.to_csv(PROCESSED / "mcap_panel_365d.csv", index=False)

    ohlcv_path = PROCESSED / "ohlcv_panel_v1.csv"
    if not ohlcv_path.exists():
        ohlcv_path = PROCESSED / "ohlcv_panel_v0.csv"
    ohlcv = pd.read_csv(ohlcv_path)

    merged = ohlcv.merge(supply, on="coin_id", how="left")
    if not mcap365.empty:
        merged = merged.merge(
            mcap365[["date", "coin_id", "market_cap_api"]],
            on=["date", "coin_id"],
            how="left",
        )
    else:
        merged["market_cap_api"] = pd.NA

    merged["market_cap_approx"] = merged["close"] * merged["circulating_supply_latest"]
    merged["market_cap"] = merged["market_cap_api"].where(
        merged["market_cap_api"].notna(), merged["market_cap_approx"]
    )
    merged["market_cap_source"] = "api"
    merged.loc[merged["market_cap_api"].isna() & merged["market_cap"].notna(), "market_cap_source"] = "approx"
    merged["circulating_supply"] = merged["circulating_supply_latest"]

    out = PROCESSED / "price_mcap_panel_v0.csv"
    merged.to_csv(out, index=False)

    api_share = float((merged["market_cap_source"] == "api").mean())
    approx_share = float((merged["market_cap_source"] == "approx").mean())
    fill = float(merged["market_cap"].notna().mean())
    print(f"\nwrote {out}")
    print(f"mcap_fill={fill:.1%}  api_share={api_share:.1%}  approx_share={approx_share:.1%}")
    print(f"ok_365d={len(ok)} failed_365d={len(bad)}")
    if bad:
        print("failed:", ", ".join(bad))
    print("NOTE: dates older than ~365d use approx = close * latest circulating_supply.")


if __name__ == "__main__":
    main()
