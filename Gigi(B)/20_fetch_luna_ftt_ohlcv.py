"""v1 step: fetch daily OHLCV for crash coins LUNA (classic) + FTT from Binance.

- LUNAUSDT: keep 2020-08-21 .. 2022-05-13 only (classic terra-luna through collapse).
  Later LUNAUSDT is terra-luna-v2 — excluded here.
- FTTUSDT: full available history through sample end 2026-09-29.
- Attach point-in-time CMC mcap/supply from cmc_snapshots_clean_v0_1.csv
  (slug terra-luna / ftx-token).
- Output: data/processed/crash_coins_panel_v1.csv
  (append into full panel later; this is the handoff for A/C crisis windows.)
"""
from __future__ import annotations

import time
 
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "binance_crash_coins"
API = "https://api.binance.com/api/v3/klines"
SAMPLE_END = pd.Timestamp("2026-09-29")

COINS = [
    {
        "symbol": "LUNAUSDT",
        "coin_id": "terra-luna",
        "ticker": "LUNA",
        "cmc_slugs": ["terra-luna"],
        "end": pd.Timestamp("2022-05-13"),  # classic collapse last day on Binance
        "end_reason": "collapsed",
    },
    {
        "symbol": "FTTUSDT",
        "coin_id": "ftx-token",
        "ticker": "FTT",
        "cmc_slugs": ["ftx-token"],
        "end": SAMPLE_END,
        "end_reason": "collapsed",  # economic collapse Nov 2022; thin tape may continue
    },
]


def fetch_klines(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_ms = int(start.tz_localize("UTC").timestamp() * 1000) if start.tzinfo is None else int(start.timestamp() * 1000)
    end_ms = int(end.tz_localize("UTC").timestamp() * 1000) if end.tzinfo is None else int(end.timestamp() * 1000)
    rows = []
    cursor = start_ms
    while cursor <= end_ms:
        r = requests.get(
            API,
            params={"symbol": symbol, "interval": "1d", "startTime": cursor, "endTime": end_ms, "limit": 1000},
            timeout=60,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for x in batch:
            rows.append(
                {
                    "date": pd.to_datetime(x[0], unit="ms").normalize(),
                    "open": float(x[1]),
                    "high": float(x[2]),
                    "low": float(x[3]),
                    "close": float(x[4]),
                    "volume": float(x[7]),  # quote USDT
                }
            )
        last = batch[-1][0]
        nxt = last + 86_400_000
        if nxt <= cursor:
            break
        cursor = nxt
        time.sleep(0.25)
        if len(batch) < 1000:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
    df = df[(df["date"] >= start.normalize()) & (df["date"] <= end.normalize())]
    return df.reset_index(drop=True)


def attach_cmc(ohlcv: pd.DataFrame, slugs: list[str], cmc: pd.DataFrame) -> pd.DataFrame:
    sub = cmc[cmc["slug"].astype(str).str.lower().isin({s.lower() for s in slugs})].copy()
    if sub.empty:
        ohlcv["market_cap"] = pd.NA
        ohlcv["circulating_supply"] = pd.NA
        ohlcv["market_cap_source"] = "missing_cmc"
        return ohlcv
    sub = sub.sort_values(["snapshot_date", "market_cap"], ascending=[True, False])
    sub = sub.drop_duplicates("snapshot_date", keep="first")
    sub = sub.rename(
        columns={
            "market_cap": "cmc_mcap",
            "circulating_supply": "cmc_circ",
        }
    )
    left = ohlcv.sort_values("date").copy()
    left["date"] = pd.to_datetime(left["date"]).astype("datetime64[ns]")
    right = sub[["snapshot_date", "cmc_mcap", "cmc_circ"]].copy()
    right["snapshot_date"] = pd.to_datetime(right["snapshot_date"]).astype("datetime64[ns]")
    m = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="snapshot_date",
        direction="backward",
    )
    on_snap = m["date"] == m["snapshot_date"]
    close = pd.to_numeric(m["close"], errors="coerce")
    circ = pd.to_numeric(m["cmc_circ"], errors="coerce")
    snap = pd.to_numeric(m["cmc_mcap"], errors="coerce")
    m["market_cap"] = snap.where(on_snap, close * circ)
    m["circulating_supply"] = circ
    m["market_cap_source"] = "cmc_snapshot"
    m.loc[~on_snap & circ.notna(), "market_cap_source"] = "cmc_supply_x_close"
    m.loc[m["market_cap"].isna(), "market_cap_source"] = "missing_cmc"
    return m.drop(columns=[c for c in ("snapshot_date", "cmc_mcap", "cmc_circ") if c in m.columns])


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    cmc_path = PROCESSED / "cmc_snapshots_clean_v0_1.csv"
    if not cmc_path.exists():
        raise SystemExit(f"Missing {cmc_path}")
    cmc = pd.read_csv(cmc_path)
    cmc["snapshot_date"] = pd.to_datetime(cmc["snapshot_date"]).dt.normalize()

    frames = []
    for spec in COINS:
        print(f"fetch {spec['symbol']} as {spec['coin_id']} ...")
        # start far back; Binance returns first available
        raw = fetch_klines(spec["symbol"], pd.Timestamp("2018-01-01"), spec["end"])
        raw_path = RAW / f"{spec['symbol']}.csv"
        raw.to_csv(raw_path, index=False)
        print(f"  klines {len(raw)} rows {raw['date'].min().date() if len(raw) else '-'} .. {raw['date'].max().date() if len(raw) else '-'} -> {raw_path}")
        if raw.empty:
            continue
        raw["coin_id"] = spec["coin_id"]
        raw["symbol"] = spec["ticker"]
        raw["price_source"] = "binance"
        raw["volume_usd"] = raw["volume"]
        raw["end_reason"] = spec["end_reason"]
        attached = attach_cmc(raw, spec["cmc_slugs"], cmc)
        frames.append(attached)
        # crash window peek
        if spec["ticker"] == "LUNA":
            w = attached[(attached["date"] >= "2022-05-08") & (attached["date"] <= "2022-05-13")]
            print("  LUNA crash closes:")
            print(w[["date", "close", "market_cap", "market_cap_source"]].to_string(index=False))
        if spec["ticker"] == "FTT":
            w = attached[(attached["date"] >= "2022-11-06") & (attached["date"] <= "2022-11-12")]
            print("  FTT crash closes:")
            print(w[["date", "close", "market_cap", "market_cap_source"]].to_string(index=False))

    out = pd.concat(frames, ignore_index=True)
    out["avg_volume_28d"] = (
        out.groupby("coin_id")["volume_usd"].transform(lambda s: s.rolling(28, min_periods=1).mean())
    )
    out_path = PROCESSED / "crash_coins_panel_v1.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} rows={len(out)} coins={out['coin_id'].nunique()}")
    print(out.groupby("coin_id").agg(n=("date", "size"), start=("date", "min"), end=("date", "max")).to_string())
    print("done")


if __name__ == "__main__":
    main()
