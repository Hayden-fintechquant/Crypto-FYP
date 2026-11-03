"""Fetch Binance daily OHLCV for CMC-universe coins still missing from panel_v1.

- Reads missing_coins.csv
- Probes SYMBOLUSDT (with aliases)
- Attaches CMC mcap by slug
- Appends into price_mcap_panel_v1.csv
- Rewrites missing_coins.csv for leftovers (no Binance / failed)
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "binance_missing_v1"
API = "https://api.binance.com/api/v3/klines"
SAMPLE_END = pd.Timestamp("2026-09-29")

# Binance symbol aliases when CMC ticker ≠ Binance pair base
ALIASES = {
    "MIOTA": ["IOTAUSDT"],
    "IOTA": ["IOTAUSDT"],
    "BTT": ["BTTCUSDT", "BTTUSDT"],
    "POL": ["POLUSDT"],
    "TON": ["TONUSDT"],
    "SHIB": ["SHIBUSDT"],
    "NEO": ["NEOUSDT"],
    "XEM": ["XEMUSDT"],
    "DASH": ["DASHUSDT"],
    "KLAY": ["KLAYUSDT"],
    "TIA": ["TIAUSDT"],
    "PEPE": ["PEPEUSDT"],
    "ACA": ["ACAUSDT"],
    "LSK": ["LSKUSDT"],
    "HYPE": ["HYPEUSDT"],
    "SYS": ["SYSUSDT"],
    "STO": ["STOUSDT"],
}


def probe_pair(symbol: str) -> str | None:
    cands = ALIASES.get(symbol.upper(), []) + [f"{symbol.upper()}USDT"]
    seen = []
    for p in cands:
        if p in seen:
            continue
        seen.append(p)
        r = requests.get(API, params={"symbol": p, "interval": "1d", "limit": 1}, timeout=30)
        if r.status_code == 200 and r.json():
            return p
        time.sleep(0.15)
    return None


def fetch_klines(pair: str, end: pd.Timestamp = SAMPLE_END) -> pd.DataFrame:
    start_ms = int(pd.Timestamp("2018-01-01", tz="UTC").timestamp() * 1000)
    end_ms = int(end.tz_localize("UTC").timestamp() * 1000)
    rows, cursor = [], start_ms
    while cursor <= end_ms:
        r = requests.get(
            API,
            params={"symbol": pair, "interval": "1d", "startTime": cursor, "endTime": end_ms, "limit": 1000},
            timeout=60,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for x in batch:
            rows.append(
                {
                    "date": pd.to_datetime(x[0], unit="ms").as_unit("ns").normalize(),
                    "open": float(x[1]),
                    "high": float(x[2]),
                    "low": float(x[3]),
                    "close": float(x[4]),
                    "volume": float(x[7]),
                }
            )
        last = batch[-1][0]
        nxt = last + 86_400_000
        if nxt <= cursor:
            break
        cursor = nxt
        time.sleep(0.2)
        if len(batch) < 1000:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
    return df[df["date"] <= end.normalize()].reset_index(drop=True)


def attach_cmc(ohlcv: pd.DataFrame, slug: str, cmc: pd.DataFrame) -> pd.DataFrame:
    sub = cmc[cmc["slug"].astype(str).str.lower() == str(slug).lower()].copy()
    if sub.empty:
        # fallback by symbol match later
        ohlcv = ohlcv.copy()
        ohlcv["market_cap"] = pd.NA
        ohlcv["circulating_supply"] = pd.NA
        ohlcv["market_cap_source"] = "missing_cmc"
        return ohlcv
    sub["snapshot_date"] = pd.to_datetime(sub["snapshot_date"]).astype("datetime64[ns]")
    sub = sub.sort_values(["snapshot_date", "market_cap"], ascending=[True, False]).drop_duplicates(
        "snapshot_date", keep="first"
    )
    left = ohlcv.copy()
    left["date"] = pd.to_datetime(left["date"]).astype("datetime64[ns]")
    m = pd.merge_asof(
        left.sort_values("date"),
        sub[["snapshot_date", "market_cap", "circulating_supply"]].rename(
            columns={"market_cap": "cmc_mcap", "circulating_supply": "cmc_circ"}
        ),
        left_on="date",
        right_on="snapshot_date",
        direction="backward",
    )
    on = m["date"] == m["snapshot_date"]
    close = pd.to_numeric(m["close"], errors="coerce")
    circ = pd.to_numeric(m["cmc_circ"], errors="coerce")
    snap = pd.to_numeric(m["cmc_mcap"], errors="coerce")
    m["market_cap"] = snap.where(on, close * circ)
    m["circulating_supply"] = circ
    m["market_cap_source"] = "cmc_snapshot"
    m.loc[~on & circ.notna(), "market_cap_source"] = "cmc_supply_x_close"
    m.loc[m["market_cap"].isna(), "market_cap_source"] = "missing_cmc"
    return m.drop(columns=[c for c in ("snapshot_date", "cmc_mcap", "cmc_circ") if c in m.columns])


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    missing = pd.read_csv(PROCESSED / "missing_coins.csv")
    panel = pd.read_csv(PROCESSED / "price_mcap_panel_v1.csv")
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    cmc = pd.read_csv(PROCESSED / "cmc_snapshots_clean_v0_1.csv")
    cmc["snapshot_date"] = pd.to_datetime(cmc["snapshot_date"])

    # prioritize: best_rank then n_weeks
    missing = missing.sort_values(["best_rank", "n_weeks"], ascending=[True, False])
    # skip junk / spam ranks that are clearly not investable aliases if weeks tiny and rank odd — still try Binance

    added = []
    still = []
    for _, row in missing.iterrows():
        sym = str(row["symbol"]).upper()
        slug = str(row["cmc_slug"])
        pair = probe_pair(sym)
        if not pair:
            still.append({**row.to_dict(), "reason": "no_binance_usdt_pair"})
            print(f"SKIP {sym} ({slug}): no Binance USDT")
            continue
        print(f"FETCH {sym} -> {pair} ...")
        try:
            ohlcv = fetch_klines(pair)
        except Exception as e:
            still.append({**row.to_dict(), "reason": f"fetch_fail:{type(e).__name__}"})
            print(f"  FAIL {e}")
            continue
        if ohlcv.empty:
            still.append({**row.to_dict(), "reason": "empty_klines"})
            print("  empty")
            continue
        ohlcv.to_csv(RAW / f"{pair}.csv", index=False)
        coin_id = slug  # use CMC slug as stable id for new coins
        ohlcv["coin_id"] = coin_id
        ohlcv["symbol"] = sym
        ohlcv["price_source"] = "binance"
        ohlcv["volume_usd"] = ohlcv["volume"]
        ohlcv["avg_volume_28d"] = ohlcv["volume_usd"].rolling(28, min_periods=1).mean()
        ohlcv = attach_cmc(ohlcv, slug, cmc)
        print(f"  rows={len(ohlcv)} {ohlcv['date'].min().date()} .. {ohlcv['date'].max().date()}")
        added.append(ohlcv)

    if added:
        new = pd.concat(added, ignore_index=True)
        # align columns
        for c in panel.columns:
            if c not in new.columns:
                new[c] = pd.NA
        for c in new.columns:
            if c not in panel.columns:
                panel[c] = pd.NA
        cols = list(dict.fromkeys(list(panel.columns) + list(new.columns)))
        panel = panel.reindex(columns=cols)
        new = new.reindex(columns=cols)
        # drop overlapping coin_ids if re-run
        ids = set(new["coin_id"].unique())
        panel = panel[~panel["coin_id"].isin(ids)]
        out = pd.concat([panel, new], ignore_index=True)
        out = out.sort_values(["coin_id", "date"]).drop_duplicates(["coin_id", "date"], keep="last")
        out_path = PROCESSED / "price_mcap_panel_v1.csv"
        out.to_csv(out_path, index=False)
        print(f"updated {out_path}: coins={out['coin_id'].nunique()} rows={len(out):,} added_coins={len(ids)}")
    else:
        print("no new coins added")

    still_df = pd.DataFrame(still)
    still_path = PROCESSED / "missing_coins.csv"
    still_df.to_csv(still_path, index=False)
    print(f"remaining missing: {len(still_df)} -> {still_path}")
    if len(still_df):
        print(still_df.sort_values("best_rank")[["symbol", "cmc_slug", "best_rank", "n_weeks", "reason"]].head(15).to_string(index=False))
    print("done")


if __name__ == "__main__":
    main()
