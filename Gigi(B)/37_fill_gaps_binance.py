"""Hayden v2 priority #3 — fill missing member-week prices via Binance Vision (HK-friendly).

Uses https://data-api.binance.vision/api/v3/klines (same REST as api.binance.com,
works when the main host is blocked in HK).

Fills:
  1) AAVE — drop old panel coin_id=='aave' rows; fetch AAVEUSDT from 2020-10-15;
     optionally prepend LENDUSDT / 100 (1 AAVE = 100 LEND) before that date.
  2) BCH — splice BCCUSDT → BCHABCUSDT → BCHUSDT into coin_id bitcoin-cash.
  3) Early history for mapped coins via *BTC pairs × BTCUSDT when USDT pair
     starts late.

Does NOT invent mcap/volume features — after this, rerun 15 / 18 / 19 / 25.

Outputs:
  data/raw/binance_vision_gaps/  (per-pair cache CSVs)
  data/processed/price_mcap_panel_v1.csv  (updated in place + .bak)
  reports/binance_gap_fill_summary_v2.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "binance_vision_gaps"
REPORTS = ROOT / "reports"
API = "https://data-api.binance.vision/api/v3/klines"
SAMPLE_START = pd.Timestamp("2018-01-01")
SAMPLE_END = pd.Timestamp("2026-09-29")
AAVE_LIST_DATE = pd.Timestamp("2020-10-15")
LEND_TO_AAVE = 100.0  # 1 AAVE = 100 LEND

# coin_id -> ordered list of Binance pairs to try / splice (earliest first)
BCH_PAIRS = ["BCCUSDT", "BCHABCUSDT", "BCHUSDT"]

# Optional early-history BTC-quoted bases (symbol without quote). Expand on PC as needed.
BTC_PAIR_BASES = [
    "ETH", "XRP", "BNB", "ADA", "LTC", "NEO", "QTUM", "XLM", "IOTA", "EOS",
    "TRX", "ETC", "ICX", "NULS", "VET", "LINK", "ONT", "WAN", "BTG", "ZRX",
    "BAT", "ZEC", "DASH", "XMR", "OMG", "ZIL", "HOT", "FET", "CELR", "ATOM",
    "ONE", "FTM", "ALGO", "DOGE", "DUSK", "ANKR", "COS", "MTL", "TOMO", "PERL",
]

# Map Binance base → panel coin_id (extend as needed)
BASE_TO_COIN = {
    "ETH": "ethereum", "XRP": "ripple", "BNB": "binancecoin", "ADA": "cardano",
    "LTC": "litecoin", "XLM": "stellar", "EOS": "eos", "TRX": "tron",
    "ETC": "ethereum-classic", "VET": "vechain", "LINK": "chainlink",
    "ZEC": "zcash", "XMR": "monero", "ATOM": "cosmos", "FTM": "fantom",
    "ALGO": "algorand", "DOGE": "dogecoin", "AAVE": "aave", "BCH": "bitcoin-cash",
}


def fetch_klines(pair: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    rows, cursor = [], start_ms
    while cursor <= end_ms:
        r = requests.get(
            API,
            params={
                "symbol": pair,
                "interval": "1d",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=60,
        )
        if r.status_code == 400:
            return pd.DataFrame()
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
                    "volume": float(x[7]),  # quote volume (USDT or BTC)
                }
            )
        last = batch[-1][0]
        nxt = last + 86_400_000
        if nxt <= cursor:
            break
        cursor = nxt
        time.sleep(0.12)
        if len(batch) < 1000:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
    return df[(df["date"] >= SAMPLE_START) & (df["date"] <= SAMPLE_END)].reset_index(drop=True)


def cache_path(pair: str) -> Path:
    return RAW / f"{pair}.csv"


def load_or_fetch(pair: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cp = cache_path(pair)
    if cp.exists():
        df = pd.read_csv(cp, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    print(f"  FETCH {pair} {start.date()}..{end.date()}")
    df = fetch_klines(pair, start, end)
    if not df.empty:
        RAW.mkdir(parents=True, exist_ok=True)
        df.to_csv(cp, index=False)
        print(f"    rows={len(df)} -> {cp.name}")
    else:
        print(f"    empty / no pair")
    return df


def make_panel_rows(ohlcv: pd.DataFrame, coin_id: str, symbol: str, price_source: str) -> pd.DataFrame:
    out = ohlcv.copy()
    out["coin_id"] = coin_id
    out["symbol"] = symbol
    out["price_source"] = price_source
    out["volume_usd"] = out["volume"]  # for USDT pairs; BTC-converted rows overwrite later
    out["market_cap"] = pd.NA
    out["market_cap_source"] = "pending_script15"
    out["circulating_supply"] = pd.NA
    return out


def build_aave() -> pd.DataFrame:
    """AAVEUSDT from listing; optional LENDUSDT / 100 before that."""
    aave = load_or_fetch("AAVEUSDT", AAVE_LIST_DATE, SAMPLE_END)
    lend = load_or_fetch("LENDUSDT", SAMPLE_START, AAVE_LIST_DATE - pd.Timedelta(days=1))
    frames = []
    if not lend.empty:
        lend = lend.copy()
        for c in ("open", "high", "low", "close"):
            lend[c] = lend[c] / LEND_TO_AAVE
        # quote volume stays in USDT; leave as-is (not /100)
        frames.append(make_panel_rows(lend, "aave", "AAVE", "binance_lend_div100"))
        print(f"  AAVE from LEND/100: {len(lend)} rows")
    if not aave.empty:
        frames.append(make_panel_rows(aave, "aave", "AAVE", "binance"))
        print(f"  AAVE from AAVEUSDT: {len(aave)} rows")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    return out


def build_bch() -> pd.DataFrame:
    frames = []
    for pair in BCH_PAIRS:
        df = load_or_fetch(pair, SAMPLE_START, SAMPLE_END)
        if df.empty:
            continue
        frames.append(make_panel_rows(df, "bitcoin-cash", "BCH", f"binance_{pair.lower()}"))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Prefer later pairs on overlapping dates (BCHUSDT over BCHABC over BCC)
    out["_prio"] = out["price_source"].map(
        {f"binance_{p.lower()}": i for i, p in enumerate(BCH_PAIRS)}
    )
    out = out.sort_values(["date", "_prio"]).drop_duplicates("date", keep="last").drop(columns="_prio")
    print(f"  BCH spliced rows={len(out)}")
    return out


def build_btc_pair_fills(panel: pd.DataFrame) -> pd.DataFrame:
    btc = load_or_fetch("BTCUSDT", SAMPLE_START, SAMPLE_END)
    if btc.empty:
        print("  WARNING: BTCUSDT empty — skip BTC-pair fills")
        return pd.DataFrame()
    btc = btc.rename(columns={"close": "btc_close", "volume": "btc_vol"})[["date", "btc_close"]]
    added = []
    for base in BTC_PAIR_BASES:
        coin_id = BASE_TO_COIN.get(base)
        if not coin_id:
            continue
        existing = panel.loc[panel["coin_id"] == coin_id, "date"]
        exist_set = set(pd.to_datetime(existing).dt.normalize()) if len(existing) else set()
        pair = f"{base}BTC"
        raw = load_or_fetch(pair, SAMPLE_START, SAMPLE_END)
        if raw.empty:
            continue
        m = raw.merge(btc, on="date", how="inner")
        if m.empty:
            continue
        m = m.copy()
        for c in ("open", "high", "low", "close"):
            m[c] = m[c] * m["btc_close"]
        # volume is in BTC; convert quote approx to USDT
        m["volume"] = m["volume"] * m["btc_close"]
        m = m.drop(columns=["btc_close"])
        # only dates missing from panel for this coin
        m = m[~m["date"].isin(exist_set)]
        if m.empty:
            continue
        rows = make_panel_rows(m, coin_id, base, f"binance_{pair.lower()}_x_btcusdt")
        rows["volume_usd"] = rows["volume"]
        added.append(rows)
        print(f"  BTC-pair fill {base}: +{len(rows)} days")
    if not added:
        return pd.DataFrame()
    return pd.concat(added, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="fetch/print only; do not rewrite panel")
    ap.add_argument("--skip-btc-pairs", action="store_true")
    args = ap.parse_args()

    panel_path = PROCESSED / "price_mcap_panel_v1.csv"
    if not panel_path.exists():
        raise SystemExit(f"Missing {panel_path}. Open desktop / copy panel first.")

    REPORTS.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    before_rows, before_coins = len(panel), panel["coin_id"].nunique()
    print(f"panel before: rows={before_rows:,} coins={before_coins}")

    summary = []

    # --- AAVE ---
    old_aave = panel[panel["coin_id"] == "aave"]
    summary.append({"action": "aave_old_rows", "n": len(old_aave),
                    "min": str(old_aave["date"].min().date()) if len(old_aave) else "",
                    "max": str(old_aave["date"].max().date()) if len(old_aave) else ""})
    new_aave = build_aave()
    summary.append({"action": "aave_new_rows", "n": len(new_aave),
                    "min": str(new_aave["date"].min().date()) if len(new_aave) else "",
                    "max": str(new_aave["date"].max().date()) if len(new_aave) else ""})

    # --- BCH ---
    new_bch = build_bch()
    summary.append({"action": "bch_new_rows", "n": len(new_bch),
                    "min": str(new_bch["date"].min().date()) if len(new_bch) else "",
                    "max": str(new_bch["date"].max().date()) if len(new_bch) else ""})

    # --- BTC pairs ---
    btc_fills = pd.DataFrame()
    if not args.skip_btc_pairs:
        # work on panel-without-aave for missing-date logic; BCH handled separately
        tmp = panel[panel["coin_id"] != "aave"]
        btc_fills = build_btc_pair_fills(tmp)
        summary.append({"action": "btc_pair_fill_rows", "n": len(btc_fills), "min": "", "max": ""})

    if args.dry_run:
        pd.DataFrame(summary).to_csv(REPORTS / "binance_gap_fill_summary_v2.csv", index=False)
        print("dry-run only; wrote summary. Re-run without --dry-run to patch panel.")
        print(pd.DataFrame(summary).to_string(index=False))
        return

    bak = PROCESSED / "price_mcap_panel_v1_before_37.bak.csv"
    panel.to_csv(bak, index=False)
    print(f"backup -> {bak}")

    # drop old aave; drop overlapping BCH dates we will replace; append new
    out = panel[panel["coin_id"] != "aave"].copy()
    if not new_bch.empty:
        out = out[out["coin_id"] != "bitcoin-cash"]
    frames = [out]
    if not new_aave.empty:
        frames.append(new_aave)
    if not new_bch.empty:
        frames.append(new_bch)
    if not btc_fills.empty:
        frames.append(btc_fills)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    # align columns
    for c in panel.columns:
        if c not in merged.columns:
            merged[c] = pd.NA
    merged = merged.reindex(columns=list(dict.fromkeys(list(panel.columns) + list(merged.columns))))
    merged = merged.sort_values(["coin_id", "date"]).drop_duplicates(["coin_id", "date"], keep="last")
    # recompute ret_1d for touched coins
    for cid in ("aave", "bitcoin-cash"):
        m = merged["coin_id"] == cid
        if m.any():
            merged.loc[m, "ret_1d"] = merged.loc[m].groupby("coin_id")["close"].pct_change()

    merged.to_csv(panel_path, index=False)
    also = PROCESSED / "price_mcap_panel_v1_2.csv"
    merged.to_csv(also, index=False)
    summary.append({"action": "panel_after_rows", "n": len(merged), "min": "", "max": ""})
    summary.append({"action": "panel_after_coins", "n": merged["coin_id"].nunique(), "min": "", "max": ""})
    pd.DataFrame(summary).to_csv(REPORTS / "binance_gap_fill_summary_v2.csv", index=False)
    print(f"updated {panel_path}: rows={len(merged):,} coins={merged['coin_id'].nunique()}")
    print(f"copy -> {also}")
    print("NEXT: rerun scripts 15, 18, 19, 25 (see docs/RUN_ORDER_Hayden_v2.md)")
    print(pd.DataFrame(summary).to_string(index=False))
    print("done")


if __name__ == "__main__":
    main()
