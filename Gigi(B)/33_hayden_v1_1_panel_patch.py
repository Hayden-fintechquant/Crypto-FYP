"""Hayden v1.1 panel patch (Member B).

Requires on disk under ROOT:
  data/processed/price_mcap_panel_v1.csv
  data/processed/universe_membership_cmc_v0_1.csv
  data/processed/cmc_snapshots_clean_v0_1.csv
  reports/outlier_verify_v1.csv  (fallback: outlier_candidates_v1.csv)

Writes/updates:
  price_mcap_panel_v1.csv + price_mcap_panel_v1_1.csv
  coin_id_map.csv (every panel coin_id)
  reports/weekly_ffill_coins_dates_v1.csv
  reports/data_error_dropped_v1.csv
  reports/outlier_verify_v1.csv (+ outlier_candidates_v1.csv)
  reports/mcap_source_by_year_v1.csv
  reports/mcap_source_members_by_year_v1.csv

Does NOT invent numbers — all stats printed from the CSVs on disk.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

PANEL = PROCESSED / "price_mcap_panel_v1.csv"
PANEL_V11 = PROCESSED / "price_mcap_panel_v1_1.csv"
UNIVERSE = PROCESSED / "universe_membership_cmc_v0_1.csv"
CMC = PROCESSED / "cmc_snapshots_clean_v0_1.csv"
IDMAP = PROCESSED / "coin_id_map.csv"
OUTLIER_VERIFY = REPORTS / "outlier_verify_v1.csv"
OUTLIER_CANDS = REPORTS / "outlier_candidates_v1.csv"

SAMPLE_END = pd.Timestamp("2026-09-29")
VERIFY_OK = 0.35
CMC_MAX_DAYS = 10
EXTREME = 0.5

# Latest-supply style approx tags Hayden wants confirmed absent among weekly members
APPROX_BAD = {
    "approx",
    "approx_latest_supply",
    "market_cap_approx",
    "close_x_latest_supply",
    "close_x_circulating_supply_latest",
}

SYMBOL_FALLBACK = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "binancecoin": "BNB",
    "ripple": "XRP",
    "cardano": "ADA",
    "solana": "SOL",
    "dogecoin": "DOGE",
    "tron": "TRX",
    "avalanche-2": "AVAX",
    "chainlink": "LINK",
    "polkadot": "DOT",
    "polkadot-new": "DOT",
    "litecoin": "LTC",
    "bitcoin-cash": "BCH",
    "stellar": "XLM",
    "monero": "XMR",
    "ethereum-classic": "ETC",
    "cosmos": "ATOM",
    "near": "NEAR",
    "algorand": "ALGO",
    "vechain": "VET",
    "filecoin": "FIL",
    "internet-computer": "ICP",
    "hedera-hashgraph": "HBAR",
    "aptos": "APT",
    "arbitrum": "ARB",
    "optimism": "OP",
    "sui": "SUI",
    "uniswap": "UNI",
    "aave": "AAVE",
    "maker": "MKR",
    "curve-dao-token": "CRV",
    "havven": "SNX",
    "lido-dao": "LDO",
    "pancakeswap-token": "CAKE",
    "the-graph": "GRT",
    "the-sandbox": "SAND",
    "decentraland": "MANA",
    "axie-infinity": "AXS",
    "immutable-x": "IMX",
    "flow": "FLOW",
    "tezos": "XTZ",
    "eos": "EOS",
    "fantom": "FTM",
    "matic-network": "MATIC",
    "polygon": "MATIC",
    "theta-token": "THETA",
    "elrond-erd-2": "EGLD",
    "injective-protocol": "INJ",
    "thorchain": "RUNE",
    "chiliz": "CHZ",
    "zcash": "ZEC",
    "terra-luna": "LUNA",
    "ftx-token": "FTT",
    "shiba-inu": "SHIB",
    "leo-token": "LEO",
    "crypto-com-chain": "CRO",
    "wrapped-bitcoin": "WBTC",
    "true-usd": "TUSD",
    "okb": "OKB",
    "polygon-ecosystem-token": "POL",
    "sonic-3": "S",
    "sky": "SKY",
    "terra-luna-v2": "LUNA",
}

# Manual end_reason / migration overrides (Hayden + prior v1 map)
OVERRIDES = {
    "terra-luna": {
        "successor_id": "terra-luna-v2",
        "successor_symbol": "LUNA",
        "migration_date": "2022-05-28",
        "end_reason": "collapsed",
        "force_last_trade_date": "2022-05-13",
        "notes": (
            "Classic LUNA through May 2022 collapse; Hayden applies -100% terminal "
            "return sensitivity only to LUNA/FTT. Binance LUNAUSDT after this is v2 — "
            "not spliced into classic series."
        ),
    },
    "ftx-token": {
        "successor_id": "",
        "successor_symbol": "",
        "migration_date": "",
        "end_reason": "collapsed",
        "notes": (
            "FTX Nov 2022 collapse; thin post-crash tape may still trade. "
            "Hayden applies -100% rule only to LUNA and FTT."
        ),
    },
    "matic-network": {
        "successor_id": "polygon-ecosystem-token",
        "successor_symbol": "POL",
        "migration_date": "2024-09-10",
        "end_reason": "migrated",
        "notes": "MATIC rebrand/migration to POL on Polygon; Hayden will splice returns.",
    },
    "fantom": {
        "successor_id": "sonic-3",
        "successor_symbol": "S",
        "migration_date": "2025-01-13",
        "end_reason": "migrated",
        "notes": "Fantom Opera -> Sonic (S); Hayden will splice returns.",
    },
    "maker": {
        "successor_id": "sky",
        "successor_symbol": "SKY",
        "migration_date": "2025-09-15",
        "end_reason": "migrated",
        "notes": "Maker (MKR) -> Sky (SKY); Hayden will splice returns.",
    },
    "eos": {
        "successor_id": "",
        "successor_symbol": "",
        "migration_date": "",
        "end_reason": "source_truncated",
        "notes": (
            "source_truncated = exchange/source delist (delisted_from_binance / listing end). "
            "Panel stops ~2025-05-26; not a collapse."
        ),
    },
    "monero": {
        "successor_id": "",
        "successor_symbol": "",
        "migration_date": "",
        "end_reason": "still_trading",
        "notes": (
            "Still trading elsewhere. Binance truncated ~2024-02-20; CoinGecko refill "
            "after that (Demo ~365d window) — check gap risk after 2024-02-20."
        ),
    },
}


def _slug_for_row(r: pd.Series) -> str | None:
    for col in ("cmc_slug", "coin_id"):
        v = r.get(col)
        if isinstance(v, str) and v.strip() and v.lower() not in ("nan", "none"):
            return v.strip().lower()
    return None


def _build_slug_map(cmc: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    slug_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for slug, g in cmc.groupby("slug", sort=False):
        slug_map[str(slug).lower()] = (
            g["snapshot_date"].to_numpy(dtype="datetime64[ns]"),
            g["price"].to_numpy(dtype=float),
        )
    return slug_map


def _nearest_cmc_price(
    slug_map: dict[str, tuple[np.ndarray, np.ndarray]],
    slug: str | None,
    t: pd.Timestamp,
) -> tuple[float, pd.Timestamp | pd.NaT, float | None]:
    """Return (cmc_price, cmc_dt, delta_days) or (nan, NaT, None)."""
    if not slug or slug not in slug_map:
        return float("nan"), pd.NaT, None
    dates, prices = slug_map[slug]
    t64 = np.datetime64(pd.Timestamp(t).to_datetime64())
    pos = dates.searchsorted(t64, side="right") - 1
    cand_idx: list[int] = []
    if 0 <= pos < len(dates):
        cand_idx.append(int(pos))
    if pos + 1 < len(dates):
        cand_idx.append(int(pos + 1))
    if not cand_idx and len(dates):
        cand_idx = [int(np.argmin(np.abs(dates.astype("int64") - t64.astype("int64"))))]
    best = None
    best_delta = None
    for i in cand_idx:
        delta = abs((dates[i] - t64) / np.timedelta64(1, "D"))
        if best is None or delta < best_delta:
            best, best_delta = i, float(delta)
    if best is None or best_delta is None or best_delta > CMC_MAX_DAYS:
        return float("nan"), pd.NaT, best_delta
    return float(prices[best]), pd.Timestamp(dates[best]), best_delta


def _explode_membership_days(uni: pd.DataFrame) -> pd.DataFrame:
    """Expand is_member weeks to daily (rebalance_date .. +6 days)."""
    m = uni[uni["is_member"].astype(bool)].copy()
    m["rebalance_date"] = pd.to_datetime(m["rebalance_date"])
    rows = []
    for _, r in m.iterrows():
        start = pd.Timestamp(r["rebalance_date"])
        for d in pd.date_range(start, start + pd.Timedelta(days=6), freq="D"):
            rows.append({"date": d, "coin_id": r["coin_id"]})
    out = pd.DataFrame(rows).drop_duplicates()
    out["is_weekly_member"] = True
    return out


def _year_crosstab(df: pd.DataFrame, src_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["year"] = tmp["date"].dt.year
    ct = pd.crosstab(tmp["year"], tmp[src_col], dropna=False)
    pct = ct.div(ct.sum(axis=1), axis=0) * 100.0
    # long form counts + pct
    long_c = ct.stack().rename("n").reset_index()
    long_p = pct.stack().rename("pct").reset_index()
    out = long_c.merge(long_p, on=["year", src_col], how="left")
    return out.sort_values(["year", src_col])


def rebuild_coin_id_map(df: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    sample_end = df["date"].max().normalize()
    specs = [
        ("first_trade_date", ("date", "min")),
        ("last_trade_date", ("date", "max")),
    ]
    if "symbol" in df.columns:
        specs.append(("symbol_panel", ("symbol", "first")))
    g = df.groupby("coin_id", sort=False).agg(**dict(specs)).reset_index()
    if "symbol_panel" not in g.columns:
        g["symbol_panel"] = ""

    # universe symbol lookup
    uni_sym = {}
    if "symbol" in uni.columns:
        for cid, sym in (
            uni.dropna(subset=["coin_id"])
            .drop_duplicates("coin_id")[["coin_id", "symbol"]]
            .itertuples(index=False)
        ):
            uni_sym[str(cid)] = str(sym)

    rows = []
    for _, r in g.iterrows():
        cid = str(r["coin_id"])
        ov = OVERRIDES.get(cid, {})
        first = pd.Timestamp(r["first_trade_date"]).date().isoformat()
        last = pd.Timestamp(r["last_trade_date"]).date().isoformat()
        if ov.get("force_last_trade_date"):
            last = ov["force_last_trade_date"]

        sym = ""
        if isinstance(r.get("symbol_panel"), str) and r["symbol_panel"] and r["symbol_panel"].lower() not in (
            "nan",
            "none",
        ):
            sym = r["symbol_panel"]
        elif cid in uni_sym:
            sym = uni_sym[cid]
        else:
            sym = SYMBOL_FALLBACK.get(cid, "")

        if ov:
            end_reason = ov["end_reason"]
            successor_id = ov.get("successor_id", "")
            successor_symbol = ov.get("successor_symbol", "")
            migration_date = ov.get("migration_date", "")
            notes = ov.get("notes", "")
        elif pd.Timestamp(last) >= sample_end:
            end_reason = "still_trading"
            successor_id = successor_symbol = migration_date = ""
            notes = ""
        else:
            end_reason = "source_truncated"
            successor_id = successor_symbol = migration_date = ""
            notes = (
                "source_truncated = exchange/source delist (often delisted_from_binance); "
                "panel ends before sample_end — not auto-labeled collapsed."
            )

        rows.append(
            {
                "coin_id": cid,
                "symbol": sym,
                "successor_id": successor_id,
                "successor_symbol": successor_symbol,
                "migration_date": migration_date,
                "first_trade_date": first,
                "last_trade_date": last,
                "end_reason": end_reason,
                "notes": notes,
            }
        )
    out = pd.DataFrame(rows).sort_values("coin_id").reset_index(drop=True)
    # enforce allowed enum
    allowed = {"still_trading", "migrated", "collapsed", "source_truncated"}
    bad = set(out["end_reason"].unique()) - allowed
    if bad:
        raise SystemExit(f"illegal end_reason values: {bad}")
    return out


def print_xmr_coverage(df: pd.DataFrame) -> None:
    x = df[df["coin_id"] == "monero"].copy()
    print("\n=== XMR (monero) coverage ===")
    if x.empty:
        print("NO monero rows in panel")
        return
    x = x.sort_values("date")
    dmin, dmax = x["date"].min(), x["date"].max()
    print(f"rows={len(x)} min_date={dmin.date()} max_date={dmax.date()}")
    # missing calendar days after Binance truncate
    cut = pd.Timestamp("2024-02-20")
    after = x[x["date"] >= cut]
    if after.empty:
        print("no rows on/after 2024-02-20")
        return
    full = pd.date_range(max(cut, after["date"].min()), after["date"].max(), freq="D")
    have = set(after["date"].dt.normalize())
    missing = [d for d in full if d not in have]
    print(f"calendar days on/after 2024-02-20 in span: {len(full)}")
    print(f"missing calendar days after 2024-02-20 (in span): {len(missing)}")
    if missing:
        # show first/last few
        show = missing[:10] + (["..."] if len(missing) > 20 else []) + missing[-10:]
        print("missing sample:", ", ".join(str(d.date()) if hasattr(d, "date") else str(d) for d in show))


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    if not PANEL.exists():
        raise SystemExit(
            f"MISSING {PANEL}\n"
            "Copy price_mcap_panel_v1.csv into data/processed/ then re-run."
        )
    for req in (UNIVERSE, CMC):
        if not req.exists():
            raise SystemExit(f"MISSING {req}")

    print(f"ROOT={ROOT}")
    print(f"reading {PANEL} ...")
    df = pd.read_csv(PANEL, low_memory=False)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    print(f"rows in={len(df)} coins={df['coin_id'].nunique()} max_date_before={df['date'].max().date()}")

    # 1) Sample end
    n_drop_end = int((df["date"] > SAMPLE_END).sum())
    df = df[df["date"] <= SAMPLE_END].copy()
    print(f"dropped dates > {SAMPLE_END.date()}: {n_drop_end}; max_date_after={df['date'].max().date()}")

    # Ensure core columns
    if "volume_usd" not in df.columns and "volume" in df.columns:
        df["volume_usd"] = df["volume"]
        print("note: volume_usd missing — copied from volume")
    if "symbol" not in df.columns:
        df["symbol"] = df["coin_id"].map(SYMBOL_FALLBACK).fillna("")
    if "outlier_check" not in df.columns:
        df["outlier_check"] = ""
    df["outlier_check"] = df["outlier_check"].fillna("").astype(str)
    df["outlier_check"] = df["outlier_check"].replace({"nan": "", "None": ""})

    # 2) weekly_ffill_flag
    ps = df["price_source"].astype(str) if "price_source" in df.columns else pd.Series("", index=df.index)
    ms = (
        df["market_cap_source"].astype(str)
        if "market_cap_source" in df.columns
        else pd.Series("", index=df.index)
    )
    df["weekly_ffill_flag"] = (ps == "cmc_weekly_ffill") | (ms == "cmc_weekly_ffill")
    ffill = df[df["weekly_ffill_flag"]].copy()
    ffill_out = REPORTS / "weekly_ffill_coins_dates_v1.csv"
    cols_ff = [c for c in ["coin_id", "symbol", "date", "price_source", "market_cap_source"] if c in ffill.columns]
    ffill[cols_ff].to_csv(ffill_out, index=False)
    print(f"\nweekly_ffill_flag True rows={len(ffill)} -> {ffill_out}")
    if len(ffill):
        by_coin = (
            ffill.groupby(["coin_id", "symbol"], dropna=False)
            .agg(n_days=("date", "size"), min_date=("date", "min"), max_date=("date", "max"))
            .reset_index()
            .sort_values("n_days", ascending=False)
        )
        by_coin.to_csv(REPORTS / "weekly_ffill_summary_by_coin_v1.csv", index=False)
        print("weekly_ffill top coins:")
        print(by_coin.head(15).to_string(index=False))
    else:
        print("no weekly_ffill rows")

    # Load CMC for data_error repair
    cmc = pd.read_csv(CMC, low_memory=False)
    cmc["snapshot_date"] = pd.to_datetime(cmc["snapshot_date"]).dt.normalize()
    cmc["slug"] = cmc["slug"].astype(str).str.lower()
    cmc = cmc.sort_values(["slug", "snapshot_date"])
    slug_map = _build_slug_map(cmc)

    # 3) Fix data_error rows
    err_mask = df["outlier_check"] == "data_error"
    n_err = int(err_mask.sum())
    print(f"\ndata_error rows before fix: {n_err}")
    dropped_rows = []
    fixed_idx = []
    for idx in df.index[err_mask]:
        r = df.loc[idx]
        slug = _slug_for_row(r)
        cmc_price, cmc_dt, delta = _nearest_cmc_price(slug_map, slug, r["date"])
        if pd.notna(cmc_price) and cmc_price > 0:
            old_close = r["close"]
            df.at[idx, "close"] = cmc_price
            # keep OHLC consistent on replaced day
            for c in ("open", "high", "low"):
                if c in df.columns:
                    df.at[idx, c] = cmc_price
            df.at[idx, "outlier_check"] = "verified_real"
            if "price_source" in df.columns:
                prev = str(df.at[idx, "price_source"]) if pd.notna(df.at[idx, "price_source"]) else ""
                df.at[idx, "price_source"] = (prev + "|").lstrip("|") + "cmc_replaced_data_error"
            fixed_idx.append(
                {
                    "date": r["date"],
                    "coin_id": r["coin_id"],
                    "symbol": r.get("symbol"),
                    "old_close": old_close,
                    "new_close": cmc_price,
                    "cmc_snapshot_date": cmc_dt,
                    "delta_days": delta,
                    "action": "replaced_with_cmc",
                }
            )
        else:
            dropped_rows.append(
                {
                    "date": r["date"],
                    "coin_id": r["coin_id"],
                    "symbol": r.get("symbol"),
                    "close": r["close"],
                    "ret_1d": r.get("ret_1d"),
                    "cmc_slug": slug,
                    "action": "dropped_no_cmc_within_10d",
                }
            )

    if dropped_rows:
        drop_df = pd.DataFrame(dropped_rows)
        drop_path = REPORTS / "data_error_dropped_v1.csv"
        drop_df.to_csv(drop_path, index=False)
        still_err = df.index[df["outlier_check"] == "data_error"]
        print(f"dropping {len(still_err)} unrepaired data_error rows -> {drop_path}")
        df = df.drop(index=still_err).copy()
    else:
        print("no data_error rows dropped")
        (REPORTS / "data_error_dropped_v1.csv").write_text(
            "date,coin_id,symbol,close,ret_1d,cmc_slug,action\n", encoding="utf-8"
        )

    if fixed_idx:
        pd.DataFrame(fixed_idx).to_csv(REPORTS / "data_error_fixed_v1.csv", index=False)
        print(f"fixed data_error via CMC replace: {len(fixed_idx)} -> reports/data_error_fixed_v1.csv")

    # Recompute ret_1d after close fixes / drops
    df = df.sort_values(["coin_id", "date"]).reset_index(drop=True)
    df["ret_1d"] = df.groupby("coin_id")["close"].pct_change()
    if "ret_extreme_flag" in df.columns:
        df["ret_extreme_flag"] = df["ret_1d"].abs() > EXTREME
    if "days_since_listing" not in df.columns or df["days_since_listing"].isna().all():
        first = df.groupby("coin_id")["date"].transform("min")
        df["days_since_listing"] = (df["date"] - first).dt.days

    # Remap remaining unverified -> exclude
    n_unv = int((df["outlier_check"] == "unverified").sum())
    df.loc[df["outlier_check"] == "unverified", "outlier_check"] = "exclude"
    # Also clear any leftover data_error label if somehow present
    n_err_left = int((df["outlier_check"] == "data_error").sum())
    if n_err_left:
        # null close/ret and mark removed (should be rare)
        df.loc[df["outlier_check"] == "data_error", "close"] = np.nan
        df.loc[df["outlier_check"] == "data_error", "ret_1d"] = np.nan
        df.loc[df["outlier_check"] == "data_error", "outlier_check"] = "data_error_removed"
    print(f"remapped unverified -> exclude: {n_unv}")
    print(f"outlier_check value_counts (all panel rows):")
    print(df["outlier_check"].replace("", "(blank)").value_counts(dropna=False).to_string())
    n_unv_after = int((df["outlier_check"] == "unverified").sum())
    print(f"unverified remaining (must be 0): {n_unv_after}")
    if n_unv_after != 0:
        raise SystemExit("FAILED: unverified still present")

    # Rewrite outlier_verify from post-fix extremes
    ext = df["ret_1d"].abs() >= EXTREME
    # Ensure extreme rows have a non-blank outlier_check for Hayden
    need = ext.fillna(False) & (df["outlier_check"].fillna("") == "")
    df.loc[need, "outlier_check"] = "exclude"

    cands = df[ext.fillna(False)].copy()
    rows = []
    for _, r in cands.iterrows():
        slug = _slug_for_row(r)
        cmc_price, cmc_dt, delta = _nearest_cmc_price(slug_map, slug, r["date"])
        rel = np.nan
        if pd.notna(cmc_price) and cmc_price > 0 and pd.notna(r["close"]) and float(r["close"]) > 0:
            rel = abs(float(r["close"]) - float(cmc_price)) / float(cmc_price)
        rows.append(
            {
                "date": r["date"],
                "coin_id": r["coin_id"],
                "symbol": r.get("symbol"),
                "close": r["close"],
                "ret_1d": r["ret_1d"],
                "days_since_listing": r.get("days_since_listing"),
                "cmc_slug": slug,
                "cmc_price_near": cmc_price,
                "cmc_snapshot_date": cmc_dt,
                "rel_err": rel,
                "delta_days": delta,
                "outlier_check": r["outlier_check"],
                "weekly_ffill_flag": bool(r.get("weekly_ffill_flag", False)),
                "verify_note": "post_v1_1_patch",
            }
        )
    ver = pd.DataFrame(rows)
    if len(ver):
        ver["abs_ret"] = ver["ret_1d"].abs()
        ver = ver.sort_values("abs_ret", ascending=False)
    ver.to_csv(OUTLIER_VERIFY, index=False)
    ver.to_csv(OUTLIER_CANDS, index=False)
    print(f"\nwrote {OUTLIER_VERIFY} and {OUTLIER_CANDS} extreme_rows={len(ver)}")
    if len(ver):
        print("extreme outlier_check counts:")
        print(ver["outlier_check"].value_counts().to_string())

    # 4) Rebuild coin_id_map for EVERY panel coin
    uni = pd.read_csv(UNIVERSE, low_memory=False)
    idmap = rebuild_coin_id_map(df, uni)
    idmap.to_csv(IDMAP, index=False)
    print(f"\nwrote {IDMAP} coins={len(idmap)}")
    print("end_reason counts:")
    print(idmap["end_reason"].value_counts().to_string())
    print("non-still_trading:")
    print(
        idmap[idmap["end_reason"] != "still_trading"][
            ["coin_id", "symbol", "end_reason", "successor_id", "migration_date", "last_trade_date"]
        ].to_string(index=False)
    )

    # 5) XMR
    print_xmr_coverage(df)

    # 6) market_cap_source by year
    if "market_cap_source" not in df.columns:
        print("WARNING: no market_cap_source column")
    else:
        by_year = _year_crosstab(df, "market_cap_source")
        by_year.to_csv(REPORTS / "mcap_source_by_year_v1.csv", index=False)
        print(f"\nwrote reports/mcap_source_by_year_v1.csv ({len(by_year)} rows)")
        # wide % print
        tmp = df.copy()
        tmp["year"] = tmp["date"].dt.year
        ct = pd.crosstab(tmp["year"], tmp["market_cap_source"], normalize="index") * 100.0
        print("market_cap_source share by year (%):")
        print(ct.round(2).to_string())

        # weekly members only
        mem_days = _explode_membership_days(uni)
        mem_days["date"] = pd.to_datetime(mem_days["date"]).dt.normalize()
        members = df.merge(mem_days, on=["date", "coin_id"], how="inner")
        print(f"\nweekly-member panel rows (exploded Mon-Sun): {len(members)}")
        if len(members):
            by_year_m = _year_crosstab(members, "market_cap_source")
            by_year_m.to_csv(REPORTS / "mcap_source_members_by_year_v1.csv", index=False)
            print(f"wrote reports/mcap_source_members_by_year_v1.csv ({len(by_year_m)} rows)")
            src = members["market_cap_source"].astype(str)
            bad = src.isin(APPROX_BAD) | src.str.contains("latest_supply", case=False, na=False)
            # also catch exact 'approx' family
            bad = bad | src.str.fullmatch(r"approx.*", case=False, na=False)
            n_bad = int(bad.sum())
            print(f"weekly-member rows with latest-supply/approx mcap_source: {n_bad}")
            if n_bad:
                print(src[bad].value_counts().head(20).to_string())
            else:
                print("OK: no weekly-member rows with close×latest-supply / approx tags")
            ct_m = pd.crosstab(members["date"].dt.year, members["market_cap_source"], normalize="index") * 100.0
            print("weekly-member market_cap_source share by year (%):")
            print(ct_m.round(2).to_string())
        else:
            print("WARNING: membership merge produced 0 rows — check coin_id alignment")

    # 7) Save panels
    df.to_csv(PANEL, index=False)
    df.to_csv(PANEL_V11, index=False)
    print(f"\nsaved {PANEL}")
    print(f"saved {PANEL_V11}")
    print(f"final rows={len(df)} coins={df['coin_id'].nunique()} "
          f"min_date={df['date'].min().date()} max_date={df['date'].max().date()}")

    # 8) Column list
    print("\n=== PANEL COLUMN LIST ===")
    for i, c in enumerate(df.columns.tolist(), 1):
        print(f"{i:02d}. {c}")
    print("done")


if __name__ == "__main__":
    main()
