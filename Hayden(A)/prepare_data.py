"""prepare_data.py  --  adapter between Student B's data layer and the research pipeline.

    python prepare_data.py PANEL.csv UNIVERSE.csv [OUT_DIR=data] [--keep-unverified]

Reads  : B's price_mcap_panel (date, coin_id, close, volume_usd, market_cap, end_reason, outlier_check,
         weekly_ffill_flag, symbol, ...) and universe_membership (rebalance_date, coin_id[CMC slug],
         panel_coin_id, is_member, ...).
Writes : OUT_DIR/panel.csv, OUT_DIR/universe.csv, OUT_DIR/adapter_log.txt,
         OUT_DIR/weekly_coverage_report.csv, OUT_DIR/coverage_by_year.csv

What it does (every rule is also written to adapter_log.txt):
  1. drops rows B marked data_error / exclude, and unverified ones unless --keep-unverified (verified_real is kept);
  2. drops weekly forward-filled rows (weekly_ffill_flag): they are not daily data and would create
     artificial zero-return days;
  3. splices token migrations into ONE series expressed in the new token's units
        MKR -> SKY (1:24000), FTM -> S (1:1), MATIC -> POL (1:1);
  4. maps universe CMC slugs to panel ids (B's panel_coin_id + manual fixes for ids B could not map);
  5. removes non-risk assets from the universe: stablecoins, wrapped assets and gold tokens (PAXG);
  6. writes the weekly coverage report (planning-report deliverable).
"""
import sys
import numpy as np
import pandas as pd

# old panel id -> (new panel id, units of OLD token per 1 NEW token)
# Rows before these dates belong to a pre-migration series in different units that B's panel does not splice
# (AAVE: LEND era, price ~0.005 USD vs ~43 USD after the 2020-10 migration, market cap also corrupted -> an 8,359x "return").
DROP_BEFORE = {"aave": "2020-10-15"}

SPLICES = {
    "maker": ("sky", 24000.0),
    "fantom": ("sonic", 1.0),
    "matic-network": ("polygon-ecosystem-token", 1.0),
}
# CMC slug -> panel id, for universe members B's panel_coin_id left empty
MANUAL_MAP = {
    "polkadot-new": "polkadot", "hedera": "hedera-hashgraph", "pancakeswap": "pancakeswap-token",
    "near-protocol": "near", "optimism-ethereum": "optimism", "injective": "injective-protocol",
    "polygon": "matic-network",
}
# panel ids that are not risk assets for this study (group rule: stablecoins, wrapped assets; PAXG decided: exclude)
EXCLUDE_IDS = {"usd1", "global-dollar-usdg", "united-stables", "pax-gold"}
EXCLUDE_SYMBOLS = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "USDE", "USDS", "PYUSD", "WBTC", "WETH",
                   "STETH", "WSTETH", "WEETH", "CBBTC", "PAXG", "XAUT", "USDD", "USD1", "USDG", "SUSDE"}


def main(panel_path, univ_path, out="data", keep_unverified=False):
    log = []
    p = pd.read_csv(panel_path, parse_dates=["date"], low_memory=False)
    n0 = len(p)
    log.append(f"input panel: {n0} rows, {p.coin_id.nunique()} coins, {p.date.min().date()} .. {p.date.max().date()}")

    # 1. data-error rows
    # B's final panel labels flagged rows verified_real / unverified / data_error (older panels: "exclude").
    # Default: keep verified_real, drop data_error / exclude / unverified (conservative; decided before looking at results).
    # --keep-unverified keeps the unverified rows (sensitivity run).
    lab = p["outlier_check"].astype(str)
    drop_labels = {"exclude", "data_error"} | (set() if keep_unverified else {"unverified"})
    bad = lab.isin(drop_labels)
    log.append("outlier_check counts: " + str(lab[lab.isin({"exclude", "data_error", "unverified", "verified_real"})].value_counts().to_dict()))
    p = p[~bad]
    log.append(f"dropped outlier_check in {sorted(drop_labels)} rows: {int(bad.sum())} (keep_unverified={keep_unverified})")

    # 2. weekly forward-filled rows
    ff = p["weekly_ffill_flag"].astype(str).str.lower().eq("true")
    p = p[~ff]
    log.append(f"dropped weekly_ffill rows (not daily data): {int(ff.sum())}")

    for cid, d0 in DROP_BEFORE.items():
        m = (p.coin_id == cid) & (p.date < pd.Timestamp(d0))
        p = p[~m]
        log.append(f"dropped {cid} rows before {d0} (pre-migration series in different units): {int(m.sum())}")

    p = p.rename(columns={"market_cap": "mcap"})
    keep = [c for c in ["date", "coin_id", "close", "volume_usd", "mcap", "end_reason", "symbol"] if c in p.columns]
    p = p[keep].copy()
    p = p[p["close"] > 0]

    # 3. splice migrations
    for old, (new, ratio) in SPLICES.items():
        a, b = p[p.coin_id == old].copy(), p[p.coin_id == new].copy()
        if a.empty or b.empty:
            log.append(f"splice {old}->{new}: SKIPPED (missing series)")
            continue
        a["close"] = a["close"] / ratio                      # express old prices in new-token units
        a["coin_id"] = new
        last_old, first_new = a.sort_values("date").iloc[-1], b.sort_values("date").iloc[0]
        gap = (first_new["date"] - last_old["date"]).days
        bridge = first_new["close"] / last_old["close"] - 1
        log.append(f"splice {old}->{new} ratio {ratio:g}: old ends {last_old['date'].date()}, new starts "
                   f"{first_new['date'].date()} (gap {gap}d), bridge return {bridge:+.1%}"
                   + ("  <-- CHECK ratio" if abs(bridge) > 0.35 else ""))
        p = pd.concat([p[~p.coin_id.isin([old, new])], a, b], ignore_index=True)
    p = p.sort_values(["coin_id", "date"])
    dup = int(p.duplicated(["date", "coin_id"]).sum())
    assert dup == 0, f"duplicate (date,coin) after splice: {dup}"
    p.to_csv(f"{out}/panel.csv", index=False)
    log.append(f"output panel: {len(p)} rows, {p.coin_id.nunique()} coins")
    # guard: a >10x daily move that survived all cleaning is a unit/splice error -> stop (never silently pass)
    _q = p.sort_values(["coin_id", "date"]).copy()
    _q["_r"] = _q.groupby("coin_id")["close"].pct_change()
    _bad = _q[_q["_r"] > 10]
    if len(_bad):
        raise SystemExit("STOP: daily return > 1000% survives cleaning (unit/splice error?):\n" + _bad[["date", "coin_id", "close", "_r"]].to_string())
    log.append("guard: no daily return > 1000% in the output panel")

    # 4./5. universe
    u = pd.read_csv(univ_path, parse_dates=["rebalance_date"])
    u0 = len(u)
    miss = u["panel_coin_id"].isna()
    u.loc[miss, "panel_coin_id"] = u.loc[miss, "coin_id"].map(MANUAL_MAP)
    log.append(f"manual id map filled {int((miss & u['panel_coin_id'].notna()).sum())} of {int(miss.sum())} unmapped member-rows")
    still = u[u["panel_coin_id"].isna()]
    log.append(f"still unmapped (dropped): {len(still)} rows, coins {sorted(still.coin_id.unique())}")
    u = u.dropna(subset=["panel_coin_id"]).copy()
    sym = u["symbol"].astype(str).str.upper() if "symbol" in u.columns else pd.Series("", index=u.index)
    ex = u["panel_coin_id"].isin(EXCLUDE_IDS) | sym.isin(EXCLUDE_SYMBOLS)
    log.append(f"excluded non-risk assets from universe: {int(ex.sum())} member-rows, coins {sorted(u.loc[ex, 'panel_coin_id'].unique())}")
    u = u[~ex]
    sp = {old: new for old, (new, _) in SPLICES.items()}
    u["panel_coin_id"] = u["panel_coin_id"].replace(sp)   # same canonical id as the spliced series
    u = u.rename(columns={"coin_id": "cmc_slug", "panel_coin_id": "coin_id"})
    u = u.drop_duplicates(["rebalance_date", "coin_id"])
    u[["rebalance_date", "coin_id", "is_member"]].to_csv(f"{out}/universe.csv", index=False)
    log.append(f"output universe: {len(u)} member-rows (from {u0}), {u.rebalance_date.nunique()} weeks")

    # 6. weekly coverage report: price + positive volume on the day before each rebalance date
    mem = u[u["is_member"].astype(int) > 0]
    pv = p[(p["close"] > 0) & (p["volume_usd"] > 0)]
    pk, pxk = set(zip(pv["date"], pv["coin_id"])), set(zip(p["date"], p["coin_id"]))
    t1 = mem["rebalance_date"] - pd.Timedelta(days=1)
    mem = mem.assign(has_px=[(d, c) in pxk for d, c in zip(t1, mem["coin_id"])],
                     has_px_vol=[(d, c) in pk for d, c in zip(t1, mem["coin_id"])])
    wk = mem.groupby("rebalance_date").agg(n_members=("coin_id", "size"), n_with_price=("has_px", "sum"),
                                           n_with_price_and_volume=("has_px_vol", "sum")).reset_index()
    wk["pct"] = (wk["n_with_price_and_volume"] / wk["n_members"] * 100).round(1)
    wk.to_csv(f"{out}/weekly_coverage_report.csv", index=False)
    wk["year"] = wk["rebalance_date"].dt.year
    by = wk.groupby("year").agg(weeks=("pct", "size"), mean_members=("n_members", "mean"),
                                mean_priced=("n_with_price_and_volume", "mean"),
                                min_priced=("n_with_price_and_volume", "min")).round(1)
    by.to_csv(f"{out}/coverage_by_year.csv")
    s = wk[wk["rebalance_date"] >= "2020-09-21"]
    log.append(f"coverage (price+volume): all weeks {wk.n_with_price_and_volume.sum() / wk.n_members.sum():.1%}; "
               f"since 2020-09-21 {s.n_with_price_and_volume.sum() / s.n_members.sum():.1%}; "
               f"weeks>=30 priced since 2020-09-21: {(s.n_with_price_and_volume >= 30).sum()}/{len(s)}")
    open(f"{out}/adapter_log.txt", "w", encoding="utf-8").write("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0], args[1], args[2] if len(args) > 2 else "data", keep_unverified="--keep-unverified" in sys.argv)
