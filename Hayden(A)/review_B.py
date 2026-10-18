#!/usr/bin/env python3
"""審查 B 的資料: python review_B.py --panel data/panel.csv --universe data/universe.csv [--end 2026-09-29]
每一項輸出 PASS / WARN / FAIL。有任何 FAIL 時結束碼為 1。"""
import argparse, sys
import numpy as np
import pandas as pd

R = []


def add(status, name, detail=""):
    R.append((status, name, detail))


STABLE = {"tether", "usd-coin", "dai", "binance-usd", "true-usd", "first-digital-usd", "usdd", "paxos-standard", "usds"}
STABLE_SYM = {"usdt", "usdc", "dai", "busd", "tusd", "fdusd", "usdd", "usdp", "usds"}
WRAPPED = {"wrapped-bitcoin", "weth", "wrapped-steth", "staked-ether", "wrapped-eeth", "coinbase-wrapped-btc"}
WRAPPED_SYM = {"wbtc", "weth", "steth", "wsteth", "cbbtc", "weeth"}


def review(panel: pd.DataFrame, universe: pd.DataFrame, end: str = "2026-09-29", top_n=None):
    R.clear()
    # 1. 欄位
    need = ["date", "coin_id", "close", "volume_usd", "mcap"]
    rec = ["circulating_supply", "total_supply", "supply_source", "mcap_source", "price_source", "days_since_listing",
           "outlier_check", "retrieved_at", "symbol"]
    miss = [c for c in need if c not in panel.columns]
    if miss:
        add("FAIL", "panel 必要欄位", f"缺少 {miss}")
        return R
    add("PASS", "panel 必要欄位", "date, coin_id, close, volume_usd, mcap 都有")
    if any(c not in universe.columns for c in ["rebalance_date", "coin_id", "is_member"]):
        add("FAIL", "universe 必要欄位", "需要 rebalance_date, coin_id, is_member")
        return R
    m_rec = [c for c in rec if c not in panel.columns]
    add("WARN" if m_rec else "PASS", "panel 建議欄位 (問題 2、6、8、9)", f"缺少 {m_rec}" if m_rec else "齊全")

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    universe = universe.assign(rebalance_date=pd.to_datetime(universe["rebalance_date"]),
                               is_member=universe["is_member"].astype(int) > 0)
    # 2. 重複與價格
    dup = int(panel.duplicated(["date", "coin_id"]).sum())
    add("FAIL" if dup else "PASS", "(date, coin_id) 不重複", f"{dup} 列重複")
    bad = int((panel["close"].isna() | (panel["close"] <= 0)).sum())
    add("FAIL" if bad else "PASS", "收盤價為正數", f"{bad} 列 NaN 或 <= 0")
    # 3. 最後日期
    last = panel["date"].max()
    add("FAIL" if last > pd.Timestamp(end) else "PASS", f"最後日期 <= {end} (不含未收盤日)", f"最後日期 {last.date()}")
    if "retrieved_at" in panel.columns:
        ra = pd.to_datetime(panel["retrieved_at"], errors="coerce", utc=True).dt.tz_localize(None)
        leak = int((panel["date"] >= ra.dt.normalize()).sum())
        add("FAIL" if leak else "PASS", "抓取時間晚於該日收盤 (沒有盤中價)", f"{leak} 列的日期 >= 抓取日")
    # 4. 缺日
    g = panel.groupby("coin_id")["date"].agg(["min", "max", "count"])
    gaps = int(((g["max"] - g["min"]).dt.days + 1 - g["count"]).sum())
    add("WARN" if gaps else "PASS", "各幣沒有缺日", f"共缺 {gaps} 天")
    # 5. 市值
    mc = panel["mcap"]
    nonpos = int((mc.isna() | (mc <= 0)).sum())
    allzero = panel.groupby("coin_id")["mcap"].apply(lambda s: bool((s.fillna(0) <= 0).all()))
    az = list(allzero[allzero].index)
    add("FAIL" if az else "PASS", "沒有『整段歷史市值都是 0』的幣 (問題 1)", f"{az}" if az else f"無 (另有 {nonpos} 列市值<=0)")
    if "mcap_source" in panel.columns:
        share = float(panel["mcap_source"].astype(str).str.contains("approx", case=False).mean())
        add("FAIL" if share > 0.5 else "WARN" if share > 0.10 else "PASS", "近似市值比例 (問題 2，目標 < 10%)", f"{share:.1%}")
    else:
        add("WARN", "近似市值比例 (問題 2)", "沒有 mcap_source 欄位，無法檢查")
    if "circulating_supply" in panel.columns:
        d = panel.dropna(subset=["circulating_supply"])
        d = d[(d["circulating_supply"] > 0) & (d["mcap"] > 0)]
        err = (d["mcap"] / (d["close"] * d["circulating_supply"]) - 1).abs().median() if len(d) else np.nan
        add("PASS" if err < 0.01 else "WARN", "mcap ≈ 價格 × 流通供給量", f"中位數相對誤差 {err:.2%}")
    # 6. universe
    wd = universe["rebalance_date"].dt.day_name().value_counts()
    add("PASS" if len(wd) == 1 else "FAIL", "再平衡日是同一個星期幾", f"{ {k: int(v) for k, v in wd.items()} }")
    n = universe[universe["is_member"]].groupby("rebalance_date")["coin_id"].nunique()
    first30 = n[n >= 30].index.min()
    after = int((n[n.index >= first30] < 30).sum()) if pd.notna(first30) else len(n)
    add("PASS" if pd.notna(first30) else "FAIL", "universe 達到 30 個幣的第一週", f"{first30.date() if pd.notna(first30) else '從未達到'}；之後仍少於 30 的週數: {after}")
    mem = universe[universe["is_member"]].groupby("rebalance_date")["coin_id"].apply(set).sort_index()
    add_, drop_ = [], []
    prev = None
    for s in mem:
        if prev is not None:
            add_.append(len(s - prev)); drop_.append(len(prev - s))
        prev = s
    ma, md = (np.mean(add_), np.mean(drop_)) if add_ else (np.nan, np.nan)
    add("WARN" if (md < 0.02 and ma < 0.15) else "PASS", "universe 每週有進有出 (問題 3)", f"平均每週新增 {ma:.2f}、剔除 {md:.2f}")
    if "mcap" in universe.columns:
        badm = int((universe["is_member"] & (universe["mcap"].isna() | (universe["mcap"] <= 0))).sum())
        add("FAIL" if badm else "PASS", "universe 成員市值都 > 0", f"{badm} 列")
    if {"rank", "mcap"}.issubset(universe.columns):
        ok = universe.sort_values(["rebalance_date", "rank"]).groupby("rebalance_date")["mcap"].apply(lambda s: s.is_monotonic_decreasing)
        add("PASS" if ok.mean() > 0.99 else "FAIL", "rank 與 mcap 順序一致", f"{ok.mean():.1%} 的週一致")
        mx = universe[universe["is_member"]].groupby("rebalance_date")["rank"].max()
        add("PASS" if mx.nunique() <= 2 else "WARN", "成員門檻 (top-N) 固定", f"最大 rank 的範圍 {mx.min()}~{mx.max()}")
    nm = set(panel["coin_id"]) - set(universe.loc[universe["is_member"], "coin_id"])
    add("WARN" if len(nm) > 3 else "PASS", "panel 中從未進過 universe 的幣", f"{len(nm)} 個: {sorted(nm)[:6]}")
    # 7. 倖存者偏差
    ends = panel.groupby("coin_id")["date"].max()
    dead = ends[ends < last - pd.Timedelta(days=30)]
    add("WARN" if len(dead) == 0 else "PASS", "有中途停止的幣 (遷移、下架或崩盤)", f"{len(dead)} 個")
    names = (panel["coin_id"].astype(str) + " " + (panel["symbol"].astype(str) if "symbol" in panel.columns else "")).str.lower()
    has_collapse = {k: bool(names.str.contains(k).any()) for k in ["luna", "ftt", "ftx"]}
    add("PASS" if any(has_collapse.values()) else "FAIL", "包含已崩盤的幣 (LUNA / FTT，問題 3)", f"{has_collapse}")
    if "end_reason" in panel.columns or "last_trade_date" in panel.columns:
        add("PASS", "end_reason / last_trade_date (問題 9)", "panel 內有")
    # 8. 穩定幣與包裝代幣
    memb = set(universe.loc[universe["is_member"], "coin_id"].astype(str))
    sym = panel.drop_duplicates("coin_id").set_index("coin_id")["symbol"].astype(str).str.lower().to_dict() if "symbol" in panel.columns else {}
    bad_ids = [c for c in memb if c.lower() in STABLE | WRAPPED or sym.get(c, "") in STABLE_SYM | WRAPPED_SYM]
    add("FAIL" if bad_ids else "PASS", "成員中沒有穩定幣與包裝代幣", f"{bad_ids}" if bad_ids else "無")
    # 9. 極端值與停滯價格
    p = panel.sort_values(["coin_id", "date"])
    cons = p.groupby("coin_id")["date"].diff().dt.days == 1
    ret = (p["close"] / p.groupby("coin_id")["close"].shift(1) - 1).where(cons)
    ext = p[ret.abs() > 0.5]
    if "outlier_check" in panel.columns:
        un = int(ext["outlier_check"].isna().sum() + (ext["outlier_check"].astype(str).str.lower() == "unverified").sum())
        add("FAIL" if un else "PASS", "所有單日 >50% 變動都已核對 (問題 6)", f"{len(ext)} 筆極端值，其中 {un} 筆未標記或 unverified")
    else:
        add("WARN", "極端值核對 (問題 6)", f"{len(ext)} 筆單日 >50% 變動，沒有 outlier_check 欄位")
    same = (p.groupby("coin_id")["close"].diff() == 0)
    run3 = same & same.groupby(p["coin_id"]).shift(1, fill_value=False)
    add("WARN" if int(run3.sum()) else "PASS", "沒有停滯價格 (連續 3 天收盤相同)", f"{int(run3.sum())} 列")
    zv = int((panel["volume_usd"].fillna(0) <= 0).sum())
    add("WARN" if zv else "PASS", "沒有零成交量", f"{zv} 列")
    if "ret_1d" in panel.columns:
        d = (p["ret_1d"] - ret).abs().dropna()
        add("PASS" if (d.max() < 1e-6 if len(d) else True) else "FAIL", "ret_1d 與收盤價一致 (同一個幣內)", f"最大誤差 {d.max():.2e}" if len(d) else "無可比列")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--universe", required=True)
    ap.add_argument("--end", default="2026-09-29")
    a = ap.parse_args()
    res = review(pd.read_csv(a.panel), pd.read_csv(a.universe), a.end)
    for s, n, d in res:
        print(f"[{s}] {n}" + (f" — {d}" if d else ""))
    c = {k: sum(1 for r in res if r[0] == k) for k in ("PASS", "WARN", "FAIL")}
    print(f"\n總結: PASS {c['PASS']} / WARN {c['WARN']} / FAIL {c['FAIL']}")
    sys.exit(1 if c["FAIL"] else 0)


if __name__ == "__main__":
    main()
