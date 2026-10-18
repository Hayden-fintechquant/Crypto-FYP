#!/usr/bin/env python3
"""審查 C 的回測: 用 B 的資料『獨立重算』C 的報酬、換手率與成本，看有沒有對上。
用法: python review_C.py --weights weights.csv --log weekly_log.csv --panel data/panel.csv --universe data/universe.csv
      [--cost_bps 15] [--cap 0.10] [--offset 1] [--delist last_price]"""
import argparse, sys
import numpy as np
import pandas as pd
from src.data_io import Market
from src.factors import forward_return_wide

R = []


def add(status, name, detail=""):
    R.append((status, name, detail))


def review(weights, log, panel, universe, cost_bps=15.0, cap=None, offset=1, delist="last_price"):
    R.clear()
    wneed, lneed = ["date", "coin_id", "weight", "allocator"], ["date", "allocator", "gross_ret", "cost", "net_ret", "turnover"]
    if any(c not in weights.columns for c in wneed) or any(c not in log.columns for c in lneed):
        add("FAIL", "欄位", f"weights 需要 {wneed}；weekly_log 需要 {lneed}")
        return R
    weights = weights.assign(date=pd.to_datetime(weights["date"]))
    log = log.assign(date=pd.to_datetime(log["date"]))
    universe = universe.assign(rebalance_date=pd.to_datetime(universe["rebalance_date"]), is_member=universe["is_member"].astype(int) > 0)
    panel = panel.assign(date=pd.to_datetime(panel["date"]))
    allocs = sorted(set(log["allocator"]))
    add("PASS", "找到的配置器", f"{allocs}")
    kw = {"equal": "等權重", "cap": "市值加權", "min": "最小變異", "risk": "風險平價", "mean": "均值變異", "black": "Black-Litterman", "btc": "BTC 買進持有"}
    missing = [v for k, v in kw.items() if not any(k in a.lower() for a in allocs)]
    add("WARN" if missing else "PASS", "六個配置器 + BTC 基準都有", f"可能缺少 {missing}" if missing else "齊全")
    nd = log.groupby("allocator")["date"].nunique()
    add("PASS" if nd.nunique() == 1 else "WARN", "各配置器的週數相同", f"{nd.to_dict()}")

    # 權重基本檢查
    s = weights.groupby(["allocator", "date"])["weight"].agg(["sum", "min", "max", "count"])
    badsum = s[(s["sum"] - 1).abs() > 1e-6]
    add("FAIL" if len(badsum) else "PASS", "每週權重加總 = 1", f"{len(badsum)} 個 (配置器, 週) 不等於 1")
    add("FAIL" if (s["min"] < -1e-9).any() else "PASS", "長倉 (沒有負權重)", f"最小權重 {s['min'].min():.4g}")
    mem = universe[universe["is_member"]].groupby("rebalance_date")["coin_id"].apply(set)
    off_u = 0
    for (d), g in weights.groupby("date"):
        ms = mem.get(d, set())
        off_u += int((~g["coin_id"].isin(ms)).sum() - ((g["weight"].abs() < 1e-12) & ~g["coin_id"].isin(ms)).sum())
    add("FAIL" if off_u else "PASS", "只持有當週 universe 成員", f"{off_u} 筆持有非成員")
    if cap is not None:
        capok = True
        for a, g in weights.groupby("allocator"):
            if any(k in a.lower() for k in ("equal", "cap", "btc")):
                continue
            n = g.groupby("date")["weight"].apply(lambda x: (x > 1e-9).sum())
            mx = g.groupby("date")["weight"].max()
            allowed = np.maximum(cap, 1.0 / n.clip(lower=1)) + 1e-6
            if (mx > allowed).any():
                capok = False
                add("FAIL", f"權重上限 {cap:.0%} ({a})", f"{int((mx > allowed).sum())} 週超過上限")
        if capok:
            add("PASS", "權重上限 (最佳化配置器)", f"上限 {cap:.0%} (N 很小時放寬到 1/N)")
    for a, g in weights.groupby("allocator"):
        if "equal" in a.lower():
            sd = g[g["weight"] > 1e-12].groupby("date")["weight"].std().fillna(0)
            add("PASS" if (sd < 1e-9).all() else "FAIL", f"等權重確實等權 ({a})", f"最大離散 {sd.max():.2g}")
        if "cap" in a.lower() and "mcap" in universe.columns:
            cs = []
            for d, gg in g.groupby("date"):
                u = universe[(universe["rebalance_date"] == d) & universe["is_member"]].set_index("coin_id")["mcap"]
                x = gg.set_index("coin_id")["weight"][lambda z: z > 1e-12]
                if len(x) > 3:
                    cs.append(np.corrcoef(x.values, u.reindex(x.index).values)[0, 1])
            add("PASS" if cs and np.nanmedian(cs) > 0.99 else "FAIL", f"市值加權與 mcap 成正比 ({a})", f"相關係數中位數 {np.nanmedian(cs):.4f}" if cs else "無法檢查")

    # 獨立重算報酬、換手率、成本
    m = Market(panel)
    fwd, _ = forward_return_wide(m, 7, delist)
    off = pd.Timedelta(days=offset)
    gross_bad, turn_bad, cost_bad, net_bad, missing_px = 0, 0, 0, 0, 0
    ratios = []
    for a in allocs:
        wa = weights[weights["allocator"] == a].pivot_table(index="date", columns="coin_id", values="weight", aggfunc="sum").fillna(0.0).sort_index()
        la = log[log["allocator"] == a].set_index("date").sort_index()
        prev_w, prev_r, prev_g = None, None, None
        for d in wa.index:
            c = d - off
            if c not in fwd.index:
                continue
            r = fwd.loc[c].reindex(wa.columns)
            w = wa.loc[d]
            held = w.abs() > 1e-12
            if r[held].isna().any():
                missing_px += int(r[held].isna().sum())
            gross = float((w[held] * r[held].fillna(0)).sum())
            if d in la.index:
                if abs(gross - la.at[d, "gross_ret"]) > 1e-6:
                    gross_bad += 1
                drift = (prev_w * (1 + prev_r.fillna(0)) / (1 + prev_g)) if prev_w is not None else pd.Series(0.0, index=wa.columns)
                turn = float((w - drift.reindex(w.index).fillna(0)).abs().sum())
                lt = float(la.at[d, "turnover"])
                if abs(turn - lt) > 1e-6:
                    turn_bad += 1
                    if turn > 1e-9:
                        ratios.append(lt / turn)
                if abs(la.at[d, "cost"] - lt * cost_bps / 1e4) > 1e-9:
                    cost_bad += 1
                if abs(la.at[d, "net_ret"] - (la.at[d, "gross_ret"] - la.at[d, "cost"])) > 1e-9:
                    net_bad += 1
            prev_w, prev_r, prev_g = w, r, gross
    add("FAIL" if missing_px else "PASS", "持有的幣在 panel 都有價格", f"{missing_px} 筆缺價格 (報酬被當成 0)")
    add("FAIL" if gross_bad else "PASS", "毛報酬 = Σ權重 × 下週報酬 (獨立重算)", f"{gross_bad} 個 (配置器, 週) 對不上")
    half = len(ratios) > 0 and abs(np.median(ratios) - 0.5) < 0.02
    add("WARN" if (turn_bad and half) else "FAIL" if turn_bad else "PASS", "換手率 = Σ|新權重 - 漂移後舊權重| (獨立重算)",
        f"{turn_bad} 個對不上" + ("；C 的定義似乎是單邊 (÷2)，請確認全組統一用 Σ|Δw|" if half else ""))
    add("FAIL" if cost_bad else "PASS", f"成本 = 換手率 × {cost_bps:g} bps", f"{cost_bad} 個對不上")
    add("FAIL" if net_bad else "PASS", "淨報酬 = 毛報酬 - 成本", f"{net_bad} 個對不上")
    first = log.sort_values("date").groupby("allocator").first()
    fw = weights.sort_values("date").groupby("allocator").apply(lambda g: g[g["date"] == g["date"].min()]["weight"].abs().sum(), include_groups=False)
    add("PASS" if ((first["turnover"] - fw).abs() < 1e-6).all() else "WARN", "第一週換手率 = 100% (從現金建倉)", f"{first['turnover'].round(3).to_dict()}")
    # 合理性
    sh = log.groupby("allocator")["net_ret"].apply(lambda x: x.mean() / x.std(ddof=1) * np.sqrt(52) if x.std() > 0 else np.nan)
    hi = sh[sh > 3]
    add("WARN" if len(hi) else "PASS", "夏普沒有好到不真實 (> 3 要懷疑前視)", f"{sh.round(2).to_dict()}")
    add("FAIL" if (log["net_ret"] < -1).any() else "PASS", "週淨報酬 > -100%", f"最小 {log['net_ret'].min():.2%}")
    return R


def main():
    ap = argparse.ArgumentParser()
    for k in ("weights", "log", "panel", "universe"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--cost_bps", type=float, default=15.0)
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--offset", type=int, default=1)
    ap.add_argument("--delist", default="last_price")
    a = ap.parse_args()
    res = review(pd.read_csv(a.weights), pd.read_csv(a.log), pd.read_csv(a.panel), pd.read_csv(a.universe),
                 a.cost_bps, a.cap, a.offset, a.delist)
    for s, n, d in res:
        print(f"[{s}] {n}" + (f" — {d}" if d else ""))
    c = {k: sum(1 for r in res if r[0] == k) for k in ("PASS", "WARN", "FAIL")}
    print(f"\n總結: PASS {c['PASS']} / WARN {c['WARN']} / FAIL {c['FAIL']}")
    sys.exit(1 if c["FAIL"] else 0)


if __name__ == "__main__":
    main()
