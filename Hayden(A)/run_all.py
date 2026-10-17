#!/usr/bin/env python3
"""Student A 一條龍: 資料 -> 因子 -> 篩選 -> IC/分組 -> Fama-MacBeth -> 訊號組合 -> BL 先驗 -> 稽核。

用法:
  python run_all.py --synthetic     # 用假資料測試整條流程(隨時可跑)
  python run_all.py --lock          # 填好 config.yaml 後，鎖定預先登記 (之後不可再改門檻)
  python run_all.py --dev           # 用真實資料測試程式 (結果放 results_dev/，不算正式結果)
  python run_all.py                 # 正式執行 (必須先 --lock，且 config 沒被改過)
"""
import argparse, os, sys, shutil
import yaml
import pandas as pd

from src.data_io import load_panel, load_universe, Market, to_long
from src.pipeline import run_pipeline
from src.audit import lookahead_audit
from src.bl import weekly_prior
from src import prereg

CFG = "config.yaml"
LOCK = "prereg.lock"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--lock", action="store_true")
    ap.add_argument("--unblind", action="store_true", help="(不建議) --dev 時也輸出真實的因子績效表")
    ap.add_argument("--audit-checks", type=int, default=4)
    a = ap.parse_args()

    if a.lock:
        info = prereg.lock(CFG, LOCK)
        print(f"已鎖定預先登記。sha256={info['sha256'][:16]}...  時間(UTC)={info['locked_at_utc']}")
        print("提醒: 鎖定之後如果還要改 config.yaml，必須重新 --lock，並在報告中說明原因。")
        return

    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    if a.synthetic:
        from src.synth import make_synthetic
        mode, out = "SYNTHETIC", "results_synthetic"
        panel, universe = make_synthetic()
        os.makedirs("data_synth", exist_ok=True)
        panel.to_csv("data_synth/panel.csv", index=False)
        universe.to_csv("data_synth/universe.csv", index=False)
        cfg["sample"]["start"], cfg["sample"]["end"] = "2019-07-01", "2024-12-23"
        panel, universe = load_panel("data_synth/panel.csv"), load_universe("data_synth/universe.csv")
        lock_info = None
    else:
        if a.dev:
            mode, out, lock_info = "DEV", "results_dev", None
        else:
            ok, msg, lock_info = prereg.verify(CFG, LOCK)
            print(msg)
            if not ok:
                sys.exit(1)
            mode, out = "OFFICIAL", "results"
        for key in ("panel_path", "universe_path"):
            if not os.path.exists(cfg["data"][key]):
                print(f"找不到 {cfg['data'][key]}。請把 B 的檔案放進 data/ 資料夾 (檔名見 config.yaml 的 data 區塊)。")
                sys.exit(1)
        panel, universe = load_panel(cfg["data"]["panel_path"]), load_universe(cfg["data"]["universe_path"])
        if cfg["sample"]["end"] and panel["date"].max() > pd.Timestamp(cfg["sample"]["end"]):
            print(f"注意: panel 最後日期 {panel['date'].max().date()} 晚於 sample.end，之後的資料不會被使用。")

    blind = (mode == "DEV" and not a.unblind)   # 預先登記鎖定前，真實資料的因子績效表不輸出(盲測)
    os.makedirs(out, exist_ok=True)
    print(f"[{mode}] 執行中..." + ("  (盲測: 不輸出因子績效，只檢查程式與資料是否正常)" if blind else ""))
    res = run_pipeline(panel, universe, cfg)
    scr = res["screen"]

    # ---- 輸出 ----
    scr["coverage"].to_csv(f"{out}/coverage.csv", header=["coverage"], encoding="utf-8-sig")
    if scr["value_corr"] is not None:
        scr["value_corr"].to_csv(f"{out}/redundancy_factor_values.csv", encoding="utf-8-sig")
    sc = res["scores"].reset_index().melt(id_vars=["date", "coin_id"], var_name="factor", value_name="score")
    sc.to_csv(f"{out}/scores.csv", index=False)                      # 介面檔: 給 C
    prior = weekly_prior(res["market"], universe, cfg)
    prior.to_csv(f"{out}/bl_prior.csv", index=False)                 # 介面檔: 給 C
    if not blind:
        res["ic_tab"].to_csv(f"{out}/ic_table.csv", encoding="utf-8-sig")
        res["q_tab"].to_csv(f"{out}/quintile_table.csv", encoding="utf-8-sig")
        res["fm_uni"].to_csv(f"{out}/fm_univariate.csv", encoding="utf-8-sig")
        res["fm_multi"].to_csv(f"{out}/fm_multivariate.csv", encoding="utf-8-sig")
        res["span"].to_csv(f"{out}/fm_spanning.csv", encoding="utf-8-sig")
        res["mt"].to_csv(f"{out}/multiple_testing.csv", encoding="utf-8-sig")
        res["verdict"].to_csv(f"{out}/verdicts.csv", encoding="utf-8-sig")
        if scr["return_corr"] is not None:
            scr["return_corr"].to_csv(f"{out}/redundancy_factor_returns.csv", encoding="utf-8-sig")
        mu = pd.concat([s_.rename("mu_weekly").reset_index().assign(method=k) for k, s_ in res["mu"].items()])
        mu.dropna(subset=["mu_weekly"]).to_csv(f"{out}/mu.csv", index=False)   # 介面檔: 給 C

    # ---- 前視稽核 ----
    tab, passed, msg = lookahead_audit(panel, universe, cfg, res, n_checks=a.audit_checks)
    tab.to_csv(f"{out}/lookahead_audit.csv", index=False)

    # ---- 摘要 ----
    pd.set_option("display.width", 200, "display.float_format", lambda x: f"{x:,.4f}")
    L = []
    L.append(f"模式: {mode}    樣本: {cfg['sample']['start']} ~ {cfg['sample']['end']}    有效週數: {scr['n_weeks']}")
    if lock_info:
        L.append(f"預先登記雜湊: {lock_info['sha256'][:16]}...  鎖定時間(UTC): {lock_info['locked_at_utc']}")
    L.append(f"候選因子: {cfg['factors']}")
    L.append(f"因子覆蓋率不足被淘汰: {scr['dropped_coverage']}")
    L.append(f"因子冗餘被淘汰: {scr['dropped_redundant']}")
    L.append(f"留下的因子: {res['factors']}")
    L.append(f"宣告的檢定總數 (多重檢定修正使用): {res['n_specs']}")
    if not blind:
        L.append("\n[IC / ICIR]\n" + res["ic_tab"][["ic_mean", "icir", "ic_t_nw", "ic_hit", "n_weeks"]].to_string())
        L.append("\n[五分位 (Q5-Q1 為多空價差；單調性 1 = 完全單調)]\n" +
                 res["q_tab"][["q1_mean", "q5_mean", "ls_mean", "ls_t_nw", "monotonicity"]].to_string())
        L.append("\n[Fama-MacBeth 多變量 (週報酬單位)]\n" + res["fm_multi"][["mean", "t", "p"]].to_string())
        L.append("\n[Spanning: 控制 " + str(cfg["validation"]["spanning_controls"]) + "]\n" + res["span"][["mean", "t", "p"]].to_string())
        L.append("\n[事前門檻下的判定]\n" + res["verdict"].to_string())
    else:
        L.append("\n(盲測模式: 因子績效表未輸出。這是為了維持預先登記的原則: 門檻鎖定之前，不看真實的因子結果。)")
    L.append(f"\n[前視稽核] {msg}\n" + (tab.to_string(index=False) if len(tab) else ""))
    txt = "\n".join(L)
    open(f"{out}/summary.txt", "w", encoding="utf-8").write(txt)
    shutil.copy(CFG, f"{out}/config_used.yaml")
    print(txt)
    print(f"\n全部結果已寫入 {out}/")
    if not passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
