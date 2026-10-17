#!/usr/bin/env python3
"""FYP II 一條龍: 預測層 -> Black-Litterman 整合 -> 風險模型 -> (拿到 C 的回測後) 評估。

用法:
  python run_fyp2.py all --synthetic            # 用假資料測試 forecast + bl + risk + evaluate 全流程
  python run_fyp2.py --lock                     # 鎖定 config_fyp2.yaml (預先登記)
  python run_fyp2.py all --dev                  # 真實資料測試程式 (盲測，不輸出樣本外績效)
  python run_fyp2.py all                        # 正式 (需 FYP I 與 FYP II 兩個鎖都通過)
  python run_fyp2.py evaluate --returns C的weekly_log.csv    # 評估 C 跑出來的回測
階段: forecast | bl | risk | all (= 前三者) | evaluate
"""
import argparse, os, sys, shutil
import numpy as np
import pandas as pd
import yaml

from src.data_io import load_panel, load_universe
from src.pipeline import run_pipeline
from src import prereg
from src.forecast import build_features, walk_forward_forecast, sigma_from_interval
from src.bl_integration import bl_weekly, distance_to_prior
from src.risk import evaluate_covariances, export_covariances, garch_vol_panel
from src.bl import iter_weekly_inputs
from src.factors import forward_return_wide
from src.evaluate import full_evaluation

CFG1, CFG2 = "config_fyp1.yaml", "config_fyp2.yaml"
LOCK1, LOCK2 = "prereg.lock", "prereg_fyp2.lock"
pd.set_option("display.width", 200, "display.float_format", lambda x: f"{x:,.4f}")


def stage_forecast(res1, cfg2, out, blind):
    X = build_features(res1["scores"], res1["mu"], cfg2["forecast"].get("include_mu_features", False))
    fo = walk_forward_forecast(X, res1["fwd"], cfg2["cv"], cfg2["forecast"])
    P = fo["preds"].copy()
    P["sigma"] = sigma_from_interval(P)
    P.reset_index().to_csv(f"{out}/oos_forecasts.csv", index=False)                  # 介面檔
    fo["params"].to_csv(f"{out}/forecast_params.csv", index=False)
    print(f"[forecast] fold 數 = {fo['diagnostics']['n_folds']}，宣告的超參數嘗試數 = {fo['n_trials']}")
    for f in fo["folds"]:
        print("   ", f)
    if not blind:
        pd.DataFrame({"raw": fo["diagnostics_raw"], "calibrated": fo["diagnostics"]}).to_csv(
            f"{out}/forecast_diagnostics.csv", encoding="utf-8-sig")
        d = fo["diagnostics"]
        print("[forecast] 樣本外診斷 (校準後): 涵蓋率(目標0.80)={:.3f}  低於q10/q50/q90的比例={:.3f}/{:.3f}/{:.3f}  "
              "相對無條件基準的技巧={:+.4f}  OOS IC={:.4f} (t={:.2f})  寬度與誤差相關={:+.3f}".format(
                  d["coverage_q10_q90"], d["share_below_q10"], d["share_below_q50"], d["share_below_q90"],
                  d["skill_vs_unconditional"], d["oos_ic_mean"], d["oos_ic_t_nw"], d["corr_width_abs_error"]))
    else:
        print("[forecast] 盲測模式: 樣本外診斷未輸出。")
    return fo


def stage_bl(res1, universe, cfg1, cfg2, fo, out):
    m = res1["market"]
    df = bl_weekly(m, universe, cfg1, fo["preds"], cfg2["bl_views"]["trust_grid"])
    df.to_csv(f"{out}/trust_sensitivity.csv", index=False)
    base = df[df["trust"] == cfg2["bl_views"]["base_trust"]]
    base.to_csv(f"{out}/mu_bl.csv", index=False)                                      # 介面檔: 給 C 的最佳化器
    d = distance_to_prior(df)
    d.to_csv(f"{out}/bl_distance_to_prior.csv", encoding="utf-8-sig")
    print("[bl] 後驗離先驗的相對距離 (應隨 trust 上升):\n" + d.to_string())
    return df


def stage_risk(res1, universe, cfg1, cfg2, out, blind):
    m, rc = res1["market"], cfg2["risk"]
    if blind:
        method = rc["default_method"]
        cut = [c for (_, _, _, _, _, _, c) in iter_weekly_inputs(m, universe, cfg1, return_window=True)]
        gv = garch_vol_panel(m.ret1d, cut, rc["refit_every_weeks"], rc["garch_min_obs"], rc["garch_max_window_days"])
        print(f"[risk] 盲測模式: 不評估績效，使用預設方法 {method}")
    else:
        fwd_wide, _ = forward_return_wide(m, int(cfg1["timing"]["horizon_days"]), cfg1["timing"]["delist_policy"])
        tab, gv = evaluate_covariances(m, universe, cfg1, rc, fwd_wide)
        tab.to_csv(f"{out}/risk_comparison.csv", encoding="utf-8-sig")
        metric = rc["selection_metric"]
        score = (tab[metric] - 1).abs() if metric == "bias_stat" else tab[metric]
        method = score.idxmin()
        print("[risk] 共變異數比較 (事前規則: 依 {} 選最佳):\n".format(metric) + tab.to_string() + f"\n    -> 選用: {method}")
    gv["weekly_vol"].stack().rename("garch_weekly_vol").reset_index().rename(columns={"level_0": "date", "level_1": "coin_id"}).to_csv(
        f"{out}/vol_forecast.csv", index=False)
    gv["params"].to_csv(f"{out}/garch_params.csv", index=False)
    cov = export_covariances(m, universe, cfg1, rc, gv, method)
    pd.to_pickle(dict(method=method, cov=cov), f"{out}/risk_cov_selected.pkl")        # 介面檔: 給 C
    print(f"[risk] 已輸出 {len(cov)} 週的共變異數 (方法: {method})")


def stage_evaluate(returns_path, cfg2, out):
    e = cfg2["evaluation"]
    log = pd.read_csv(returns_path, parse_dates=["date"])
    wide = log.pivot(index="date", columns="allocator", values="net_ret").sort_index()
    if e["benchmark"] not in wide.columns:
        sys.exit(f"weekly_log 裡找不到基準 allocator '{e['benchmark']}'。現有: {list(wide.columns)}")
    stress = {k: tuple(v) for k, v in e["stress"].items()}
    ev = full_evaluation(wide, e["benchmark"], rf=e["rf_weekly"], n_boot=e["bootstrap"]["n"], block=e["bootstrap"]["block"],
                         es_level=e["es_level"], seed=e["bootstrap"]["seed"], stress=stress, extra_trials=e["extra_trials"])
    per = pd.concat({k: v for k, v in ev["tables"].items()}, names=["period", "allocator"])
    per.to_csv(f"{out}/performance_by_period.csv", encoding="utf-8-sig")
    ev["tables"]["full"].to_csv(f"{out}/performance_full.csv", encoding="utf-8-sig")
    ev["dsr"].to_csv(f"{out}/deflated_sharpe.csv", encoding="utf-8-sig")
    ev["sharpe_ci"].to_csv(f"{out}/sharpe_bootstrap_ci.csv", encoding="utf-8-sig")
    ev["sharpe_diff_vs_benchmark"].to_csv(f"{out}/sharpe_diff_vs_benchmark.csv", encoding="utf-8-sig")
    print(f"[evaluate] 宣告的嘗試總數 n_trials = {ev['n_trials']}\n\n全樣本績效:\n{ev['tables']['full'].to_string()}")
    print(f"\nDeflated Sharpe (>0.95 才算在多重檢定後仍顯著):\n{ev['dsr'].to_string()}")
    print(f"\nSharpe 差距 vs {e['benchmark']} (配對 bootstrap):\n{ev['sharpe_diff_vs_benchmark'].to_string()}")
    print("\n提醒: 如果簡單基準贏了，請如實寫進報告。")


def _synthetic_weekly_log(res1):
    """只用於 --synthetic 的示範: 用 FYP I 的 fwd 與 mu 造出幾個假的『週淨報酬』。"""
    fwd, mu = res1["fwd"].dropna(), res1["mu"]["fm"]
    rows = []
    rng = np.random.default_rng(0)
    for d, g in fwd.groupby(level=0):
        z = mu.reindex(g.index)
        w_tilt = np.exp(5 * (z - z.mean()).fillna(0)); w_tilt = w_tilt / w_tilt.sum()
        rows.append((d, "equal_weight", g.mean()))
        rows.append((d, "mu_tilt", float((w_tilt * g).sum())))
        rows.append((d, "noise_tilt", float(((lambda x: x / x.sum())(pd.Series(np.exp(rng.normal(0, 1, len(g))), index=g.index)) * g).sum())))
    return pd.DataFrame(rows, columns=["date", "allocator", "net_ret"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all", choices=["forecast", "bl", "risk", "all", "evaluate"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--lock", action="store_true")
    ap.add_argument("--unblind", action="store_true")
    ap.add_argument("--returns", default=None)
    a = ap.parse_args()

    if a.lock:
        info = prereg.lock(CFG2, LOCK2)
        print(f"已鎖定 FYP II 預先登記。sha256={info['sha256'][:16]}...  時間(UTC)={info['locked_at_utc']}")
        return
    cfg1 = yaml.safe_load(open(CFG1, encoding="utf-8"))
    cfg2 = yaml.safe_load(open(CFG2, encoding="utf-8"))

    if a.synthetic:
        from src.synth import make_synthetic
        mode, out = "SYNTHETIC", "results_fyp2_synthetic"
        panel, universe = make_synthetic()
        os.makedirs("data_synth", exist_ok=True)
        panel.to_csv("data_synth/panel.csv", index=False); universe.to_csv("data_synth/universe.csv", index=False)
        panel, universe = load_panel("data_synth/panel.csv"), load_universe("data_synth/universe.csv")
        cfg1["sample"]["start"], cfg1["sample"]["end"] = "2019-07-01", "2024-12-23"
        cfg2["risk"]["refit_every_weeks"] = 26
        cfg2["cv"]["min_train_weeks"] = 104
    else:
        if a.dev:
            mode, out = "DEV", "results_fyp2_dev"
        else:
            for cfgp, lk in ((CFG1, LOCK1), (CFG2, LOCK2)):
                ok, msg, _ = prereg.verify(cfgp, lk)
                print(f"{cfgp}: {msg}")
                if not ok:
                    sys.exit(1)
            mode, out = "OFFICIAL", "results_fyp2"
        if a.stage != "evaluate":
            for key in ("panel_path", "universe_path"):
                if not os.path.exists(cfg1["data"][key]):
                    sys.exit(f"找不到 {cfg1['data'][key]}。請把 B 的檔案放進 data/ (見 config_fyp1.yaml)")
            panel, universe = load_panel(cfg1["data"]["panel_path"]), load_universe(cfg1["data"]["universe_path"])
    blind = (mode == "DEV" and not a.unblind)
    os.makedirs(out, exist_ok=True)
    print(f"[{mode}] 階段: {a.stage}" + ("  (盲測)" if blind else ""))

    if a.stage in ("forecast", "bl", "risk", "all"):
        res1 = run_pipeline(panel, universe, cfg1, light=True)
        if a.stage in ("forecast", "bl", "all"):
            fo = stage_forecast(res1, cfg2, out, blind)
        if a.stage in ("bl", "all"):
            stage_bl(res1, universe, cfg1, cfg2, fo, out)
        if a.stage in ("risk", "all"):
            stage_risk(res1, universe, cfg1, cfg2, out, blind)
    if a.stage == "evaluate" or (a.synthetic and a.stage == "all"):
        path = a.returns or cfg2["evaluation"]["returns_path"]
        if a.synthetic:
            path = f"{out}/fake_weekly_log.csv"
            _synthetic_weekly_log(res1).to_csv(path, index=False)
            cfg2["evaluation"]["benchmark"] = "equal_weight"
        if not os.path.exists(path):
            sys.exit(f"找不到 {path}。請指定 C 的 weekly_log.csv: --returns 路徑")
        stage_evaluate(path, cfg2, out)
    shutil.copy(CFG2, f"{out}/config_fyp2_used.yaml")
    print(f"\n結果已寫入 {out}/")


if __name__ == "__main__":
    main()
