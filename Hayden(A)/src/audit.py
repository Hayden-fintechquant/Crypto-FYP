"""前視偏差稽核 (look-ahead audit)。
做法: 隨機選幾個日期 t，把資料「截斷到 t 當時已知」，重跑整條流程，
      結果在 t 的分數與預期報酬必須和用全部資料跑出來的完全一樣。不一樣 = 偷看了未來。"""
import numpy as np
import pandas as pd
from .pipeline import run_pipeline


def _default_fn(panel, universe, cfg, fixed):
    return run_pipeline(panel, universe, cfg, fixed=fixed, light=True)


def _maxdiff(a, b) -> float:
    """兩個以 coin_id 為 index 的結果的最大差距；A 有值而 B 沒有 -> 視為無限大(不一致)。"""
    b = b.reindex(a.index)
    d = (a - b).abs()
    d = d.where(~(a.notna() & b.isna() if isinstance(a, pd.Series) else (a.notna() & b.isna())), np.inf)
    return float(np.nanmax(np.asarray(d, dtype=float))) if np.asarray(d).size else 0.0


def lookahead_audit(panel, universe, cfg, full, n_checks=4, seed=0, pipeline_fn=None, tol=1e-9):
    fn = pipeline_fn or _default_fn
    fixed = full["screen"]["fixed"]
    off = pd.Timedelta(days=int(cfg["timing"]["info_cutoff_offset_days"]))
    ref = fn(panel, universe, cfg, fixed) if pipeline_fn else full
    methods = list(ref["mu"].keys())
    valid = sorted(set.intersection(*[set(ref["mu"][k].dropna().index.get_level_values(0)) for k in methods]))
    if len(valid) < n_checks:
        return pd.DataFrame(), False, "可稽核的日期太少 (歷史不足以產生預期報酬)"
    rng = np.random.default_rng(seed)
    picks = sorted(rng.choice(len(valid), size=n_checks, replace=False))
    rows = []
    for i in picks:
        t = valid[i]
        cut = t - off
        p2 = panel[panel["date"] <= cut]
        u2 = universe[universe["rebalance_date"] <= t]
        r2 = fn(p2, u2, cfg, fixed)
        row = dict(date=t.date(), max_diff_scores=_maxdiff(ref["scores"].loc[t], r2["scores"].loc[t]))
        for k in methods:
            row[f"max_diff_mu_{k}"] = _maxdiff(ref["mu"][k].xs(t, level=0), r2["mu"][k].xs(t, level=0))
        rows.append(row)
    tab = pd.DataFrame(rows)
    passed = bool((tab.drop(columns="date").fillna(np.inf) <= tol).all().all())
    return tab, passed, "通過: 截斷資料後結果完全一致" if passed else "失敗: 結果依賴未來資料，請檢查!"
