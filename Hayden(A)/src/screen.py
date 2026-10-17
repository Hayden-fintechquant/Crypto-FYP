"""篩選流程: 覆蓋率 -> 完整樣本 -> 縮尾與標準化 -> 最少幣數 -> 冗餘剔除。
所有門檻都來自 config.yaml (事前固定)。"""
import numpy as np
import pandas as pd
from .fama_macbeth import fm_univariate


def coverage_table(fp: pd.DataFrame, factors) -> pd.Series:
    return pd.Series({f: float(fp[f].notna().mean()) for f in factors})


def standardize(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    lo, hi = cfg["screening"]["winsor_pct"]
    mode = cfg["screening"]["standardize"]

    def one(g):
        if mode == "rank":
            g = g.rank(pct=True)
        else:
            g = g.clip(g.quantile(lo), g.quantile(hi), axis=1)
        sd = g.std(ddof=1).replace(0, np.nan)
        return (g - g.mean()) / sd

    return df.groupby(level=0, group_keys=False).apply(one)


def redundancy_tables(scores: pd.DataFrame, fwd: pd.Series, factors, lags="auto"):
    """兩個層次: (1) 因子值的相關 (2) 因子報酬序列(單變量 FM 斜率)的相關。"""
    cv = scores[factors].groupby(level=0).apply(lambda g: g.corr())
    value_corr = cv.groupby(level=1).mean().reindex(index=factors, columns=factors)
    _, gam = fm_univariate(scores, fwd, factors, lags)
    return value_corr, gam.corr().reindex(index=factors, columns=factors)


def screen(fp: pd.DataFrame, cfg: dict, fixed: dict = None):
    """回傳 (scores, fwd, report)。
    fixed=None: 正常篩選 (覆蓋率 + 冗餘)，並把結果記在 report['fixed'] 供稽核使用。
    fixed={'kept':[...], 'survivors':[...]}: 沿用既定的因子清單 (用於前視稽核)。"""
    scfg = cfg["screening"]
    factors = list(cfg["factors"])
    cov = coverage_table(fp, factors)
    kept = list(fixed["kept"]) if fixed else [f for f in factors if cov[f] >= scfg["min_factor_coverage"]]
    cc = fp.dropna(subset=kept)
    z = standardize(cc[kept], cfg)
    n = z.groupby(level=0).size()
    good = n[n >= scfg["min_cross_section"]].index
    z = z.loc[z.index.get_level_values(0).isin(good)].dropna()
    fwd = cc["fwd_ret"].reindex(z.index)

    report = dict(coverage=cov, n_weeks=len(good), n_rows=len(z), dropped_coverage=[f for f in factors if f not in kept])
    if fixed:
        surv = list(fixed["survivors"])
        report.update(value_corr=None, return_corr=None, dropped_redundant={})
    else:
        lags = cfg["validation"]["nw_lags"]
        vc, rc = redundancy_tables(z, fwd, kept, lags)
        cut = scfg["redundancy_corr_cutoff"]
        surv, dropped = [], {}
        for f in kept:                                   # 依 config 的優先順序
            clash = [(j, float(vc.loc[f, j])) for j in surv if abs(vc.loc[f, j]) > cut]
            if clash:
                dropped[f] = clash
            else:
                surv.append(f)
        report.update(value_corr=vc, return_corr=rc, dropped_redundant=dropped)
    report["fixed"] = dict(kept=kept, survivors=surv)
    report["survivors"] = surv
    return z[surv], fwd, report
