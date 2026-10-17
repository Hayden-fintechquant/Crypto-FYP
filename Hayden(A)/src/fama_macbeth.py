"""Fama-MacBeth: 每週做橫截面迴歸，再對時間平均，用 Newey-West 算標準誤。"""
import numpy as np
import pandas as pd
from .stats import mean_tstat


def fm_gammas(scores: pd.DataFrame, fwd: pd.Series, factors, min_extra: int = 5) -> pd.DataFrame:
    """每個日期: fwd ~ const + factors。回傳 日期 x (const + factors) 的係數表。"""
    factors = list(factors)
    out = {}
    for d, g in scores[factors].groupby(level=0):
        y = fwd.reindex(g.index).values
        X = g.values
        ok = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        if ok.sum() < len(factors) + min_extra:
            continue
        Xd = np.column_stack([np.ones(ok.sum()), X[ok]])
        b = np.linalg.lstsq(Xd, y[ok], rcond=None)[0]
        out[d] = b
    return pd.DataFrame(out, index=["const"] + factors).T


def fm_summary(gammas: pd.DataFrame, lags="auto") -> pd.DataFrame:
    rows = {}
    for c in gammas.columns:
        r = mean_tstat(gammas[c].values, lags)
        rows[c] = r
    return pd.DataFrame(rows).T


def fm_univariate(scores, fwd, factors, lags="auto"):
    gam, rows = {}, {}
    for f in factors:
        g = fm_gammas(scores, fwd, [f])
        gam[f] = g[f]
        rows[f] = mean_tstat(g[f].values, lags)
    return pd.DataFrame(rows).T, pd.DataFrame(gam)


def fm_multivariate(scores, fwd, factors, lags="auto"):
    g = fm_gammas(scores, fwd, factors)
    return fm_summary(g, lags), g


def fm_spanning(scores, fwd, candidates, controls, lags="auto"):
    """檢驗候選因子在控制 (截距=市場) + controls 之後，是否仍有溢酬。"""
    rows = {}
    for f in candidates:
        if f in controls:
            continue
        g = fm_gammas(scores, fwd, list(controls) + [f])
        rows[f] = mean_tstat(g[f].values, lags)
    return pd.DataFrame(rows).T
