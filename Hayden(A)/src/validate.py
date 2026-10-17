"""IC / ICIR / 五分位排序。"""
import numpy as np
import pandas as pd
from .stats import mean_tstat


def ic_series(scores: pd.DataFrame, fwd: pd.Series, factor: str, min_n: int = 10) -> pd.Series:
    """每週的 rank IC = 因子分數與下一週報酬的 Spearman 相關。"""
    out = {}
    for d, g in scores[[factor]].groupby(level=0):
        y = fwd.reindex(g.index)
        ok = y.notna() & g[factor].notna()
        if ok.sum() < min_n:
            continue
        out[d] = g[factor][ok].rank().corr(y[ok].rank())
    return pd.Series(out, dtype=float).sort_index()


def ic_summary(ic: pd.Series, rolling_weeks: int = 52, lags="auto") -> dict:
    r = mean_tstat(ic.values, lags)
    sd = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    roll = ic.rolling(rolling_weeks, min_periods=rolling_weeks // 2).mean()
    return dict(ic_mean=r["mean"], ic_std=sd, icir=(r["mean"] / sd if sd and sd > 0 else np.nan),
                ic_t_nw=r["t"], ic_p=r["p"], ic_hit=float((ic > 0).mean()), n_weeks=r["n"],
                roll_ic_min=float(roll.min()), roll_ic_max=float(roll.max()))


def quantile_sorts(scores: pd.DataFrame, fwd: pd.Series, factor: str, nq: int = 5, lags="auto"):
    """每週依因子分數分成 nq 組，算各組等權報酬。回傳 (逐週各組報酬, 摘要 dict)。"""
    per = {}
    for d, g in scores[[factor]].groupby(level=0):
        y = fwd.reindex(g.index)
        ok = y.notna() & g[factor].notna()
        if ok.sum() < nq * 3:
            continue
        s = g[factor][ok]
        q = pd.qcut(s.rank(method="first"), nq, labels=False)
        per[d] = y[ok].groupby(q.values).mean()
    wk = pd.DataFrame(per).T.sort_index()
    means = wk.mean()
    ls = wk[nq - 1] - wk[0]
    r = mean_tstat(ls.values, lags)
    steps = np.diff(means.values)
    mono = float(pd.Series(range(nq)).corr(pd.Series(means.values), method="spearman"))
    summ = dict(ls_mean=r["mean"], ls_t_nw=r["t"], ls_p=r["p"], monotonicity=mono,
                frac_up_steps=float((steps > 0).mean()))
    for i in range(nq):
        summ[f"q{i + 1}_mean"] = float(means.iloc[i])
    return wk, summ
