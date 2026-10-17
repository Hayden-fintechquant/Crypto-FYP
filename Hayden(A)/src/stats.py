"""統計小工具: Newey-West、多重檢定修正。"""
import numpy as np
from scipy import stats


def auto_nw_lags(T: int) -> int:
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


def nw_se_of_mean(x, lags: int) -> float:
    """序列平均數的 Newey-West (Bartlett kernel) 標準誤。等同於「只有截距的 OLS + HAC」。"""
    x = np.asarray(x, dtype=float)
    T = len(x)
    xm = x - x.mean()
    s = xm @ xm / T
    for l in range(1, lags + 1):
        if l >= T:
            break
        w = 1.0 - l / (lags + 1.0)
        s += 2.0 * w * (xm[l:] @ xm[:-l] / T)
    return float(np.sqrt(max(s, 0.0) / T))


def mean_tstat(x, lags="auto") -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    T = len(x)
    if T < 8:
        return dict(mean=np.nan, se=np.nan, t=np.nan, p=np.nan, n=T, lags=0)
    L = auto_nw_lags(T) if lags == "auto" else int(lags)
    se = nw_se_of_mean(x, L)
    m = float(x.mean())
    t = m / se if se > 0 else np.nan
    p = float(2 * (1 - stats.t.cdf(abs(t), df=T - 1))) if np.isfinite(t) else np.nan
    return dict(mean=m, se=se, t=t, p=p, n=T, lags=L)


def holm(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx])
        adj[idx] = min(run, 1.0)
    return adj


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    run = 1.0
    for k, idx in enumerate(order):
        rank = m - k
        run = min(run, p[idx] * m / rank)
        adj[idx] = min(run, 1.0)
    return adj
