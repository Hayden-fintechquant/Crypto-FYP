import pandas as pd
import numpy as np
from scipy.optimize import minimize

def equal_weight(date, members, hist):
    """Equal weight: put the same amount in every coin"""
    n = len(members)
    if n == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / n, index=members)

def market_cap_weighted(date, members, hist):
    """Market-cap weighted: larger market cap gets higher weight"""
    if "market_cap" not in hist.columns:
        return equal_weight(date, members, hist)
    latest = hist[hist["date"] < date].groupby("coin_id")["market_cap"].last()
    mcap = latest.reindex(members).fillna(0)
    if mcap.sum() <= 0:
        return equal_weight(date, members, hist)
    return mcap / mcap.sum()

def minimum_variance(date, members, hist, lookback=60):
    """Minimum variance (long-only)"""
    recent = hist[hist["date"] < date].pivot(index="date", columns="coin_id", values="close")
    recent = recent.reindex(columns=members).pct_change().dropna().tail(lookback)
    if len(recent) < 20 or recent.shape[1] < 2:
        return equal_weight(date, members, hist)

    cov = recent.cov().values
    n = len(members)

    def objective(w):
        return w @ cov @ w

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        return equal_weight(date, members, hist)
    return pd.Series(res.x, index=members)

def risk_parity(date, members, hist, lookback=60):
    """Risk parity"""
    recent = hist[hist["date"] < date].pivot(index="date", columns="coin_id", values="close")
    recent = recent.reindex(columns=members).pct_change().dropna().tail(lookback)
    if len(recent) < 20 or recent.shape[1] < 2:
        return equal_weight(date, members, hist)

    cov = recent.cov().values
    n = len(members)

    def objective(w):
        port_var = w @ cov @ w
        if port_var <= 0:
            return 1e6
        mrc = cov @ w
        rc = w * mrc
        target = rc.mean()
        return ((rc - target) ** 2).sum()

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0.001, 1)] * n
    w0 = np.ones(n) / n
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        return equal_weight(date, members, hist)
    w = np.clip(res.x, 0, None)
    w = w / w.sum()
    return pd.Series(w, index=members)