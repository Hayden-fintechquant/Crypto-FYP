"""Six benchmark allocators + BTC buy-and-hold.  Every allocator takes a Context `ctx` and returns weights (pd.Series, sum 1)
over ctx.coins (coins that are members, have a price at the information cut-off and pass the liquidity rule).

Fixes vs the first version: optimisers work on a rescaled problem (the first risk-parity returned exactly 1/N because
the objective was ~1e-12), the cap is a CONSTRAINT of the optimisers (not clip-and-renormalise afterwards), the
covariance uses Ledoit-Wolf shrinkage on 180 days (as in the planning report), fall-backs are logged, and the
mean-variance and Black-Litterman benchmarks exist.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


# ---------------------------------------------------------------- helpers
def cap_weights(w, cap, tol=1e-12):
    """Water-filling cap: clip the weights above `cap` and give the excess to the others pro-rata, repeat until none
    exceeds the cap. Sum stays 1. If cap*N < 1 the cap is infeasible -> equal weight 1/N."""
    w = pd.Series(w, dtype=float).clip(lower=0.0)
    n = len(w)
    if n == 0:
        return w
    if cap is None:
        return w / w.sum() if w.sum() > 0 else pd.Series(1.0 / n, index=w.index)
    if cap * n < 1 - 1e-12:
        return pd.Series(1.0 / n, index=w.index)
    w = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / n, index=w.index)
    for _ in range(200):
        over = w > cap + tol
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = w < cap - tol
        base = w[free].sum()
        if base <= 0:
            w[free] += excess / free.sum()
        else:
            w[free] += excess * w[free] / base
    return w


def estimate_risk(ctx):
    """Weekly covariance (Ledoit-Wolf on daily returns, x7) and trailing mean weekly return for the coins with enough history.
    Missing daily returns are filled with that day's cross-sectional mean (documented; keeps the matrix rectangular)."""
    key = ("risk", ctx.date)
    if key in ctx.data.cache:
        return ctx.data.cache[key]
    R = ctx.window_returns()
    ok = R.columns[R.notna().sum() >= ctx.cfg["risk_model"]["min_obs_days"]]
    R = R[ok]
    if len(ok) < 5:
        out = None
    else:
        day_mean = R.mean(axis=1)
        Rf = R.apply(lambda s: s.fillna(day_mean)).fillna(0.0)
        if ctx.cfg["risk_model"]["shrinkage"] == "ledoit_wolf":
            S = LedoitWolf().fit(Rf.values).covariance_
        else:
            S = np.cov(Rf.values, rowvar=False)
        Sigma = pd.DataFrame(S * 7.0, index=ok, columns=ok)
        mu = R.mean(axis=0) * 7.0
        out = dict(coins=list(ok), Sigma=Sigma, mu=mu)
    ctx.data.cache[key] = out
    return out


def _qp_long_only(Sigma, mu, gamma, cap):
    """max mu'w - gamma/2 w'Sigma w   s.t. sum w = 1, 0 <= w <= cap   (SLSQP on a rescaled problem)."""
    S = Sigma.values
    n = len(S)
    s = float(np.mean(np.diag(S)))
    S_, m_ = S / s, (np.zeros(n) if mu is None else np.asarray(mu, float) / s)
    ub = 1.0 if cap is None else max(cap, 1.0 / n)
    f = lambda w: -(m_ @ w) + 0.5 * gamma * w @ S_ @ w
    g = lambda w: -m_ + gamma * S_ @ w
    res = minimize(f, np.full(n, 1.0 / n), jac=g, method="SLSQP", bounds=[(0.0, ub)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
                   options={"maxiter": 1000, "ftol": 1e-14})
    w = np.clip(res.x, 0, None)
    w = w / w.sum()
    # SLSQP often reports "failure" at machine precision although the point is optimal -> judge by the KKT conditions:
    # on the coins strictly between the bounds the gradient of the utility must be (nearly) equal.
    grad = m_ - gamma * S_ @ w
    free = (w > 1e-7) & (w < ub - 1e-7)
    spread = float(np.ptp(grad[free])) if free.sum() > 1 else 0.0
    scale = max(1e-12, float(np.abs(gamma * S_ @ w).mean()))
    ok = bool(res.success) or (spread / scale < 1e-5 and abs(w.sum() - 1) < 1e-9)
    return pd.Series(w, index=Sigma.index), ok


def _fallback(ctx, why):
    ctx.notes.append(why)
    return pd.Series(1.0 / len(ctx.coins), index=ctx.coins)


def _embed(w, ctx):
    """Weights on the coins used by the risk model -> weights on all tradeable coins (others get 0); renormalised."""
    out = pd.Series(0.0, index=ctx.coins)
    out[w.index] = w
    return out / out.sum()


# ---------------------------------------------------------------- allocators
def equal_weight(ctx):
    return cap_weights(pd.Series(1.0, index=ctx.coins), ctx.cfg["portfolio"]["weight_cap"])


def market_cap(ctx, capped=True):
    mc = ctx.mcap.reindex(ctx.coins)
    if mc.isna().any() or (mc <= 0).any():
        ctx.notes.append(f"market_cap: {int((mc.isna() | (mc <= 0)).sum())} coins without market cap get the median")
        mc = mc.where(mc > 0).fillna(mc[mc > 0].median())
    return cap_weights(mc, ctx.cfg["portfolio"]["weight_cap"] if capped else None)


def market_cap_uncapped(ctx):
    return market_cap(ctx, capped=False)


def minimum_variance(ctx):
    r = estimate_risk(ctx)
    if r is None:
        return _fallback(ctx, "min_variance: not enough history -> equal weight")
    w, ok = _qp_long_only(r["Sigma"], None, 1.0, ctx.cfg["portfolio"]["weight_cap"])
    if not ok:
        ctx.notes.append("min_variance: optimiser did not converge")
    return _embed(w, ctx)


def risk_parity(ctx):
    """Equal risk contribution, long-only: minimise 0.5 x'Sx - (1/n) sum ln x, then w = x / sum x; cap by water-filling."""
    r = estimate_risk(ctx)
    if r is None:
        return _fallback(ctx, "risk_parity: not enough history -> equal weight")
    S = r["Sigma"].values
    n = len(S)
    S_ = S / np.mean(np.diag(S))
    f = lambda x: 0.5 * x @ S_ @ x - np.log(x).sum() / n
    g = lambda x: S_ @ x - 1.0 / (n * x)
    res = minimize(f, np.full(n, 1.0 / n), jac=g, method="L-BFGS-B", bounds=[(1e-10, None)] * n,
                   options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-12})
    w = pd.Series(res.x / res.x.sum(), index=r["Sigma"].index)
    rc = w.values * (S @ w.values)
    if rc.std() / rc.mean() > 1e-3:                         # judge by the result (equal risk contributions), not by the solver flag
        ctx.notes.append(f"risk_parity: risk contributions not equal (CoV {rc.std() / rc.mean():.3g})")
    w = _embed(w, ctx)
    return cap_weights(w, ctx.cfg["portfolio"]["weight_cap"])


def mean_variance(ctx):
    """Classic Markowitz benchmark: trailing sample mean as expected return (deliberately naive)."""
    r = estimate_risk(ctx)
    if r is None:
        return _fallback(ctx, "mean_variance: not enough history -> equal weight")
    w, ok = _qp_long_only(r["Sigma"], r["mu"].reindex(r["Sigma"].index).values, ctx.cfg["benchmarks"]["mv_risk_aversion"],
                          ctx.cfg["portfolio"]["weight_cap"])
    if not ok:
        ctx.notes.append("mean_variance: optimiser did not converge")
    return _embed(w, ctx)


def black_litterman(ctx, views=None):
    """Black-Litterman with NO views (FYP I benchmark): posterior = equilibrium prior pi = delta * Sigma * w_mkt, with delta set
    so the market portfolio has the pre-specified annual Sharpe ratio. Then long-only, capped mean-variance on pi.
    Without the cap this returns exactly the market-cap weights (checked in the tests). FYP II passes `views`."""
    if views is not None:
        raise NotImplementedError("views are added in FYP II")
    r = estimate_risk(ctx)
    if r is None:
        return _fallback(ctx, "black_litterman: not enough history -> equal weight")
    Sig = r["Sigma"]
    mc = ctx.mcap.reindex(Sig.index)
    mc = mc.where(mc > 0).fillna(mc[mc > 0].median())
    wm = (mc / mc.sum()).values
    sigma_m = float(np.sqrt(wm @ Sig.values @ wm))
    sr_w = ctx.cfg["benchmarks"]["bl_implied_sharpe"] / np.sqrt(ctx.cfg["timing"]["periods_per_year"])
    delta = sr_w / sigma_m
    pi = delta * Sig.values @ wm
    w, ok = _qp_long_only(Sig, pi, delta, ctx.cfg["portfolio"]["weight_cap"])
    if not ok:
        ctx.notes.append("black_litterman: optimiser did not converge")
    return _embed(w, ctx)


def btc_buy_hold(ctx):
    if "bitcoin" not in ctx.coins:
        ctx.notes.append("btc_buy_hold: bitcoin not tradeable this week")
        return pd.Series(0.0, index=ctx.coins)
    w = pd.Series(0.0, index=ctx.coins)
    w["bitcoin"] = 1.0
    return w


# name -> (function, apply_engine_cap)
ALLOCATORS = {
    "equal_weight": (equal_weight, True),
    "market_cap": (market_cap, True),
    "min_variance": (minimum_variance, True),
    "risk_parity": (risk_parity, True),
    "mean_variance": (mean_variance, True),
    "black_litterman": (black_litterman, True),
    "btc_buy_hold": (btc_buy_hold, False),
    "market_cap_uncapped": (market_cap_uncapped, False),   # reference only: shows what the 10% cap does to the market portfolio
}
