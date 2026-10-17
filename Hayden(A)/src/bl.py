"""Black-Litterman: 反向最佳化(隱含均衡報酬)、後驗報酬、極限情況。"""
import numpy as np
import pandas as pd


def sample_cov(X):
    return np.cov(X, rowvar=False)


def ledoit_wolf_cov(X):
    try:
        from sklearn.covariance import LedoitWolf
        return LedoitWolf().fit(X).covariance_
    except Exception:
        return sample_cov(X)


def implied_returns(Sigma, w_mkt, delta):
    """反向最佳化: pi = delta * Sigma * w_mkt"""
    return delta * Sigma @ w_mkt


def reverse_opt_weights(Sigma, pi, delta):
    """正向: 無限制均值變異的最佳權重 w = inv(Sigma) pi / delta (用來檢查 pi 是否正確)"""
    return np.linalg.solve(Sigma, pi) / delta


def market_delta(R_weekly_mkt, rf_weekly=0.0, lo=1.0, hi=10.0):
    """delta = (E[r_m] - rf) / Var(r_m)，夾在 [lo, hi]。R_weekly_mkt: 市值加權組合的週報酬序列。"""
    mu, var = float(np.mean(R_weekly_mkt)), float(np.var(R_weekly_mkt, ddof=1))
    if var <= 0:
        return lo
    return float(np.clip((mu - rf_weekly) / var, lo, hi))


def black_litterman(Sigma, pi, P, Q, Omega, tau):
    """回傳 (後驗報酬, 後驗共變異數)。"""
    tS = tau * Sigma
    A = P @ tS @ P.T + Omega
    mu = pi + tS @ P.T @ np.linalg.solve(A, Q - P @ pi)
    post = Sigma + tS - tS @ P.T @ np.linalg.solve(A, P @ tS)
    return mu, post


def omega_from_trust(P, Sigma, tau, trust):
    """view 不確定性: Omega = diag(P tau Sigma P') / trust。trust 越大越相信 view。"""
    return np.diag(np.diag(P @ (tau * Sigma) @ P.T)) / trust


def relative_views(mu, w_mkt):
    """FYP II 用: 每個幣相對於市值加權組合的相對看法。P = I - 1 w', Q = P mu。"""
    n = len(mu)
    P = np.eye(n) - np.outer(np.ones(n), w_mkt)
    return P, P @ np.asarray(mu)


def weekly_prior(m, universe: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """每個再平衡日: 市值權重 w_mkt、隱含均衡報酬 pi (週報酬單位)。交給 C 的 BL 基準使用。"""
    universe = universe.assign(is_member=universe["is_member"].astype(int) > 0)   # 0/1 或 True/False 都可以
    b, off = cfg["black_litterman"], pd.Timedelta(days=int(cfg["timing"]["info_cutoff_offset_days"]))
    reb = pd.DatetimeIndex(sorted(universe["rebalance_date"].unique()))
    reb = reb[(reb >= pd.Timestamp(cfg["sample"]["start"])) & (reb <= pd.Timestamp(cfg["sample"]["end"]))]
    cov_fn = ledoit_wolf_cov if b["shrinkage"] == "ledoit_wolf" else sample_cov
    mem = universe[universe["is_member"]].groupby("rebalance_date")["coin_id"].apply(set)
    rows = []
    for t in reb:
        c = t - off
        if c not in m.close.index or t not in mem.index:
            continue
        win = m.ret1d.loc[c - pd.Timedelta(days=b["cov_window_days"] - 1): c]
        cand = [k for k in mem[t] if k in win.columns]
        ok = [k for k in cand if win[k].notna().sum() >= b["min_history_days"]
              and pd.notna(m.close.at[c, k]) and m.mcap.at[c, k] > 0]
        if len(ok) < 5:
            continue
        R = win[ok]
        R = R.apply(lambda col: col.fillna(R.mean(axis=1))).fillna(0.0)   # 缺值用當天橫截面平均補
        Sigma = cov_fn(R.values) * 7.0
        cap = m.mcap.loc[c, ok].values.astype(float)
        w = cap / cap.sum()
        delta = b["delta"] if b["delta_mode"] == "fixed" else market_delta(7 * (R.values @ w), b["risk_free_weekly"])
        pi = implied_returns(Sigma, w, delta)
        for k, wk, pk in zip(ok, w, pi):
            rows.append((t, k, wk, pk, delta))
    return pd.DataFrame(rows, columns=["date", "coin_id", "w_mkt", "pi_weekly", "delta"])
