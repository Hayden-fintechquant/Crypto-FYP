"""FYP II: 把預測層的輸出放進 Black-Litterman。
- 預測的中位數 (q50)         -> view 的數值 Q  (相對於市值加權組合的相對看法)
- 預測區間寬度 (q90 - q10)   -> 每個 view 的不確定性 (Omega 對角線)
- 信任參數 trust             -> 整體要多相信這些 view (trust 越大，離先驗越遠)
介面只有這兩樣: (q50, 區間寬度)。"""
import numpy as np
import pandas as pd
from . import bl
from .forecast import sigma_from_interval


def omega_from_forecast(P, Sigma, tau, sigma_i, trust):
    """Omega_ii = diag(P tau Sigma P')_ii * (sigma_i / mean(sigma))^2 / trust。
    - 不確定性「相對」大的幣 (區間比平均寬)，view 被打折。
    - trust 決定整體尺度；trust -> 0 回到先驗，trust -> 大，view 幾乎被全盤接受。"""
    sigma_i = np.asarray(sigma_i, float)
    base = np.diag(P @ (tau * Sigma) @ P.T)
    rel = (sigma_i / np.nanmean(sigma_i)) ** 2
    return np.diag(base * rel) / trust


def posterior_one_date(Sigma, w, delta, q50, sigma_i, tau, trust):
    """回傳 (pi, mu_bl)。q50 / sigma_i 中的 NaN = 該幣沒有 view (給極大的不確定性，等於忽略)。"""
    pi = bl.implied_returns(Sigma, w, delta)
    has = np.isfinite(q50) & np.isfinite(sigma_i) & (sigma_i > 0)
    if has.sum() < 3:
        return pi, pi.copy()
    q = np.where(has, q50, 0.0)
    sig = np.where(has, sigma_i, np.nanmean(sigma_i[has]))
    P, Q = bl.relative_views(q, w)
    Q = np.where(has, Q, P @ pi)                       # 沒有 view 的幣: view = 先驗 (無影響)
    Om = omega_from_forecast(P, Sigma, tau, sig, trust)
    idx = np.where(~has)[0]
    Om[idx, idx] = 1e12                                 # 沒有 view 的幣: 極大不確定性 = 忽略
    mu, _ = bl.black_litterman(Sigma, pi, P, Q, Om, tau)
    return pi, mu


def bl_weekly(m, universe, cfg, preds: pd.DataFrame, trust_grid, base_trust=None):
    """preds: 預測層的樣本外輸出 (index=(date, coin_id)，欄含 q10,q50,q90)。
    回傳 long 表: date, coin_id, trust, pi_weekly, mu_bl_weekly, mu_naive_weekly。
    mu_naive = 直接把 q50 當預期報酬 (用來做『同一份預測丟進樸素均值變異』的對照)。"""
    b = cfg["black_litterman"]
    tau = b["tau"]
    q50_all = preds["q50"]
    sig_all = sigma_from_interval(preds)
    pred_dates = set(preds.index.get_level_values(0))
    rows = []
    for t, ok, w, Sigma, delta in bl.iter_weekly_inputs(m, universe, cfg):
        if t not in pred_dates:
            continue
        q50 = q50_all.xs(t, level=0).reindex(ok).values
        sg = sig_all.xs(t, level=0).reindex(ok).values
        for tr in trust_grid:
            pi, mu = posterior_one_date(Sigma, w, delta, q50, sg, tau, tr)
            for k, a, c_, n_ in zip(ok, pi, mu, q50):
                rows.append((t, k, tr, a, c_, n_))
    return pd.DataFrame(rows, columns=["date", "coin_id", "trust", "pi_weekly", "mu_bl_weekly", "mu_naive_weekly"])


def distance_to_prior(df: pd.DataFrame) -> pd.DataFrame:
    """每個 trust 下，後驗離先驗多遠: mean|mu_bl - pi| / mean|pi|。應隨 trust 單調上升。"""
    g = df.assign(d=(df["mu_bl_weekly"] - df["pi_weekly"]).abs(), p=df["pi_weekly"].abs()).groupby("trust")
    return (g["d"].mean() / g["p"].mean()).rename("relative_distance_to_prior").to_frame()
