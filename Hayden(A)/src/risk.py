"""FYP II 風險模型: 每個幣用 GARCH(1,1) 預測波動，再配上「因子/主成分」或收縮的相關結構。
Sigma = diag(波動) @ 相關矩陣 @ diag(波動)。並提供比較各種共變異數估計的評估函式。
所有估計都只用截止日 (週日) 以前的資料。"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .bl import ledoit_wolf_cov, sample_cov, iter_weekly_inputs

EWMA_PARAMS = (0.0, 0.06, 0.94)       # RiskMetrics = omega 為 0 的 GARCH(1,1) (IGARCH)


# ---------- 單一幣的 GARCH ----------
def garch11_fit(r: np.ndarray):
    """用 arch 套件以 Student-t 分佈估計零均值 GARCH(1,1)。失敗或不穩定回傳 None。r 為日報酬(小數)。"""
    try:
        from arch import arch_model
        am = arch_model(np.asarray(r) * 100.0, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        p = res.params
        om, al, be = float(p["omega"]) / 1e4, float(p["alpha[1]"]), float(p["beta[1]"])
    except Exception:
        return None
    if not (np.isfinite([om, al, be]).all() and om > 0 and al >= 0 and be >= 0 and al + be < 0.9999):
        return None
    return om, al, be


def garch_filter(r: np.ndarray, om, al, be, v0):
    """條件變異數路徑。s2[t] 是『給定 t 以前的報酬』下 r[t] 的條件變異數。"""
    n = len(r)
    s2 = np.empty(n)
    s2[0] = v0
    for t in range(1, n):
        s2[t] = om + al * r[t - 1] ** 2 + be * s2[t - 1]
    return s2


def garch_forecast_variance(s2_t, r_t, om, al, be, h=7):
    """在 t 日收盤後，預測未來 h 天的『累積變異數』(= 週變異數)。"""
    s2_next = om + al * r_t ** 2 + be * s2_t
    pers = al + be
    if pers >= 0.9999 or om <= 0:
        return h * s2_next
    V = om / (1 - pers)
    return h * V + (s2_next - V) * (1 - pers ** h) / (1 - pers)


def garch_vol_panel(ret1d: pd.DataFrame, cutoffs, refit_every=13, min_obs=250, max_window=1095, h=7, use_garch=True):
    """每個幣、每個截止日的『週波動預測』，以及逐日的條件波動路徑 (用來把報酬標準化)。
    參數每 refit_every 個截止日重新估計一次，只用該日以前的資料 (expanding，最多 max_window 天)。
    歷史不足 min_obs 天的幣，改用 EWMA (RiskMetrics)。回傳 dict(weekly_vol, daily_vol, params)。"""
    cutoffs = pd.DatetimeIndex(sorted(cutoffs))
    daily = pd.DataFrame(np.nan, index=ret1d.index, columns=ret1d.columns)
    weekly = pd.DataFrame(np.nan, index=cutoffs, columns=ret1d.columns)
    prow = []
    for coin in ret1d.columns:
        s = ret1d[coin]
        first = s.first_valid_index()
        if first is None:
            continue
        s = s.loc[first:].fillna(0.0)
        # 這個幣第一個「歷史 >= 30 天」的截止日；之後每 refit_every 個截止日重估一次 (每個幣各自的排程)
        i0 = next((j for j, c in enumerate(cutoffs) if len(s.loc[:c]) >= 30 and c >= s.index[0]), None)
        first_seg = True
        while i0 is not None and i0 < len(cutoffs):
            seg = cutoffs[i0: i0 + refit_every]
            c0, c_last = seg[0], seg[-1]
            hist = s.loc[:c0].tail(max_window)
            fit = garch11_fit(hist.values) if (use_garch and len(hist) >= min_obs) else None
            kind = "garch" if fit is not None else "ewma"
            om, al, be = fit if fit is not None else EWMA_PARAMS
            path_s = s.loc[hist.index[0]: c_last]
            v0 = float(hist.var()) if hist.var() > 0 else 1e-4
            s2 = garch_filter(path_s.values, om, al, be, v0)
            vol = pd.Series(np.sqrt(s2), index=path_s.index)
            # 逐日路徑: 第一段包含暖機期；之後每一天都用「該日以前已估好」的參數
            start = path_s.index[0] - pd.Timedelta(days=1) if first_seg else cutoffs[i0 - 1]
            sl = vol.loc[vol.index > start]
            daily.loc[sl.index, coin] = sl.values
            for c in seg:
                if c in path_s.index:
                    pos = path_s.index.get_loc(c)
                    weekly.at[c, coin] = np.sqrt(garch_forecast_variance(s2[pos], path_s.values[pos], om, al, be, h))
            prow.append(dict(coin_id=coin, refit_date=c0, omega=om, alpha=al, beta=be, kind=kind))
            first_seg = False
            i0 += refit_every
    return dict(weekly_vol=weekly, daily_vol=daily, params=pd.DataFrame(prow))


# ---------- 相關結構 ----------
def _to_corr(C):
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    return C / np.outer(d, d)


def correlation_estimate(Z: np.ndarray, method="sample", k=3) -> np.ndarray:
    """Z: T x N 標準化報酬 (用條件波動除過)。
    sample: 樣本相關；ledoit_wolf: 收縮；pca: 前 k 個主成分 + 對角殘差；const: 常數相關。"""
    N = Z.shape[1]
    if method == "sample":
        return np.corrcoef(Z, rowvar=False)
    if method == "ledoit_wolf":
        return _to_corr(ledoit_wolf_cov(Z))
    if method == "const":
        C = np.corrcoef(Z, rowvar=False)
        rho = (C.sum() - N) / (N * (N - 1))
        return rho * np.ones((N, N)) + (1 - rho) * np.eye(N)
    if method == "pca":
        C = np.corrcoef(Z, rowvar=False)
        vals, vecs = np.linalg.eigh(C)
        order = np.argsort(vals)[::-1][:k]
        L = vecs[:, order] * np.sqrt(np.clip(vals[order], 0, None))
        Ck = L @ L.T
        D = np.clip(1 - np.diag(Ck), 1e-6, None)               # 特異(個別)變異
        return Ck + np.diag(D)
    raise ValueError(method)


def build_cov(vol_week: np.ndarray, corr: np.ndarray) -> np.ndarray:
    return corr * np.outer(vol_week, vol_week)


# ---------- 每週各種共變異數估計 ----------
def covariance_models(R: pd.DataFrame, daily_vol: pd.DataFrame, weekly_vol: pd.Series, methods, k=3):
    """R: 截止日前 window 天的日報酬 (欄=幣，已補缺值)。回傳 {方法名: Sigma (週單位)}。
    方法名: sample | ledoit_wolf (FYP I 的做法) ；garch_sample | garch_lw | garch_pca | garch_const。"""
    coins = list(R.columns)
    out = {}
    Z = (R / daily_vol.reindex(index=R.index, columns=coins)).replace([np.inf, -np.inf], np.nan)
    Z = Z.apply(lambda col: col.fillna(Z.mean(axis=1))).fillna(0.0)
    Z = Z - Z.mean()
    v = weekly_vol.reindex(coins).values
    for name in methods:
        if name == "sample":
            out[name] = sample_cov(R.values) * 7.0
        elif name == "ledoit_wolf":
            out[name] = ledoit_wolf_cov(R.values) * 7.0
        elif name.startswith("garch_"):
            if np.isnan(v).any() or np.isnan(Z.values).any():
                continue
            cm = {"garch_sample": "sample", "garch_lw": "ledoit_wolf", "garch_pca": "pca", "garch_const": "const"}[name]
            out[name] = build_cov(v, correlation_estimate(Z.values, cm, k))
    return out


# ---------- 評估 ----------
def gmv_longonly(Sigma: np.ndarray, cap=0.10):
    """長倉、每個幣有上限的全域最小變異權重 (SLSQP)。上限會自動提高到至少 1/N。"""
    n = len(Sigma)
    cap = max(cap, 1.0 / n + 1e-6)
    w0 = np.ones(n) / n
    res = minimize(lambda w: w @ Sigma @ w, w0, jac=lambda w: 2 * Sigma @ w, method="SLSQP",
                   bounds=[(0, cap)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 200, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def evaluate_covariances(m, universe, cfg, risk_cfg, fwd_wide: pd.DataFrame, max_dates=None, seed=0):
    """逐週: 用各方法預測共變異數，與『下一週實際報酬』比較。
    指標:
      qlike_mean   測試組合 (等權、市值權重、隨機長倉) 的平均 QLIKE = log(var_hat) + r^2/var_hat，越低越好
      bias_stat    (實際組合報酬 / 預測波動) 的標準差，理想 = 1 (<1 高估風險, >1 低估風險)
      gmv_vol_ann  用該共變異數做長倉最小變異，實現的年化波動，越低越好
      gmv_turnover 該組合每週平均換手率
      cond_median  共變異數矩陣條件數的中位數"""
    rng = np.random.default_rng(seed)
    methods = list(risk_cfg["methods"])
    cut_off = pd.Timedelta(days=int(cfg["timing"]["info_cutoff_offset_days"]))
    gen = list(iter_weekly_inputs(m, universe, cfg, return_window=True))
    if max_dates:
        gen = gen[-max_dates:]
    cutoffs = [c for (_, _, _, _, _, _, c) in gen]
    gv = garch_vol_panel(m.ret1d, cutoffs, risk_cfg["refit_every_weeks"], risk_cfg["garch_min_obs"],
                         risk_cfg["garch_max_window_days"], 7, True)
    rec = {k: dict(q=[], z=[], gmv_r=[], gmv_w=[], cond=[]) for k in methods}
    prev_w = {k: None for k in methods}
    turn = {k: [] for k in methods}
    for (t, ok, w, Sig_fyp1, delta, R, c) in gen:
        if c not in fwd_wide.index:
            continue
        y = fwd_wide.loc[c, ok].values.astype(float)
        if np.isnan(y).any():
            continue
        covs = covariance_models(R, gv["daily_vol"], gv["weekly_vol"].loc[c], methods, risk_cfg["pca_components"])
        if len(covs) < len(methods):
            continue                                  # 所有方法都算得出來的週才納入，才能公平比較 (同一批週)
        ports = [np.ones(len(ok)) / len(ok), w] + [rng.dirichlet(np.ones(len(ok))) for _ in range(risk_cfg["n_random_portfolios"])]
        for k, S in covs.items():
            for p in ports:
                var_hat = float(p @ S @ p)
                rp = float(p @ y)
                if var_hat > 0:
                    rec[k]["q"].append(np.log(var_hat) + rp ** 2 / var_hat)
            ew = ports[0]
            rec[k]["z"].append(float(ew @ y) / np.sqrt(float(ew @ S @ ew)))
            wg = gmv_longonly(S, risk_cfg["eval_cap"])
            rec[k]["gmv_r"].append(float(wg @ y))
            rec[k]["cond"].append(float(np.linalg.cond(S)))
            if prev_w[k] is not None:
                a = pd.Series(prev_w[k][1], index=prev_w[k][0]).reindex(ok).fillna(0.0)
                b = pd.Series(wg, index=ok)
                turn[k].append(float((a - b).abs().sum()))
            prev_w[k] = (ok, wg)
    rows = {}
    for k in methods:
        r = rec[k]
        if not r["z"]:
            continue
        rows[k] = dict(n_weeks=len(r["z"]), qlike_mean=float(np.mean(r["q"])), bias_stat=float(np.std(r["z"], ddof=1)),
                       gmv_vol_ann=float(np.std(r["gmv_r"], ddof=1) * np.sqrt(52)),
                       gmv_turnover=float(np.mean(turn[k])) if turn[k] else np.nan, cond_median=float(np.median(r["cond"])))
    return pd.DataFrame(rows).T, gv


def export_covariances(m, universe, cfg, risk_cfg, gv, method):
    """把所選方法的『週共變異數』逐週存成 {date: (幣清單, Sigma)}，交給 C 的最佳化器使用。
    讀取方式: pandas.read_pickle('risk_cov_selected.pkl')。"""
    out = {}
    for (t, ok, w, S1, delta, R, c) in iter_weekly_inputs(m, universe, cfg, return_window=True):
        covs = covariance_models(R, gv["daily_vol"], gv["weekly_vol"].loc[c], [method], risk_cfg["pca_components"])
        if method in covs:
            out[t] = (ok, covs[method])
    return out
