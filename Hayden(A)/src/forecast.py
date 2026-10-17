"""預測層: 梯度提升的分位數模型 (10% / 50% / 90%)。
中位數預測 = view 的數值；(q90 - q10) 的寬度 = 該幣預測有多不確定。
所有調參都只在「訓練視窗內部」做，測試區塊只用來評估一次。"""
import numpy as np
import pandas as pd
from scipy import stats
from .cv import purged_walk_forward, check_no_overlap
from .stats import mean_tstat

Z90 = 1.2815515655446004      # 常態下 q90 的 z 值


def _has_xgb():
    try:
        import xgboost  # noqa
        return True
    except Exception:
        return False


class QuantileForecaster:
    """每個分位數一個模型。後端: xgboost (預設)，沒有安裝時自動退回 sklearn。"""

    def __init__(self, quantiles=(0.1, 0.5, 0.9), params=None, backend="auto", seed=0):
        self.q = tuple(quantiles)
        self.params = dict(n_estimators=150, max_depth=2, learning_rate=0.1, min_child_weight=20,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0)
        self.params.update(params or {})
        self.backend = ("xgboost" if _has_xgb() else "sklearn") if backend == "auto" else backend
        self.seed = seed
        self.models = {}

    def _make(self, tau):
        p = self.params
        if self.backend == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(objective="reg:quantileerror", quantile_alpha=tau, tree_method="hist",
                                n_jobs=1, random_state=self.seed, verbosity=0, **p)
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(loss="quantile", alpha=tau, n_estimators=p["n_estimators"],
                                         max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                                         min_samples_leaf=int(p["min_child_weight"]), subsample=p["subsample"],
                                         random_state=self.seed)

    def _prep(self, X):
        A = X.values.astype(float)
        return A if self.backend == "xgboost" else np.nan_to_num(A)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        ok = y.notna().values
        A, t = self._prep(X)[ok], y.values[ok]
        self.models = {tau: self._make(tau).fit(A, t) for tau in self.q}
        self.columns = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        A = self._prep(X[self.columns])
        P = np.column_stack([self.models[tau].predict(A) for tau in self.q])
        P = np.sort(P, axis=1)                                  # 避免分位數交叉 (q10 <= q50 <= q90)
        return pd.DataFrame(P, index=X.index, columns=[f"q{int(round(t * 100))}" for t in self.q])


# ---------- 損失與診斷 ----------
def pinball(y, q, tau):
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.nanmean(np.maximum(tau * d, (tau - 1) * d)))


def mean_pinball(y, preds: pd.DataFrame, quantiles):
    return float(np.mean([pinball(y, preds[f"q{int(round(t * 100))}"], t) for t in quantiles]))


def sigma_from_interval(preds: pd.DataFrame, lo="q10", hi="q90", z=Z90):
    """把 [q10, q90] 的寬度換成「相當於常態的標準差」: (q90 - q10) / (2 * 1.2816)。"""
    return (preds[hi] - preds[lo]) / (2 * z)


def ic_based_sigma(sigma_cs: pd.Series, ic_rolling: pd.Series) -> pd.Series:
    """較簡單的替代做法 (報告 4.5 節): 用滾動 IC 決定不確定性。IC 越高 → 不確定性越小。
    sigma = sigma_cs * sqrt(1 - max(IC, 0)^2)。輸入都以日期為 index。"""
    return sigma_cs * np.sqrt(1 - np.clip(ic_rolling, 0, 0.999) ** 2)


def diagnostics(y: pd.Series, preds: pd.DataFrame, quantiles=(0.1, 0.5, 0.9), baseline: pd.DataFrame = None) -> dict:
    """樣本外診斷。y 與 preds 以 (date, coin) 對齊。"""
    d = preds.join(y.rename("y"), how="left").dropna(subset=["y"])     # 只評估「有預測值」的列
    out = {}
    out["n_obs"] = len(d)
    out["coverage_q10_q90"] = float(((d["y"] >= d["q10"]) & (d["y"] <= d["q90"])).mean())    # 理想 ≈ 0.80
    for t in quantiles:
        c = f"q{int(round(t * 100))}"
        out[f"share_below_{c}"] = float((d["y"] < d[c]).mean())                                # 理想 ≈ t
    out["mean_pinball"] = mean_pinball(d["y"], d, quantiles)
    if baseline is not None:
        b = baseline.reindex(d.index)
        out["mean_pinball_baseline"] = mean_pinball(d["y"], b, quantiles)
        out["skill_vs_unconditional"] = 1 - out["mean_pinball"] / out["mean_pinball_baseline"]  # >0 才表示有加分
    ic = {}
    for dt, g in d.groupby(level=0):
        if len(g) >= 10:
            ic[dt] = g["q50"].rank().corr(g["y"].rank())
    ic = pd.Series(ic, dtype=float).sort_index()
    r = mean_tstat(ic.values)
    out.update(oos_ic_mean=r["mean"], oos_ic_t_nw=r["t"], oos_icir=float(ic.mean() / ic.std(ddof=1)) if len(ic) > 2 else np.nan,
               oos_ic_weeks=r["n"])
    width = (d["q90"] - d["q10"])
    err = (d["y"] - d["q50"]).abs()
    out["corr_width_abs_error"] = float(width.rank().corr(err.rank()))     # 寬度有資訊量的話應 > 0
    return out


# ---------- 調參 (只用訓練視窗內部) ----------
def select_params(X, y, grid, quantiles, inner_val_weeks=26, purge_weeks=2, backend="auto", seed=0):
    """訓練視窗內最後 inner_val_weeks 週當驗證，驗證前清除 purge_weeks 週。回傳 (最佳參數, 每組的損失表)。"""
    dates = X.index.get_level_values(0).unique().sort_values()
    if len(dates) < inner_val_weeks + purge_weeks + 26 or len(grid) == 1:
        return grid[0], pd.DataFrame({"loss": [np.nan] * len(grid)})
    val_d = dates[-inner_val_weeks:]
    tr_d = dates[: len(dates) - inner_val_weeks - purge_weeks]
    Xtr, ytr = X[X.index.get_level_values(0).isin(tr_d)], y.reindex(X.index)[X.index.get_level_values(0).isin(tr_d)]
    Xva, yva = X[X.index.get_level_values(0).isin(val_d)], y.reindex(X.index)[X.index.get_level_values(0).isin(val_d)]
    rows = []
    for p in grid:
        m = QuantileForecaster(quantiles, p, backend, seed).fit(Xtr, ytr)
        rows.append(dict(p, loss=mean_pinball(yva, m.predict(Xva), quantiles)))
    tab = pd.DataFrame(rows)
    best = int(tab["loss"].idxmin())
    return grid[best], tab


def make_grid(grid_cfg: dict):
    import itertools
    keys = list(grid_cfg.keys())
    return [dict(zip(keys, v)) for v in itertools.product(*[grid_cfg[k] for k in keys])]


# ---------- 保形校準 (Conformalized Quantile Regression, Romano et al. 2019) ----------
def cqr_adjustment(y: pd.Series, preds: pd.DataFrame, lo="q10", hi="q90", target_coverage=0.8) -> float:
    """在「校準集」上算出要把區間往兩邊各擴大(或縮小)多少，使區間涵蓋率 ≈ target_coverage。
    不確定性之所以需要校準，是因為梯度提升的分位數模型在尾端常常欠擬合 (區間偏窄)。"""
    d = pd.concat([y.rename("y"), preds[[lo, hi]]], axis=1).dropna()
    if len(d) < 30:
        return 0.0
    score = np.maximum(d[lo] - d["y"], d["y"] - d[hi])
    n = len(score)
    level = min(np.ceil((n + 1) * target_coverage) / n, 1.0)
    return float(np.quantile(score, level))


def apply_adjustment(preds: pd.DataFrame, adj: float, lo="q10", hi="q90") -> pd.DataFrame:
    out = preds.copy()
    out[lo] = out[lo] - adj
    out[hi] = out[hi] + adj
    return out


# ---------- walk-forward ----------
def build_features(scores: pd.DataFrame, mu: dict = None, include_mu=False) -> pd.DataFrame:
    X = scores.copy()
    if include_mu and mu:
        for k, s in mu.items():
            X[f"mu_{k}"] = s.reindex(X.index)
    return X


def demean_by_date(y: pd.Series) -> pd.Series:
    return y - y.groupby(level=0).transform("mean")


def _qcols(quantiles):
    return [f"q{int(round(t * 100))}" for t in quantiles]


def walk_forward_forecast(X: pd.DataFrame, y_raw: pd.Series, cv_cfg: dict, fc_cfg: dict):
    """清除+禁運的 expanding walk-forward。
    每個 fold: 訓練視窗最後 calib_weeks 週留作『校準集』(其前再清除 purge_weeks 週)，
    模型只用更早的資料訓練；校準集用來把 [q10, q90] 調整到目標涵蓋率；測試區塊只評估一次。"""
    quantiles = tuple(fc_cfg["quantiles"])
    qc = _qcols(quantiles)
    y = demean_by_date(y_raw) if fc_cfg.get("target", "demeaned") == "demeaned" else y_raw
    y = y.reindex(X.index)
    dates = X.index.get_level_values(0).unique().sort_values()
    purge = cv_cfg["purge_weeks"]
    folds = purged_walk_forward(dates, cv_cfg["min_train_weeks"], cv_cfg["test_weeks"], purge,
                                cv_cfg["embargo_weeks"], cv_cfg.get("horizon_weeks", 1))
    if not folds:
        raise ValueError("歷史不夠長，切不出任何 fold。請檢查 min_train_weeks 與樣本長度")
    assert check_no_overlap(folds, cv_cfg.get("horizon_weeks", 1))
    grid = make_grid(fc_cfg["grid"])
    use_cqr = fc_cfg.get("calibration", "cqr") == "cqr"
    calib_w, target_cov = int(fc_cfg.get("calib_weeks", 26)), float(fc_cfg.get("target_coverage", 0.8))
    lvl = lambda idx, ds: idx.get_level_values(0).isin(ds)
    raw_p, cal_p, base, chosen = [], [], [], []
    for f in folds:
        tr_dates = f.train_dates
        if use_cqr:
            calib_d = tr_dates[-calib_w:]
            proper_d = tr_dates[: max(len(tr_dates) - calib_w - purge, 0)]
        else:
            calib_d, proper_d = tr_dates[:0], tr_dates
        Xtr, ytr = X[lvl(X.index, proper_d)], y[lvl(X.index, proper_d)]
        if fc_cfg.get("tune", True):
            best, _ = select_params(Xtr, ytr, grid, quantiles, fc_cfg.get("inner_val_weeks", 26), purge,
                                    fc_cfg.get("backend", "auto"), fc_cfg.get("seed", 0))
        else:
            best = grid[0]
        model = QuantileForecaster(quantiles, best, fc_cfg.get("backend", "auto"), fc_cfg.get("seed", 0)).fit(Xtr, ytr)
        P = model.predict(X[lvl(X.index, f.test_dates)])
        adj = 0.0
        if use_cqr and len(calib_d):
            Xc = X[lvl(X.index, calib_d)]
            adj = cqr_adjustment(y.reindex(Xc.index), model.predict(Xc), target_coverage=target_cov)
        Pc = apply_adjustment(P, adj)
        for d_ in (P, Pc):
            d_["fold"] = f.k
        Pc["cqr_adj"] = adj
        raw_p.append(P), cal_p.append(Pc)
        yy = y[lvl(X.index, tr_dates)].dropna()
        base.append(pd.DataFrame({c: float(np.nanquantile(yy, t)) for c, t in zip(qc, quantiles)}, index=P.index))
        chosen.append(dict(best, fold=f.k, train_weeks=len(proper_d), calib_weeks=len(calib_d), test_weeks=len(f.test_dates),
                           cqr_adj=adj, backend=model.backend))
    R, C, B = pd.concat(raw_p), pd.concat(cal_p), pd.concat(base)
    diag_raw = diagnostics(y, R[qc], quantiles, B)
    diag = diagnostics(y, C[qc], quantiles, B)
    diag["n_folds"] = len(folds)
    return dict(preds=C, preds_raw=R, folds=folds, params=pd.DataFrame(chosen), baseline=B,
                diagnostics=diag, diagnostics_raw=diag_raw, n_trials=len(grid), target=y)
