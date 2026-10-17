import numpy as np, pandas as pd, pytest, yaml, os, json
from scipy import stats
from src.cv import purged_walk_forward, check_no_overlap
from src import evaluate as ev
from src.forecast import (QuantileForecaster, cqr_adjustment, apply_adjustment, walk_forward_forecast,
                          build_features, sigma_from_interval, pinball, diagnostics)
from src import bl, bl_integration as bi
from src import risk
from src.pipeline import run_pipeline
from src.data_io import Market
from src import prereg


# ================= 清除與禁運切分 =================
def _weeks(n=313, start="2020-09-21"):
    return pd.date_range(start, periods=n, freq="W-MON")


def test_cv_no_overlap_and_contiguous_blocks():
    d = _weeks()
    folds = purged_walk_forward(d, 156, 26, 2, 2, 1)
    assert len(folds) == 6                                   # 約 6 個 fold (不是報告原先寫的 9 個)
    assert check_no_overlap(folds, 1)
    prev_end = None
    for f in folds:
        assert f.train_dates.max() < f.test_dates.min()
        gap_weeks = (f.test_dates.min() - f.train_dates.max()).days // 7
        assert gap_weeks >= 2 + 1 - 1 and gap_weeks >= 3     # 清除 2 週 (+ 基本 1 週間隔)
        assert len(f.train_dates) >= 156 - 2
        if prev_end is not None:
            assert f.test_dates.min() > prev_end             # 測試區塊不重疊、依序排列
        prev_end = f.test_dates.max()


def test_cv_purge_must_cover_horizon():
    with pytest.raises(ValueError):
        purged_walk_forward(_weeks(), 156, 26, purge_weeks=0, embargo_weeks=0, horizon_weeks=1)


def test_cv_train_set_is_expanding():
    folds = purged_walk_forward(_weeks(), 156, 26, 2, 2, 1)
    sizes = [len(f.train_dates) for f in folds]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)


# ================= 評估套件 (已知答案) =================
def test_sharpe_es_drawdown_known_values():
    r = np.array([0.1, -0.5, 0.2])
    assert abs(ev.max_drawdown(r) - (-0.5)) < 1e-12
    assert abs(ev.expected_shortfall(np.arange(-9, 11) / 100, 0.1) - (-0.085)) < 1e-12     # 最差 10% = {-9%, -8%}
    x = np.array([0.01, 0.02, 0.03, 0.04])
    assert abs(ev.sharpe(x, annualise=False) - x.mean() / x.std(ddof=1)) < 1e-12


def test_psr_matches_closed_form_for_normal_returns():
    sr, n = 0.1, 100
    expected = stats.norm.cdf(sr * np.sqrt(n - 1) / np.sqrt(1 + sr ** 2 / 2))
    assert abs(ev.probabilistic_sharpe(sr, n, 0.0, 3.0, 0.0) - expected) < 1e-12


def test_dsr_properties():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.004, 0.05, 260))
    one = ev.deflated_sharpe(r, 1, 0.001)
    sr = one["sr_per_period"]
    assert abs(one["dsr"] - ev.probabilistic_sharpe(sr, 260, float(stats.skew(r)), float(stats.kurtosis(r, fisher=False)), 0.0)) < 1e-12
    d10 = ev.deflated_sharpe(r, 10, 0.001)["dsr"]
    d100 = ev.deflated_sharpe(r, 100, 0.001)["dsr"]
    assert one["dsr"] > d10 > d100                           # 試的次數越多，越難顯著
    assert ev.expected_max_sharpe(100, 0.001) > ev.expected_max_sharpe(10, 0.001) > 0


def test_bootstrap_ci_covers_truth_and_paired_diff():
    rng = np.random.default_rng(3)
    r = rng.normal(0.01, 0.05, 400)
    ci = ev.bootstrap_ci(r, np.mean, n_boot=600, block=4, seed=1)
    assert ci["lo"] < 0.01 < ci["hi"] and ci["lo"] < ci["estimate"] < ci["hi"]
    d = ev.bootstrap_diff_ci(r + 0.02, r, np.mean, n_boot=600, seed=1)
    assert abs(d["estimate"] - 0.02) < 1e-12 and d["lo"] > 0 and d["p_value_le_0"] == 0.0


def test_subperiod_tables_contain_stress_windows():
    idx = pd.date_range("2020-01-06", "2023-01-02", freq="W-MON")
    df = pd.DataFrame({"a": np.random.default_rng(0).normal(0, 0.05, len(idx))}, index=idx)
    t = ev.subperiod_tables(df)
    assert {"full", "year_2021", "stress_COVID", "stress_Luna", "stress_FTX"} <= set(t)


# ================= 預測層 =================
@pytest.fixture(scope="module")
def res1(synth, cfg):
    panel, universe = synth
    return run_pipeline(panel, universe, cfg, light=True)


CV = dict(min_train_weeks=104, test_weeks=26, purge_weeks=2, embargo_weeks=2, horizon_weeks=1)
FC = dict(quantiles=[0.1, 0.5, 0.9], target="demeaned", grid={"max_depth": [2], "n_estimators": [150]},
          tune=False, calibration="cqr", calib_weeks=26, target_coverage=0.8, seed=0)


@pytest.fixture(scope="module")
def fo(res1):
    X = build_features(res1["scores"])
    return walk_forward_forecast(X, res1["fwd"], CV, FC)


def test_quantiles_do_not_cross(fo):
    P = fo["preds"]
    assert (P["q10"] <= P["q50"] + 1e-12).all() and (P["q50"] <= P["q90"] + 1e-12).all()


def test_forecast_interval_is_calibrated_out_of_sample(fo):
    d = fo["diagnostics"]
    assert 0.74 <= d["coverage_q10_q90"] <= 0.86             # 80% 區間實際涵蓋率
    assert 0.44 <= d["share_below_q50"] <= 0.56
    assert d["oos_ic_t_nw"] > 2.5                           # 預先植入的流動性訊號在樣本外找得到


def test_predictions_only_on_test_dates_after_purged_training(fo, res1):
    folds = fo["folds"]
    for f in folds:
        pf = fo["preds"][fo["preds"]["fold"] == f.k].index.get_level_values(0).unique()
        assert set(pf) <= set(f.test_dates) and f.train_dates.max() < pf.min()


def test_cqr_widens_a_too_narrow_interval():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(0, 1, 5000))
    narrow = pd.DataFrame({"q10": -0.5, "q90": 0.5}, index=y.index)
    adj = cqr_adjustment(y.iloc[:2500], narrow.iloc[:2500], target_coverage=0.8)
    new = apply_adjustment(narrow.iloc[2500:], adj)
    cov = ((y.iloc[2500:] >= new["q10"]) & (y.iloc[2500:] <= new["q90"])).mean()
    assert adj > 0.3 and 0.77 < cov < 0.83


def test_forecast_has_no_lookahead(res1):
    """未來(測試期之後)的目標被亂改，fold 0 的預測必須完全不變；改『訓練期』的目標則必須改變(對照組)。"""
    X = build_features(res1["scores"])
    base = walk_forward_forecast(X, res1["fwd"], CV, FC)
    f0, f1 = base["folds"][0], base["folds"][1]
    p0 = base["preds"][base["preds"]["fold"] == 0]
    rng = np.random.default_rng(1)
    y2 = res1["fwd"].copy()
    late = y2.index.get_level_values(0) >= f1.test_dates.min()
    y2[late] = rng.normal(0, 0.1, late.sum())
    alt = walk_forward_forecast(X, y2, CV, FC)
    q0 = alt["preds"][alt["preds"]["fold"] == 0]
    assert np.allclose(p0[["q10", "q50", "q90"]].values, q0[["q10", "q50", "q90"]].values, atol=1e-12)
    y3 = res1["fwd"].copy()
    inside = y3.index.get_level_values(0).isin(f0.train_dates[:60])
    y3[inside] = rng.normal(0, 0.1, inside.sum())
    ctrl = walk_forward_forecast(X, y3, CV, FC)
    c0 = ctrl["preds"][ctrl["preds"]["fold"] == 0]
    assert not np.allclose(p0[["q10", "q50", "q90"]].values, c0[["q10", "q50", "q90"]].values, atol=1e-6)


# ================= Black-Litterman 整合 =================
def _toy_bl(n=6):
    Sigma = np.eye(n) * 0.01
    w = np.ones(n) / n
    return Sigma, w


def test_bl_trust_moves_posterior_away_from_prior_monotonically():
    Sigma, w = _toy_bl()
    q50 = np.array([0.02, 0.01, 0, 0, -0.01, -0.02])
    sg = np.full(6, 0.05)
    d = []
    for tr in [0.001, 0.1, 1, 10, 1000]:
        pi, mu = bi.posterior_one_date(Sigma, w, 2.5, q50, sg, 0.05, tr)
        d.append(np.abs(mu - pi).mean())
    assert d == sorted(d) and d[0] < 1e-3 * max(d[-1], 1e-9) * 1e3
    pi0, mu0 = bi.posterior_one_date(Sigma, w, 2.5, q50, sg, 0.05, 1e-9)
    assert np.allclose(pi0, mu0, atol=1e-8)                  # trust -> 0: 回到先驗


def test_bl_wider_interval_means_smaller_tilt():
    Sigma, w = _toy_bl()
    q50 = np.array([0.01, 0.01, 0, 0, 0, 0])
    sg = np.array([1.0, 3.0, 1, 1, 1, 1])
    pi, mu = bi.posterior_one_date(Sigma, w, 2.5, q50, sg, 0.05, 1.0)
    assert abs(mu[0] - pi[0]) > abs(mu[1] - pi[1]) > 0


def test_bl_handles_missing_views():
    Sigma, w = _toy_bl()
    q50 = np.array([0.02, np.nan, 0, np.nan, -0.01, -0.02])
    sg = np.array([0.05, np.nan, 0.05, np.nan, 0.05, 0.05])
    pi, mu = bi.posterior_one_date(Sigma, w, 2.5, q50, sg, 0.05, 1.0)
    assert np.isfinite(mu).all()


def test_bl_weekly_runs_and_distance_is_monotone(synth, cfg, res1, fo):
    panel, universe = synth
    df = bi.bl_weekly(res1["market"], universe, cfg, fo["preds"], [0.01, 1.0, 100.0])
    dist = bi.distance_to_prior(df)["relative_distance_to_prior"].values
    assert len(df) > 0 and (np.diff(dist) > 0).all()


# ================= 風險模型 =================
def test_garch_recovers_persistence_and_forecast_formula():
    rng = np.random.default_rng(1)
    n, om, al, be = 4000, 2e-5, 0.08, 0.88
    s2, r = np.empty(n), np.empty(n)
    s2[0] = om / (1 - al - be)
    for t in range(n):
        if t:
            s2[t] = om + al * r[t - 1] ** 2 + be * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * rng.standard_t(6) / np.sqrt(6 / 4)
    f = risk.garch11_fit(r)
    assert f is not None and abs((f[1] + f[2]) - (al + be)) < 0.04
    o, a, b = f
    assert abs(risk.garch_forecast_variance(1e-3, 0.01, o, a, b, h=1) - (o + a * 0.01 ** 2 + b * 1e-3)) < 1e-15
    V = o / (1 - a - b)
    # 已經在無條件變異數 (s2 = V 且 r^2 = V) 時，任何期數的預測都應恰好等於 h * V
    assert abs(risk.garch_forecast_variance(V, np.sqrt(V), o, a, b, h=50) / 50 - V) / V < 1e-9
    # 從 5V 開始，期數拉很長時平均變異數趨向 V (均值回歸)
    assert abs(risk.garch_forecast_variance(5 * V, 0.0, o, a, b, h=20000) / 20000 - V) / V < 0.02


def test_garch_vol_panel_uses_only_past_data(synth):
    panel, _ = synth
    m = Market(panel)
    ret = m.ret1d.iloc[:, :6]
    cutoffs = pd.date_range("2020-01-05", periods=40, freq="W-SUN")
    full = risk.garch_vol_panel(ret, cutoffs, refit_every=8, min_obs=250)
    k = 25
    trunc = risk.garch_vol_panel(ret.loc[:cutoffs[k]], cutoffs[: k + 1], refit_every=8, min_obs=250)
    a, b = full["weekly_vol"].iloc[: k + 1], trunc["weekly_vol"]
    assert np.allclose(a.values, b.values, equal_nan=True, atol=1e-12)
    da, db = full["daily_vol"].loc[:cutoffs[k]], trunc["daily_vol"].loc[:cutoffs[k]]
    assert np.allclose(da.values, db.values, equal_nan=True, atol=1e-12)


def test_correlation_estimators_are_valid():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(300, 1))
    Z = f @ rng.normal(size=(1, 12)) + rng.normal(size=(300, 12))
    for method in ["sample", "ledoit_wolf", "pca", "const"]:
        C = risk.correlation_estimate(Z, method, k=2)
        assert np.allclose(np.diag(C), 1, atol=1e-8) and np.allclose(C, C.T, atol=1e-10)
        assert np.linalg.eigvalsh(C).min() > -1e-8
    Cc = risk.correlation_estimate(Z, "const")
    off = Cc[~np.eye(12, dtype=bool)]
    assert np.allclose(off, off[0])
    S = risk.build_cov(np.full(12, 0.1), risk.correlation_estimate(Z, "pca", k=2))
    assert np.linalg.eigvalsh(S).min() > -1e-10


def test_gmv_longonly_constraints_and_optimality():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(10, 10))
    S = A @ A.T / 10 + 0.02 * np.eye(10)
    w = risk.gmv_longonly(S, cap=0.25)
    assert abs(w.sum() - 1) < 1e-8 and w.min() >= -1e-12 and w.max() <= 0.25 + 1e-6
    ew = np.ones(10) / 10
    assert w @ S @ w <= ew @ S @ ew + 1e-10
    # 上限小於 1/N 時會自動提高到 1/N，不會無解
    w2 = risk.gmv_longonly(S, cap=0.01)
    assert abs(w2.sum() - 1) < 1e-8


# ================= 預先登記 =================
def test_prereg_lock_detects_changes(tmp_path):
    cfgp, lockp = tmp_path / "c.yaml", tmp_path / "c.lock"
    cfgp.write_text("a: 1\nb: [1, 2]\n", encoding="utf-8")
    prereg.lock(str(cfgp), str(lockp))
    assert prereg.verify(str(cfgp), str(lockp))[0]
    cfgp.write_text("a: 2\nb: [1, 2]\n", encoding="utf-8")
    assert not prereg.verify(str(cfgp), str(lockp))[0]
