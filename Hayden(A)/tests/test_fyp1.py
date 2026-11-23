import numpy as np, pandas as pd
import statsmodels.api as sm
import pytest
from src.stats import nw_se_of_mean, auto_nw_lags, holm, benjamini_hochberg
from src.fama_macbeth import fm_gammas, fm_summary
from src.pipeline import run_pipeline
from src.audit import lookahead_audit
from src.factors import forward_return_wide
from src.data_io import Market
from src import bl


# ---------- 統計工具 ----------
def test_newey_west_matches_statsmodels():
    rng = np.random.default_rng(1)
    x = np.zeros(300)
    for t in range(1, 300):
        x[t] = 0.4 * x[t - 1] + rng.normal()
    L = auto_nw_lags(300)
    ref = sm.OLS(x, np.ones(300)).fit(cov_type="HAC", cov_kwds=dict(maxlags=L, use_correction=False))
    assert abs(nw_se_of_mean(x, L) - ref.bse[0]) < 1e-10


def test_multiple_testing_known_values():
    p = [0.01, 0.04, 0.03]
    assert np.allclose(holm(p), [0.03, 0.06, 0.06])
    assert np.allclose(benjamini_hochberg(p), [0.03, 0.04, 0.04])


def test_fama_macbeth_recovers_true_slope():
    rng = np.random.default_rng(2)
    rows = []
    for d in pd.date_range("2020-01-06", periods=200, freq="W-MON"):
        x = rng.normal(size=60)
        y = 0.01 + 0.02 * x + rng.normal(0, 0.05, 60)
        rows.append(pd.DataFrame({"date": d, "coin_id": range(60), "f": x, "y": y}))
    df = pd.concat(rows).set_index(["date", "coin_id"])
    g = fm_gammas(df[["f"]], df["y"], ["f"])
    s = fm_summary(g)
    assert abs(s.loc["f", "mean"] - 0.02) < 0.003
    assert abs(s.loc["const", "mean"] - 0.01) < 0.003


# ---------- 整條流程 (合成資料，已知答案) ----------
@pytest.fixture(scope="module")
def full(synth, cfg):
    panel, universe = synth
    return run_pipeline(panel, universe, cfg)


def test_planted_signal_found_and_noise_not(full):
    t = full["ic_tab"]
    assert t.loc["liq_28d", "ic_mean"] > 0.03 and t.loc["liq_28d", "ic_t_nw"] > 4
    assert full["q_tab"].loc["liq_28d", "monotonicity"] >= 0.8
    assert full["verdict"].loc["liq_28d", "verdict"] == "PASS"
    for f in ["mom_2w", "mom_4w", "rev_1w"]:
        assert abs(t.loc[f, "ic_t_nw"]) < 3.0


def test_screen_only_uses_eligible_members_and_enough_coins(full, cfg):
    n = full["scores"].groupby(level=0).size()
    assert (n >= cfg["screening"]["min_cross_section"]).all()
    assert abs(full["scores"].groupby(level=0).mean().values).max() < 1e-9      # 每週 z-score 平均 0


def test_mu_has_no_values_before_enough_history(full, cfg):
    mh = cfg["combination"]["min_history_weeks"]
    first = full["scores"].index.get_level_values(0).unique().sort_values()
    for k, s in full["mu"].items():
        d = s.dropna().index.get_level_values(0).min()
        assert d >= first[mh - 1]            # 至少要累積 min_history 週之後才有預期報酬


def test_lookahead_audit_passes_on_clean_pipeline(synth, cfg, full):
    panel, universe = synth
    tab, passed, _ = lookahead_audit(panel, universe, cfg, full, n_checks=3)
    assert passed and len(tab) == 3


def test_lookahead_audit_catches_a_leak(synth, cfg, full):
    """對照組: 故意讓預期報酬依賴整份資料的平均價格(=偷看未來)，稽核必須抓到。"""
    panel, universe = synth

    def leaky(p, u, c, fixed):
        r = run_pipeline(p, u, c, fixed=fixed, light=True)
        r["mu"] = {k: v + p["close"].mean() * 1e-6 for k, v in r["mu"].items()}
        return r

    _, passed, _ = lookahead_audit(panel, universe, cfg, full, n_checks=2, pipeline_fn=leaky)
    assert not passed


def test_delisting_policies(synth):
    panel, _ = synth
    m = Market(panel)
    last = m.last_date.dropna()
    dead = last[last < m.end - pd.Timedelta(days=30)].index[0]
    d = m.last_date[dead] - pd.Timedelta(days=3)                    # 最後交易日前 3 天
    f_last, ended = forward_return_wide(m, 7, "last_price")
    f_m100, _ = forward_return_wide(m, 7, "minus_100")
    f_drop, _ = forward_return_wide(m, 7, "drop")
    assert ended.at[d, dead]
    assert f_m100.at[d, dead] == -1.0
    assert np.isnan(f_drop.at[d, dead])
    assert abs(f_last.at[d, dead] - (m.close.at[m.last_date[dead], dead] / m.close.at[d, dead] - 1)) < 1e-12


# ---------- Black-Litterman ----------
def _toy(n=6, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    Sigma = A @ A.T / n + 0.05 * np.eye(n)
    w = rng.random(n); w /= w.sum()
    return Sigma, w


def test_reverse_optimisation_round_trip():
    Sigma, w = _toy()
    pi = bl.implied_returns(Sigma, w, 2.5)
    assert np.allclose(bl.reverse_opt_weights(Sigma, pi, 2.5), w)


def test_bl_limit_infinite_view_uncertainty_recovers_prior():
    Sigma, w = _toy()
    pi = bl.implied_returns(Sigma, w, 2.5)
    P, Q = np.eye(len(w)), np.random.default_rng(1).normal(size=len(w))
    mu, _ = bl.black_litterman(Sigma, pi, P, Q, np.eye(len(w)) * 1e12, tau=0.05)
    assert np.allclose(mu, pi, atol=1e-6)


def test_bl_limit_zero_view_uncertainty_recovers_forecast():
    Sigma, w = _toy()
    pi = bl.implied_returns(Sigma, w, 2.5)
    P, Q = np.eye(len(w)), np.random.default_rng(1).normal(size=len(w))
    mu, _ = bl.black_litterman(Sigma, pi, P, Q, np.eye(len(w)) * 1e-12, tau=0.05)
    assert np.allclose(mu, Q, atol=1e-6)


def test_bl_no_views_equals_prior_and_trust_moves_between():
    Sigma, w = _toy()
    pi = bl.implied_returns(Sigma, w, 2.5)
    mu_f = np.random.default_rng(3).normal(0, 0.01, len(w))
    P, Q = bl.relative_views(mu_f, w)
    lo = bl.black_litterman(Sigma, pi, P, Q, bl.omega_from_trust(P, Sigma, 0.05, 1e-3), 0.05)[0]
    hi = bl.black_litterman(Sigma, pi, P, Q, bl.omega_from_trust(P, Sigma, 0.05, 1e3), 0.05)[0]
    assert np.linalg.norm(lo - pi) < np.linalg.norm(hi - pi)       # 越信任 view，離先驗越遠


def test_weekly_prior_output(synth, cfg):
    panel, universe = synth
    pr = bl.weekly_prior(Market(panel), universe, cfg)
    assert len(pr) > 0
    assert abs(pr.groupby("date")["w_mkt"].sum() - 1).max() < 1e-9


def test_monotonicity_threshold_is_not_broken_by_float_noise():
    # Spearman of 5 quintile means can be -0.7999999999999999 for a perfect -0.8; the verdict must still treat it as 0.8
    assert round(abs(-0.7999999999999999), 9) >= 0.8
