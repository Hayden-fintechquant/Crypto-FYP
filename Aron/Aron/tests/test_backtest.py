"""Known-answer tests for the fixed engine and allocators (run: python -m pytest tests -q  |  or any runner that calls test_*)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.backtest_engine import BacktestData, load_config, run_backtest, performance, coin_period_returns
from src.allocators import cap_weights, equal_weight, market_cap, minimum_variance, risk_parity, mean_variance, black_litterman, btc_buy_hold

CFG = load_config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_backtest.yaml"))
CFG["portfolio"]["min_names"] = 1
CFG["tradeable"]["min_avg_volume_usd"] = 0
CFG["costs"] = {"fee_bps_one_way": 0, "slippage_bps_one_way": 0}


def make_data(prices, mcaps=None, weeks=None, cfg=None, collapsed=()):
    """prices: {coin: array of daily closes starting 2023-01-01 (Sunday)}; universe Mondays 2023-01-02, +7d, ..."""
    n = max(len(v) for v in prices.values())
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    rows = []
    for k, v in prices.items():
        for d, x in zip(idx[:len(v)], v):
            if not np.isnan(x):
                rows.append(dict(date=d, coin_id=k, close=x, volume_usd=5e6, mcap=(mcaps or {}).get(k, 1e9),
                                 end_reason="collapsed" if k in collapsed else np.nan))
    mondays = pd.date_range("2023-01-02", periods=weeks or (n // 7), freq="7D")
    uni = pd.DataFrame([dict(rebalance_date=d, coin_id=k, is_member=1) for d in mondays for k in prices])
    return BacktestData(pd.DataFrame(rows), uni, cfg or CFG)


def synth_prices(n_coins=15, T=400, seed=0, vols=None):
    rng = np.random.default_rng(seed)
    vols = vols if vols is not None else np.linspace(0.02, 0.07, n_coins)
    common = rng.normal(0, 0.02, T)
    r = common[:, None] * 0.6 + rng.normal(0, 1, (T, n_coins)) * vols
    return {f"c{j}": 100 * np.cumprod(1 + r[:, j]) for j in range(n_coins)}


# ------------------------------------------------------------ timing / returns
def test_weekly_return_is_7_days():
    p = {"A": 100 * 1.01 ** np.arange(40)}
    d = make_data(p, weeks=4)
    log, _, _ = run_backtest(d, "equal_weight", start="2023-01-02", end="2023-01-30")
    assert np.allclose(log["gross_ret"], 1.01 ** 7 - 1)            # first version gave 1.01**6 - 1


def test_entry_and_exit_use_sunday_close():
    p = {"A": np.array([100.0] * 7 + [100, 110, 110, 110, 110, 110, 110, 121, 121])}   # Sunday Jan1..; Sun Jan8=100->...
    d = make_data(p, weeks=2)
    c = pd.Timestamp("2023-01-08"); c_next = pd.Timestamp("2023-01-15")
    r, n = coin_period_returns(d, ["A"], c, c_next)
    assert abs(r["A"] - (d.close.loc[c_next, "A"] / d.close.loc[c, "A"] - 1)) < 1e-12 and n == 0


def test_missing_exit_price_uses_last_traded_price():
    a = 100.0 * np.ones(30)
    b = np.concatenate([np.full(10, 100.0), [105.0, 90.0], np.full(18, np.nan)])      # trades until Jan 12 (90), then gone
    d = make_data({"A": a, "B": b}, weeks=3)
    c, c_next = pd.Timestamp("2023-01-08"), pd.Timestamp("2023-01-15")
    r, n_noexit = coin_period_returns(d, ["A", "B"], c, c_next)
    assert abs(r["B"] - (90 / 100 - 1)) < 1e-12 and abs(r["A"]) < 1e-12 and n_noexit == 0


def test_collapsed_coin_shows_real_loss():
    b = np.concatenate([np.full(9, 100.0), [60.0, 20.0, 1.0, 0.01, 0.0001], np.full(20, np.nan)])
    d = make_data({"A": np.full(30, 100.0), "LUNA": b}, weeks=3, collapsed=("LUNA",))
    c, c_next = pd.Timestamp("2023-01-08"), pd.Timestamp("2023-01-15")
    r, _ = coin_period_returns(d, ["A", "LUNA"], c, c_next)
    assert r["LUNA"] < -0.999          # first version: silently 0%


def test_held_coin_with_no_price_after_entry_is_counted_not_silent():
    b = np.concatenate([np.full(8, 100.0), np.full(22, np.nan)])
    d = make_data({"A": np.full(30, 100.0), "B": b}, weeks=3)
    r, n = coin_period_returns(d, ["A", "B"], pd.Timestamp("2023-01-08"), pd.Timestamp("2023-01-15"))
    assert n == 1 and r["B"] == 0.0


def test_coin_without_price_at_cutoff_is_not_tradeable():
    b = np.concatenate([np.full(7, np.nan), np.full(23, 100.0)])         # first price Jan 8 -> not tradeable for Monday Jan 2
    d = make_data({"A": np.full(30, 100.0), "B": b}, weeks=3)
    assert d.make_ctx("2023-01-02").coins == ["A"] and set(d.make_ctx("2023-01-09").coins) == {"A", "B"}


# ------------------------------------------------------------ costs / turnover
def test_turnover_first_week_is_100pct_and_cost_formula():
    cfg = {**CFG, "costs": {"fee_bps_one_way": 15, "slippage_bps_one_way": 5}}
    d = make_data({"A": np.full(40, 100.0), "B": np.full(40, 100.0)}, weeks=5, cfg=cfg)
    log, _, _ = run_backtest(d, "equal_weight", start="2023-01-02", end="2023-02-06")
    assert abs(log["turnover"].iloc[0] - 1.0) < 1e-12 and abs(log["cost"].iloc[0] - 20 / 1e4) < 1e-12
    assert log["turnover"].iloc[1:].max() < 1e-12                       # flat prices, same weights -> no trading
    assert abs(log["net_ret"].iloc[0] - (log["gross_ret"].iloc[0] - log["cost"].iloc[0])) < 1e-15


def test_drift_creates_turnover_and_matches_manual_value():
    a = 100 * 1.02 ** np.arange(40); b = np.full(40, 100.0)
    d = make_data({"A": a, "B": b}, weeks=5)
    log, w, _ = run_backtest(d, "equal_weight", start="2023-01-02", end="2023-02-06")
    gA = 1.02 ** 7
    pre_A = 0.5 * gA / (0.5 * gA + 0.5)                                 # drifted weight of A before rebalancing to 50/50
    assert abs(log["turnover"].iloc[1] - 2 * abs(0.5 - pre_A)) < 1e-12


# ------------------------------------------------------------ cap
def test_cap_water_filling_respects_cap_and_sum():
    w = pd.Series([0.6] + [0.4 / 14] * 14)
    out = cap_weights(w, 0.10)
    assert abs(out.sum() - 1) < 1e-12 and out.max() <= 0.10 + 1e-12     # clip+renormalise gave 0.20


def test_cap_infeasible_falls_back_to_equal_weight():
    out = cap_weights(pd.Series([5.0, 3.0, 1.0]), 0.10)
    assert np.allclose(out, 1 / 3)


# ------------------------------------------------------------ allocators
def _ctx(prices, mcaps=None, weeks=None):
    d = make_data(prices, mcaps, weeks=weeks)
    return d, d.make_ctx(pd.Timestamp("2023-12-04"))


def test_equal_weight_and_market_cap():
    d, ctx = _ctx(synth_prices(12), {f"c{j}": 1e9 * (j + 1) for j in range(12)})
    ctx.cfg = {**CFG, "portfolio": {**CFG["portfolio"], "weight_cap": 1.0}}
    assert np.allclose(equal_weight(ctx), 1 / 12)
    w = market_cap(ctx)
    assert np.allclose(w, ctx.mcap / ctx.mcap.sum())                    # first version silently returned equal weight


def test_minimum_variance_matches_analytic_solution_when_cap_not_binding():
    d, ctx = _ctx(synth_prices(10, vols=np.linspace(0.02, 0.05, 10)))
    ctx.cfg = {**CFG, "portfolio": {**CFG["portfolio"], "weight_cap": 1.0}}
    from src.allocators import estimate_risk
    S = estimate_risk(ctx)["Sigma"].values
    w = minimum_variance(ctx).values
    ref = np.linalg.solve(S, np.ones(len(S))); ref = ref / ref.sum()
    if (ref >= -1e-9).all():
        assert np.allclose(w, ref, atol=1e-4)
    else:                                                                # long-only binds: must still be no worse than 1/N
        assert w @ S @ w <= np.full(len(S), 1 / len(S)) @ S @ np.full(len(S), 1 / len(S)) + 1e-12
    assert (w >= -1e-12).all() and abs(w.sum() - 1) < 1e-9


def test_risk_parity_equalises_risk_contributions_and_is_not_equal_weight():
    d, ctx = _ctx(synth_prices(12, vols=np.linspace(0.02, 0.08, 12)))
    ctx.cfg = {**CFG, "portfolio": {**CFG["portfolio"], "weight_cap": 1.0}}
    from src.allocators import estimate_risk
    S = estimate_risk(ctx)["Sigma"].values
    w = risk_parity(ctx).values
    rc = w * (S @ w); rc = rc / rc.sum()
    assert rc.std() / rc.mean() < 0.01                                   # risk contributions equal within 1%
    assert w.max() / w.min() > 2.0                                       # first version returned exactly 1/N


def test_optimisers_respect_cap():
    d, ctx = _ctx(synth_prices(20))
    for f in (minimum_variance, mean_variance, black_litterman, risk_parity):
        w = f(ctx)
        assert w.max() <= 0.10 + 1e-6 and abs(w.sum() - 1) < 1e-8 and (w >= -1e-12).all(), f.__name__


def test_black_litterman_without_views_equals_market_cap_when_uncapped():
    d, ctx = _ctx(synth_prices(12), {f"c{j}": 1e9 * (j + 1) ** 2 for j in range(12)})
    ctx.cfg = {**CFG, "portfolio": {**CFG["portfolio"], "weight_cap": 1.0}}
    assert np.allclose(black_litterman(ctx), ctx.mcap / ctx.mcap.sum(), atol=1e-4)


def test_four_risk_allocators_give_different_portfolios():
    # skewed market caps (BTC-like dominance) so the 10% cap binds; with no binding cap BL-without-views == market cap by design
    d, ctx = _ctx(synth_prices(20), {f"c{j}": 1e9 * (j + 1) ** 3 for j in range(20)})
    ws = {f.__name__: f(ctx).values for f in (equal_weight, market_cap, minimum_variance, risk_parity, mean_variance, black_litterman)}
    names = list(ws)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not np.allclose(ws[names[i]], ws[names[j]], atol=1e-4), (names[i], names[j])   # first version: all identical


def test_btc_buy_hold_is_all_bitcoin():
    p = synth_prices(5); p["bitcoin"] = p.pop("c0")
    d, ctx = _ctx(p)
    w = btc_buy_hold(ctx)
    assert w["bitcoin"] == 1.0 and w.sum() == 1.0


# ------------------------------------------------------------ no look-ahead
def test_weights_do_not_depend_on_future_data():
    p = synth_prices(15, T=420)
    full = make_data(p, weeks=58)
    t = pd.Timestamp("2023-10-02")
    cut = t - pd.Timedelta(days=1)
    trunc = {k: v[: (cut - pd.Timestamp("2023-01-01")).days + 1] for k, v in p.items()}
    part = make_data(trunc, weeks=40)
    for f in (equal_weight, minimum_variance, risk_parity, mean_variance, black_litterman):
        a, b = f(full.make_ctx(t)), f(part.make_ctx(t))
        assert np.allclose(a.values, b.values, atol=1e-9), f.__name__


def test_performance_metrics_known_values():
    log = pd.DataFrame({"net_ret": [0.01] * 52, "turnover": [0.1] * 52, "n_assets": [10] * 52, "cost": [0.0] * 52})
    pf = performance(log)
    assert abs(pf["ann_return"] - (1.01 ** 52 - 1)) < 1e-9 and pf["max_drawdown"] == 0.0
