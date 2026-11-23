import pandas as pd
import numpy as np
import pytest
from src.backtest_engine import run_backtest, apply_drift

def make_synthetic():
    dates = pd.date_range("2023-01-02", periods=21, freq="D")
    rows = []
    rng = np.random.default_rng(0)
    for coin in ["A", "B", "C"]:
        px = 100.0
        for d in dates:
            px *= (1 + rng.normal(0, 0.015))
            rows.append({"date": d, "coin_id": coin, "close": px})
    return pd.DataFrame(rows)

def equal_weight(date, members, hist):
    return pd.Series(1.0 / len(members), index=members)

def test_known_answer():
    w = pd.Series([0.5, 0.5], index=["A", "B"])
    r = pd.Series([0.10, -0.05], index=["A", "B"])
    drifted = apply_drift(w, r)
    assert abs(drifted.sum() - 1) < 1e-8
    assert drifted["A"] > drifted["B"]

def test_cost_difference():
    panel = make_synthetic()
    universe = pd.DataFrame({
        "rebalance_date": ["2023-01-02", "2023-01-09"] * 3,
        "coin_id": ["A", "B", "C"] * 2,
        "is_member": [1] * 6,
    })
    log0, _ = run_backtest(panel, universe, equal_weight, {"one_side_cost_bps": 0})
    log15, _ = run_backtest(panel, universe, equal_weight, {"one_side_cost_bps": 15})
    assert (log0["net_ret"] - log15["net_ret"]).abs().sum() > 0

def test_weight_sum_long_only():
    panel = make_synthetic()
    universe = pd.DataFrame({
        "rebalance_date": ["2023-01-02"] * 3,
        "coin_id": ["A", "B", "C"],
        "is_member": [1, 1, 1],
    })
    _, weights = run_backtest(panel, universe, equal_weight)
    assert abs(weights.groupby("date")["weight"].sum().iloc[0] - 1) < 1e-6
    assert (weights["weight"] >= -1e-9).all()

def test_lookahead():
    panel = make_synthetic()
    universe = pd.DataFrame({
        "rebalance_date": ["2023-01-02"] * 3,
        "coin_id": ["A", "B", "C"],
        "is_member": [1, 1, 1],
    })
    log1, w1 = run_backtest(panel, universe, equal_weight)
    panel2 = panel.copy()
    panel2.loc[panel2["date"] > "2023-01-10", "close"] *= 1.5
    log2, w2 = run_backtest(panel2, universe, equal_weight)
    pd.testing.assert_frame_equal(log1.reset_index(drop=True), log2.reset_index(drop=True))

def test_first_week_turnover():
    panel = make_synthetic()
    universe = pd.DataFrame({
        "rebalance_date": ["2023-01-02"] * 3,
        "coin_id": ["A", "B", "C"],
        "is_member": [1, 1, 1],
    })
    log, _ = run_backtest(panel, universe, equal_weight)
    assert abs(log.iloc[0]["turnover"] - 1.0) < 1e-6