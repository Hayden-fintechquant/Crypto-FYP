"""讀取設定檔
對於每一個再平衡日 t：
    1. 取得當週 universe 成員
    2. 只用 t 以前的資料
    3. 呼叫配置器得到目標權重
    4. 如果是第一週 → 換手率 = 100%
       否則 → 用漂移後的舊權重計算換手率
    5. 成本 = 換手率 × 15 bps
    6. 計算本週毛報酬
    7. 淨報酬 = 毛報酬 − 成本
    8. 記錄結果，進入下一週
最後回傳 weekly_log 與 weights"""

import pandas as pd
import numpy as np
import yaml

def load_config(path="config_backtest.yaml"):
    """讀取設定檔"""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def apply_drift(weights: pd.Series, period_returns: pd.Series) -> pd.Series:
    """計算一週後漂移的權重"""
    gross = weights * (1 + period_returns.fillna(0))
    total = gross.sum()
    if total <= 0:
        return weights
    return gross / total

def calculate_turnover(target: pd.Series, drifted: pd.Series) -> float:
    """換手率 = Σ |目標權重 - 漂移後權重|"""
    aligned = pd.concat([target, drifted], axis=1, keys=["target", "drift"]).fillna(0)
    return float((aligned["target"] - aligned["drift"]).abs().sum())

def run_backtest(panel: pd.DataFrame, universe: pd.DataFrame, allocator_fn, config=None):
    """走向前回測引擎"""
    if config is None:
        config = load_config()
    cost_rate = config["one_side_cost_bps"] / 10000.0

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    universe = universe.copy()
    universe["rebalance_date"] = pd.to_datetime(universe["rebalance_date"])
    if "is_member" in universe.columns:
        universe = universe[universe["is_member"] == 1]

    reb_dates = sorted(universe["rebalance_date"].unique())
    logs = []
    weights_records = []
    prev_weights = None
    prev_period_returns = None

    for i, t in enumerate(reb_dates):
        members = universe[universe["rebalance_date"] == t]["coin_id"].tolist()
        hist = panel[panel["date"] < t]

        w_target = allocator_fn(t, members, hist)
        w_target = w_target.reindex(members).fillna(0.0)
        if w_target.sum() > 0:
            w_target = w_target / w_target.sum()

        if prev_weights is None:
            turnover = 1.0
        else:
            drifted = apply_drift(prev_weights, prev_period_returns.reindex(prev_weights.index).fillna(0))
            turnover = calculate_turnover(w_target, drifted.reindex(w_target.index).fillna(0))

        cost = turnover * cost_rate

        next_t = reb_dates[i + 1] if i + 1 < len(reb_dates) else t + pd.Timedelta(days=7)
        period_data = panel[(panel["date"] >= t) & (panel["date"] < next_t)]
        period_rets = {}
        for coin in members:
            cdata = period_data[period_data["coin_id"] == coin]
            if len(cdata) >= 2:
                period_rets[coin] = cdata["close"].iloc[-1] / cdata["close"].iloc[0] - 1
            else:
                period_rets[coin] = 0.0
        period_returns = pd.Series(period_rets)

        gross_ret = float((w_target * period_returns.reindex(w_target.index).fillna(0)).sum())
        net_ret = gross_ret - cost

        logs.append({
            "date": t,
            "allocator": "equal_weight",
            "gross_ret": gross_ret,
            "cost": cost,
            "net_ret": net_ret,
            "turnover": turnover,
            "n_assets": len(members),
        })

        for coin, w in w_target.items():
            weights_records.append({
                "date": t,
                "coin_id": coin,
                "weight": float(w),
                "allocator": "equal_weight",
            })

        prev_weights = w_target.copy()
        prev_period_returns = period_returns

    return pd.DataFrame(logs), pd.DataFrame(weights_records)