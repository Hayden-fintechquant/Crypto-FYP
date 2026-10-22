"""1. 建立合成資料（3 個幣、幾週）
2. 定義等權重配置器
3. 呼叫回測引擎
4. 把結果存成 weekly_log.csv 與 weights.csv"""

import pandas as pd
import numpy as np
from src.backtest_engine import run_backtest

dates = pd.date_range("2023-01-02", periods=28, freq="D")
rows = []
rng = np.random.default_rng(42)
for coin in ["A", "B", "C"]:
    px = 100.0
    for d in dates:
        px *= (1 + rng.normal(0, 0.02))
        rows.append({"date": d, "coin_id": coin, "close": px})
panel = pd.DataFrame(rows)

universe = pd.DataFrame({
    "rebalance_date": ["2023-01-02"] * 3 + ["2023-01-09"] * 3 + ["2023-01-16"] * 3,
    "coin_id": ["A", "B", "C"] * 3,
    "is_member": [1] * 9,
})

def equal_weight_allocator(date, members, hist):
    return pd.Series(1.0 / len(members), index=members)

weekly_log, weights = run_backtest(panel, universe, equal_weight_allocator)
weekly_log.to_csv("weekly_log.csv", index=False)
weights.to_csv("weights.csv", index=False)

print("已產生 weekly_log.csv 與 weights.csv")
print(weekly_log)