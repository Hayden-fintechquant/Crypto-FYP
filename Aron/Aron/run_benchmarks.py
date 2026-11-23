"""Run all benchmark allocators on the SAME weeks and write weights.csv, weekly_log.csv, benchmark_table.csv.
    python run_benchmarks.py --panel data/panel.csv --universe data/universe.csv [--out results_bench]"""
import argparse, os
import pandas as pd
from src.backtest_engine import BacktestData, load_config, run_backtest, performance

ap = argparse.ArgumentParser()
ap.add_argument("--panel", default="data/panel.csv"); ap.add_argument("--universe", default="data/universe.csv")
ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config_backtest.yaml")); ap.add_argument("--out", default="results_bench")
a = ap.parse_args()
cfg = load_config(a.config)
os.makedirs(a.out, exist_ok=True)
data = BacktestData(pd.read_csv(a.panel, parse_dates=["date"], low_memory=False), pd.read_csv(a.universe), cfg)
logs, ws, skipped = [], [], None
for name in ["equal_weight", "market_cap", "min_variance", "risk_parity", "mean_variance", "black_litterman", "btc_buy_hold", "market_cap_uncapped"]:
    l, w, sk = run_backtest(data, name); logs.append(l); ws.append(w); skipped = sk
    print(f"{name:22s} weeks {len(l)}")
log, w = pd.concat(logs), pd.concat(ws)
log.to_csv(f"{a.out}/weekly_log.csv", index=False); w.to_csv(f"{a.out}/weights.csv", index=False)
tab = pd.DataFrame({n: performance(g, cfg["timing"]["periods_per_year"], cfg["risk_free_weekly"]) for n, g in log.groupby("allocator")}).T
gross = pd.DataFrame({n: performance(g, cfg["timing"]["periods_per_year"], cfg["risk_free_weekly"], "gross_ret") for n, g in log.groupby("allocator")}).T
tab["sharpe_gross"] = gross["sharpe"]
tab.round(4).to_csv(f"{a.out}/benchmark_table.csv")
pd.DataFrame(skipped, columns=["date", "reason"]).to_csv(f"{a.out}/skipped_weeks.csv", index=False)
fb = log[log["notes"].astype(str).str.len() > 0].groupby("allocator").size()
print("\n", tab.round(3).to_string()); print("\nweeks with a logged fall-back/note:", fb.to_dict()); print("skipped weeks:", len(skipped))
