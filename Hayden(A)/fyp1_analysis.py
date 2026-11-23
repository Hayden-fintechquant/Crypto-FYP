"""fyp1_analysis.py -- evidence tables for the FYP I report.
    python fyp1_analysis.py --panel data/panel.csv --universe data/universe.csv --bench results_bench --out results_analysis
Writes CSV tables + analysis_summary.txt. Nothing here changes any pre-registered result; it only describes data and benchmarks.
"""
import argparse, os
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--panel", default="data/panel.csv"); ap.add_argument("--universe", default="data/universe.csv")
ap.add_argument("--bench", default="results_bench"); ap.add_argument("--out", default="results_analysis")
ap.add_argument("--start", default="2020-09-21"); ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
S = []                                       # summary lines


def say(s=""):
    S.append(s); print(s)


panel = pd.read_csv(a.panel, parse_dates=["date"], low_memory=False)
uni = pd.read_csv(a.universe, parse_dates=["rebalance_date"])
uni = uni[(uni["is_member"].astype(int) > 0) & (uni["rebalance_date"] >= a.start)]
close = panel.pivot(index="date", columns="coin_id", values="close")
close = close.reindex(pd.date_range(close.index.min(), close.index.max()))
r1 = close / close.shift(1) - 1
members = set(uni["coin_id"])

# ---------------------------------------------------------------- 1. return distribution (daily, member coins, sample period)
R = r1.loc[a.start:, [c for c in r1.columns if c in members]]
x = R.stack().dropna()
z = (R - R.mean()) / R.std()
zs = z.stack().dropna()
from scipy import stats
exp5 = 2 * stats.norm.sf(5) * len(zs)
per_coin = R.apply(lambda s: stats.kurtosis(s.dropna()) if s.notna().sum() > 250 else np.nan).dropna()
btc = r1.loc[a.start:, "bitcoin"].dropna()
dist = pd.DataFrame({
    "pooled_member_coin_days": [len(x)],
    "pooled_excess_kurtosis": [stats.kurtosis(x)],
    "pooled_skew": [stats.skew(x)],
    "median_coin_excess_kurtosis": [per_coin.median()],
    "share_abs_ret_gt_20pct": [(x.abs() > 0.20).mean()],
    "coin_days_beyond_5_sigma": [int((zs.abs() > 5).sum())],
    "expected_beyond_5_sigma_if_gaussian": [exp5],
    "btc_excess_kurtosis": [stats.kurtosis(btc)], "btc_days_beyond_5_sigma": [int(((btc - btc.mean()).abs() / btc.std() > 5).sum())],
    "btc_ann_vol": [btc.std() * np.sqrt(365)]})
dist.round(4).T.to_csv(f"{a.out}/return_distribution.csv", header=["value"])
say("1. DAILY RETURN DISTRIBUTION (member coins, %s onward)" % a.start)
say(f"   coin-days {len(x):,}; pooled excess kurtosis {stats.kurtosis(x):.1f}; median per-coin excess kurtosis {per_coin.median():.1f}; BTC {stats.kurtosis(btc):.1f}")
say(f"   |z|>5 events: {int((zs.abs() > 5).sum())} observed vs {exp5:.4f} expected under normality; share of |daily return|>20%: {(x.abs() > 0.20).mean():.2%}")

# ---------------------------------------------------------------- 2. correlation by regime
def mean_corr(start, end, min_cov=0.9):
    w = r1.loc[start:end]
    cols = [c for c in w.columns if c in members and w[c].notna().mean() >= min_cov]
    c = w[cols].corr().values
    return len(cols), float(c[np.triu_indices(len(cols), 1)].mean())


wins = {"2023 (calm benchmark year)": ("2023-01-01", "2023-12-31"), "2022 pre-Luna (Mar-Apr)": ("2022-03-01", "2022-04-30"),
        "Terra/Luna window (2022-05-07..05-20)": ("2022-05-07", "2022-05-20"), "FTX window (2022-11-07..11-20)": ("2022-11-07", "2022-11-20")}
rows = []
for k, (s, e) in wins.items():
    n, m = mean_corr(s, e)
    rows.append(dict(window=k, n_coins=n, mean_pairwise_corr=m))
corr = pd.DataFrame(rows)
corr.round(3).to_csv(f"{a.out}/correlation_by_regime.csv", index=False)
say("\n2. MEAN PAIRWISE DAILY CORRELATION (coins with >=90% coverage in the window)")
for _, r in corr.iterrows():
    say(f"   {r['window']:42s} n={int(r['n_coins']):3d}  corr={r['mean_pairwise_corr']:.2f}")

# ---------------------------------------------------------------- 3. benchmark evaluation
log = pd.read_csv(f"{a.bench}/weekly_log.csv", parse_dates=["date"])
P = {n: g.set_index("date").sort_index() for n, g in log.groupby("allocator")}
names = [n for n in ["equal_weight", "market_cap", "min_variance", "risk_parity", "mean_variance", "black_litterman", "btc_buy_hold", "market_cap_uncapped"] if n in P]
rng = np.random.default_rng(20260901)


def sharpe(r):
    return r.mean() / r.std(ddof=1) * np.sqrt(52)


def block_bootstrap_sharpes(R, B=2000, block=4):
    """Stationary-style circular block bootstrap, SAME resampled weeks for all allocators (keeps the cross-correlation)."""
    T = len(R)
    out = np.empty((B, R.shape[1]))
    vals = R.values
    for b in range(B):
        idx = []
        while len(idx) < T:
            s0 = rng.integers(0, T); idx.extend([(s0 + j) % T for j in range(block)])
        v = vals[idx[:T]]
        out[b] = v.mean(0) / v.std(0, ddof=1) * np.sqrt(52)
    return out


net = pd.DataFrame({n: P[n]["net_ret"] for n in names})
bs = block_bootstrap_sharpes(net, a.boot)
tab = []
for j, n in enumerate(names):
    lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
    tab.append(dict(allocator=n, sharpe_net=sharpe(net[n]), ci95_low=lo, ci95_high=hi))
bt = pd.DataFrame(tab).set_index("allocator")
# paired comparison vs equal weight
if "equal_weight" in names:
    je = names.index("equal_weight")
    for j, n in enumerate(names):
        d = bs[:, j] - bs[:, je]
        bt.loc[n, "sharpe_minus_EW"] = sharpe(net[n]) - sharpe(net["equal_weight"])
        bt.loc[n, "diff_ci95_low"], bt.loc[n, "diff_ci95_high"] = np.percentile(d, [2.5, 97.5])
bt.round(3).to_csv(f"{a.out}/sharpe_bootstrap.csv")
say("\n3. BENCHMARKS (net of 20 bps, %d weeks): net Sharpe with 95%% block-bootstrap CI (4-week blocks, %d resamples)" % (len(net), a.boot))
for n, r in bt.iterrows():
    extra = f"  | vs EW: {r['sharpe_minus_EW']:+.2f} [{r['diff_ci95_low']:+.2f}, {r['diff_ci95_high']:+.2f}]" if "sharpe_minus_EW" in bt else ""
    say(f"   {n:22s} {r['sharpe_net']:5.2f} [{r['ci95_low']:5.2f}, {r['ci95_high']:5.2f}]{extra}")

# by calendar year (net compounded return)
yr = pd.DataFrame({n: (1 + net[n]).groupby(net.index.year).prod() - 1 for n in names})
yr.round(3).to_csv(f"{a.out}/benchmark_by_year.csv")
say("\n   Net return by calendar year (2020 = from Sep 21; 2026 = to Sep 21):")
say(yr.round(2).to_string())

# stress weeks
stress = {"Terra/Luna week (rebalance 2022-05-09)": "2022-05-09", "FTX week (rebalance 2022-11-07)": "2022-11-07", "May-2021 crash week (2021-05-17)": "2021-05-17"}
st = pd.DataFrame({k: net.loc[pd.Timestamp(d)] for k, d in stress.items() if pd.Timestamp(d) in net.index}).T
st.round(3).to_csv(f"{a.out}/benchmark_stress_weeks.csv")
say("\n   Net weekly return in stress weeks:")
say(st.round(3).to_string())

# cost sensitivity (net = gross - turnover * bps)
rows = []
for bps in [0, 20, 40, 80]:
    row = {"one_way_cost_bps": bps}
    for n in names:
        g = P[n]["gross_ret"] - P[n]["turnover"] * bps / 1e4
        row[n] = sharpe(g)
    rows.append(row)
cs = pd.DataFrame(rows).set_index("one_way_cost_bps")
cs.round(3).to_csv(f"{a.out}/cost_sensitivity_sharpe.csv")
say("\n   Net Sharpe vs one-way trading cost (fee+slippage):")
say(cs.round(2).to_string())

# ---------------------------------------------------------------- 4. selection / survivorship diagnostics
tr = P["equal_weight"]
mw = int(tr["n_members"].sum()); tw = int(tr["n_tradeable"].sum())
say("\n4. SELECTION DIAGNOSTICS (benchmark engine)")
say(f"   member-weeks {mw:,}; tradeable (price at cut-off + US$1M liquidity) {tw:,} = {tw / mw:.1%}; avg tradeable per week {tr['n_tradeable'].mean():.1f} of {tr['n_members'].mean():.1f}")
say(f"   held positions without any price after entry (valued 0%, counted): {int(log['n_no_exit_price'].sum())}")
open(f"{a.out}/analysis_summary.txt", "w", encoding="utf-8").write("\n".join(S) + "\n")
