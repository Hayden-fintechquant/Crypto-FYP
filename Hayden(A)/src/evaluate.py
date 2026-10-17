"""評估套件: 夏普、最大回撤、Expected Shortfall、Deflated Sharpe、區塊 bootstrap、子期間與壓力期間表。
輸入都是「週淨報酬」序列 (index = 日期)，與 C 的 weekly_log 對接。"""
import numpy as np
import pandas as pd
from scipy import stats

PERIODS_PER_YEAR = 52
EULER_GAMMA = 0.5772156649015329

DEFAULT_STRESS = {
    "COVID": ("2020-02-15", "2020-04-15"),
    "Luna": ("2022-05-01", "2022-05-31"),
    "FTX": ("2022-11-01", "2022-11-30"),
}


# ---------- 基本指標 ----------
def sharpe(r, rf=0.0, periods=PERIODS_PER_YEAR, annualise=True):
    x = np.asarray(pd.Series(r).dropna(), float) - rf
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan
    s = x.mean() / x.std(ddof=1)
    return float(s * np.sqrt(periods)) if annualise else float(s)


def ann_return(r, periods=PERIODS_PER_YEAR):
    x = np.asarray(pd.Series(r).dropna(), float)
    if len(x) == 0:
        return np.nan
    return float(np.prod(1 + x) ** (periods / len(x)) - 1)


def ann_vol(r, periods=PERIODS_PER_YEAR):
    return float(pd.Series(r).dropna().std(ddof=1) * np.sqrt(periods))


def max_drawdown(r):
    x = pd.Series(r).dropna()
    eq = (1 + x).cumprod()
    return float((eq / eq.cummax() - 1).min()) if len(x) else np.nan


def expected_shortfall(r, level=0.05):
    """最糟 level 比例的週的平均報酬 (負數表示虧損)。"""
    x = np.sort(np.asarray(pd.Series(r).dropna(), float))
    if len(x) == 0:
        return np.nan
    k = max(int(np.ceil(level * len(x))), 1)
    return float(x[:k].mean())


# ---------- Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) ----------
def probabilistic_sharpe(sr, n_obs, skew=0.0, kurt=3.0, sr_benchmark=0.0):
    """PSR: 真實夏普 > sr_benchmark 的機率。sr 是「每期(非年化)」夏普，kurt 是一般峰度(常態=3)。"""
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n_obs - 1) / denom))


def expected_max_sharpe(n_trials, var_sr_trials):
    """在「沒有任何真實技巧」時，n_trials 個嘗試中最佳夏普的期望值 (每期單位)。"""
    if n_trials <= 1:
        return 0.0
    g = EULER_GAMMA
    return float(np.sqrt(var_sr_trials) * ((1 - g) * stats.norm.ppf(1 - 1.0 / n_trials)
                                           + g * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))))


def deflated_sharpe(r, n_trials, var_sr_trials, rf=0.0):
    """DSR: 把「試了 n_trials 種設定才挑到最好」考慮進去之後，夏普仍然顯著的機率。
    r: 被挑出來的那個策略的週報酬；var_sr_trials: 所有嘗試之『每期夏普』的變異數。"""
    x = pd.Series(r).dropna()
    sr = float((x - rf).mean() / (x - rf).std(ddof=1))
    sr0 = expected_max_sharpe(n_trials, var_sr_trials)
    return dict(sr_per_period=sr, sr_annualised=sr * np.sqrt(PERIODS_PER_YEAR), sr0_per_period=sr0,
                dsr=probabilistic_sharpe(sr, len(x), float(stats.skew(x)), float(stats.kurtosis(x, fisher=False)), sr0),
                n_trials=n_trials, n_obs=len(x))


# ---------- 區塊 bootstrap ----------
def _block_indices(n, block, rng):
    """circular block bootstrap 的一組索引 (長度 n)。"""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=nb)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return idx.reshape(-1)[:n]


def bootstrap_ci(r, stat_fn=sharpe, n_boot=2000, block=4, alpha=0.05, seed=0):
    x = np.asarray(pd.Series(r).dropna(), float)
    rng = np.random.default_rng(seed)
    draws = np.array([stat_fn(x[_block_indices(len(x), block, rng)]) for _ in range(n_boot)])
    return dict(estimate=float(stat_fn(x)), lo=float(np.nanquantile(draws, alpha / 2)),
                hi=float(np.nanquantile(draws, 1 - alpha / 2)))


def bootstrap_diff_ci(r_a, r_b, stat_fn=sharpe, n_boot=2000, block=4, alpha=0.05, seed=0):
    """配對 bootstrap: 同一批抽樣週同時套到 A、B 兩個策略，估計 stat(A) - stat(B) 的信賴區間。
    p_value_le_0 = 差距 <= 0 的比例 (單尾)。"""
    df = pd.concat([pd.Series(r_a), pd.Series(r_b)], axis=1).dropna()
    a, b = df.iloc[:, 0].values, df.iloc[:, 1].values
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = _block_indices(len(a), block, rng)
        diffs.append(stat_fn(a[i]) - stat_fn(b[i]))
    diffs = np.array(diffs)
    return dict(estimate=float(stat_fn(a) - stat_fn(b)), lo=float(np.nanquantile(diffs, alpha / 2)),
                hi=float(np.nanquantile(diffs, 1 - alpha / 2)), p_value_le_0=float(np.nanmean(diffs <= 0)))


# ---------- 績效表 ----------
def performance_row(r, rf=0.0, es_level=0.05):
    x = pd.Series(r).dropna()
    return dict(n_weeks=len(x), ann_return=ann_return(x), ann_vol=ann_vol(x), sharpe=sharpe(x, rf),
                max_drawdown=max_drawdown(x), es_5pct=expected_shortfall(x, es_level),
                hit_rate=float((x > 0).mean()) if len(x) else np.nan,
                skew=float(stats.skew(x)) if len(x) > 2 else np.nan,
                kurt_excess=float(stats.kurtosis(x)) if len(x) > 3 else np.nan)


def performance_table(returns: pd.DataFrame, rf=0.0, es_level=0.05, mask=None):
    """returns: 欄=策略, 列=週。回傳每個策略一列。"""
    r = returns if mask is None else returns[mask]
    return pd.DataFrame({c: performance_row(r[c], rf, es_level) for c in r.columns}).T


def subperiod_tables(returns: pd.DataFrame, stress=None, rf=0.0, es_level=0.05):
    """依年份與壓力期間切開，各給一張績效表。回傳 {名稱: DataFrame}。"""
    stress = DEFAULT_STRESS if stress is None else stress
    out = {"full": performance_table(returns, rf, es_level)}
    for y in sorted(set(returns.index.year)):
        m = returns.index.year == y
        if m.sum() >= 8:
            out[f"year_{y}"] = performance_table(returns, rf, es_level, m)
    for name, (a, b) in stress.items():
        m = (returns.index >= pd.Timestamp(a)) & (returns.index <= pd.Timestamp(b))
        if m.sum() >= 2:
            out[f"stress_{name}"] = performance_table(returns, rf, es_level, m)
    return out


def full_evaluation(returns: pd.DataFrame, benchmark: str, selected=None, rf=0.0, n_boot=2000, block=4,
                    es_level=0.05, seed=0, stress=None, extra_trials=0):
    """一條龍。returns: 欄=所有「試過的設定」(含基準)，列=週。
    n_trials = 欄數，var_sr_trials = 各欄每期夏普的變異數 (DSR 的輸入)。"""
    tabs = subperiod_tables(returns, stress, rf, es_level)
    per_sr = returns.apply(lambda s: sharpe(s, rf, annualise=False))
    n_trials = returns.shape[1] + int(extra_trials)      # extra_trials: 沒有出現在 weekly_log、但試過的設定數 (例如超參數組合)
    var_sr = float(per_sr.var(ddof=1)) if returns.shape[1] > 1 else 0.0
    sel = list(selected) if selected else [c for c in returns.columns if c != benchmark]
    dsr = pd.DataFrame({c: deflated_sharpe(returns[c], n_trials, var_sr, rf) for c in sel}).T
    ci = pd.DataFrame({c: bootstrap_ci(returns[c], lambda z: sharpe(z, rf), n_boot, block, seed=seed) for c in returns.columns}).T
    diff = pd.DataFrame({c: bootstrap_diff_ci(returns[c], returns[benchmark], lambda z: sharpe(z, rf), n_boot, block, seed=seed)
                         for c in sel if c != benchmark}).T
    return dict(tables=tabs, dsr=dsr, sharpe_ci=ci, sharpe_diff_vs_benchmark=diff, n_trials=n_trials)
