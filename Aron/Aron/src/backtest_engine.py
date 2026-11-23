"""Weekly walk-forward backtest engine (fixed version).

Timing (same as the factor code): at rebalance Monday t the information cut-off is c = t - 1 day (Sunday close). Weights are
computed from data up to c, the portfolio is entered at the CLOSE of c and held to the close of the next cut-off c' = t' - 1
day, i.e. exactly 7 days. (The first version held Monday close -> Sunday close, 6 days, and lost one day of return every week.)

Missing prices: a coin without a price at c is not tradeable that week. A held coin without a price at c' is valued at its last
traded price inside (c, c'] (policy 'last_price'; a collapsed coin therefore shows its real loss, e.g. LUNA -99.99%).
A held coin with no price at all after c is valued at 0% and counted in `n_no_exit_price` (never silently).
Costs: (fee + slippage) x turnover, turnover = sum |w_new - w_drifted|, first week 100% (build-up from cash). Not halved.
"""
import numpy as np
import pandas as pd
import yaml
from .allocators import ALLOCATORS, cap_weights


def load_config(path):
    return yaml.safe_load(open(path, encoding="utf-8"))


class BacktestData:
    def __init__(self, panel, universe, cfg):
        self.cfg = cfg
        panel = panel.copy()
        panel["date"] = pd.to_datetime(panel["date"])
        close = panel.pivot(index="date", columns="coin_id", values="close")
        idx = pd.date_range(close.index.min(), close.index.max(), freq="D")
        self.close = close.reindex(idx)
        vol = panel.pivot(index="date", columns="coin_id", values="volume_usd") if "volume_usd" in panel else None
        self.volume = vol.reindex(idx) if vol is not None else pd.DataFrame(np.nan, index=idx, columns=self.close.columns)
        mc = panel.pivot(index="date", columns="coin_id", values="mcap") if "mcap" in panel else None
        self.mcap = (mc.reindex(idx).ffill(limit=14) if mc is not None else pd.DataFrame(np.nan, index=idx, columns=self.close.columns))
        self.ret1d = self.close / self.close.shift(1) - 1.0          # only when both days exist
        self.avgvol = self.volume.rolling(28, min_periods=int(cfg["tradeable"]["min_volume_obs"])).mean()
        self.collapsed = set(panel.loc[panel["end_reason"].astype(str).eq("collapsed"), "coin_id"]) if "end_reason" in panel else set()
        u = universe.copy()
        u["rebalance_date"] = pd.to_datetime(u["rebalance_date"])
        u = u[u["is_member"].astype(int) > 0]
        self.members = {d: sorted(g["coin_id"].unique()) for d, g in u.groupby("rebalance_date")}
        self.dates = sorted(self.members)
        self.cache = {}

    def make_ctx(self, t):
        return Context(self, pd.Timestamp(t))


class Context:
    def __init__(self, data, t):
        self.data, self.cfg, self.date = data, data.cfg, t
        off = pd.Timedelta(days=int(self.cfg["timing"]["info_cutoff_offset_days"]))
        self.c = t - off
        self.notes = []
        d = data
        mem = [k for k in d.members.get(t, []) if k in d.close.columns]
        if self.c not in d.close.index:
            self.coins = []
            return
        px = d.close.loc[self.c, mem]
        av = d.avgvol.loc[self.c, mem]
        vmin = float(self.cfg["tradeable"]["min_avg_volume_usd"])
        ok = px.notna() & ((av >= vmin) if vmin > 0 else True)
        self.n_members = len(d.members.get(t, []))
        self.coins = [k for k in mem if bool(ok[k])]
        self.mcap = d.mcap.loc[self.c].reindex(self.coins)

    def window_returns(self):
        w = int(self.cfg["risk_model"]["cov_window_days"])
        R = self.data.ret1d
        return R.loc[self.c - pd.Timedelta(days=w - 1): self.c, self.coins]


def coin_period_returns(data, coins, c, c_next, policy="last_price"):
    """Return of each coin from close(c) to close(c_next); also the number of coins with no price after c."""
    p0 = data.close.loc[c, coins]
    seg = data.close.loc[c + pd.Timedelta(days=1): c_next, coins]
    p1 = seg.ffill().iloc[-1] if len(seg) else pd.Series(np.nan, index=coins)
    ret = p1 / p0 - 1.0
    no_exit = ret.isna()
    if policy == "minus_100":
        last_in_window = seg.index.max() if len(seg) else None
        gone = seg.iloc[-1].isna() if len(seg) else pd.Series(True, index=coins)
        ret = ret.where(~gone, -1.0)
    return ret.fillna(0.0), int(no_exit.sum())


def run_backtest(data, name, allocator=None, apply_cap=None, start=None, end=None):
    cfg = data.cfg
    fn, cap_flag = ALLOCATORS[name] if allocator is None else (allocator, True if apply_cap is None else apply_cap)
    cap = cfg["portfolio"]["weight_cap"]
    cost_bps = cfg["costs"]["fee_bps_one_way"] + cfg["costs"]["slippage_bps_one_way"]
    off = pd.Timedelta(days=int(cfg["timing"]["info_cutoff_offset_days"]))
    start = pd.Timestamp(start or cfg["sample"]["start"])
    end = pd.Timestamp(end or cfg["sample"]["end"])
    dates = [d for d in data.dates if start <= d <= end]
    rows, wrows, skipped = [], [], []
    prev_w, prev_r, prev_g = None, None, None
    for i, t in enumerate(dates[:-1]):
        t_next = dates[i + 1]
        if (t_next - t).days != 7:
            skipped.append((t, "gap in rebalance dates"))
            prev_w = None
            continue
        ctx = data.make_ctx(t)
        c_next = t_next - off
        if len(ctx.coins) < cfg["portfolio"]["min_names"] or c_next not in data.close.index:
            skipped.append((t, f"only {len(ctx.coins)} tradeable"))
            prev_w = None
            continue
        w = fn(ctx).reindex(ctx.coins).fillna(0.0)
        if cap_flag:
            w = cap_weights(w, cap)
        ret, n_noexit = coin_period_returns(data, ctx.coins, ctx.c, c_next, cfg["exits"]["missing_exit_policy"])
        gross = float((w * ret).sum())
        if prev_w is None:
            pre = pd.Series(0.0, index=w.index)
        else:
            pre = (prev_w * (1 + prev_r.reindex(prev_w.index).fillna(0.0)) / (1 + prev_g)).reindex(w.index).fillna(0.0)
        # coins held last week but no longer tradeable are sold: their drifted weight is part of turnover
        dropped = 0.0 if prev_w is None else float((prev_w * (1 + prev_r.reindex(prev_w.index).fillna(0.0)) / (1 + prev_g)).drop(w.index, errors="ignore").sum())
        turnover = float((w - pre).abs().sum() + dropped)
        cost = turnover * cost_bps / 1e4
        rows.append(dict(date=t, allocator=name, n_members=ctx.n_members, n_tradeable=len(ctx.coins), n_assets=int((w > 1e-9).sum()),
                         gross_ret=gross, turnover=turnover, cost=cost, net_ret=gross - cost, max_weight=float(w.max()),
                         n_no_exit_price=n_noexit, notes="; ".join(ctx.notes)))
        for k in w[w > 1e-12].index:
            wrows.append(dict(date=t, allocator=name, coin_id=k, weight=float(w[k]), coin_ret=float(ret[k])))
        prev_w, prev_r, prev_g = w, ret, gross
    return pd.DataFrame(rows), pd.DataFrame(wrows), skipped


def performance(log, periods=52, rf_weekly=0.0, col="net_ret"):
    r = log[col].astype(float)
    ex = r - rf_weekly
    nav = (1 + r).cumprod()
    dd = nav / nav.cummax() - 1
    q = r.quantile(0.05)
    return dict(weeks=len(r), ann_return=float((1 + r).prod() ** (periods / len(r)) - 1), ann_vol=float(r.std(ddof=1) * np.sqrt(periods)),
                sharpe=float(ex.mean() / r.std(ddof=1) * np.sqrt(periods)) if r.std() > 0 else np.nan,
                max_drawdown=float(dd.min()), es_5pct_weekly=float(r[r <= q].mean()), worst_week=float(r.min()),
                avg_turnover=float(log["turnover"].mean()), avg_names=float(log["n_assets"].mean()), total_cost_drag_ann=float(log["cost"].mean() * periods))
