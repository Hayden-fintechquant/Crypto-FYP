"""EXPLORATORY (not part of the locked FYP I pre-registration).

Explanatory (time-series) factor model on the point-in-time top-40 universe:
    MKT = cap-weighted return of eligible coins
    SMB = equal-weighted bottom-30% minus top-30% by market cap at the cut-off
    MOM = equal-weighted top-30% minus bottom-30% by past 4-week return (mom_4w)
Then, for each coin with >= 52 eligible weeks:  r_i = a + b'F + e  (CAPM: MKT only; 3F: MKT+SMB+MOM).

Timing = benchmark engine: rebalance Monday d, cut-off Sunday c = d-1, hold c close -> c+7 close.
Eligible = universe member at d, price on c, 28-day avg volume >= US$1M (>= 14 obs). rf = 0.
Exit price = last close in (c, c+7] (collapsed coins keep their last traded price).

    python exploratory/factor_model_explanatory.py --panel data/panel.csv --universe data/universe.csv --out results_explanatory
"""
import argparse, os
import numpy as np, pandas as pd

START, END = "2020-09-21", "2026-09-28"


def nw_t(x, lags=4):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x); m = x.mean(); e = x - m
    v = e @ e / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return m / np.sqrt(v / n)


def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b; r2 = 1 - res @ res / ((y - y.mean()) @ (y - y.mean()))
    s2 = res @ res / (len(y) - X.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b, b / se, r2


def main(panel, universe, out):
    os.makedirs(out, exist_ok=True)
    p = pd.read_csv(panel, parse_dates=["date"], low_memory=False)
    u = pd.read_csv(universe, parse_dates=["rebalance_date"])
    u = u[u.is_member == 1]
    close = p.pivot_table(index="date", columns="coin_id", values="close").sort_index()
    close = close.reindex(pd.date_range(close.index.min(), close.index.max()))
    mcap = p.pivot_table(index="date", columns="coin_id", values="mcap").reindex(index=close.index, columns=close.columns)
    vol = p.pivot_table(index="date", columns="coin_id", values="volume_usd").reindex(index=close.index, columns=close.columns)
    vavg = vol.rolling(28, min_periods=14).mean()

    rows, wk = [], []
    for d in sorted(u.rebalance_date.unique()):
        d = pd.Timestamp(d)
        if d < pd.Timestamp(START) or d > pd.Timestamp(END):
            continue
        c, e = d - pd.Timedelta(days=1), d + pd.Timedelta(days=6)
        if e > close.index.max():
            continue
        mem = [x for x in u.loc[u.rebalance_date == d, "coin_id"] if x in close.columns]
        p0, m0, v0 = close.loc[c, mem], mcap.loc[c, mem], vavg.loc[c, mem]
        p4 = close.loc[c - pd.Timedelta(days=28), mem]
        win = close.loc[c + pd.Timedelta(days=1):e, mem]
        p1 = win.ffill().iloc[-1]
        ok = p0.notna() & p1.notna() & m0.gt(0) & v0.ge(1e6)
        if ok.sum() < 20:
            continue
        df = pd.DataFrame({"ret": p1[ok] / p0[ok] - 1, "mcap": m0[ok], "mom4": p0[ok] / p4[ok] - 1})
        df["date"], df["coin_id"] = d, df.index
        rows.append(df)
        n = len(df); k = max(int(round(0.3 * n)), 1)
        bys = df.sort_values("mcap"); bym = df.dropna(subset=["mom4"]).sort_values("mom4")
        wk.append({"date": d, "n": n,
                   "MKT": np.average(df.ret, weights=df.mcap),
                   "SMB": bys.ret.iloc[:k].mean() - bys.ret.iloc[-k:].mean(),
                   "MOM": bym.ret.iloc[-k:].mean() - bym.ret.iloc[:k].mean(),
                   "EW": df.ret.mean()})
    F = pd.DataFrame(wk).set_index("date"); R = pd.concat(rows, ignore_index=True)
    F.to_csv(f"{out}/factor_returns.csv")

    fac = ["MKT", "SMB", "MOM"]
    summ = pd.DataFrame({f: {"mean_weekly": F[f].mean(), "t_NW": nw_t(F[f]), "ann_sharpe": F[f].mean() / F[f].std() * np.sqrt(52),
                             "ann_vol": F[f].std() * np.sqrt(52)} for f in fac}).T
    corr = F[fac].corr()

    per = []
    for cid, g in R.groupby("coin_id"):
        g = g.set_index("date").join(F[fac], how="inner")
        if len(g) < 52:
            continue
        y = g.ret.values
        b1, t1, r1 = ols(y, g[["MKT"]].values)
        b3, t3, r3 = ols(y, g[fac].values)
        per.append({"coin_id": cid, "weeks": len(g), "R2_capm": r1, "R2_3f": r3, "beta_capm": b1[1],
                    "alpha_3f_wk": b3[0], "t_alpha_3f": t3[0], "b_MKT": b3[1], "b_SMB": b3[2], "b_MOM": b3[3],
                    "t_MKT": t3[1], "t_SMB": t3[2], "t_MOM": t3[3]})
    per = pd.DataFrame(per).sort_values("weeks", ascending=False)
    per.to_csv(f"{out}/per_coin_regressions.csv", index=False)

    # pooled: share of all coin-week return variance explained (coins weighted by observations)
    J = R.set_index("date").join(F[fac], how="inner")
    lines = [
        "EXPLORATORY explanatory factor model (NOT in the FYP I pre-registration)",
        f"weeks {len(F)} ({F.index.min().date()} .. {F.index.max().date()}), avg eligible coins/week {F.n.mean():.1f}, rf = 0",
        "", "[factor returns]", summ.round(4).to_string(),
        "", "[factor correlations]", corr.round(3).to_string(),
        f"\ncorr(MKT, equal-weight universe return) = {F.MKT.corr(F.EW):.3f}",
        "", f"[per-coin time-series regressions: {len(per)} coins with >= 52 eligible weeks]",
        per[["R2_capm", "R2_3f", "beta_capm", "b_MKT", "b_SMB", "b_MOM"]].describe(percentiles=[.25, .5, .75]).round(3).to_string(),
        "", f"share of coins with |t| >= 2:  MKT {np.mean(per.t_MKT.abs() >= 2):.0%}, SMB {np.mean(per.t_SMB.abs() >= 2):.0%}, "
            f"MOM {np.mean(per.t_MOM.abs() >= 2):.0%}, alpha {np.mean(per.t_alpha_3f.abs() >= 2):.0%} (5% expected by chance)",
        f"median R2 gain from adding SMB+MOM: {(per.R2_3f - per.R2_capm).median():.3f}",
        "", "[largest coins by eligible weeks]",
        per.head(12)[["coin_id", "weeks", "R2_capm", "R2_3f", "beta_capm", "b_SMB", "b_MOM", "t_SMB", "t_MOM"]].round(2).to_string(index=False),
    ]
    open(f"{out}/summary.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/panel.csv"); ap.add_argument("--universe", default="data/universe.csv")
    ap.add_argument("--out", default="results_explanatory")
    a = ap.parse_args(); main(a.panel, a.universe, a.out)
