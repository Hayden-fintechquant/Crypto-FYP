"""把所有步驟串起來: 因子 -> 篩選 -> IC/分組 -> Fama-MacBeth -> 多重檢定 -> 訊號組合 -> mu。"""
import numpy as np
import pandas as pd
from .data_io import Market
from .factors import build_factor_panel
from .screen import screen
from .validate import ic_series, ic_summary, quantile_sorts
from .fama_macbeth import fm_univariate, fm_multivariate, fm_spanning
from .stats import holm, benjamini_hochberg
from .combine import make_mu


def run_pipeline(panel, universe, cfg, fixed=None, light=False):
    m = Market(panel)
    fp = build_factor_panel(m, universe, cfg)
    scores, fwd, scr = screen(fp, cfg, fixed)
    factors = list(scores.columns)
    if not factors:
        raise ValueError("篩選後沒有任何因子留下，請檢查資料覆蓋率與 config")
    vcfg = cfg["validation"]
    lags = vcfg["nw_lags"]

    fm_multi, gam_multi = fm_multivariate(scores, fwd, factors, lags)
    mu = make_mu(scores, fwd, gam_multi, cfg)
    res = dict(market=m, scores=scores, fwd=fwd, screen=scr, mu=mu, gam_multi=gam_multi, factors=factors)
    if light:
        return res

    ic = {f: ic_series(scores, fwd, f) for f in factors}
    ic_tab = pd.DataFrame({f: ic_summary(ic[f], vcfg["rolling_ic_weeks"], lags) for f in factors}).T
    quint, qweeks = {}, {}
    for f in factors:
        qweeks[f], quint[f] = quantile_sorts(scores, fwd, f, vcfg["n_quantiles"], lags)
    q_tab = pd.DataFrame(quint).T
    fm_uni, gam_uni = fm_univariate(scores, fwd, factors, lags)
    span = fm_spanning(scores, fwd, factors, vcfg["spanning_controls"], lags)

    # --- 多重檢定: 把「所有檢定」放進同一個家族，宣告總數 ---
    fam = [("uni:" + f, fm_uni.loc[f, "p"]) for f in factors]
    fam += [("multi:" + f, fm_multi.loc[f, "p"]) for f in factors]
    fam += [("span:" + f, span.loc[f, "p"]) for f in span.index]
    names, ps = zip(*fam)
    ps = np.array([1.0 if pd.isna(p) else p for p in ps])
    mt = pd.DataFrame({"p": ps, "p_holm": holm(ps), "p_bh": benjamini_hochberg(ps)}, index=list(names))
    n_specs = len(mt)

    # --- 事前門檻下的判定 ---
    v = vcfg["verdict"]
    rows = {}
    for f in factors:
        why = []
        if not abs(ic_tab.loc[f, "ic_t_nw"]) >= v["min_abs_t"]:
            why.append("IC t 值不足")
        if not abs(q_tab.loc[f, "monotonicity"]) >= v["min_abs_monotonicity"]:
            why.append("五分位單調性不足")
        if not abs(fm_multi.loc[f, "t"]) >= v["min_abs_t"]:
            why.append("多變量 FM t 值不足")
        if not mt.loc["multi:" + f, "p_holm"] < v["alpha"]:
            why.append("多重檢定(Holm)後不顯著")
        rows[f] = dict(verdict="PASS" if not why else "FAIL", reasons="; ".join(why))
    verdict = pd.DataFrame(rows).T

    res.update(ic=ic, ic_tab=ic_tab, q_tab=q_tab, qweeks=qweeks, fm_uni=fm_uni, fm_multi=fm_multi,
               span=span, mt=mt, n_specs=n_specs, verdict=verdict, gam_uni=gam_uni)
    return res
