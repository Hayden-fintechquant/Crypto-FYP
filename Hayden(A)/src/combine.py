"""訊號組合與預期報酬校準。
鐵則: 第 t 週用到的任何統計量，都只能來自「第 t-1 週以前」已經實現的資料 (shift(1))。"""
import numpy as np
import pandas as pd
from .validate import ic_series


def _composite(scores: pd.DataFrame, W: pd.DataFrame) -> pd.Series:
    dates = scores.index.get_level_values(0)
    Wa = W[scores.columns].reindex(dates).values
    Wn = np.nan_to_num(Wa)
    num = (scores.values * Wn).sum(axis=1)
    den = np.abs(Wn).sum(axis=1)
    return pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), index=scores.index)


def _zscore_by_date(s: pd.Series) -> pd.Series:
    return s.groupby(level=0).transform(lambda x: (x - x.mean()) / (x.std(ddof=1) if x.std(ddof=1) > 0 else np.nan))


def _grinold(zc: pd.Series, fwd: pd.Series, all_dates, min_hist: int) -> pd.Series:
    ic_c = ic_series(zc.dropna().to_frame("c"), fwd, "c").reindex(all_dates)
    ic_bar = ic_c.expanding(min_periods=min_hist).mean().shift(1)
    sig = fwd.groupby(level=0).std(ddof=1).reindex(all_dates)
    sig_bar = sig.expanding(min_periods=min_hist).mean().shift(1)
    mult = (ic_bar * sig_bar).reindex(zc.index.get_level_values(0)).values
    return zc * mult


def make_mu(scores: pd.DataFrame, fwd: pd.Series, gam_multi: pd.DataFrame, cfg: dict) -> dict:
    """回傳 {'ew': mu, 'icir': mu, 'fm': mu}，mu 為 (date, coin_id) 的每週預期報酬。"""
    ccfg = cfg["combination"]
    mh, fw = int(ccfg["min_history_weeks"]), int(ccfg["fm_window_weeks"])
    all_dates = scores.index.get_level_values(0).unique().sort_values()
    factors = list(scores.columns)
    ic = {f: ic_series(scores, fwd, f).reindex(all_dates) for f in factors}
    out = {}

    # --- 正負號 ---
    if ccfg["sign_mode"] == "config":
        sg = pd.DataFrame({f: float(ccfg["expected_sign"].get(f, 0)) for f in factors}, index=all_dates)
    else:
        sg = pd.DataFrame({f: np.sign(ic[f].expanding(min_periods=mh).mean().shift(1)) for f in factors})
    # --- 1) 等權重組合 ---
    comp_ew = _zscore_by_date(_composite(scores, sg))
    # --- 2) ICIR 加權 ---
    icir = pd.DataFrame({f: ic[f].expanding(min_periods=mh).mean().shift(1) /
                            ic[f].expanding(min_periods=mh).std().shift(1) for f in factors})
    comp_icir = _zscore_by_date(_composite(scores, icir))
    if ccfg["calibration"] == "grinold":
        out["ew"] = _grinold(comp_ew, fwd, all_dates, mh)
        out["icir"] = _grinold(comp_icir, fwd, all_dates, mh)
    else:
        out["ew"], out["icir"] = comp_ew, comp_icir
    # --- 3) 滾動 Fama-MacBeth 溢酬加權 (已經是報酬單位) ---
    g = gam_multi.drop(columns=["const"], errors="ignore").reindex(all_dates)
    prem = g.rolling(fw, min_periods=mh).mean().shift(1)
    dates = scores.index.get_level_values(0)
    P = prem[scores.columns].reindex(dates).values
    out["fm"] = pd.Series(np.where(np.isnan(P).any(axis=1), np.nan, (scores.values * P).sum(axis=1)), index=scores.index)
    return out
