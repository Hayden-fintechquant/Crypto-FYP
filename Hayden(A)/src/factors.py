"""因子與預測報酬的計算。重點: 每個因子只用「資訊截止日(週日)以前」的資料。"""
import re
import numpy as np
import pandas as pd
from .data_io import Market, to_long


def raw_factor_wide(m: Market, name: str) -> pd.DataFrame:
    """回傳 日期 x 幣 的寬表；每個日期 d 的值只用 <= d 的資料。"""
    if name == "size":
        return np.log(m.mcap.where(m.mcap > 0))
    mm = re.fullmatch(r"mom_(\d+)w", name)
    if mm:
        k = 7 * int(mm.group(1))
        return m.close / m.close.shift(k) - 1.0
    mm = re.fullmatch(r"rev_(\d+)w", name)
    if mm:
        k = 7 * int(mm.group(1))
        return -(m.close / m.close.shift(k) - 1.0)
    mm = re.fullmatch(r"vol_(\d+)d", name)
    if mm:
        n = int(mm.group(1))
        return m.ret1d.rolling(n, min_periods=int(n * 0.75)).std()
    mm = re.fullmatch(r"liq_(\d+)d", name)
    if mm:
        n = int(mm.group(1))
        avg = m.volume.rolling(n, min_periods=int(n * 0.75)).mean()
        return np.log(avg.where(avg > 0))
    mm = re.fullmatch(r"amihud_(\d+)d", name)
    if mm:
        n = int(mm.group(1))
        x = m.ret1d.abs() / m.volume.where(m.volume > 0)
        return x.rolling(n, min_periods=int(n * 0.75)).mean()
    raise ValueError(f"不認識的因子名稱: {name}")


def forward_return_wide(m: Market, horizon: int, policy: str, scope: str = "collapsed_only"):
    """從日期 d 收盤到 d+horizon 收盤的報酬。回傳 (fwd, ended)。
    ended = 該幣在 d+horizon 之前就沒有價格了(下市/歸零/資料中斷)，依 policy 處理。"""
    c = m.close
    fwd = c.shift(-horizon) / c - 1.0
    h_end = (c.index + pd.Timedelta(days=horizon)).values
    last = m.last_date.values  # 每個幣最後一個有價格的日期
    ended = (last[None, :] < h_end[:, None]) & (h_end[:, None] <= np.datetime64(m.end)) & c.notna().values
    ended = pd.DataFrame(ended, index=c.index, columns=c.columns)
    # 為什麼只限崩盤的幣: 用「整份面板的最後一筆資料日」判斷 ended 會用到未來 (資料缺口 vs 真的下市
    # 在截斷資料下無法區分，例如 Monero 2024-25 的缺口)。崩盤是 B 的資料屬性，不依賴未來。
    if scope == "collapsed_only" and m.collapsed is not None:
        ended = ended & pd.Series({k: (k in m.collapsed) for k in c.columns}).reindex(c.columns).values[None, :]
    if policy == "last_price":
        fwd_last = c.ffill().shift(-horizon) / c - 1.0
        fwd = fwd.where(~ended, fwd_last)
    elif policy == "minus_100":
        fwd = fwd.where(~ended, -1.0)
    elif policy == "drop":
        pass
    else:
        raise ValueError("delist_policy 必須是 last_price | minus_100 | drop")
    return fwd, ended


def build_factor_panel(m: Market, universe: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """每個再平衡日 t、每個合格的幣一列。index=(date=再平衡日, coin_id)。
    欄位: 各因子原始值、fwd_ret、ended。"""
    universe = universe.assign(is_member=universe["is_member"].astype(int) > 0)   # 0/1 或 True/False 都可以
    tcfg, scfg = cfg["timing"], cfg["screening"]
    off = pd.Timedelta(days=int(tcfg["info_cutoff_offset_days"]))
    H = int(tcfg["horizon_days"])
    reb = pd.DatetimeIndex(sorted(universe["rebalance_date"].unique()))
    reb = reb[(reb >= pd.Timestamp(cfg["sample"]["start"])) & (reb <= pd.Timestamp(cfg["sample"]["end"]))]
    cut = reb - off
    ok = cut.isin(m.close.index)
    reb, cut = reb[ok], cut[ok]
    if len(reb) == 0:
        raise ValueError("樣本期間內沒有可用的再平衡日，請檢查 sample.start/end 與 panel 日期")

    # 成員矩陣 (再平衡日 x 幣)
    mem = (universe[universe["is_member"]].assign(v=1)
           .pivot_table(index="rebalance_date", columns="coin_id", values="v", aggfunc="max")
           .reindex(index=reb, columns=m.close.columns).fillna(0).astype(bool))
    # 合格: 是成員、有價格、市值>0、上線超過 N 天
    px = m.close.loc[cut]; px.index = reb
    mc = m.mcap.loc[cut]; mc.index = reb
    dsl = pd.DataFrame((cut.values[:, None] - m.first_date.values[None, :]).astype("timedelta64[D]").astype(float),
                       index=reb, columns=m.close.columns)
    elig = mem & px.notna() & (mc > 0) & (dsl >= scfg["min_days_since_listing"])
    # 小組規則 3: 近 28 天平均日成交額 >= 門檻 (預設 0 = 不篩; 沒有成交量資料 = 不合格)
    vmin = float(scfg.get("min_avg_volume_usd", 0) or 0)
    if vmin > 0:
        av = m.volume.rolling(28, min_periods=int(scfg.get("min_volume_obs", 14))).mean().loc[cut]
        av.index = reb
        elig = elig & (av >= vmin)

    cols = {}
    for name in cfg["factors"]:
        w = raw_factor_wide(m, name).loc[cut]
        w.index = reb
        cols[name] = to_long(w, name)
    fwd, ended = forward_return_wide(m, H, tcfg["delist_policy"], tcfg.get("delist_scope", "collapsed_only"))
    f = fwd.loc[cut]; f.index = reb
    e = ended.loc[cut]; e.index = reb
    cols["fwd_ret"] = to_long(f, "fwd_ret")
    cols["ended"] = to_long(e.astype(float), "ended")
    fp = pd.concat(cols, axis=1)
    keep = to_long(elig.astype(float), "elig")
    keep = keep[keep > 0].index
    return fp.reindex(keep).sort_index()
