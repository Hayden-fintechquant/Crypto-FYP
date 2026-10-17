"""讀取 B 的檔案，並整理成「日期 x 幣」的寬表。"""
import numpy as np
import pandas as pd

REQ_PANEL = ["date", "coin_id", "close", "volume_usd", "mcap"]
REQ_UNIV = ["rebalance_date", "coin_id", "is_member"]


def load_panel(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    miss = [c for c in REQ_PANEL if c not in df.columns]
    if miss:
        raise ValueError(f"panel 缺少欄位: {miss}")
    if df.duplicated(["date", "coin_id"]).any():
        raise ValueError("panel 的 (date, coin_id) 有重複，請 B 修正")
    if (df["close"] <= 0).any():
        raise ValueError("panel 有 close <= 0")
    return df.sort_values(["coin_id", "date"]).reset_index(drop=True)


def load_universe(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["rebalance_date"])
    miss = [c for c in REQ_UNIV if c not in df.columns]
    if miss:
        raise ValueError(f"universe 缺少欄位: {miss}")
    df["is_member"] = df["is_member"].astype(int) > 0
    return df


def to_long(wide: pd.DataFrame, name: str) -> pd.Series:
    """寬表(日期 x 幣) -> 長表 Series，index=(date, coin_id)，NaN 保留。"""
    w = wide.copy()
    w.index.name = "date"
    w.columns.name = "coin_id"
    out = w.reset_index().melt(id_vars="date", var_name="coin_id", value_name=name)
    return out.set_index(["date", "coin_id"])[name]


class Market:
    """把 panel 轉成寬表，並預先算好日報酬、第一/最後一筆資料日。"""

    def __init__(self, panel: pd.DataFrame):
        close = panel.pivot(index="date", columns="coin_id", values="close")
        full = pd.date_range(close.index.min(), close.index.max(), freq="D")
        self.close = close.reindex(full)
        self.volume = panel.pivot(index="date", columns="coin_id", values="volume_usd").reindex(full)
        self.mcap = panel.pivot(index="date", columns="coin_id", values="mcap").reindex(full)
        self.ret1d = self.close / self.close.shift(1) - 1.0   # 只在兩天都有價格時才有值
        self.first_date = self.close.apply(lambda s: s.first_valid_index())
        self.last_date = self.close.apply(lambda s: s.last_valid_index())
        self.end = full[-1]
