"""清除與禁運 (purged and embargoed) 的 expanding walk-forward 切分。
每個觀察值在日期 s 的「目標視窗」是 [s, s+horizon]。訓練資料的目標視窗不可以和測試期重疊，
所以在測試期開始前要「清除 (purge)」掉最後幾週；測試期結束後的幾週要「禁運 (embargo)」。"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Fold:
    k: int
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex

    def __repr__(self):
        return (f"Fold({self.k}: train {self.train_dates[0].date()}~{self.train_dates[-1].date()} "
                f"({len(self.train_dates)}w), test {self.test_dates[0].date()}~{self.test_dates[-1].date()} "
                f"({len(self.test_dates)}w))")


def purged_walk_forward(dates, min_train_weeks=156, test_weeks=26, purge_weeks=2, embargo_weeks=2,
                        horizon_weeks=1):
    """dates: 每週一個的日期(已排序)。回傳 Fold 清單。
    - 訓練集永遠只用測試期「以前」的資料 (expanding window)。
    - 訓練集尾端少 purge_weeks 週 (必須 >= horizon_weeks，否則目標視窗會重疊)。
    - embargo_weeks: 前一個測試區塊結束後的 embargo_weeks 週，若落在下一個訓練集中，也會被移除。
    - 測試區塊不重疊、連續排列。"""
    if purge_weeks < horizon_weeks:
        raise ValueError("purge_weeks 必須 >= horizon_weeks，否則訓練與測試的目標視窗會重疊")
    d = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(dates))))
    n = len(d)
    folds, start, k = [], min_train_weeks + purge_weeks, 0
    prev_blocks = []
    while start + 1 <= n:
        end = min(start + test_weeks, n)
        test = d[start:end]
        tr_idx = np.arange(0, start - purge_weeks)
        keep = np.ones(len(tr_idx), dtype=bool)
        for (ps, pe) in prev_blocks:                         # 禁運: 前一個測試區塊之後的 embargo 週
            keep &= ~((tr_idx >= pe) & (tr_idx < pe + embargo_weeks))
        train = d[tr_idx[keep]]
        if len(train) >= min_train_weeks - embargo_weeks * len(prev_blocks) - 1:
            folds.append(Fold(k, train, test))
            k += 1
        prev_blocks.append((start, end))
        start = end
    return folds


def check_no_overlap(folds, horizon_weeks=1, step_days=7):
    """自我檢查: 每個訓練觀察值的目標視窗結束日 < 該 fold 測試期第一天。回傳通過與否。"""
    for f in folds:
        last_target_end = f.train_dates.max() + pd.Timedelta(days=step_days * horizon_weeks)
        if not last_target_end < f.test_dates.min():
            return False
        if f.train_dates.max() >= f.test_dates.min():
            return False
    return True
