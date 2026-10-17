"""合成資料產生器: 用來在拿到 B 的真實資料「之前」測試整條流程。
內建一個已知訊號: 流動性高的幣，每天有固定的額外報酬 (beta_liq)，所以 liq_28d 應該有正的 IC。"""
import numpy as np
import pandas as pd


def make_synthetic(n_coins=60, start="2019-01-01", end="2024-12-29", seed=7, beta_liq=0.0010, top_n=40):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    T = len(dates)
    liq_level = rng.normal(0, 1, n_coins)
    supply = np.exp(rng.normal(19, 1.0, n_coins))
    start_idx = np.r_[np.zeros(40, int), rng.integers(100, T - 400, n_coins - 40)]
    end_idx = np.full(n_coins, T)
    collapse = rng.choice(np.arange(5, 40), 3, replace=False)       # 3 個幣中途崩盤
    for k in collapse:
        end_idx[k] = rng.integers(T // 2, T - 200)
    mkt = rng.normal(0.0004, 0.03, T)
    rows = []
    for i in range(n_coins):
        s, e = start_idx[i], end_idx[i]
        n = e - s
        eps = rng.standard_t(4, n) * 0.035 / np.sqrt(2)
        r = 0.6 * mkt[s:e] + eps + beta_liq * liq_level[i]
        if i in collapse:
            r[-5:] = -0.4
        price = 10 * np.exp(np.cumsum(r))
        volume = np.exp(14 + 1.0 * liq_level[i] + rng.normal(0, 0.3, n))
        df = pd.DataFrame(dict(date=dates[s:e], coin_id=f"coin{i:02d}", close=price, volume_usd=volume,
                               mcap=price * supply[i]))
        rows.append(df)
    panel = pd.concat(rows, ignore_index=True)

    mcap = panel.pivot(index="date", columns="coin_id", values="mcap")
    urows = []
    for d in pd.date_range(dates[0] + pd.Timedelta(days=1), dates[-1], freq="W-MON"):
        sunday = d - pd.Timedelta(days=1)
        if sunday not in mcap.index:
            continue
        cap = mcap.loc[sunday].dropna().sort_values(ascending=False)
        for rank, (coin, c) in enumerate(cap.items(), start=1):
            urows.append((d, coin, rank, c, int(rank <= top_n)))
    universe = pd.DataFrame(urows, columns=["rebalance_date", "coin_id", "rank", "mcap", "is_member"])
    return panel, universe
