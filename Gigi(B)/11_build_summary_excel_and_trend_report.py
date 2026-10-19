"""Build Member B midterm summary Excel + trend report for A/C."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

CRISIS = [
    ("COVID crash", "2020-02-15", "2020-04-15"),
    ("2021 bull peak / late 2021", "2021-10-01", "2021-12-31"),
    ("Luna / Terra", "2022-05-01", "2022-05-31"),
    ("FTX collapse", "2022-11-01", "2022-11-30"),
    ("2022 bear trough", "2022-06-01", "2022-06-30"),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def load_clean() -> pd.DataFrame:
    p = PROCESSED / "price_mcap_panel_CLEAN_v0.csv"
    if not p.exists():
        raise SystemExit(f"Missing {p}")
    df = pd.read_csv(p, parse_dates=["date"])
    return df


def overview_sheet(df: pd.DataFrame) -> pd.DataFrame:
    n_coins = df["coin_id"].nunique()
    src = (
        df["market_cap_source"].value_counts(dropna=False)
        if "market_cap_source" in df.columns
        else pd.Series(dtype=int)
    )
    rows = [
        ("built_at", now_iso()),
        ("panel_file", "price_mcap_panel_CLEAN_v0.csv"),
        ("rows", len(df)),
        ("coins", n_coins),
        ("date_min", str(df["date"].min().date())),
        ("date_max", str(df["date"].max().date())),
        ("target_sample_end", "2026-09-30"),
        ("ret_1d_count", int(df["ret_1d"].notna().sum())),
        ("ret_1d_mean", float(df["ret_1d"].mean())),
        ("ret_1d_std", float(df["ret_1d"].std())),
        ("ret_1d_min", float(df["ret_1d"].min())),
        ("ret_1d_max", float(df["ret_1d"].max())),
        ("extreme_return_flags", int(df.get("ret_extreme_flag", pd.Series(dtype=bool)).fillna(False).sum())),
        ("mcap_source_api_rows", int(src.get("api", 0))),
        ("mcap_source_approx_rows", int(src.get("approx", 0))),
        ("mcap_api_share", float(src.get("api", 0) / len(df)) if len(df) else np.nan),
        ("mcap_approx_share", float(src.get("approx", 0) / len(df)) if len(df) else np.nan),
        ("universe_rule", "historical mcap top 30-50 among downloaded candidates; exclude stables/wrapped per report"),
        ("onchain_gate", "no free on-chain metric passes full-universe; BTC/ETH timing only"),
    ]
    return pd.DataFrame(rows, columns=["key", "value"])


def coin_coverage(df: pd.DataFrame) -> pd.DataFrame:
    if "ret_extreme_flag" not in df.columns:
        df = df.copy()
        df["ret_extreme_flag"] = False
    if "market_cap" not in df.columns:
        df = df.copy()
        df["market_cap"] = np.nan
    g = df.groupby("coin_id", as_index=False).agg(
        n_days=("date", "count"),
        date_min=("date", "min"),
        date_max=("date", "max"),
        close_last=("close", "last"),
        ret_mean=("ret_1d", "mean"),
        ret_std=("ret_1d", "std"),
        ret_min=("ret_1d", "min"),
        ret_max=("ret_1d", "max"),
        extreme_flags=("ret_extreme_flag", "sum"),
        mcap_last=("market_cap", "last"),
    )
    if "market_cap_source" in df.columns:
        share = (
            df.assign(is_api=df["market_cap_source"].eq("api"))
            .groupby("coin_id")["is_api"]
            .mean()
            .rename("mcap_api_share")
        )
        g = g.merge(share, on="coin_id", how="left")
    g["date_min"] = g["date_min"].dt.date.astype(str)
    g["date_max"] = g["date_max"].dt.date.astype(str)
    return g.sort_values("n_days", ascending=False)


def yearly_returns(df: pd.DataFrame) -> pd.DataFrame:
    x = df.dropna(subset=["ret_1d"]).copy()
    x["year"] = x["date"].dt.year
    out = (
        x.groupby("year", as_index=False)
        .agg(
            n=("ret_1d", "count"),
            mean=("ret_1d", "mean"),
            std=("ret_1d", "std"),
            p05=("ret_1d", lambda s: s.quantile(0.05)),
            p50=("ret_1d", "median"),
            p95=("ret_1d", lambda s: s.quantile(0.95)),
            min=("ret_1d", "min"),
            max=("ret_1d", "max"),
            frac_gt_5pct=("ret_1d", lambda s: (s.abs() > 0.05).mean()),
            frac_gt_10pct=("ret_1d", lambda s: (s.abs() > 0.10).mean()),
        )
    )
    return out


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    x = df.dropna(subset=["ret_1d"]).copy()
    x["ym"] = x["date"].dt.to_period("M").astype(str)
    # equal-weight cross-section mean return per day, then monthly stats
    daily = x.groupby("date", as_index=False)["ret_1d"].mean().rename(columns={"ret_1d": "ew_mean_ret"})
    daily["ym"] = daily["date"].dt.to_period("M").astype(str)
    m = daily.groupby("ym", as_index=False).agg(
        n_days=("ew_mean_ret", "count"),
        ew_mean_daily=("ew_mean_ret", "mean"),
        ew_vol_daily=("ew_mean_ret", "std"),
        ew_cum_approx=("ew_mean_ret", lambda s: (1 + s).prod() - 1),
    )
    # breadth: share of coins with positive month return (compound within coin)
    x["month_ret"] = x.groupby(["coin_id", "ym"])["ret_1d"].transform(lambda s: (1 + s).prod() - 1)
    # one value per coin-month
    cm = x.drop_duplicates(["coin_id", "ym"])[["coin_id", "ym", "month_ret"]]
    breadth = cm.groupby("ym", as_index=False).agg(
        n_coins=("coin_id", "count"),
        share_up=("month_ret", lambda s: (s > 0).mean()),
        median_month_ret=("month_ret", "median"),
    )
    return m.merge(breadth, on="ym", how="left")


def crisis_windows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    x = df.dropna(subset=["ret_1d"])
    for name, a, b in CRISIS:
        w = x[(x["date"] >= a) & (x["date"] <= b)]
        if w.empty:
            rows.append({"window": name, "start": a, "end": b, "n": 0})
            continue
        daily = w.groupby("date")["ret_1d"].mean()
        rows.append(
            {
                "window": name,
                "start": a,
                "end": b,
                "n_obs": len(w),
                "n_days": daily.shape[0],
                "ew_mean_daily": float(daily.mean()),
                "ew_vol_daily": float(daily.std()),
                "ew_cum_approx": float((1 + daily).prod() - 1),
                "worst_day_ew": float(daily.min()),
                "best_day_ew": float(daily.max()),
                "tail_share_abs_gt_10pct": float((w["ret_1d"].abs() > 0.10).mean()),
            }
        )
    return pd.DataFrame(rows)


def corr_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """Mean pairwise corr of daily returns among top coins by coverage in windows."""
    windows = [
        ("full_sample", None, None),
        ("calm_2023", "2023-01-01", "2023-12-31"),
        ("crisis_luna", "2022-05-01", "2022-05-31"),
        ("crisis_ftx", "2022-11-01", "2022-11-30"),
        ("recent_365d", "2025-09-30", "2026-09-30"),
    ]
    # pick coins with most days
    top = (
        df.groupby("coin_id")["date"].count().sort_values(ascending=False).head(15).index.tolist()
    )
    rows = []
    for name, a, b in windows:
        w = df[df["coin_id"].isin(top)].copy()
        if a:
            w = w[(w["date"] >= a) & (w["date"] <= b)]
        piv = w.pivot_table(index="date", columns="coin_id", values="ret_1d")
        if piv.shape[1] < 3 or piv.shape[0] < 10:
            rows.append({"window": name, "n_days": piv.shape[0], "n_coins": piv.shape[1], "mean_pairwise_corr": np.nan})
            continue
        c = piv.corr()
        mask = np.triu(np.ones(c.shape, dtype=bool), k=1)
        vals = c.where(mask).stack()
        rows.append(
            {
                "window": name,
                "start": a or str(w["date"].min().date()),
                "end": b or str(w["date"].max().date()),
                "n_days": int(piv.dropna(how="all").shape[0]),
                "n_coins": int(piv.shape[1]),
                "mean_pairwise_corr": float(vals.mean()),
                "median_pairwise_corr": float(vals.median()),
            }
        )
    return pd.DataFrame(rows)


def universe_sheets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mem = PROCESSED / "universe_membership_v0.csv"
    summ = PROCESSED / "universe_membership_summary_v0.csv"
    if not mem.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    m = pd.read_csv(mem)
    m["rebalance_date"] = pd.to_datetime(m["rebalance_date"], errors="coerce")
    s = pd.read_csv(summ) if summ.exists() else pd.DataFrame()
    # membership count over time
    cnt = (
        m.groupby("rebalance_date", as_index=False)
        .agg(n_members=("coin_id", "count"), mean_rank=("rank_by_mcap", "mean"))
        .sort_values("rebalance_date")
    )
    cnt["rebalance_date"] = pd.to_datetime(cnt["rebalance_date"]).dt.date.astype(str)
    # turnover: Jaccard distance between consecutive sets
    turns = []
    dates = sorted(m["rebalance_date"].unique())
    prev = None
    for d in dates:
        cur = set(m.loc[m["rebalance_date"] == d, "coin_id"])
        if prev is not None:
            inter = len(prev & cur)
            union = len(prev | cur) or 1
            added = len(cur - prev)
            dropped = len(prev - cur)
            turns.append(
                {
                    "rebalance_date": str(pd.Timestamp(d).date()),
                    "n_members": len(cur),
                    "added": added,
                    "dropped": dropped,
                    "jaccard": inter / union,
                    "turnover_share": (added + dropped) / (2 * max(len(prev), 1)),
                }
            )
        prev = cur
    turn_df = pd.DataFrame(turns)
    # top persistence: how often each coin appears
    persist = (
        m.groupby("coin_id", as_index=False)
        .agg(n_rebalances=("rebalance_date", "count"), mean_rank=("rank_by_mcap", "mean"), min_rank=("rank_by_mcap", "min"))
        .sort_values("n_rebalances", ascending=False)
    )
    return cnt, turn_df, persist


def coverage_sheet() -> pd.DataFrame:
    p = DOCS / "coverage_gate_decisions_FILLED.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(
        [
            {"metric": "ohlcv_daily", "decision": "price_input_ready"},
            {"metric": "market_cap", "decision": "price_input_ready"},
            {"metric": "active_addresses", "decision": "btc_eth_timing"},
            {"metric": "tx_count", "decision": "btc_eth_timing"},
            {"metric": "nvt", "decision": "btc_eth_timing"},
            {"metric": "exchange_net_flow", "decision": "btc_eth_timing"},
            {"metric": "realised_cap", "decision": "btc_timing_only"},
            {"metric": "mvrv", "decision": "btc_timing_only"},
            {"metric": "sopr", "decision": "btc_timing_only"},
            {"metric": "sth_lth_cost_basis", "decision": "btc_timing_only"},
            {"note": "No free on-chain metric clears full-universe gate"},
        ]
    )


def cleaning_sheet() -> pd.DataFrame:
    p = PROCESSED / "cleaning_log_v0.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def write_report(
    overview: pd.DataFrame,
    yearly: pd.DataFrame,
    crisis: pd.DataFrame,
    corr: pd.DataFrame,
    turn: pd.DataFrame,
    persist: pd.DataFrame,
) -> Path:
    ov = {r.key: r.value for r in overview.itertuples()}
    # turnover summary
    turn_note = ""
    if len(turn):
        turn_note = (
            f"- Rebalances with turnover stats: {len(turn)}\n"
            f"- Mean Jaccard overlap consecutive membership: {turn['jaccard'].mean():.3f}\n"
            f"- Mean turnover_share: {turn['turnover_share'].mean():.3f}\n"
            f"- Mean added/dropped per rebalance: {turn['added'].mean():.2f} / {turn['dropped'].mean():.2f}\n"
        )
    top_persist = ""
    if len(persist):
        top = persist.head(10)
        lines = [f"  - {r.coin_id}: {int(r.n_rebalances)} times, mean rank {r.mean_rank:.1f}" for r in top.itertuples()]
        top_persist = "\n".join(lines)

    yearly_tbl = yearly.to_string(index=False)
    crisis_tbl = crisis.to_string(index=False)
    corr_tbl = corr.to_string(index=False)

    md = f"""# Data Trend Summary for Members A & C

**Prepared by:** Member B (Edward Lau)  
**Built:** {now_iso()}  
**Primary panel:** `data/processed/price_mcap_panel_CLEAN_v0.csv`  
**Companion Excel:** `reports/MemberB_data_summary_midterm.xlsx`

---

## 1. Sample coverage

| Item | Value |
|------|-------|
| Rows | {ov.get('rows')} |
| Coins in download universe | {ov.get('coins')} |
| Date range | {ov.get('date_min')} to {ov.get('date_max')} |
| Target sample end (report) | {ov.get('target_sample_end')} |
| Market-cap api share | {float(ov.get('mcap_api_share', 0)):.1%} |
| Market-cap approx share | {float(ov.get('mcap_approx_share', 0)):.1%} |
| Extreme return flags (abs daily > 50%) | {ov.get('extreme_return_flags')} |

**Universe rule (report only):** historical market-cap top 30??0 among *downloaded* candidates each rebalance; exclude stables / wrapped. Ranking is **not** global all-crypto (honest midterm note).

**On-chain coverage gate:** no free metric passes full-universe cross-section. FYP I cross-section uses **price factors**; on-chain metrics are **BTC / ETH timing** only. See `docs/coverage_gate_decisions_FILLED.csv`.

**Look-ahead disclosure:** early-sample `market_cap` with `market_cap_source=approx` uses **latest circulating supply ? historical close**. Disclose to A/C; optional sensitivity on `api`-only (~last 365 days).

---

## 2. Return characteristics (pooled daily `ret_1d`)

| Stat | Value |
|------|-------|
| Count | {ov.get('ret_1d_count')} |
| Mean | {float(ov.get('ret_1d_mean', 0)):.6f} |
| Std | {float(ov.get('ret_1d_std', 0)):.6f} |
| Min | {float(ov.get('ret_1d_min', 0)):.4f} |
| Max | {float(ov.get('ret_1d_max', 0)):.4f} |

Interpretation for A/C: mean near zero with **fat tails** (large min/max, extreme flags). Do not assume Gaussian residuals for factors / BL views without robust or truncated treatments.

### By calendar year

```
{yearly_tbl}
```

---

## 3. Crisis / stress windows (equal-weight cross-section of available coins)

```
{crisis_tbl}
```

Use these windows when A designs factor tests and C designs covariance / backtest stress checks.

---

## 4. Cross-sectional correlation snapshots

Mean pairwise correlation among ~15 longest-history coins:

```
{corr_tbl}
```

Expect **higher average correlation in crashes** than in calm years ??relevant for risk models (Member C) and for how much diversification crypto long-only books actually get.

---

## 5. Universe membership dynamics

{turn_note}

**Most persistent members (top 10 by rebalance count):**

{top_persist}

Full series: Excel sheets `Universe_size`, `Universe_turnover`, `Universe_persistence`.

---

## 6. What A / C should take as inputs

1. **Prices / returns:** `price_mcap_panel_CLEAN_v0.csv` (`ret_1d` already lagged within coin).
2. **Membership:** `universe_membership_v0.csv` (weekly, that-day mcap ranks).
3. **Size signal:** `market_cap` with source flag; prefer documenting approx bias.
4. **Do not** treat free on-chain series as full-universe factors for FYP I.
5. **Extend to 2026-09-30:** re-pull OHLCV after sample end date, then re-run clean + this summary.

---

## 7. File map

| File | Role |
|------|------|
| `price_mcap_panel_CLEAN_v0.csv` | Clean price + mcap + ret_1d |
| `universe_membership_v0.csv` | Survivorship-aware membership |
| `cleaning_log_v0.csv` | Clean steps |
| `docs/data_dictionary_v0.md` | Column meanings |
| `docs/lookahead_audit_checklist_v0.md` | Audit ticks |
| `docs/coverage_gate_decisions_FILLED.csv` | Gate decisions |
| `reports/MemberB_data_summary_midterm.xlsx` | This Excel pack |
| `reports/data_summary_for_AC_Oct2026.md` | This narrative |

---

*Generated by `scripts/11_build_summary_excel_and_trend_report.py`.*
"""
    out = REPORTS / "data_summary_for_AC_Oct2026.md"
    out.write_text(md, encoding="utf-8")
    return out


def main() -> None:
    print("Loading clean panel...")
    df = load_clean()
    print(f"rows={len(df)} coins={df['coin_id'].nunique()}")

    overview = overview_sheet(df)
    coins = coin_coverage(df)
    yearly = yearly_returns(df)
    monthly = monthly_trend(df)
    crisis = crisis_windows(df)
    corr = corr_snapshots(df)
    cov = coverage_sheet()
    clog = cleaning_sheet()
    u_size, u_turn, u_persist = universe_sheets()

    xlsx = REPORTS / "MemberB_data_summary_midterm.xlsx"
    print(f"Writing {xlsx} ...")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        overview.to_excel(w, sheet_name="Overview", index=False)
        coins.to_excel(w, sheet_name="Coin_coverage", index=False)
        yearly.to_excel(w, sheet_name="Returns_by_year", index=False)
        monthly.to_excel(w, sheet_name="Monthly_EW_trend", index=False)
        crisis.to_excel(w, sheet_name="Crisis_windows", index=False)
        corr.to_excel(w, sheet_name="Corr_snapshots", index=False)
        if len(u_size):
            u_size.to_excel(w, sheet_name="Universe_size", index=False)
        if len(u_turn):
            u_turn.to_excel(w, sheet_name="Universe_turnover", index=False)
        if len(u_persist):
            u_persist.to_excel(w, sheet_name="Universe_persistence", index=False)
        if len(cov):
            cov.to_excel(w, sheet_name="Coverage_gate", index=False)
        if len(clog):
            clog.to_excel(w, sheet_name="Cleaning_log", index=False)

    md = write_report(overview, yearly, crisis, corr, u_turn, u_persist)
    print(f"wrote {xlsx}")
    print(f"wrote {md}")
    print("sheets: Overview, Coin_coverage, Returns_by_year, Monthly_EW_trend, Crisis_windows, Corr_snapshots, Universe_*, Coverage_gate, Cleaning_log")


if __name__ == "__main__":
    main()
