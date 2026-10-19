"""Day 9/28: clean panel, look-ahead rules, data dictionary."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
SAMPLE_END = "2026-09-30"
SAMPLE_START = "2018-01-01"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    src = PROCESSED / "price_mcap_panel_v0.csv"
    if not src.exists():
        raise SystemExit(f"Missing {src}")

    raw = pd.read_csv(src)
    n0 = len(raw)
    log = []

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = int(df["date"].isna().sum())
    df = df.dropna(subset=["date", "coin_id"])
    log.append(
        {
            "step": "drop_null_date_or_coin",
            "rows_removed": n0 - len(df),
            "detail": f"bad_dates={bad_dates}",
        }
    )

    before = len(df)
    df = df[(df["date"] >= SAMPLE_START) & (df["date"] <= SAMPLE_END)]
    log.append(
        {
            "step": "clip_sample_window",
            "rows_removed": before - len(df),
            "detail": f"{SAMPLE_START}..{SAMPLE_END}",
        }
    )

    before = len(df)
    df = df.sort_values(["coin_id", "date"])
    dups = int(df.duplicated(["date", "coin_id"]).sum())
    df = df.drop_duplicates(["date", "coin_id"], keep="last")
    log.append(
        {
            "step": "dedupe_date_coin",
            "rows_removed": before - len(df),
            "detail": f"dup_flags={dups}",
        }
    )

    before = len(df)
    df = df[df["close"].notna() & (df["close"] > 0)]
    log.append(
        {
            "step": "drop_nonpositive_close",
            "rows_removed": before - len(df),
            "detail": "",
        }
    )

    if {"high", "low"}.issubset(df.columns):
        bad_hl = (df["high"] < df["low"]).fillna(False)
        n_bad = int(bad_hl.sum())
        if n_bad:
            df.loc[bad_hl, "high"] = df.loc[bad_hl, "close"]
            df.loc[bad_hl, "low"] = df.loc[bad_hl, "close"]
        log.append(
            {"step": "fix_high_lt_low", "rows_removed": 0, "detail": f"fixed={n_bad}"}
        )

    df["ret_1d"] = df.groupby("coin_id")["close"].pct_change()
    df["ret_extreme_flag"] = df["ret_1d"].abs() > 0.5
    log.append(
        {
            "step": "flag_extreme_returns_gt_50pct",
            "rows_removed": 0,
            "detail": f"flagged={int(df['ret_extreme_flag'].sum())}",
        }
    )

    if "market_cap" in df.columns:
        df["mcap_bad_flag"] = df["market_cap"].notna() & (df["market_cap"] <= 0)
        log.append(
            {
                "step": "flag_nonpositive_mcap",
                "rows_removed": 0,
                "detail": f"flagged={int(df['mcap_bad_flag'].sum())}",
            }
        )

    df["clean_timestamp"] = now_iso()
    df["date"] = df["date"].dt.date.astype(str)

    out = PROCESSED / "price_mcap_panel_CLEAN_v0.csv"
    df.to_csv(out, index=False)
    pd.DataFrame(log).to_csv(PROCESSED / "cleaning_log_v0.csv", index=False)

    dict_md = DOCS / "data_dictionary_v0.md"
    dict_md.write_text(
        f"""# Data Dictionary - Member B cleaned panel

**File:** `data/processed/price_mcap_panel_CLEAN_v0.csv`
**Built:** {now_iso()}
**Sample window:** {SAMPLE_START} to {SAMPLE_END}

## Columns

| Column | Meaning | Look-ahead note |
|--------|---------|-----------------|
| date | UTC calendar date | End of that UTC day for daily bars |
| coin_id | Asset id | Fixed label |
| open, high, low, close | USD prices (Binance daily) | For signal on day T, prefer using info known by T; portfolio return usually T to T+1 |
| volume | Quote volume USDT | Same-day |
| market_cap | Cap used for size/universe | api = last ~365d CoinGecko; approx = close * latest circulating supply |
| market_cap_source | api or approx | approx uses latest supply snapshot (disclose) |
| circulating_supply / circulating_supply_latest | Latest supply snapshot | Do not treat as known in 2018 |
| ret_1d | close_t / close_t-1 - 1 | Lagged close only within coin - OK |
| ret_extreme_flag | abs(ret_1d) > 0.5 | Flag only, not auto-dropped |
| mcap_bad_flag | market_cap <= 0 | Flag only |
| clean_timestamp | When cleaned | Metadata |

## Related files

- `universe_membership_v0.csv` - weekly membership by that day's mcap rank
- `coverage_report_FINAL_MemberB.md` - coverage gate
- `cleaning_log_v0.csv` - this clean step log
- `lookahead_audit_checklist_v0.md` - audit ticks
""",
        encoding="utf-8",
    )

    la = DOCS / "lookahead_audit_checklist_v0.md"
    la.write_text(
        """# Look-ahead Audit Checklist - Member B

## Prices / returns
- [x] `ret_1d` uses only lagged close within each coin
- [ ] A/C backtests: signals dated T must not use T+1 prices
- [x] No forward-filled future prices into the past

## Market cap / universe
- [x] Membership uses that rebalance date's market_cap (not today's ranks)
- [ ] Disclose early-sample approx = latest supply * historical price
- [ ] Optional sensitivity: size/universe on `market_cap_source=api` only (~365d)

## On-chain
- [x] Gate: no full-universe on-chain factors on free data
- [ ] Any later BTC timing series must be available as of date T

## Reproducibility
- [x] Script `scripts/10_clean_panel_and_docs.py`
- [x] Cleaning log + data dictionary saved
- [ ] A/C confirmed snapshot `price_mcap_panel_CLEAN_v0.csv`

## Sign-off
Member B: __________ date ____
Member A: __________ date ____
Member C: __________ date ____
""",
        encoding="utf-8",
    )

    print(f"wrote {out}")
    print(f"rows_in={n0} rows_out={len(df)}")
    print(f"wrote {PROCESSED / 'cleaning_log_v0.csv'}")
    print(f"wrote {dict_md}")
    print(f"wrote {la}")
    print("date range", df["date"].min(), df["date"].max())
    print("extreme_return_flags", int(df["ret_extreme_flag"].sum()))


if __name__ == "__main__":
    main()
