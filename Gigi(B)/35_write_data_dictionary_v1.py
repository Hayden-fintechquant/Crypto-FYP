"""Rewrite data_dictionary_v1.md from actual panel columns when present."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
PANEL_V11 = PROCESSED / "price_mcap_panel_v1_1.csv"
PANEL_V1 = PROCESSED / "price_mcap_panel_v1.csv"
UNI = PROCESSED / "universe_membership_cmc_v0_1.csv"
OUT = DOCS / "data_dictionary_v1.md"
OLD = DOCS / "data_dictionary_v0.md"

# Descriptions for known columns (others get generic note)
DESC = {
    "date": "UTC calendar date of the daily bar",
    "coin_id": "Stable asset id used across panel / universe / id map",
    "symbol": "Ticker symbol (display)",
    "open": "USD open (USDT treated as USD)",
    "high": "USD high",
    "low": "USD low",
    "close": "USD close — primary price for returns",
    "volume": "Raw quote volume (usually USDT)",
    "volume_usd": "Quote volume treated as USD (USDT≈USD)",
    "avg_volume_28d": "28-day average of volume_usd",
    "market_cap": "Market cap used for size / ranking (Hayden: mcap)",
    "market_cap_source": "Provenance of market_cap (cmc_snapshot, cmc_supply_x_close, cmc_weekly_ffill, api, …)",
    "price_source": "Provenance of OHLC (binance, cmc_snapshot, cmc_weekly_ffill, coingecko, …)",
    "circulating_supply": "Circulating supply paired with mcap when available",
    "circulating_supply_latest": "Latest supply snapshot — do not treat as known historically",
    "ret_1d": "close_t / close_{t-1} - 1 within coin_id",
    "ret_extreme_flag": "True if abs(ret_1d) > 0.5",
    "outlier_check": "verified_real | exclude | data_error_removed | '' (blank=not extreme)",
    "weekly_ffill_flag": "True if price_source or market_cap_source is cmc_weekly_ffill",
    "days_since_listing": "Days since first panel price for this coin_id",
    "stale_price_flag": "Stale / flat price heuristic flag",
    "zero_volume_flag": "Zero or missing volume flag",
    "cmc_slug": "CoinMarketCap slug used for joins",
    "mcap_bad_flag": "market_cap <= 0",
    "end_reason": "Optional copy of id-map end_reason (prefer coin_id_map.csv)",
}


OUTLIER_DOC = """
## outlier_check values (v1.1)

| Value | Meaning for Member A |
|-------|----------------------|
| `` (blank) | Not an extreme daily return row |
| `verified_real` | Extreme move confirmed (crash window or CMC second source) — usable |
| `exclude` | Extreme move not second-source verified — **keep prices for levels**, but **exclude from daily-return factor estimation** (vol / momentum) |
| `data_error_removed` | Unrepairable bad extreme; close/ret nulled or row dropped (see `reports/data_error_dropped_v1.csv`) |

There must be **zero** `unverified` labels after script 33 (Hayden review script rejects unverified).
"""


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    panel_path = PANEL_V11 if PANEL_V11.exists() else PANEL_V1
    cols: list[str] = []
    n_rows = None
    n_coins = None
    dmin = dmax = None
    if panel_path.exists():
        df = pd.read_csv(panel_path, low_memory=False, nrows=5)
        cols = df.columns.tolist()
        # full scan for bounds without loading all object cols twice
        full = pd.read_csv(panel_path, usecols=["date", "coin_id"], low_memory=False)
        full["date"] = pd.to_datetime(full["date"])
        n_rows = len(full)
        n_coins = int(full["coin_id"].nunique())
        dmin = full["date"].min().date().isoformat()
        dmax = full["date"].max().date().isoformat()
        # prefer full column list
        cols = pd.read_csv(panel_path, nrows=0).columns.tolist()
    else:
        cols = [
            "date",
            "coin_id",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "volume_usd",
            "market_cap",
            "market_cap_source",
            "price_source",
            "ret_1d",
            "weekly_ffill_flag",
            "outlier_check",
            "days_since_listing",
        ]
        print(f"WARNING: panel not found at {PANEL_V1} / {PANEL_V11}; writing expected columns only")

    uni_cols = []
    if UNI.exists():
        uni_cols = pd.read_csv(UNI, nrows=0).columns.tolist()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Data Dictionary v1 / v1.1 — Member B panel",
        "",
        f"**Primary file:** `data/processed/price_mcap_panel_v1.csv` (also `price_mcap_panel_v1_1.csv` after script 33)",
        f"**Dictionary built:** {now}",
        f"**Sample window:** {dmin or '2018-01-01'} → {dmax or '2026-09-29'} (target end **2026-09-29**, not partial 2026-09-30)",
        f"**Rows × coins (if panel present):** {n_rows} rows, {n_coins} coins",
        "",
        "## Quote currency",
        "- Binance pairs are `*USDT`. For this project **USDT ≈ USD** (1:1) for `close` and `volume_usd`.",
        "",
        "## Panel columns",
        "",
        "| Column | Meaning | Look-ahead / usage note |",
        "|--------|---------|-------------------------|",
    ]
    for c in cols:
        meaning = DESC.get(c, "See pipeline scripts / known_issues_log — pass-through column")
        note = ""
        if c == "close":
            note = "Hayden primary price"
        elif c == "market_cap":
            note = "Hayden mcap; do not use approx latest-supply for members"
        elif c == "volume_usd":
            note = "Prefer over raw volume"
        elif c == "weekly_ffill_flag":
            note = "Daily-return factors INVALID when True"
        elif c == "outlier_check":
            note = "exclude → drop from vol/momentum estimation"
        elif c == "days_since_listing":
            note = "A may drop first 30 days"
        lines.append(f"| `{c}` | {meaning} | {note} |")

    lines.append(OUTLIER_DOC)
    lines += [
        "## Universe file",
        "",
        "**File:** `data/processed/universe_membership_cmc_v0_1.csv`",
        "",
        "Hayden needs: `rebalance_date`, `coin_id`, `rank` (alias of `rank_by_mcap`), and will add `is_member` if needed.",
        "",
        "| Column | Meaning |",
        "|--------|---------|",
    ]
    for c in uni_cols or [
        "snapshot_date",
        "rebalance_date",
        "coin_id",
        "symbol",
        "rank_by_mcap",
        "is_member",
    ]:
        extra = ""
        if c == "rank_by_mcap":
            extra = " — Hayden `rank`"
        if c == "rebalance_date":
            extra = " — membership effective date (Monday after Sunday CMC snapshot)"
        lines.append(f"| `{c}` | universe field{extra} |")

    lines += [
        "",
        "## Related files",
        "",
        "- `coin_id_map.csv` — `end_reason` ∈ {still_trading, migrated, collapsed, source_truncated}",
        "- `reports/outlier_verify_v1.csv`",
        "- `reports/weekly_ffill_coins_dates_v1.csv`",
        "- `reports/mcap_source_by_year_v1.csv`",
        "- `docs/known_issues_log.md`",
        "- `docs/handoff_v1_for_AC.md`",
        "",
        "## Deprecated",
        "",
        "- `data_dictionary_v0.md` described `price_mcap_panel_CLEAN_v0.csv` — use **this v1 dictionary** instead.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} cols={len(cols)}")

    # Point v0 header to v1
    pointer = (
        "# Data Dictionary v0 (DEPRECATED)\n\n"
        f"> **Superseded.** Use `docs/data_dictionary_v1.md` for the v1 / v1.1 panel "
        f"(`price_mcap_panel_v1.csv`). This file remains only for midterm CLEAN_v0 archaeology.\n\n"
        f"Updated pointer: {now}\n\n"
    )
    if OLD.exists():
        old = OLD.read_text(encoding="utf-8")
        if "Superseded" not in old:
            OLD.write_text(pointer + old, encoding="utf-8")
            print(f"prepended deprecation pointer to {OLD}")
        else:
            # refresh pointer block
            rest = old.split("\n\n", 2)
            OLD.write_text(pointer + (rest[-1] if len(rest) > 2 else old), encoding="utf-8")
            print(f"refreshed deprecation pointer on {OLD}")
    else:
        OLD.write_text(pointer, encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
