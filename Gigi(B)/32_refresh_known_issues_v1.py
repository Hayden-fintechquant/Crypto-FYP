"""Refresh known_issues_log.md after outlier verify + Coin Metrics gate."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    missing = PROCESSED / "missing_coins.csv"
    n_miss = len(pd.read_csv(missing)) if missing.exists() else 0

    ver_path = REPORTS / "outlier_verify_v1.csv"
    if not ver_path.exists():
        ver_path = REPORTS / "outlier_candidates_v1.csv"
    vc = {}
    n_ext = 0
    if ver_path.exists():
        ver = pd.read_csv(ver_path)
        n_ext = len(ver)
        if "outlier_check" in ver.columns:
            vc = ver["outlier_check"].value_counts().to_dict()

    cm_note = ""
    cm_md = DOCS / "coinmetrics_coverage_note_v1.md"
    if cm_md.exists():
        cm_note = cm_md.read_text(encoding="utf-8")[:800]

    text = f"""# Known issues log (Member B data layer)

Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC

## Assumptions
- **USDT ≈ USD** for all Binance USDT quote volumes and prices used as USD.
- Rebalance: Sunday CMC snapshot ranks → membership effective next Monday (UTC).
- Classic **LUNA** series ends **2022-05-13** (collapse). Later Binance `LUNAUSDT` is terra-luna-v2 and is **not** spliced into classic LUNA.
- **FTT** kept through sample end; crisis analysis should focus on Nov 2022 window (`end_reason=collapsed`).
- Early FTM/MATIC rows before CMC top-100 entry: `market_cap_source=missing_pre_cmc_top100` (no look-ahead supply fill).
- Daily mcap between weekly CMC Sundays uses `cmc_supply_x_close` (prior snapshot circulating supply × daily close).

## Source truncations
- **XMR**: Binance delist ~2024-02-20; refilled via CoinGecko where Demo allows (~365d). Gap between delist and CG window (if any) remains `source_truncated` risk — check panel dates.
- **EOS**: panel ends 2025-05-26 (`source_truncated`); not treated as collapse.
- Some thin CMC-only fills (script 28/29) are weekly-price expanded to daily; treat as lower resolution for microstructure / high-freq factors.

## Migrations (see coin_id_map.csv)
- MATIC → POL (`polygon-ecosystem-token`)
- FTM → S (`sonic-3`)
- MKR → SKY (`sky`)
- Return splicing across migration is **not** auto-applied in v1 panel; A/C decide join rules.

## Missing prices
- `missing_coins.csv` count = **{n_miss}** after CMC weekly fill (script 29). Target for v1 handoff: 0 for ever top-40 members.

## Outliers (problem 6)
- Flag rule: `|ret_1d| >= 50%` → needs `outlier_check`.
- Second source: nearest CMC weekly snapshot price (≤10 days), relative error thresholds 35% / 100%.
- Extreme rows checked: **{n_ext}**. Counts: **{vc}**.
- Detail: `reports/outlier_verify_v1.csv` (also mirrored to `outlier_candidates_v1.csv`).
- Panel also has `days_since_listing` for A’s first-30-day exclusion.

## On-chain coverage (Coin Metrics re-check)
- Community API catalog + timeseries spot-check completed (script 31).
- Ever-universe members with **both** `AdrActCnt` + `TxCnt`: **partial (~29%)** → does **not** clear full cross-section gate.
- Decision kept: **`btc_eth_timing`** for active addresses / tx count.
- Files: `reports/coinmetrics_coverage_v1.csv`, `docs/coinmetrics_coverage_note_v1.md`, updated `docs/coverage_gate_decisions_FILLED.csv`.

## Handoff
- Primary panel: `data/processed/price_mcap_panel_v1.csv`
- Universe: `data/processed/universe_membership_cmc_v0_1.csv`
- Summary for A/C: `docs/handoff_v1_for_AC.md`
"""
    path = DOCS / "known_issues_log.md"
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")
    print(f"missing={n_miss}; extreme={n_ext}; counts={vc}")
    if cm_note:
        print("coinmetrics note present")
    print("done")


if __name__ == "__main__":
    main()
