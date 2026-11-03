"""Write known_issues_log.md + append USDT≈USD note to data dictionary if present."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    panel = PROCESSED / "price_mcap_panel_v1.csv"
    missing = PROCESSED / "missing_coins.csv"
    n_miss = len(pd_read(missing)) if missing.exists() else "n/a"
    text = f"""# Known issues log (Member B data layer)

Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC

## Assumptions
- **USDT ≈ USD** for all Binance USDT quote volumes and prices used as USD.
- Rebalance: Sunday CMC snapshot ranks → membership effective next Monday (UTC).
- Classic **LUNA** series ends **2022-05-13** (collapse). Later Binance `LUNAUSDT` is terra-luna-v2 and is **not** spliced into classic LUNA.
- **FTT** kept through sample end; crisis analysis should focus on Nov 2022 window (`end_reason=collapsed`).
- Early FTM/MATIC rows before CMC top-100 entry: `market_cap_source=missing_pre_cmc_top100` (no look-ahead supply fill).

## Source truncations
- **XMR**: Binance delist ~2024-02-20; refilled via CoinGecko where Demo allows (~365d). Gap between delist and CG window (if any) remains `source_truncated` risk — check panel dates.
- **EOS**: panel ends 2025-05-26 (`source_truncated`); not treated as collapse.

## Migrations (see coin_id_map.csv)
- MATIC → POL (`polygon-ecosystem-token`)
- FTM → S (`sonic-3`)
- MKR → SKY (`sky`)
- Return splicing across migration is **not** auto-applied in v1 panel; A/C decide join rules.

## Missing prices
- CMC top-40 ever-members without Binance (or failed fetch) remain in `missing_coins.csv` (count≈{n_miss}).
- Notable non-Binance: LEO, BSV, CRO (often), etc. Need CoinGecko/Coin Metrics or accept missing.

## Outliers
- `|ret_1d|>=50%` flagged `outlier_check=unverified` except LUNA/FTT crash windows → `verified_real`.
- Full second-source verification still open for non-crash extremes (`reports/outlier_candidates_v1.csv`).

## On-chain coverage
- Re-check active addresses / tx counts via Coin Metrics Community API still pending (brief 8.5).
"""
    path = DOCS / "known_issues_log.md"
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")

    dd = DOCS / "data_dictionary_v0.md"
    note = (
        "\n\n## Quote currency (v0.1+)\n"
        "- Binance pairs are `*USDT`. For this project we treat **USDT as USD** "
        "(1:1) for prices and `volume_usd`.\n"
    )
    if dd.exists():
        cur = dd.read_text(encoding="utf-8")
        if "USDT as USD" not in cur:
            dd.write_text(cur + note, encoding="utf-8")
            print(f"appended USDT note to {dd}")
        else:
            print("dictionary already has USDT note")
    else:
        dd.write_text("# Data dictionary\n" + note, encoding="utf-8")
        print(f"created {dd}")
    print("done")


def pd_read(p: Path):
    import pandas as pd

    return pd.read_csv(p)


if __name__ == "__main__":
    main()
