"""
Step 3 (Week 1-2 deliverable): Coverage gate template.

Fill the CSV after you check each on-chain metric's availability.
Then classify: full_universe_factor vs btc_timing_only vs drop.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PROCESSED = ROOT / "data" / "processed"


CANDIDATE_METRICS = [
    ("active_addresses", "How many addresses were active"),
    ("tx_count", "On-chain transaction count"),
    ("nvt", "Network value to transactions"),
    ("exchange_net_flow", "Coins moving onto/off exchanges"),
    ("realised_cap", "Cost-basis style network valuation"),
    ("mvrv", "Market value / realised value"),
    ("sopr", "Spent output profit ratio"),
    ("sth_lth_cost_basis", "Short/long-term holder cost basis"),
]


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    coins = ["bitcoin", "ethereum", "solana", "OTHER_COIN_1", "OTHER_COIN_2"]
    panel_path = PROCESSED / "pilot_panel.csv"
    if panel_path.exists():
        coins = sorted(pd.read_csv(panel_path)["coin_id"].dropna().unique().tolist())[:12]

    rows = []
    for metric, note in CANDIDATE_METRICS:
        for c in coins:
            rows.append(
                {
                    "metric": metric,
                    "coin_id": c,
                    "available": "",  # fill yes/no
                    "start_date": "",
                    "end_date": "",
                    "source": "",
                    "free_tier_ok": "",
                    "notes": note,
                }
            )
    matrix = pd.DataFrame(rows)
    matrix_path = DOCS / "coverage_matrix_TEMPLATE.csv"
    matrix.to_csv(matrix_path, index=False)

    report = DOCS / "coverage_report_WEEK2_TEMPLATE.md"
    report.write_text(
        """# On-chain Coverage Gate Report (Week 2)

## Decision rule
- Available for (almost) the whole 30–50 coin universe → **cross-sectional factor**
- Available mainly for BTC/ETH → **Bitcoin-level timing / regime signal**
- Cannot obtain on free/affordable tier → **drop for FYP I**

## Summary table (fill after matrix)

| Metric | Classification | Reason |
|--------|----------------|--------|
| active_addresses |  |  |
| nvt |  |  |
| exchange_net_flow |  |  |
| mvrv |  |  |
| sopr |  |  |

## Locked factor list for the group
_Price-based (always on if market data exists): size, momentum, volatility, liquidity_

_Crypto-native cross-sectional:_ …

_BTC timing only:_ …

_Dropped:_ …

## Sources and free-tier limits
…

## Sign-off
- Member B (data): __________ date ____
- Member A (factors): __________ date ____
- Member C (portfolio): __________ date ____
""",
        encoding="utf-8",
    )
    print(f"Wrote {matrix_path}")
    print(f"Wrote {report}")
    print("Fill the CSV, then complete the markdown — that is your Week 2 handoff.")


if __name__ == "__main__":
    main()
