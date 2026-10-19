"""Day 9/26: build coverage matrix from existing panels + on-chain placeholders.

Fills what Member B already has (OHLCV, market_cap) from CSVs.
Leaves on-chain metrics for manual classification (report coverage gate).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"

ONCHAIN_METRICS = [
    ("active_addresses", "Active addresses / users on chain"),
    ("tx_count", "On-chain transaction count"),
    ("nvt", "Network value to transactions"),
    ("exchange_net_flow", "Exchange inflow minus outflow"),
    ("realised_cap", "Realised capitalisation"),
    ("mvrv", "Market value / realised value"),
    ("sopr", "Spent output profit ratio"),
    ("sth_lth_cost_basis", "Short/long-term holder cost basis"),
]


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    panel_path = PROCESSED / "price_mcap_panel_v0.csv"
    if not panel_path.exists():
        raise SystemExit("Missing price_mcap_panel_v0.csv")

    df = pd.read_csv(panel_path)
    df["date"] = pd.to_datetime(df["date"])
    coins = sorted(df["coin_id"].dropna().unique())

    rows = []
    for cid in coins:
        sub = df[df["coin_id"] == cid]
        # OHLCV
        rows.append(
            {
                "metric": "ohlcv_daily",
                "coin_id": cid,
                "available": "yes",
                "start_date": sub["date"].min().date().isoformat(),
                "end_date": sub["date"].max().date().isoformat(),
                "coverage_frac": 1.0,
                "source": "Binance public klines",
                "classification": "price_input",
                "notes": "Full for listing period on Binance",
            }
        )
        # market_cap
        m = sub["market_cap"].notna().mean()
        api = (sub.get("market_cap_source") == "api").mean() if "market_cap_source" in sub else 0
        rows.append(
            {
                "metric": "market_cap",
                "coin_id": cid,
                "available": "yes" if m > 0 else "no",
                "start_date": sub["date"].min().date().isoformat(),
                "end_date": sub["date"].max().date().isoformat(),
                "coverage_frac": float(m),
                "source": "CoinGecko Demo 365d api + approx fill",
                "classification": "price_input",
                "notes": f"api_share_approx={api:.2f}; older dates often approx",
            }
        )
        for metric, note in ONCHAIN_METRICS:
            # Manual gate fields — default blank for student to fill after checking free sources
            suggest = ""
            if cid in {"bitcoin", "ethereum"} and metric in {
                "active_addresses",
                "tx_count",
                "nvt",
                "mvrv",
                "sopr",
                "sth_lth_cost_basis",
                "realised_cap",
                "exchange_net_flow",
            }:
                suggest = "often available on free dashboards for majors; confirm yourself"
            rows.append(
                {
                    "metric": metric,
                    "coin_id": cid,
                    "available": "",  # fill yes/no/partial
                    "start_date": "",
                    "end_date": "",
                    "coverage_frac": "",
                    "source": "",
                    "classification": "",  # full_universe_factor | btc_timing_only | drop
                    "notes": note + (f"; {suggest}" if suggest else ""),
                }
            )

    out = DOCS / "coverage_matrix_v1.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    # Aggregate template for gate decision
    gate = []
    for metric, note in ONCHAIN_METRICS:
        gate.append(
            {
                "metric": metric,
                "economic_rationale": note,
                "covers_full_candidate_universe": "",  # yes/no
                "decision": "",  # full_universe_factor | btc_eth_timing | drop
                "free_source_checked": "",
                "comment": "",
            }
        )
    gate_path = DOCS / "coverage_gate_decisions_TEMPLATE.csv"
    pd.DataFrame(gate).to_csv(gate_path, index=False)

    report = DOCS / "coverage_report_DRAFT_week2.md"
    report.write_text(
        f"""# Coverage Gate Report (Draft) — Member B

**Date:** 2026-09-22  
**Universe candidates in panel:** {len(coins)} coins  
**Sample window targeted:** 2018-01-01 to 2026-09-30  

## 1. Price / market data (already retrieved)

| Metric | Status | Notes |
|--------|--------|-------|
| Daily OHLCV | Available for candidates on Binance listing history | See `coverage_matrix_v1.csv` metric `ohlcv_daily` |
| Market cap | Filled 100% in panel | Last ~365d: CoinGecko API; earlier: approx = close × latest circulating supply |
| Circulating supply | Latest snapshot used for approx mcap | Documented in panel / summary for A/C |

## 2. On-chain coverage gate (to complete today)

Rule from project report:
- Available for (almost) whole candidate universe → **cross-sectional factor**
- Mainly BTC/ETH → **timing / regime signal**
- Cannot obtain affordably → **drop for FYP I**

Fill `coverage_gate_decisions_TEMPLATE.csv` and the on-chain rows in `coverage_matrix_v1.csv`.

Candidate on-chain metrics from report: active addresses, tx count, NVT, exchange net flows, realised cap, MVRV, SOPR, STH/LTH cost basis.

## 3. Preliminary recommendations (edit after you check sources)

| Metric | Tentative decision | Why |
|--------|--------------------|-----|
| active_addresses |  |  |
| nvt |  |  |
| exchange_net_flow |  |  |
| mvrv / sopr / STH-LTH | often BTC-heavy | likely timing, not full cross-section |

## 4. Implications for Member A / C

- A: which factors can be tested cross-sectionally vs BTC-only
- C: earliest reliable backtest date; mcap approx caveat for early sample

## 5. Sign-off

- Member B: __________ date ____
- Member A: __________ date ____
- Member C: __________ date ____
""",
        encoding="utf-8",
    )

    print(f"wrote {out}")
    print(f"wrote {gate_path}")
    print(f"wrote {report}")
    print(f"coins={len(coins)}")
    print("Next: manually fill on-chain available/classification columns, then decisions CSV.")


if __name__ == "__main__":
    main()
