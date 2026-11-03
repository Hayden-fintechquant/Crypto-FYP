"""Build v1 handoff notes + missing_coins.csv for Student A/C.

Compares CMC weekly universe (top-40 after stable/wrap exclude) against
price coverage in price_mcap_panel_v1.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"


def main() -> None:
    panel = pd.read_csv(PROCESSED / "price_mcap_panel_v1.csv", usecols=["date", "coin_id", "close"])
    panel["date"] = pd.to_datetime(panel["date"])
    mem = pd.read_csv(PROCESSED / "universe_membership_cmc_v0_1.csv")
    pool = pd.read_csv(PROCESSED / "candidate_pool_v0_1.csv")
    cmap = pd.read_csv(PROCESSED / "coin_id_map.csv")

    # Map CMC slug/symbol -> panel coin_id where we know it
    # panel uses coingecko ids + terra-luna/ftx-token
    sym_to_panel = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "BNB": "binancecoin",
        "XRP": "ripple",
        "ADA": "cardano",
        "SOL": "solana",
        "DOGE": "dogecoin",
        "TRX": "tron",
        "AVAX": "avalanche-2",
        "LINK": "chainlink",
        "DOT": "polkadot",
        "LTC": "litecoin",
        "BCH": "bitcoin-cash",
        "XLM": "stellar",
        "XMR": "monero",
        "ETC": "ethereum-classic",
        "ATOM": "cosmos",
        "NEAR": "near",
        "ALGO": "algorand",
        "VET": "vechain",
        "FIL": "filecoin",
        "ICP": "internet-computer",
        "HBAR": "hedera-hashgraph",
        "APT": "aptos",
        "ARB": "arbitrum",
        "OP": "optimism",
        "SUI": "sui",
        "UNI": "uniswap",
        "AAVE": "aave",
        "MKR": "maker",
        "CRV": "curve-dao-token",
        "SNX": "havven",
        "LDO": "lido-dao",
        "CAKE": "pancakeswap-token",
        "GRT": "the-graph",
        "SAND": "the-sandbox",
        "MANA": "decentraland",
        "AXS": "axie-infinity",
        "IMX": "immutable-x",
        "FLOW": "flow",
        "XTZ": "tezos",
        "EOS": "eos",
        "FTM": "fantom",
        "MATIC": "matic-network",
        "THETA": "theta-token",
        "EGLD": "elrond-erd-2",
        "INJ": "injective-protocol",
        "RUNE": "thorchain",
        "CHZ": "chiliz",
        "ZEC": "zcash",
        "LUNA": "terra-luna",
        "FTT": "ftx-token",
    }

    panel_ids = set(panel["coin_id"].astype(str))
    mem["panel_coin_id"] = mem["symbol"].astype(str).str.upper().map(sym_to_panel)
    # also match when panel coin_id == CMC slug (batch / weekly fills)
    mem.loc[mem["panel_coin_id"].isna(), "panel_coin_id"] = mem.loc[mem["panel_coin_id"].isna(), "coin_id"]
    mem["has_price"] = mem["panel_coin_id"].astype(str).isin(panel_ids)

    # missing: ever in CMC universe top-40 but no panel price
    ever = (
        mem.groupby(["symbol", "coin_id"], dropna=False)
        .agg(
            n_weeks=("rebalance_date", "nunique"),
            best_rank=("rank_by_mcap", "min"),
            first=("rebalance_date", "min"),
            last=("rebalance_date", "max"),
            panel_coin_id=("panel_coin_id", "first"),
            has_price=("has_price", "max"),
        )
        .reset_index()
        .rename(columns={"coin_id": "cmc_slug"})
    )
    missing = ever[~ever["has_price"]].copy()
    missing["reason"] = missing["panel_coin_id"].apply(
        lambda x: "not_in_panel_v1_yet" if pd.isna(x) else "mapped_but_no_rows"
    )
    missing_path = PROCESSED / "missing_coins.csv"
    missing.sort_values(["best_rank", "symbol"]).to_csv(missing_path, index=False)

    # weekly n_members with price
    wk = mem.copy()
    wk["has_price"] = wk["has_price"].fillna(False)
    weekly = (
        wk.groupby("rebalance_date")
        .agg(n_members=("symbol", "nunique"), n_with_price=("has_price", "sum"))
        .reset_index()
    )
    weekly_path = PROCESSED / "universe_weekly_coverage_v1.csv"
    weekly.to_csv(weekly_path, index=False)

    DOCS.mkdir(parents=True, exist_ok=True)
    md_path = DOCS / "handoff_v1_for_AC.md"
    src = panel["coin_id"].nunique()
    lines = [
        "# Member B handoff v1 (for Students A / C)",
        "",
        f"- Panel: `data/processed/price_mcap_panel_v1.csv` ({len(panel):,} date×coin rows after read cols; full file has more columns).",
        f"- Coins in panel: **{src}** (50 midterm survivors + terra-luna + ftx-token).",
        f"- CMC candidate pool: **{len(pool)}** (ever in weekly top-100 after stable/wrap exclude).",
        f"- CMC universe: top-40 weekly → `universe_membership_cmc_v0_1.csv` (mean turnover ~1.55 adds/week).",
        f"- Missing prices among CMC top-40 ever-members: **{len(missing)}** → `missing_coins.csv`.",
        f"- Weekly coverage: `universe_weekly_coverage_v1.csv` (n_members vs n_with_price).",
        f"- ID / end reasons: `coin_id_map.csv`.",
        "",
        "## Crash coins (problem 3)",
        "- LUNA (classic) in panel through **2022-05-13**; in CMC universe before collapse.",
        "- FTT in panel; in CMC universe around FTX window.",
        "",
        "## Market-cap sources (v0.1 panel core)",
        "- Prefer `cmc_snapshot` / `cmc_supply_x_close`; remaining `approx` / `missing_pre_cmc_top100` flagged.",
        "",
        "## Still open for full v1",
        "- Download OHLCV for high-priority missing CMC members (see missing_coins.csv).",
        "- XMR price refill after Binance delist (source_truncated).",
        "- Outlier verification (problem 6).",
        "- On-chain coverage re-check (problem 8.5).",
        "",
        "## Top missing (by best CMC rank)",
        "",
    ]
    top = missing.sort_values("best_rank").head(15)
    if top.empty:
        lines.append("(none)")
    else:
        lines.append("| symbol | cmc_slug | best_rank | weeks | reason |")
        lines.append("|---|---|---:|---:|---|")
        for _, r in top.iterrows():
            lines.append(
                f"| {r['symbol']} | {r['cmc_slug']} | {int(r['best_rank'])} | {int(r['n_weeks'])} | {r['reason']} |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {missing_path} ({len(missing)} missing)")
    print(f"wrote {weekly_path} ({len(weekly)} weeks)")
    print(f"wrote {md_path}")
    print("weekly n_with_price: min", int(weekly["n_with_price"].min()), "median", int(weekly["n_with_price"].median()), "max", int(weekly["n_with_price"].max()))
    print("top missing:")
    print(top[["symbol", "best_rank", "n_weeks", "reason"]].head(10).to_string(index=False))
    # LUNA/FTT in mem with price
    for sym in ("LUNA", "FTT"):
        sub = mem[mem["symbol"].astype(str).str.upper() == sym]
        print(f"{sym}: universe weeks={len(sub)} has_price={bool(sub['has_price'].any())}")
    print("done")


if __name__ == "__main__":
    main()
