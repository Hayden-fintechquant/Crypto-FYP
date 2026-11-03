"""v0.1: coin_id_map.csv + end_reason (modification brief problems 8 & 9).

Columns: coin_id, symbol, successor_id, migration_date, last_trade_date, end_reason
end_reason in {still_trading, source_truncated, migrated, collapsed}

Known migrations (do not invent prices for successors here — map only):
  matic-network (MATIC) -> polygon-ecosystem-token (POL) ~2024-09-10
  fantom (FTM) -> sonic-3 (S) ~2025-01-13
  maker (MKR) -> sky (SKY) ~2025-09-15

Elrond/EGLD: panel coin_id elrond-erd-2 still has rows through sample end → still_trading.
Monero: last Binance day 2024-02-20 → source_truncated (coin still trades elsewhere).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

SYMBOL = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "binancecoin": "BNB",
    "ripple": "XRP",
    "cardano": "ADA",
    "solana": "SOL",
    "dogecoin": "DOGE",
    "tron": "TRX",
    "avalanche-2": "AVAX",
    "chainlink": "LINK",
    "polkadot": "DOT",
    "litecoin": "LTC",
    "bitcoin-cash": "BCH",
    "stellar": "XLM",
    "monero": "XMR",
    "ethereum-classic": "ETC",
    "cosmos": "ATOM",
    "near": "NEAR",
    "algorand": "ALGO",
    "vechain": "VET",
    "filecoin": "FIL",
    "internet-computer": "ICP",
    "hedera-hashgraph": "HBAR",
    "aptos": "APT",
    "arbitrum": "ARB",
    "optimism": "OP",
    "sui": "SUI",
    "uniswap": "UNI",
    "aave": "AAVE",
    "maker": "MKR",
    "curve-dao-token": "CRV",
    "havven": "SNX",
    "lido-dao": "LDO",
    "pancakeswap-token": "CAKE",
    "the-graph": "GRT",
    "the-sandbox": "SAND",
    "decentraland": "MANA",
    "axie-infinity": "AXS",
    "immutable-x": "IMX",
    "flow": "FLOW",
    "tezos": "XTZ",
    "eos": "EOS",
    "fantom": "FTM",
    "matic-network": "MATIC",
    "theta-token": "THETA",
    "elrond-erd-2": "EGLD",
    "injective-protocol": "INJ",
    "thorchain": "RUNE",
    "chiliz": "CHZ",
    "zcash": "ZEC",
}

# Manual overrides from brief + Binance listing history in this panel
OVERRIDES = {
    "matic-network": {
        "successor_id": "polygon-ecosystem-token",
        "successor_symbol": "POL",
        "migration_date": "2024-09-10",
        "end_reason": "migrated",
        "notes": "MATIC rebrand/migration to POL on Polygon",
    },
    "fantom": {
        "successor_id": "sonic-3",
        "successor_symbol": "S",
        "migration_date": "2025-01-13",
        "end_reason": "migrated",
        "notes": "Fantom Opera -> Sonic (S); panel ends at migration",
    },
    "maker": {
        "successor_id": "sky",
        "successor_symbol": "SKY",
        "migration_date": "2025-09-15",
        "end_reason": "migrated",
        "notes": "Maker (MKR) -> Sky (SKY); confirm exact swap day with A if needed",
    },
    "monero": {
        "successor_id": "",
        "successor_symbol": "",
        "migration_date": "",
        "end_reason": "source_truncated",
        "notes": "Binance delisted XMR; coin still trades elsewhere — refill price in v1",
    },
    "eos": {
        "successor_id": "",
        "successor_symbol": "",
        "migration_date": "",
        "end_reason": "source_truncated",
        "notes": "Panel stops 2025-05-26 (likely Binance/listing end); not a collapse — review for v1",
    },
}


def main() -> None:
    panel_path = PROCESSED / "price_mcap_panel_v0_1.csv"
    if not panel_path.exists():
        raise SystemExit(f"Missing {panel_path}")

    df = pd.read_csv(panel_path, usecols=["date", "coin_id"])
    df["date"] = pd.to_datetime(df["date"])
    sample_end = df["date"].max().date().isoformat()
    g = (
        df.groupby("coin_id")["date"]
        .agg(first_trade_date="min", last_trade_date="max")
        .reset_index()
    )
    g["first_trade_date"] = g["first_trade_date"].dt.date.astype(str)
    g["last_trade_date"] = g["last_trade_date"].dt.date.astype(str)

    rows = []
    for _, r in g.iterrows():
        cid = r["coin_id"]
        ov = OVERRIDES.get(cid, {})
        last = r["last_trade_date"]
        if ov:
            end_reason = ov["end_reason"]
            successor_id = ov.get("successor_id", "")
            successor_symbol = ov.get("successor_symbol", "")
            migration_date = ov.get("migration_date", "")
            notes = ov.get("notes", "")
        elif last >= sample_end:
            end_reason = "still_trading"
            successor_id = successor_symbol = migration_date = ""
            notes = ""
        else:
            end_reason = "source_truncated"
            successor_id = successor_symbol = migration_date = ""
            notes = "Panel ends before sample_end; classify with A (truncate vs collapse)"

        rows.append(
            {
                "coin_id": cid,
                "symbol": SYMBOL.get(cid, ""),
                "successor_id": successor_id,
                "successor_symbol": successor_symbol,
                "migration_date": migration_date,
                "first_trade_date": r["first_trade_date"],
                "last_trade_date": last,
                "end_reason": end_reason,
                "notes": notes,
            }
        )

    out = pd.DataFrame(rows).sort_values("coin_id")
    out_path = PROCESSED / "coin_id_map.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(out)} coins)")
    print("\n=== non-still_trading ===")
    print(
        out[out["end_reason"] != "still_trading"][
            ["coin_id", "symbol", "end_reason", "successor_id", "migration_date", "last_trade_date"]
        ].to_string(index=False)
    )
    print("\nend_reason counts:")
    print(out["end_reason"].value_counts().to_string())
    print("done")


if __name__ == "__main__":
    main()
