"""v0.1: overlay point-in-time CMC mcap onto existing CLEAN_v0 panel.

For each panel coin_id mapped to CMC slug(s) / symbol:
- On Sunday snapshot dates: use CMC market_cap (mcap_source=cmc_snapshot)
- Other days: last Sunday circulating_supply * that day's close
  (mcap_source=cmc_supply_x_close) — known then, not look-ahead

Keeps prior api/approx only where CMC has no coverage for that coin/date.
Writes price_mcap_panel_v0_1.csv + QA for the four zero-mcap coins.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# coin_id -> (preferred_symbol, slug_aliases)  — aliases cover CMC renames
CG_MAP: dict[str, tuple[str, list[str]]] = {
    "bitcoin": ("BTC", ["bitcoin"]),
    "ethereum": ("ETH", ["ethereum"]),
    "binancecoin": ("BNB", ["bnb", "binance-coin"]),
    "ripple": ("XRP", ["xrp"]),
    "cardano": ("ADA", ["cardano"]),
    "solana": ("SOL", ["solana"]),
    "dogecoin": ("DOGE", ["dogecoin"]),
    "tron": ("TRX", ["tron"]),
    "avalanche-2": ("AVAX", ["avalanche"]),
    "chainlink": ("LINK", ["chainlink"]),
    "polkadot": ("DOT", ["polkadot-new", "polkadot"]),
    "litecoin": ("LTC", ["litecoin"]),
    "bitcoin-cash": ("BCH", ["bitcoin-cash"]),
    "stellar": ("XLM", ["stellar"]),
    "monero": ("XMR", ["monero"]),
    "ethereum-classic": ("ETC", ["ethereum-classic"]),
    "cosmos": ("ATOM", ["cosmos"]),
    "near": ("NEAR", ["near-protocol", "near"]),
    "algorand": ("ALGO", ["algorand"]),
    "vechain": ("VET", ["vechain"]),
    "filecoin": ("FIL", ["filecoin"]),
    "internet-computer": ("ICP", ["internet-computer"]),
    "hedera-hashgraph": ("HBAR", ["hedera", "hedera-hashgraph"]),
    "aptos": ("APT", ["aptos"]),
    "arbitrum": ("ARB", ["arbitrum"]),
    "optimism": ("OP", ["optimism-ethereum", "optimism"]),
    "sui": ("SUI", ["sui"]),
    "uniswap": ("UNI", ["uniswap"]),
    "aave": ("AAVE", ["aave"]),
    "maker": ("MKR", ["maker"]),
    "curve-dao-token": ("CRV", ["curve-dao-token"]),
    "havven": ("SNX", ["synthetix", "synthetix-network-token"]),
    "lido-dao": ("LDO", ["lido-dao"]),
    "pancakeswap-token": ("CAKE", ["pancakeswap"]),
    "the-graph": ("GRT", ["the-graph"]),
    "the-sandbox": ("SAND", ["the-sandbox"]),
    "decentraland": ("MANA", ["decentraland"]),
    "axie-infinity": ("AXS", ["axie-infinity"]),
    "immutable-x": ("IMX", ["immutable-x"]),
    "flow": ("FLOW", ["flow"]),
    "tezos": ("XTZ", ["tezos"]),
    "eos": ("EOS", ["eos"]),
    "fantom": ("FTM", ["fantom"]),
    "matic-network": ("MATIC", ["polygon", "matic-network"]),
    "theta-token": ("THETA", ["theta-network", "theta"]),
    "elrond-erd-2": ("EGLD", ["multiversx-egld", "elrond-egld"]),
    "injective-protocol": ("INJ", ["injective"]),
    "thorchain": ("RUNE", ["thorchain"]),
    "chiliz": ("CHZ", ["chiliz"]),
    "zcash": ("ZEC", ["zcash"]),
}

FOUR = ["eos", "fantom", "matic-network", "maker"]


def cmc_series_for_coin(cmc: pd.DataFrame, symbol: str, slugs: list[str]) -> pd.DataFrame:
    """All Sunday rows for this project coin (union of slug aliases + symbol)."""
    slugs_l = {s.lower() for s in slugs}
    sym_u = symbol.upper()
    m = cmc["slug"].astype(str).str.lower().isin(slugs_l) | (
        cmc["symbol"].astype(str).str.upper() == sym_u
    )
    sub = cmc.loc[m].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values(["snapshot_date", "cmc_mcap"], ascending=[True, False])
    # one row per Sunday: prefer alias slug match over pure symbol collision
    sub["_slug_hit"] = sub["slug"].astype(str).str.lower().isin(slugs_l).astype(int)
    sub = sub.sort_values(["snapshot_date", "_slug_hit", "cmc_mcap"], ascending=[True, False, False])
    sub = sub.drop_duplicates("snapshot_date", keep="first")
    return sub.drop(columns=["_slug_hit"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(PROCESSED / "price_mcap_panel_CLEAN_v0.csv"))
    ap.add_argument("--cmc-clean", default=str(PROCESSED / "cmc_snapshots_clean_v0_1.csv"))
    ap.add_argument("--out", default=str(PROCESSED / "price_mcap_panel_v0_1.csv"))
    args = ap.parse_args()

    panel_path, cmc_path = Path(args.panel), Path(args.cmc_clean)
    if not panel_path.exists():
        raise SystemExit(f"Missing panel: {panel_path}")
    if not cmc_path.exists():
        raise SystemExit(f"Missing CMC clean: {cmc_path}. Run script 14 first.")

    print(f"loading panel {panel_path} ...")
    panel = pd.read_csv(panel_path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["coin_id"] = panel["coin_id"].astype(str)
    print(f"  rows={len(panel):,} coins={panel['coin_id'].nunique()}")

    print(f"loading CMC clean {cmc_path} ...")
    cmc = pd.read_csv(cmc_path)
    cmc["snapshot_date"] = pd.to_datetime(cmc["snapshot_date"]).dt.normalize()
    cmc = cmc.rename(
        columns={
            "market_cap": "cmc_mcap",
            "circulating_supply": "cmc_circ",
            "total_supply": "cmc_total",
        }
    )
    print(f"  CMC rows={len(cmc):,}")

    pieces = []
    for cid, g in panel.groupby("coin_id", sort=False):
        g = g.sort_values("date").copy()
        if cid not in CG_MAP:
            print(f"  WARNING unmapped coin_id={cid}")
            g["cmc_mcap"] = np.nan
            g["cmc_circ"] = np.nan
            g["cmc_total"] = np.nan
            g["cmc_slug"] = pd.NA
            g["_on_snapshot"] = False
            pieces.append(g)
            continue
        sym, slugs = CG_MAP[cid]
        sub = cmc_series_for_coin(cmc, sym, slugs)
        g["cmc_slug"] = slugs[0]
        if sub.empty:
            print(f"  no CMC rows for {cid} ({sym} / {slugs})")
            g["cmc_mcap"] = np.nan
            g["cmc_circ"] = np.nan
            g["cmc_total"] = np.nan
            g["_on_snapshot"] = False
            pieces.append(g)
            continue
        merged = pd.merge_asof(
            g,
            sub[["snapshot_date", "cmc_mcap", "cmc_circ", "cmc_total"]],
            left_on="date",
            right_on="snapshot_date",
            direction="backward",
        )
        merged["_on_snapshot"] = merged["date"] == merged["snapshot_date"]
        pieces.append(merged)

    out = pd.concat(pieces, ignore_index=True)

    close = pd.to_numeric(out["close"], errors="coerce")
    circ = pd.to_numeric(out["cmc_circ"], errors="coerce")
    snap_mcap = pd.to_numeric(out["cmc_mcap"], errors="coerce")
    mcap_from_supply = close * circ

    use_snap = out["_on_snapshot"].fillna(False) & snap_mcap.notna() & (snap_mcap > 0)
    use_ffill = (~use_snap) & circ.notna() & (circ > 0) & close.notna() & (close > 0)

    new_mcap = pd.to_numeric(out["market_cap"], errors="coerce").copy()
    new_src = out["market_cap_source"].astype(str).copy()
    new_mcap = new_mcap.where(~use_ffill, mcap_from_supply)
    new_src = new_src.where(~use_ffill, "cmc_supply_x_close")
    new_mcap = new_mcap.where(~use_snap, snap_mcap)
    new_src = new_src.where(~use_snap, "cmc_snapshot")

    out["market_cap_old"] = out["market_cap"]
    out["market_cap_source_old"] = out["market_cap_source"]
    out["market_cap"] = new_mcap
    out["market_cap_source"] = new_src
    out["circulating_supply"] = circ.where(circ.notna(), out.get("circulating_supply"))
    if "total_supply" not in out.columns:
        out["total_supply"] = np.nan
    tot = pd.to_numeric(out["cmc_total"], errors="coerce")
    out["total_supply"] = tot.where(tot.notna(), out["total_supply"])
    out["supply_source"] = np.where(circ.notna(), "cmc_snapshot_ffill", "")
    out["mcap_bad_flag"] = out["market_cap"].notna() & (out["market_cap"] <= 0)

    out = out.drop(columns=[c for c in ("snapshot_date", "_on_snapshot") if c in out.columns])

    out_path = Path(args.out)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(out):,} rows)")

    print("\n=== market_cap_source mix ===")
    print(out["market_cap_source"].value_counts(dropna=False).to_string())

    print("\n=== four problem coins ===")
    for cid in FOUR:
        s = out[out["coin_id"] == cid]
        bad = int(s["mcap_bad_flag"].fillna(False).sum())
        pos = int((pd.to_numeric(s["market_cap"], errors="coerce") > 0).sum())
        print(
            f"{cid}: rows={len(s)} mcap>0={pos} still_bad={bad} "
            f"src={s['market_cap_source'].value_counts().to_dict()}"
        )
        if pos:
            print(
                f"  mcap range ${s['market_cap'].min():,.0f} .. ${s['market_cap'].max():,.0f}"
            )

    cmc_used = int(out["market_cap_source"].isin(["cmc_snapshot", "cmc_supply_x_close"]).sum())
    print(f"\nrows now CMC-sourced: {cmc_used:,} / {len(out):,}")
    print("done")


if __name__ == "__main__":
    main()
