"""Hayden v2 #6 — add panel_coin_id to universe_membership_cmc_v0_1.csv.

Universe coin_id is CMC slug; panel uses CoinGecko ids. Mapping table:
  data/processed/cmc_slug_to_panel_coin_id_v2.csv

Also drops uni-coin rows (CMC 4113, not Uniswap) from the universe file copy
written as universe_membership_cmc_v0_2.csv (original kept).

Optional --fetch-coingecko: fill UNMAPPED slugs via public /coins/list (needs net).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# Hayden-confirmed overrides (CMC slug → CoinGecko id)
OVERRIDES = {
    "xrp": "ripple",
    "bnb": "binancecoin",
    "binance-coin": "binancecoin",
    "theta": "theta-token",
    "theta-network": "theta-token",
    "elrond-egld": "elrond-erd-2",
    "multiversx-egld": "elrond-erd-2",
    "avalanche": "avalanche-2",
    "synthetix-network-token": "havven",  # CONFIRMED CoinGecko API id=havven, symbol=snx
}
DROP_SLUGS = {"uni-coin"}  # CMC 4113 UNI COIN ≠ Uniswap
# Defensive: also strip Hayden-v2 stables if script 14 not yet re-run on snapshots
DROP_STABLE_SLUGS = {"usd1", "global-dollar-usdg", "united-stables"}
DROP_STABLE_SYMBOLS = {"USD1", "USDG", "U"}


def load_map() -> pd.DataFrame:
    path = PROCESSED / "cmc_slug_to_panel_coin_id_v2.csv"
    if path.exists():
        m = pd.read_csv(path)
    else:
        m = pd.DataFrame(columns=["cmc_slug", "panel_coin_id", "notes"])
    return m


def enrich_from_coingecko(slugs: list[str], existing: dict[str, str]) -> dict[str, str]:
    print("fetch CoinGecko /coins/list ...")
    r = requests.get("https://api.coingecko.com/api/v3/coins/list", timeout=60)
    r.raise_for_status()
    coins = r.json()
    by_id = {c["id"]: c for c in coins}
    # direct id match
    out = dict(existing)
    for s in slugs:
        if s in out and out[s]:
            continue
        if s in by_id:
            out[s] = s
    time.sleep(1.2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-coingecko", action="store_true")
    args = ap.parse_args()

    uni_path = PROCESSED / "universe_membership_cmc_v0_1.csv"
    if not uni_path.exists():
        raise SystemExit(f"Missing {uni_path}")

    u = pd.read_csv(uni_path)
    m = load_map()
    mapping = {str(r.cmc_slug): str(r.panel_coin_id) for _, r in m.iterrows() if str(r.panel_coin_id).strip()}
    mapping.update(OVERRIDES)

    if args.fetch_coingecko:
        mapping = enrich_from_coingecko(sorted(u["coin_id"].astype(str).unique()), mapping)

    # identity: if slug already a known panel id from coin_id_map
    cmap = PROCESSED / "coin_id_map.csv"
    if cmap.exists():
        known = set(pd.read_csv(cmap)["coin_id"].astype(str))
        for s in u["coin_id"].astype(str).unique():
            if s not in mapping and s in known:
                mapping[s] = s

    u2 = u.copy()
    u2["panel_coin_id"] = u2["coin_id"].astype(str).map(lambda s: mapping.get(s, ""))
    # drop uni-coin + Hayden-v2 stables (until script 14 rebuild)
    before = len(u2)
    drop_ids = DROP_SLUGS | DROP_STABLE_SLUGS
    u2 = u2[~u2["coin_id"].astype(str).isin(drop_ids)].copy()
    u2 = u2[~u2["symbol"].astype(str).str.upper().isin(DROP_STABLE_SYMBOLS)].copy()
    print(f"dropped uni-coin/stables rows: {before - len(u2)}")

    out_uni = PROCESSED / "universe_membership_cmc_v0_2.csv"
    u2.to_csv(out_uni, index=False)
    # Sibling with panel_coin_id for Hayden join (original v0_1 slug file left intact until rebuild)
    sibling = PROCESSED / "universe_membership_cmc_v0_1_with_panel_id.csv"
    u2.to_csv(sibling, index=False)
    print(f"wrote {out_uni} rows={len(u2):,}")
    print(f"wrote {sibling}")

    # refresh map file
    rows = []
    for s in sorted(u["coin_id"].astype(str).unique()):
        pid = mapping.get(s, "")
        note = ""
        if s in DROP_SLUGS:
            note = "DROP — CMC 4113 UNI COIN, not Uniswap"
        elif s in OVERRIDES:
            note = "hayden_override"
            if s == "synthetix-network-token":
                note = "CONFIRMED CoinGecko id=havven (API check 2026-11-05)"
        elif not pid:
            note = "UNMAPPED"
        elif pid == s:
            note = "identity"
        rows.append({"cmc_slug": s, "panel_coin_id": pid, "notes": note})
    map_out = PROCESSED / "cmc_slug_to_panel_coin_id_v2.csv"
    pd.DataFrame(rows).to_csv(map_out, index=False)

    REPORTS.mkdir(parents=True, exist_ok=True)
    def _filled(v) -> bool:
        s = str(v).strip()
        return bool(s) and s.lower() != "nan"
    mapped_n = sum(1 for r in rows if _filled(r["panel_coin_id"]))
    unmapped = [r for r in rows if not _filled(r["panel_coin_id"]) and r["cmc_slug"] not in DROP_SLUGS]
    pd.DataFrame(unmapped).to_csv(REPORTS / "universe_panel_id_unmapped_v2.csv", index=False)
    print(f"map: mapped={mapped_n} / {len(rows)}; unmapped={len(unmapped)} -> reports/universe_panel_id_unmapped_v2.csv")
    print("SNX check: synthetix-network-token ->", mapping.get("synthetix-network-token"))
    print("done")


if __name__ == "__main__":
    main()
