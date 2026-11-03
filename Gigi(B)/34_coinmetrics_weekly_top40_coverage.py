"""Week-by-week Coin Metrics AdrActCnt+TxCnt coverage among weekly top-40 members.

Hayden gate: pct_both >= 80%. If typically below, keep btc_eth_timing.

Reuses MANUAL map from script 31. Needs network for Coin Metrics Community API.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

UNIVERSE = PROCESSED / "universe_membership_cmc_v0_1.csv"
OUT = REPORTS / "coinmetrics_weekly_top40_coverage_v1.csv"
NOTE = DOCS / "coinmetrics_weekly_top40_note_v1.md"
CM_BASE = "https://community-api.coinmetrics.io/v4"

# Copied from scripts/31_coinmetrics_coverage_gate.py
MANUAL = {
    "bitcoin": "btc",
    "ethereum": "eth",
    "ripple": "xrp",
    "bitcoin-cash": "bch",
    "bitcoin-sv": "bsv",
    "litecoin": "ltc",
    "cardano": "ada",
    "dogecoin": "doge",
    "polkadot-new": "dot",
    "polkadot": "dot",
    "chainlink": "link",
    "stellar": "xlm",
    "tron": "trx",
    "eos": "eos",
    "ethereum-classic": "etc",
    "tezos": "xtz",
    "neo": "neo",
    "dash": "dash",
    "zcash": "zec",
    "decred": "dcr",
    "digibyte": "dgb",
    "verge": "xvg",
    "nem": "xem",
    "aave": "aave",
    "uniswap": "uni",
    "maker": "mkr",
    "compound-governance-token": "comp",
    "curve-dao-token": "crv",
    "sushi": "sushi",
    "yearn-finance": "yfi",
    "ftx-token": "ftt",
    "binancecoin": "bnb",
    "avalanche-2": "avaxp",
    "internet-computer": "icp",
    "algorand": "algo",
    "the-graph": None,
    "shiba-inu": "shib_eth",
    "polygon": "matic_eth",
    "matic-network": "matic_eth",
    "leo-token": "leo_eth",
    "crypto-com-chain": "cro",
    "okb": None,
    "wrapped-bitcoin": "wbtc",
    "true-usd": "tusd",
    "terra-luna": "luna",
    "solana": "sol",
    "cosmos": "atom",
    "near": "near",
    "filecoin": "fil",
    "hedera-hashgraph": "hbar",
    "vechain": "vet",
    "monero": "xmr",
    "fantom": "ftm",
}


def _get(url: str, retries: int = 3) -> dict:
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def load_cm_community_network_assets() -> dict[str, set[str]]:
    d = _get(f"{CM_BASE}/catalog/assets")
    out: dict[str, set[str]] = {}
    for a in d.get("data", []):
        asset = a.get("asset")
        for m in a.get("metrics") or []:
            mid = m.get("metric")
            if mid not in ("AdrActCnt", "TxCnt"):
                continue
            for f in m.get("frequencies") or []:
                if f.get("frequency") == "1d" and f.get("community") is True:
                    out.setdefault(asset, set()).add(mid)
                    break
    return out


def map_to_cm(coin_id: str, symbol: str | None, cm_assets: set[str]) -> str | None:
    cid = (coin_id or "").strip().lower()
    sym = (symbol or "").strip().lower()
    if cid in MANUAL:
        return MANUAL[cid]
    for cand in (cid, sym, cid.replace("-", "_"), sym.replace("-", "_") if sym else ""):
        if cand and cand in cm_assets:
            return cand
    for cand in (f"{sym}_eth", f"{cid}_eth"):
        if cand in cm_assets:
            return cand
    return None


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    if not UNIVERSE.exists():
        raise SystemExit(f"missing {UNIVERSE}")

    print("loading Coin Metrics catalog (community AdrActCnt/TxCnt)...")
    cm_map = load_cm_community_network_assets()
    cm_assets = set(cm_map)
    print(f"community network assets: {len(cm_assets)}")

    uni = pd.read_csv(UNIVERSE, low_memory=False)
    uni["rebalance_date"] = pd.to_datetime(uni["rebalance_date"])
    members = uni[uni["is_member"].astype(bool)].copy()
    print(f"member-rows={len(members)} weeks={members['rebalance_date'].nunique()}")

    # Cache coin_id -> covers_both
    coins = members.drop_duplicates("coin_id")
    cache: dict[str, bool] = {}
    mapped = 0
    for _, r in coins.iterrows():
        cid = str(r["coin_id"])
        sym = str(r["symbol"]) if "symbol" in r and pd.notna(r["symbol"]) else None
        cm = map_to_cm(cid, sym, cm_assets)
        if cm:
            mapped += 1
        mets = cm_map.get(cm, set()) if cm else set()
        cache[cid] = ("AdrActCnt" in mets and "TxCnt" in mets)
    print(f"unique member coins={len(cache)} mapped_nonzero={mapped} both_true={sum(cache.values())}")

    rows = []
    for rb, g in members.groupby("rebalance_date", sort=True):
        ids = g["coin_id"].astype(str).tolist()
        n = len(ids)
        n_both = sum(1 for c in ids if cache.get(c, False))
        pct = 100.0 * n_both / n if n else 0.0
        rows.append(
            {
                "rebalance_date": pd.Timestamp(rb).date().isoformat(),
                "n_members": n,
                "n_with_both": n_both,
                "pct_both": round(pct, 4),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    s = out["pct_both"]
    n80 = int((s >= 80.0).sum())
    print(f"wrote {OUT} weeks={len(out)}")
    print(
        f"pct_both min={s.min():.2f} median={s.median():.2f} "
        f"mean={s.mean():.2f} max={s.max():.2f}"
    )
    print(f"weeks with pct_both >= 80%: {n80}/{len(out)}")

    decision = "btc_eth_timing"
    if s.median() >= 80.0 and s.mean() >= 80.0:
        decision = "full_universe_ok_tentative"
    note = (
        f"# Coin Metrics weekly top-40 coverage (v1.1)\n\n"
        f"Date: generated by script 34\n\n"
        f"## Method\n"
        f"- Universe: `is_member` rows in `universe_membership_cmc_v0_1.csv` (weekly top-40)\n"
        f"- Per `rebalance_date`: share of that week's 40 members with community "
        f"`AdrActCnt` **and** `TxCnt` (catalog), mapped via script-31 MANUAL map\n"
        f"- Hayden gate rule: **80%**\n\n"
        f"## Result\n"
        f"- Weeks: **{len(out)}**\n"
        f"- pct_both min/median/mean/max: "
        f"**{s.min():.2f} / {s.median():.2f} / {s.mean():.2f} / {s.max():.2f}**\n"
        f"- Weeks >= 80%: **{n80}/{len(out)}**\n\n"
        f"## Gate note\n"
        f"Recommended decision: **`{decision}`** "
        f"(if typically below 80%, keep `btc_eth_timing`).\n\n"
        f"Detail: `{OUT.as_posix()}`\n"
    )
    NOTE.write_text(note, encoding="utf-8")
    print(f"wrote {NOTE}")
    print(f"gate_recommendation={decision}")
    print("done")


if __name__ == "__main__":
    main()
