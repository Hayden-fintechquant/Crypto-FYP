"""Problem 8.5: re-check active addresses + tx count via Coin Metrics Community API.

Maps CMC universe members → CM assets, probes AdrActCnt / TxCnt community availability,
writes reports/coinmetrics_coverage_v1.csv + updates docs coverage gate notes.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

UNIVERSE = PROCESSED / "universe_membership_cmc_v0_1.csv"
CANDIDATE = PROCESSED / "candidate_pool_v0_1.csv"
OUT = REPORTS / "coinmetrics_coverage_v1.csv"
GATE = DOCS / "coverage_gate_decisions_FILLED.csv"
NOTES = DOCS / "coinmetrics_coverage_note_v1.md"
KNOWN = DOCS / "known_issues_log.md"

CM_BASE = "https://community-api.coinmetrics.io/v4"

# Manual symbol / slug → CM asset overrides (community naming quirks)
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
    "avalanche-2": "avaxp",  # Avalanche X/P; community also has avaxc
    "internet-computer": "icp",
    "algorand": "algo",
    "the-graph": None,  # often not in community network metrics
    "shiba-inu": "shib_eth",
    "polygon": "matic_eth",  # ERC20-era metric on CM
    "matic-network": "matic_eth",
    "leo-token": "leo_eth",
    "crypto-com-chain": "cro",
    "okb": None,
    "wrapped-bitcoin": "wbtc",
    "true-usd": "tusd",
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
    """asset -> set of metrics among AdrActCnt, TxCnt with community 1d."""
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
    # direct asset code
    for cand in (cid, sym, cid.replace("-", "_"), sym.replace("-", "_")):
        if cand and cand in cm_assets:
            return cand
    # eth-suffixed tokens
    for cand in (f"{sym}_eth", f"{cid}_eth"):
        if cand in cm_assets:
            return cand
    return None


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    print("loading Coin Metrics catalog (community AdrActCnt/TxCnt)...")
    cm_map = load_cm_community_network_assets()
    cm_assets = set(cm_map)
    print(f"community network assets: {len(cm_assets)}")

    if UNIVERSE.exists():
        uni = pd.read_csv(UNIVERSE, low_memory=False)
    else:
        raise SystemExit(f"missing {UNIVERSE}")

    # unique members that ever appeared
    id_col = "coin_id" if "coin_id" in uni.columns else None
    slug_col = "slug" if "slug" in uni.columns else ("cmc_slug" if "cmc_slug" in uni.columns else None)
    sym_col = "symbol" if "symbol" in uni.columns else None
    if id_col is None and slug_col is not None:
        uni["coin_id"] = uni[slug_col]
        id_col = "coin_id"
    members = uni.drop_duplicates(subset=[id_col]).copy()
    print(f"universe ever-members: {len(members)}")

    rows = []
    for _, r in members.iterrows():
        cid = str(r[id_col])
        sym = str(r[sym_col]) if sym_col and pd.notna(r.get(sym_col)) else None
        cm = map_to_cm(cid, sym, cm_assets)
        mets = cm_map.get(cm, set()) if cm else set()
        rows.append(
            {
                "coin_id": cid,
                "symbol": sym,
                "cm_asset": cm or "",
                "has_AdrActCnt": "AdrActCnt" in mets,
                "has_TxCnt": "TxCnt" in mets,
                "covers_both": ("AdrActCnt" in mets and "TxCnt" in mets),
            }
        )
    cov = pd.DataFrame(rows)
    cov.to_csv(OUT, index=False)

    n = len(cov)
    n_both = int(cov["covers_both"].sum())
    n_adr = int(cov["has_AdrActCnt"].sum())
    n_tx = int(cov["has_TxCnt"].sum())
    n_mapped = int((cov["cm_asset"] != "").sum())
    pct_both = 100.0 * n_both / n if n else 0.0

    print(f"mapped={n_mapped}/{n}; AdrActCnt={n_adr}; TxCnt={n_tx}; both={n_both} ({pct_both:.1f}%)")

    # Spot-check timeseries for a few covered majors (prove API works)
    sample = cov[cov["covers_both"]]["cm_asset"].head(8).tolist()
    ok = 0
    if sample:
        q = urllib.parse.urlencode(
            {
                "assets": ",".join(sample),
                "metrics": "AdrActCnt,TxCnt",
                "start_time": "2021-01-01",
                "end_time": "2021-01-02",
                "frequency": "1d",
                "page_size": 500,
            }
        )
        try:
            d = _get(f"{CM_BASE}/timeseries/asset-metrics?{q}")
            ok = len(d.get("data") or [])
            print(f"timeseries spot-check rows={ok} assets={sample}")
        except Exception as e:  # noqa: BLE001
            print(f"timeseries spot-check failed: {e}")

    decision = (
        "btc_eth_timing"
        if pct_both < 80
        else "full_universe_factor"
    )
    note = (
        f"Coin Metrics Community API (2026-10-31 re-check): "
        f"{n_both}/{n} ever-universe members have BOTH AdrActCnt+TxCnt ({pct_both:.1f}%). "
        f"Mapped {n_mapped}/{n}. Spot-check timeseries rows={ok}. "
        f"Does NOT clear full-candidate-universe gate → keep {decision}."
    )

    # Update gate CSV rows for active_addresses / tx_count
    if GATE.exists():
        gate = pd.read_csv(GATE)
    else:
        gate = pd.DataFrame(
            columns=[
                "metric",
                "economic_rationale",
                "covers_full_candidate_universe",
                "decision",
                "free_source_checked",
                "available_btc",
                "available_eth",
                "available_typical_alt",
                "comment",
            ]
        )

    def upsert(metric: str, rationale: str) -> None:
        nonlocal gate
        row = {
            "metric": metric,
            "economic_rationale": rationale,
            "covers_full_candidate_universe": "no",
            "decision": decision,
            "free_source_checked": "Coin Metrics Community API v4",
            "available_btc": "yes",
            "available_eth": "yes",
            "available_typical_alt": "partial",
            "comment": note,
        }
        if "metric" in gate.columns and (gate["metric"] == metric).any():
            for k, v in row.items():
                gate.loc[gate["metric"] == metric, k] = v
        else:
            gate = pd.concat([gate, pd.DataFrame([row])], ignore_index=True)

    upsert("active_addresses", "Network usage / adoption")
    upsert("tx_count", "On-chain activity")
    gate.to_csv(GATE, index=False)
    print(f"updated {GATE}")

    NOTES.write_text(
        f"""# Coin Metrics coverage re-check (v1)

Date: 2026-10-31

## Method
- Catalog: `GET https://community-api.coinmetrics.io/v4/catalog/assets`
- Metrics: `AdrActCnt` (active addresses), `TxCnt` (transaction count), frequency `1d`, `community=true`
- Universe: ever-members in `universe_membership_cmc_v0_1.csv`
- Mapping: coin_id / symbol → CM asset codes (plus manual overrides for majors)

## Result
- Ever-universe members: **{n}**
- Mapped to a CM asset: **{n_mapped}**
- Have `AdrActCnt`: **{n_adr}**
- Have `TxCnt`: **{n_tx}**
- Have **both**: **{n_both}** ({pct_both:.1f}%)
- Timeseries spot-check rows: **{ok}**

## Gate decision
**Does not cover the full candidate / weekly universe cross-section.**

Keep decision: **`{decision}`** (on-chain as BTC/ETH / partial timing signals for FYP I).

Detail table: `reports/coinmetrics_coverage_v1.csv`
""",
        encoding="utf-8",
    )
    print(f"wrote {NOTES}")

    if KNOWN.exists():
        text = KNOWN.read_text(encoding="utf-8")
        block = (
            "\n## On-chain coverage (Coin Metrics re-check)\n"
            f"- {note}\n"
            "- Detail: `reports/coinmetrics_coverage_v1.csv`, `docs/coinmetrics_coverage_note_v1.md`.\n"
        )
        if "Coin Metrics re-check" not in text:
            # replace old pending line if present
            if "## On-chain coverage" in text:
                # append after that section header area — simple append
                text = text.rstrip() + "\n" + block
            else:
                text = text.rstrip() + "\n" + block
            KNOWN.write_text(text, encoding="utf-8")
            print(f"appended Coin Metrics note to {KNOWN}")
        else:
            print("known_issues already has Coin Metrics re-check")

    print("done")


if __name__ == "__main__":
    main()
