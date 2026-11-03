"""Problem 6: second-source verify |ret_1d|>=50% vs CMC weekly snapshot prices.

Also adds days_since_listing (first price date in panel per coin_id).
Writes reports/outlier_verify_v1.csv and updates panel outlier_check + days_since_listing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

PANEL = PROCESSED / "price_mcap_panel_v1.csv"
CMC = PROCESSED / "cmc_snapshots_clean_v0_1.csv"
OUT_REPORT = REPORTS / "outlier_verify_v1.csv"
OUT_CANDS = REPORTS / "outlier_candidates_v1.csv"

# Relative error thresholds vs nearest CMC Sunday price
VERIFY_OK = 0.35  # within 35% → verified_real
ERROR_BAD = 1.0   # >100% away with CMC present → data_error


def _slug_for_row(r: pd.Series) -> str | None:
    for col in ("cmc_slug", "coin_id"):
        v = r.get(col)
        if isinstance(v, str) and v.strip() and v.lower() not in ("nan", "none"):
            return v.strip().lower()
    return None


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PANEL, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["coin_id", "date"]).reset_index(drop=True)

    # days_since_listing
    first = df.groupby("coin_id")["date"].transform("min")
    df["days_since_listing"] = (df["date"] - first).dt.days

    df["ret_1d"] = df.groupby("coin_id")["close"].pct_change()
    ext = df["ret_1d"].abs() >= 0.5
    df["outlier_check"] = ""
    df.loc[ext.fillna(False), "outlier_check"] = "unverified"

    # Known crash windows (already confirmed real events)
    luna = (
        (df["coin_id"] == "terra-luna")
        & (df["date"] >= "2022-05-09")
        & (df["date"] <= "2022-05-13")
        & ext.fillna(False)
    )
    ftt = (
        (df["coin_id"] == "ftx-token")
        & (df["date"] >= "2022-11-07")
        & (df["date"] <= "2022-11-14")
        & ext.fillna(False)
    )
    df.loc[luna | ftt, "outlier_check"] = "verified_real"

    cmc = pd.read_csv(CMC, low_memory=False)
    cmc["snapshot_date"] = pd.to_datetime(cmc["snapshot_date"])
    cmc["slug"] = cmc["slug"].astype(str).str.lower()
    cmc = cmc.sort_values(["slug", "snapshot_date"])

    # Build per-slug lookup arrays for nearest Sunday
    slug_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for slug, g in cmc.groupby("slug", sort=False):
        slug_map[slug] = (
            g["snapshot_date"].to_numpy(dtype="datetime64[ns]"),
            g["price"].to_numpy(dtype=float),
        )

    cands = df[ext.fillna(False)].copy()
    rows = []
    for idx, r in cands.iterrows():
        if r["outlier_check"] == "verified_real":
            rows.append(
                {
                    "date": r["date"],
                    "coin_id": r["coin_id"],
                    "symbol": r.get("symbol"),
                    "close": r["close"],
                    "ret_1d": r["ret_1d"],
                    "days_since_listing": r["days_since_listing"],
                    "cmc_slug": _slug_for_row(r),
                    "cmc_price_near": np.nan,
                    "cmc_snapshot_date": pd.NaT,
                    "rel_err": np.nan,
                    "outlier_check": "verified_real",
                    "verify_note": "crash_window_LUNA_or_FTT",
                }
            )
            continue

        slug = _slug_for_row(r)
        note = ""
        cmc_price = np.nan
        cmc_dt = pd.NaT
        rel = np.nan
        label = "unverified"

        if slug and slug in slug_map:
            dates, prices = slug_map[slug]
            t = np.datetime64(pd.Timestamp(r["date"]).to_datetime64())
            # nearest snapshot (prefer on/before within 7d, else absolute nearest)
            pos = dates.searchsorted(t, side="right") - 1
            cand_idx = []
            if 0 <= pos < len(dates):
                cand_idx.append(pos)
            if pos + 1 < len(dates):
                cand_idx.append(pos + 1)
            if not cand_idx and len(dates):
                cand_idx = [int(np.argmin(np.abs(dates.astype("int64") - t.astype("int64"))))]
            best = None
            best_delta = None
            for i in cand_idx:
                delta = abs((dates[i] - t) / np.timedelta64(1, "D"))
                if best is None or delta < best_delta:
                    best, best_delta = i, delta
            if best is not None and best_delta is not None and best_delta <= 10:
                cmc_price = float(prices[best])
                cmc_dt = pd.Timestamp(dates[best])
                if cmc_price > 0 and pd.notna(r["close"]) and float(r["close"]) > 0:
                    rel = abs(float(r["close"]) - cmc_price) / cmc_price
                    if rel <= VERIFY_OK:
                        label = "verified_real"
                        note = f"cmc_near_ok delta_days={best_delta:.0f}"
                    elif rel >= ERROR_BAD:
                        label = "data_error"
                        note = f"cmc_near_mismatch delta_days={best_delta:.0f}"
                    else:
                        label = "unverified"
                        note = f"cmc_near_borderline delta_days={best_delta:.0f}"
                else:
                    note = "bad_price"
            else:
                note = "no_cmc_within_10d"
        else:
            note = "slug_not_in_cmc"

        # Listing-day / first-week mania still real if CMC agrees, else keep label
        df.at[idx, "outlier_check"] = label
        rows.append(
            {
                "date": r["date"],
                "coin_id": r["coin_id"],
                "symbol": r.get("symbol"),
                "close": r["close"],
                "ret_1d": r["ret_1d"],
                "days_since_listing": r["days_since_listing"],
                "cmc_slug": slug,
                "cmc_price_near": cmc_price,
                "cmc_snapshot_date": cmc_dt,
                "rel_err": rel,
                "outlier_check": label,
                "verify_note": note,
            }
        )

    ver = pd.DataFrame(rows)
    ver["abs_ret"] = ver["ret_1d"].abs()
    ver = ver.sort_values("abs_ret", ascending=False)
    ver.to_csv(OUT_REPORT, index=False)
    ver.to_csv(OUT_CANDS, index=False)

    df.to_csv(PANEL, index=False)

    vc = ver["outlier_check"].value_counts()
    print(f"updated {PANEL}")
    print(f"wrote {OUT_REPORT} and {OUT_CANDS} rows={len(ver)}")
    print("outlier_check counts (extreme rows):")
    print(vc.to_string())
    print("top10 after verify:")
    cols = [
        "date",
        "coin_id",
        "ret_1d",
        "close",
        "cmc_price_near",
        "rel_err",
        "outlier_check",
        "verify_note",
    ]
    print(ver.head(10)[cols].to_string(index=False))
    print("done")


if __name__ == "__main__":
    main()
