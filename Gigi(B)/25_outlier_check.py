"""Problem 6: flag extreme daily returns.

Always recomputes ret_1d. Marks |ret|>=50% as unverified; LUNA/FTT crash windows as verified_real.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def main() -> None:
    path = PROCESSED / "price_mcap_panel_v1.csv"
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["coin_id", "date"])
    df["ret_1d"] = df.groupby("coin_id")["close"].pct_change()
    df["outlier_check"] = ""
    ext = df["ret_1d"].abs() >= 0.5
    df.loc[ext.fillna(False), "outlier_check"] = "unverified"

    luna_crash = (
        (df["coin_id"] == "terra-luna")
        & (df["date"] >= "2022-05-09")
        & (df["date"] <= "2022-05-13")
        & ext.fillna(False)
    )
    ftt_crash = (
        (df["coin_id"] == "ftx-token")
        & (df["date"] >= "2022-11-07")
        & (df["date"] <= "2022-11-14")
        & ext.fillna(False)
    )
    df.loc[luna_crash, "outlier_check"] = "verified_real"
    df.loc[ftt_crash, "outlier_check"] = "verified_real"

    df.to_csv(path, index=False)
    REPORTS.mkdir(parents=True, exist_ok=True)
    cands = df[ext.fillna(False)].copy()
    cands["abs_ret"] = cands["ret_1d"].abs()
    top20 = cands.nlargest(20, "abs_ret")
    out = pd.concat([top20, cands]).drop_duplicates(["coin_id", "date"])
    out_path = REPORTS / "outlier_candidates_v1.csv"
    out.sort_values("abs_ret", ascending=False).to_csv(out_path, index=False)

    n_ver = int((df["outlier_check"] == "verified_real").sum())
    print(f"updated panel outlier_check; wrote {out_path}")
    print(f"extreme rows |ret|>=50%: {int(ext.sum())}; verified_real={n_ver}")
    print("verified_real sample:")
    print(
        df[df["outlier_check"] == "verified_real"][["date", "coin_id", "close", "ret_1d", "outlier_check"]]
        .head(12)
        .to_string(index=False)
    )
    print("done")


if __name__ == "__main__":
    main()
