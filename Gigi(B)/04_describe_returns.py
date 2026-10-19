"""Step 4: Descriptive return stats on the pilot panel (fat tails motivation)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    src = PROCESSED / "pilot_panel.csv"
    if not src.exists():
        raise SystemExit("Run scripts/01_fetch_pilot_panel.py first.")
    df = pd.read_csv(src)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["coin_id", "date"])
    df["ret"] = df.groupby("coin_id")["close"].pct_change()

    stats = (
        df.groupby("coin_id")["ret"]
        .agg(["count", "mean", "std", "skew", lambda s: s.kurtosis()])
        .rename(columns={"<lambda_0>": "kurtosis"})
        .reset_index()
    )
    out = PROCESSED / "pilot_return_stats.csv"
    stats.to_csv(out, index=False)
    print(stats.to_string(index=False))
    print(f"\nSaved {out}")

    # stress windows (if sample long enough)
    for name, start, end in [
        ("terra_window", "2022-05-01", "2022-05-20"),
        ("ftx_window", "2022-11-01", "2022-11-20"),
    ]:
        w = df[(df["date"] >= start) & (df["date"] <= end)]
        if w.empty:
            print(f"{name}: no overlap in pilot sample (need longer history later)")
            continue
        wide = w.pivot_table(index="date", columns="coin_id", values="ret")
        corr = wide.corr()
        corr.to_csv(PROCESSED / f"pilot_corr_{name}.csv")
        print(f"{name}: mean pairwise corr ≈ {corr.stack().mean():.3f}")


if __name__ == "__main__":
    main()
