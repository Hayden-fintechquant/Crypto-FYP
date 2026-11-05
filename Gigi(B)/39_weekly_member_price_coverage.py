"""Hayden v2 #7 — weekly coverage: share of top-40 members with daily price+volume at Sunday cut-off.

Does NOT invent percentages: requires live panel + universe on disk.
Writes reports/weekly_member_price_vol_coverage_v2.csv and a short note with [RUN] if panel missing.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
SAMPLE_END = pd.Timestamp("2026-09-29")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    uni_path = PROCESSED / "universe_membership_cmc_v0_2.csv"
    if not uni_path.exists():
        uni_path = PROCESSED / "universe_membership_cmc_v0_1.csv"
    panel_path = PROCESSED / "price_mcap_panel_v1.csv"
    if not panel_path.exists():
        note = DOCS / "weekly_member_coverage_note_v2.md"
        note.write_text(
            "# Weekly member price+volume coverage (v2)\n\n"
            "**Status:** panel not on this machine — [RUN] on JGigiK after scripts 14/37/38.\n\n"
            "Command:\n```powershell\n"
            ".\\.venv\\Scripts\\python.exe scripts\\39_weekly_member_price_coverage.py\n"
            "```\n"
            "Hayden reference (his local join, not ours): overall ~72.9%; ~85–91% in 2021–2024; ~75% in 2025–2026.\n"
            "Do not cite fabricated box numbers.\n"
        )
        raise SystemExit(f"Missing {panel_path}. Wrote placeholder note {note}")

    uni = pd.read_csv(uni_path)
    panel = pd.read_csv(panel_path, usecols=lambda c: c in {"date", "coin_id", "close", "volume", "volume_usd", "symbol"})
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel[panel["date"] <= SAMPLE_END]
    # join key
    id_col = "panel_coin_id" if "panel_coin_id" in uni.columns and uni["panel_coin_id"].astype(str).str.len().gt(0).any() else "coin_id"
    uni = uni.copy()
    uni["join_id"] = uni[id_col].astype(str).where(uni[id_col].astype(str).str.len() > 0, uni["coin_id"].astype(str))
    uni["snapshot_date"] = pd.to_datetime(uni["snapshot_date"]).dt.normalize()

    # Sunday cut-off = snapshot_date
    # member has price+volume if join_id has non-null close and (volume_usd or volume) that day
    p = panel.copy()
    p["has_px"] = pd.to_numeric(p["close"], errors="coerce").notna()
    vol = pd.to_numeric(p["volume_usd"] if "volume_usd" in p.columns else p.get("volume"), errors="coerce")
    if "volume_usd" not in p.columns and "volume" in p.columns:
        vol = pd.to_numeric(p["volume"], errors="coerce")
    elif "volume_usd" in p.columns:
        vol = pd.to_numeric(p["volume_usd"], errors="coerce")
        if "volume" in p.columns:
            vol = vol.fillna(pd.to_numeric(p["volume"], errors="coerce"))
    p["has_vol"] = vol.notna() & (vol >= 0)
    p["ok"] = p["has_px"] & p["has_vol"]
    ok = p.loc[p["ok"], ["date", "coin_id"]].drop_duplicates()
    ok_set = set(zip(ok["date"], ok["coin_id"].astype(str)))

    rows = []
    for snap, g in uni.groupby("snapshot_date"):
        members = g["join_id"].astype(str).tolist()
        n = len(members)
        n_ok = sum((snap, m) in ok_set for m in members)
        rows.append(
            {
                "snapshot_date": snap.date().isoformat(),
                "n_members": n,
                "n_with_price_and_volume": n_ok,
                "pct": round(100.0 * n_ok / n, 2) if n else 0.0,
            }
        )
    cov = pd.DataFrame(rows)
    out = REPORTS / "weekly_member_price_vol_coverage_v2.csv"
    cov.to_csv(out, index=False)
    overall = cov["n_with_price_and_volume"].sum() / cov["n_members"].sum() * 100 if cov["n_members"].sum() else 0
    print(f"weeks={len(cov)} overall_pct={overall:.2f}")
    print(f"pct min/median/mean/max = {cov['pct'].min():.1f} / {cov['pct'].median():.1f} / {cov['pct'].mean():.1f} / {cov['pct'].max():.1f}")
    print(f"-> {out}")

    note = DOCS / "weekly_member_coverage_note_v2.md"
    note.write_text(
        "# Weekly member price+volume coverage (v2)\n\n"
        f"- File: `{out.relative_to(ROOT)}`\n"
        f"- Overall member-weeks with price+volume at Sunday cut-off: **{overall:.2f}%**\n"
        f"- Weekly pct: min {cov['pct'].min():.1f} / median {cov['pct'].median():.1f} / "
        f"mean {cov['pct'].mean():.1f} / max {cov['pct'].max():.1f}\n"
        f"- Join key used: `{id_col}`\n"
        "- Hayden prior local estimate (~72.9% overall) was before v2 fills; re-compare after 37.\n"
    )
    print(f"-> {note}")
    print("done")


if __name__ == "__main__":
    main()
