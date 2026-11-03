"""Zip Drive/GitHub handoff for Hayden (Member A)."""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELIVER = ROOT / "deliver"
OUT = DELIVER / "MemberB_v1_1_for_Hayden.zip"

FILES = [
    "data/processed/price_mcap_panel_v1.csv",
    "data/processed/price_mcap_panel_v1_1.csv",
    "data/processed/universe_membership_cmc_v0_1.csv",
    "data/processed/coin_id_map.csv",
    "data/processed/missing_coins.csv",
    "data/processed/universe_weekly_coverage_v1.csv",
    "data/processed/cmc_snapshots_clean_v0_1.csv",
    "reports/outlier_verify_v1.csv",
    "reports/outlier_candidates_v1.csv",
    "reports/weekly_ffill_coins_dates_v1.csv",
    "reports/weekly_ffill_summary_by_coin_v1.csv",
    "reports/data_error_dropped_v1.csv",
    "reports/data_error_fixed_v1.csv",
    "reports/mcap_source_by_year_v1.csv",
    "reports/mcap_source_members_by_year_v1.csv",
    "reports/coinmetrics_weekly_top40_coverage_v1.csv",
    "reports/coinmetrics_coverage_v1.csv",
    "docs/data_dictionary_v1.md",
    "docs/data_dictionary_v0.md",
    "docs/known_issues_log.md",
    "docs/handoff_v1_for_AC.md",
    "docs/REPLY_Hayden_checklist_v1_1.md",
    "docs/coinmetrics_coverage_note_v1.md",
    "docs/coinmetrics_weekly_top40_note_v1.md",
    "docs/RUN_ORDER_Hayden_v1_1.md",
    "docs/coverage_gate_decisions_FILLED.csv",
]


def main() -> None:
    DELIVER.mkdir(parents=True, exist_ok=True)
    scripts_dir = ROOT / "scripts"
    script_files = sorted(scripts_dir.glob("*.py"))
    # Prefer 13–36 plus earlier if present
    wanted_scripts = [
        p for p in script_files if p.name[:2].isdigit() and 13 <= int(p.name[:2]) <= 36
    ]

    missing = []
    added = []
    panel_missing = not (ROOT / "data/processed/price_mcap_panel_v1.csv").exists() and not (
        ROOT / "data/processed/price_mcap_panel_v1_1.csv"
    ).exists()

    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for rel in FILES:
            path = ROOT / rel
            if path.exists():
                z.write(path, arcname=rel)
                added.append(rel)
            else:
                missing.append(rel)
        for p in wanted_scripts:
            arc = f"scripts/{p.name}"
            z.write(p, arcname=arc)
            added.append(arc)
        # manifest
        man = ["# MemberB_v1_1_for_Hayden manifest", ""]
        man += ["## Added", *[f"- {a}" for a in added], "", "## Missing at pack time", *[f"- {m}" for m in missing]]
        if panel_missing:
            man += ["", "WARNING: price_mcap_panel_v1*.csv missing — re-run pack after script 33."]
        z.writestr("MANIFEST.txt", "\n".join(man) + "\n")

    print(f"wrote {OUT}")
    print(f"added={len(added)} missing_listed={len(missing)}")
    if panel_missing:
        print("WARNING: panel CSV not on disk — zip is incomplete for Hayden data share")
    for m in missing:
        print(f"  missing: {m}")
    print("done")


if __name__ == "__main__":
    main()
