"""Zip Hayden v2 share package (scripts + docs; panel if present)."""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELIVER = ROOT / "deliver"
OUT = DELIVER / "MemberB_v2_for_Hayden.zip"

INCLUDE = [
    "scripts/14_build_cmc_candidate_pool.py",
    "scripts/15_apply_cmc_mcap_to_panel.py",
    "scripts/18_add_volume_features.py",
    "scripts/19_flag_stale_and_zero_volume.py",
    "scripts/25_outlier_check.py",
    "scripts/33_hayden_v1_1_panel_patch.py",
    "scripts/34_coinmetrics_weekly_top40_coverage.py",
    "scripts/35_write_data_dictionary_v1.py",
    "scripts/36_pack_hayden_share.py",
    "scripts/37_fill_gaps_binance.py",
    "scripts/38_add_panel_coin_id_to_universe.py",
    "scripts/39_weekly_member_price_coverage.py",
    "scripts/40_pack_hayden_v2.py",
    "docs/REPLY_Hayden_v2.md",
    "docs/RUN_ORDER_Hayden_v2.md",
    "docs/known_issues_log.md",
    "docs/data_dictionary_v1.md",
    "docs/weekly_member_coverage_note_v2.md",
    "docs/gaps_priority4_5_note_v2.md",
    "docs/panel_share_note_v2.md",
    "docs/REPLY_Hayden_v2.docx",
    "docs/RUN_ORDER_Hayden_v2.docx",
    "docs/known_issues_log.docx",
    "docs/data_dictionary_v1.docx",
    "docs/weekly_member_coverage_note_v2.docx",
    "docs/gaps_priority4_5_note_v2.docx",
    "docs/panel_share_note_v2.docx",
    "data/processed/cmc_slug_to_panel_coin_id_v2.csv",
    "data/processed/universe_membership_cmc_v0_1.csv",
    "data/processed/coin_id_map.csv",
]
OPTIONAL = [
    "data/processed/price_mcap_panel_v1.csv",
    "data/processed/universe_membership_cmc_v0_2.csv",
    "data/processed/universe_membership_cmc_v0_1_with_panel_id.csv",
    "reports/binance_gap_fill_summary_v2.csv",
    "reports/weekly_member_price_vol_coverage_v2.csv",
    "reports/universe_panel_id_unmapped_v2.csv",
    "reports/flag_not_auto_exclude_v2.csv",
]


def main() -> None:
    DELIVER.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for rel in INCLUDE + OPTIONAL:
            p = ROOT / rel
            if p.exists():
                z.write(p, rel)
                print("+", rel, f"({p.stat().st_size:,} bytes)")
            else:
                print("skip missing", rel)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
