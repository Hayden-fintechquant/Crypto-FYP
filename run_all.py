#!/usr/bin/env python3
"""run_all.py -- rerun the whole FYP I pipeline on a (new) data panel with ONE command.

Place this file in the repository root (the folder that contains `Hayden(A)/`, `Aron/`, `Gigi(B)/`) and run:

    python run_all.py --panel PATH/price_mcap_panel_v1.csv --universe "Gigi(B)/universe_membership_cmc_v0_2.csv"

Steps (stops at the first failure):
  1. prepare_data.py   B's panel + universe  ->  Hayden(A)/data/panel.csv, universe.csv, coverage reports
  2. review_B.py       data audit (must have 0 FAIL)
  3. unit tests        Hayden(A) and Aron/Aron (skipped with a warning if pytest is not installed; --skip-tests to skip)
  4. run_fyp1.py       OFFICIAL pre-registered factor run (needs a valid prereg.lock; the config must be unchanged)
  5. run_benchmarks.py six benchmark allocators + BTC on the same weeks
  6. review_C.py       independent recomputation of the benchmark returns / turnover / cost (must have 0 FAIL)
  7. fyp1_analysis.py  tables for the report (distribution, correlation, bootstrap, sub-periods, costs)
Everything is copied to results_final/ together with a run log and the SHA-256 of the input panel.
"""
import argparse, hashlib, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
A, C = ROOT / "Hayden(A)", ROOT / "Aron" / "Aron"
OUT = ROOT / "results_final"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(step, cmd, cwd, log, allow_fail=False):
    print(f"\n=== {step}\n$ {' '.join(map(str, cmd))}   (in {cwd})")
    t0 = time.time()
    p = subprocess.run([str(x) for x in cmd], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = p.stdout + p.stderr
    log.write(f"\n=== {step}\n$ {' '.join(map(str, cmd))}\n{out}\n")
    tail = "\n".join(out.strip().splitlines()[-14:])
    print(tail)
    print(f"--- {step}: exit {p.returncode} ({time.time() - t0:.0f}s)")
    if p.returncode != 0 and not allow_fail:
        sys.exit(f"STOP: step '{step}' failed. See {OUT / 'run_log.txt'}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="Gigi's FINAL price_mcap_panel csv")
    ap.add_argument("--universe", default=str(ROOT / "Gigi(B)" / "universe_membership_cmc_v0_2.csv"))
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    panel, universe = Path(a.panel).resolve(), Path(a.universe).resolve()
    for p in (panel, universe, A / "prereg.lock"):
        if not p.exists():
            sys.exit(f"missing: {p}")
    OUT.mkdir(exist_ok=True)
    py = sys.executable
    with open(OUT / "run_log.txt", "w", encoding="utf-8") as log:
        log.write(f"input panel: {panel}\nsha256: {sha256(panel)}\nuniverse: {universe}\nsha256: {sha256(universe)}\n")
        print(f"input panel sha256: {sha256(panel)}")
        (A / "data").mkdir(exist_ok=True)
        run("1 adapter", [py, A / "prepare_data.py", panel, universe, A / "data"], A, log)
        run("2 review_B (data audit)", [py, "review_B.py", "--panel", "data/panel.csv", "--universe", "data/universe.csv"], A, log)
        if not a.skip_tests:
            for name, d in (("Hayden(A)", A), ("Aron", C)):
                try:
                    import pytest  # noqa: F401
                    run(f"3 tests {name}", [py, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"], d, log)
                except ImportError:
                    print("pytest not installed -> tests skipped (pip install pytest)")
        run("4 OFFICIAL factor run", [py, "run_fyp1.py"], A, log)
        run("5 benchmarks", [py, "run_benchmarks.py", "--panel", A / "data" / "panel.csv", "--universe", A / "data" / "universe.csv", "--out", "results_bench"], C, log)
        run("6 review_C", [py, "review_C.py", "--weights", C / "results_bench" / "weights.csv", "--log", C / "results_bench" / "weekly_log.csv",
                           "--panel", "data/panel.csv", "--universe", "data/universe.csv", "--cost_bps", "20", "--cap", "0.10", "--offset", "1"], A, log)
        run("7 analysis", [py, A / "fyp1_analysis.py", "--panel", A / "data" / "panel.csv", "--universe", A / "data" / "universe.csv",
                           "--bench", C / "results_bench", "--out", OUT / "analysis"], A, log)
    for src, dst in ((A / "results", OUT / "factors"), (C / "results_bench", OUT / "benchmarks")):
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    for f in ("adapter_log.txt", "weekly_coverage_report.csv", "coverage_by_year.csv"):
        shutil.copy(A / "data" / f, OUT / f)
    print(f"\nDONE. Everything is in {OUT}\nNext: git add -A && git commit -m 'FYP I final-panel results' && git push")


if __name__ == "__main__":
    main()
