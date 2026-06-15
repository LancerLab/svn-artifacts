#!/usr/bin/env python3
"""scripts/collect_all_stats.py

Run all four per-system assertion-statistics collectors and merge the results
into a single unified CSV for cross-system comparison.

Systems
-------
  choreo  : scripts/choreo_assertion_stats.py
  mlir    : scripts/mlir_assertion_stats.py   (tensor+scf)
  memref  : scripts/memref_assertion_stats.py  (memref dialect)
  iree    : scripts/iree_assertion_stats.py    (instrumented mode)
  triton  : scripts/triton_assertion_stats.py

Unified CSV schema
------------------
  system, category, case_name,
  generated_total,   # total generation attempts (user + internal / evaluated)
  generated_user,    # user/programmer-written portion  (N/A for choreo/iree)
  generated_internal,# compiler-generated portion       (N/A for triton/iree)
  discharged,        # proved at compile time
  runtime,           # surviving as runtime checks
  resolution_rate,   # discharged / generated_total
  status             # "ok" | "error" | "N/A"

Outputs
-------
  benchmark/results/all_stats.csv
  benchmark/results/all_stats_summary.csv  (per-system aggregate)

Usage
-----
  python scripts/collect_all_stats.py [--skip choreo] [--skip mlir] ...
  python scripts/collect_all_stats.py --only iree
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR    = Path(__file__).parent
RESULTS_DIR    = WORKSPACE_ROOT / "benchmark/results"

ALL_SYSTEMS = ["choreo", "mlir", "memref", "iree", "triton"]

INDIVIDUAL_CSVS = {
    "choreo": RESULTS_DIR / "choreo_stats.csv",
    "mlir":   RESULTS_DIR / "mlir_stats.csv",
    "memref": RESULTS_DIR / "memref_stats.csv",
    "iree":   RESULTS_DIR / "iree_stats.csv",
    "triton": RESULTS_DIR / "triton_stats.csv",
}

RUNNER_SCRIPTS = {
    "choreo": SCRIPTS_DIR / "choreo_assertion_stats.py",
    "mlir":   SCRIPTS_DIR / "mlir_assertion_stats.py",
    "memref": SCRIPTS_DIR / "memref_assertion_stats.py",
    "iree":   SCRIPTS_DIR / "iree_assertion_stats.py",
    "triton": SCRIPTS_DIR / "triton_assertion_stats.py",
}

# Extra flags for iree instrumented mode
SYSTEM_EXTRA_ARGS = {
    "iree": ["--instrumented"],
}

UNIFIED_OUT = RESULTS_DIR / "all_stats.csv"
SUMMARY_OUT = RESULTS_DIR / "all_stats_summary.csv"


def run_collector(system: str, verbose: bool) -> bool:
    script = RUNNER_SCRIPTS[system]
    if not script.exists():
        print(f"  SKIP {system}: runner not found at {script}", file=sys.stderr)
        return False
    cmd = [sys.executable, str(script)]
    cmd += SYSTEM_EXTRA_ARGS.get(system, [])
    if verbose:
        cmd.append("--verbose")
    print(f"  Running {system} collector …")
    try:
        result = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT),
                                stdin=subprocess.DEVNULL, timeout=3600)
    except subprocess.TimeoutExpired:
        print(f"  WARN: {system} collector timed out after 3600s", file=sys.stderr)
        return False
    if result.returncode != 0:
        print(f"  WARN: {system} collector exited with code {result.returncode}",
              file=sys.stderr)
        return False
    return True


def _safe_int(v: str) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def norm_choreo(row: dict) -> dict:
    gen    = _safe_int(row.get("generated", "0")) or 0
    dis    = _safe_int(row.get("discharged", "0")) or 0
    rt     = _safe_int(row.get("runtime",    "0")) or 0
    status = row.get("status", "ok")
    rate   = row.get("resolution_rate", "0.0000")
    return dict(
        system="choreo",
        category=row["category"],
        case_name=row["case_name"],
        generated_total=gen,
        generated_user="N/A",
        generated_internal="N/A",
        discharged=dis,
        runtime=rt,
        resolution_rate=rate,
        status=status,
    )


def norm_mlir(row: dict, system: str = "mlir") -> dict:
    gen   = _safe_int(row.get("generated_total", "0")) or 0
    gusr  = _safe_int(row.get("generated_user",  "0")) or 0
    gint  = _safe_int(row.get("generated_internal", "0")) or 0
    dis   = _safe_int(row.get("discharged", "0")) or 0
    rt    = _safe_int(row.get("runtime",    "0")) or 0
    rate  = row.get("resolution_rate", "0.0000")
    status = "error" if row.get("discharged") == "error" else "ok"
    return dict(
        system=system,
        category=row["category"],
        case_name=row["case_name"],
        generated_total=gen,
        generated_user=gusr,
        generated_internal=gint,
        discharged=dis,
        runtime=rt,
        resolution_rate=rate,
        status=status,
    )


def norm_iree(row: dict) -> dict:
    gen  = row.get("generated", "N/A")
    surv = row.get("surviving", "N/A")
    dis  = row.get("discharged", "N/A")
    notes = row.get("notes", "")
    gen_i  = _safe_int(str(gen))
    surv_i = _safe_int(str(surv))
    dis_i  = _safe_int(str(dis))

    # IREE has no discharge (no fold pattern for hal.buffer_view.assert)
    # IREE has no discharge (no fold pattern for hal.buffer_view.assert)
    # generated = surviving, discharged = 0
    # When generated is "unknown", use surviving as the effective generated count
    effective_gen = gen_i if gen_i is not None else surv_i
    if effective_gen is not None:
        rate = "0.0000"
        status = "ok"
    else:
        rate = "N/A"
        status = notes if notes else "N/A"

    return dict(
        system="iree",
        category=row["category"],
        case_name=row["case_name"],
        generated_total=effective_gen if effective_gen is not None else "N/A",
        generated_user="N/A",
        generated_internal=effective_gen if effective_gen is not None else "N/A",
        discharged=dis_i if dis_i is not None else 0,
        runtime=surv_i if surv_i is not None else "N/A",
        resolution_rate=rate,
        status=status,
    )


def norm_triton(row: dict) -> dict:
    gen  = _safe_int(row.get("generated_total", "0")) or 0
    dis  = _safe_int(row.get("discharged", "0")) or 0
    rt   = _safe_int(row.get("runtime",    "0")) or 0
    rate = row.get("resolution_rate", "0.0000")
    return dict(
        system="triton",
        category=row["category"],
        case_name=row["case_name"],
        generated_total=gen,
        generated_user=gen,   # all are user-written
        generated_internal=0,
        discharged=dis,
        runtime=rt,
        resolution_rate=rate,
        status="ok",
    )


NORMALIZERS = {
    "choreo": norm_choreo,
    "mlir":   lambda r: norm_mlir(r, "mlir"),
    "memref": lambda r: norm_mlir(r, "memref"),
    "iree":   norm_iree,
    "triton": norm_triton,
}


def merge_csvs(systems: list[str]) -> list[dict]:
    rows: list[dict] = []
    for system in systems:
        csv_path = INDIVIDUAL_CSVS[system]
        if not csv_path.exists():
            print(f"  WARN: {csv_path} not found, skipping {system}",
                  file=sys.stderr)
            continue
        norm = NORMALIZERS[system]
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(norm(r))
    return rows


def compute_summary(rows: list[dict]) -> list[dict]:
    from collections import defaultdict
    agg: dict[str, dict] = defaultdict(lambda: dict(
        cases=0, ok=0, errors=0,
        generated_total=0, discharged=0, runtime=0,
    ))
    for r in rows:
        s = r["system"]
        agg[s]["cases"] += 1
        if str(r.get("status", "ok")).lower() == "ok":
            agg[s]["ok"] += 1
            gen = _safe_int(str(r["generated_total"]))
            dis = _safe_int(str(r["discharged"]))
            rt  = _safe_int(str(r["runtime"]))
            if gen is not None: agg[s]["generated_total"] += gen
            if dis is not None: agg[s]["discharged"]      += dis
            if rt  is not None: agg[s]["runtime"]         += rt
        else:
            agg[s]["errors"] += 1

    summary_rows = []
    for system, d in sorted(agg.items()):
        gen = d["generated_total"]
        dis = d["discharged"]
        rt  = d["runtime"]
        rate = (dis / gen) if gen > 0 else 1.0
        summary_rows.append(dict(
            system=system,
            cases=d["cases"],
            ok=d["ok"],
            errors=d["errors"],
            generated_total=gen,
            discharged=dis,
            runtime=rt,
            resolution_rate=f"{rate:.4f}",
        ))
    return summary_rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip", action="append", default=[],
                    choices=ALL_SYSTEMS, metavar="SYSTEM",
                    help="Skip running this system's collector (can repeat)")
    ap.add_argument("--only", action="append", default=[],
                    choices=ALL_SYSTEMS, metavar="SYSTEM",
                    help="Only run these systems (can repeat)")
    ap.add_argument("--no-run", action="store_true",
                    help="Skip running collectors; just merge existing CSVs")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    systems_to_run = args.only if args.only else ALL_SYSTEMS
    systems_to_run = [s for s in systems_to_run if s not in args.skip]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.no_run:
        print("=== Running per-system collectors ===")
        for system in systems_to_run:
            run_collector(system, args.verbose)

    print("\n=== Merging CSVs ===")
    all_rows = merge_csvs(systems_to_run)
    print(f"  Merged {len(all_rows)} rows across {len(systems_to_run)} systems")

    unified_fields = ["system", "category", "case_name",
                      "generated_total", "generated_user", "generated_internal",
                      "discharged", "runtime", "resolution_rate", "status"]
    with open(UNIFIED_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=unified_fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Wrote {UNIFIED_OUT}")

    summary = compute_summary(all_rows)
    summary_fields = ["system", "cases", "ok", "errors",
                      "generated_total", "discharged", "runtime", "resolution_rate"]
    with open(SUMMARY_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n=== Summary ===")
    print(f"{'system':<10} {'cases':>6} {'ok':>5} {'errors':>7} "
          f"{'generated':>10} {'discharged':>11} {'runtime':>8} {'rate':>6}")
    print("-" * 70)
    for r in summary:
        print(f"{r['system']:<10} {r['cases']:>6} {r['ok']:>5} {r['errors']:>7} "
              f"{r['generated_total']:>10} {r['discharged']:>11} "
              f"{r['runtime']:>8} {r['resolution_rate']:>6}")
    print(f"\nWrote {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
