#!/usr/bin/env python3
"""scripts/choreo_assertion_stats.py

Collect assessment statistics for the Choreo benchmark suite by running
``choreo --stats -t cute`` on every ``.co`` file under ``benchmark/choreo/``.

Assessment model (Choreo DSVN)
-------------------------------
Choreo's DSVN analyses every constraint that arises from kernel semantics
(DMA bounds, tiled-loop bounds, alignment checks, etc.) at compile time.
The ``--stats`` flag emits:

  Assessments evaluated          → total generation attempts
  Resolved at compile time       → compile-time discharged (static-true)
  Proven false at compile time   → compile-time errors detected
  Runtime assertions generated   → materialize as GPU runtime checks

Metric columns
--------------
  category, case_name,
  generated    : total assessments evaluated (generation attempts)
  discharged   : resolved at compile time (static-true + static-false)
  static_true  : provably correct at compile time
  static_false : provably incorrect (compile error)
  runtime      : assertions emitted as runtime GPU checks
  resolution_rate : discharged / generated
  status       : "ok" | "error" (compilation failure)

Outputs
-------
  benchmark/results/choreo_stats.csv

Usage
-----
  python scripts/choreo_assertion_stats.py [--choreo BIN] [--cases-dir DIR] [--out CSV]
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).parent.parent

# build-debug supports --stats; build-release does not (treats it as positional arg)
_CHOREO_CANDIDATES = [
    WORKSPACE_ROOT / "choreo" / "build-debug"   / "choreo",
    WORKSPACE_ROOT / "choreo" / "build-release" / "choreo",
    WORKSPACE_ROOT / "build-release" / "choreo",
    WORKSPACE_ROOT / "choreo" / "choreo",
]
DEFAULT_CASES_DIR = WORKSPACE_ROOT / "benchmark/choreo"
DEFAULT_OUT       = WORKSPACE_ROOT / "benchmark/results/choreo_stats.csv"

_STATS_BLOCK_RE = re.compile(
    r"===[-]+===\s*\.\.\. Assessment Statistics \.\.\.\s*===[-]+==="
    r"(.*?)"
    r"===[-]+===",
    re.DOTALL,
)
_STAT_LINE_RE = re.compile(r"^\s*(\d+)\s+assess\s+-\s+(.+)$", re.MULTILINE)


def find_choreo_bin(hint: Path | None) -> Path:
    if hint is not None and hint.exists():
        return hint
    for cand in _CHOREO_CANDIDATES:
        if cand.exists():
            return cand
    sys.exit(
        "Choreo binary not found. Build with: make -C choreo release\n"
        "Or specify with --choreo PATH"
    )


def run_choreo(choreo: Path, co_file: Path, timeout_s: int = 120) -> dict:
    """Run choreo --stats (no GPU target) to measure DSVN analysis only.

    We use '-t cute' because the benchmark kernels are written for the CuTe
    target (block-level parallel patterns require the target to be set).
    The DSVN assessment statistics are still reported identically; the target
    flag merely enables the code-generation paths these kernels require.
    """
    result = subprocess.run(
        [str(choreo), "--stats", "-es", "-fc", "-t", "cute", str(co_file)],
        capture_output=True, text=True, timeout=timeout_s,
        stdin=subprocess.DEVNULL,
    )
    output = result.stdout + result.stderr

    m = _STATS_BLOCK_RE.search(output)
    if m is None:
        # Compilation error – no stats block emitted
        return {"status": "error", "generated": 0, "discharged": 0,
                "static_true": 0, "static_false": 0, "runtime": 0,
                "resolution_rate": "0.0000",
                "ut_shape": 0, "ut_elem": 0, "ut_loop": 0, "ut_hw": 0,
                "rt_shape": 0, "rt_elem": 0, "rt_loop": 0, "rt_hw": 0}

    stats: dict[str, int] = {}
    for line in _STAT_LINE_RE.finditer(m.group(1)):
        count = int(line.group(1))
        label = line.group(2).strip()
        stats[label] = count

    generated    = stats.get("Assessments evaluated", 0)
    static_true  = stats.get("Resolved at compile time (static-true)", 0)
    static_false = stats.get("Proven false at compile time (static-false)", 0)
    runtime      = stats.get("Runtime assertions generated", 0)
    discharged   = static_true + static_false
    rate = (discharged / generated) if generated > 0 else 1.0

    # Per-usage-type breakdown
    ut_shape   = stats.get("Assessments (shape-compatibility)", 0)
    ut_elem    = stats.get("Assessments (element-access)", 0)
    ut_loop    = stats.get("Assessments (loop-bound)", 0)
    ut_hw      = stats.get("Assessments (hw-constraint)", 0)
    rt_shape   = stats.get("Runtime assertions (shape-compatibility)", 0)
    rt_elem    = stats.get("Runtime assertions (element-access)", 0)
    rt_loop    = stats.get("Runtime assertions (loop-bound)", 0)
    rt_hw      = stats.get("Runtime assertions (hw-constraint)", 0)

    return {
        "status":          "ok",
        "generated":       generated,
        "discharged":      discharged,
        "static_true":     static_true,
        "static_false":    static_false,
        "runtime":         runtime,
        "resolution_rate": f"{rate:.4f}",
        "ut_shape":        ut_shape,
        "ut_elem":         ut_elem,
        "ut_loop":         ut_loop,
        "ut_hw":           ut_hw,
        "rt_shape":        rt_shape,
        "rt_elem":         rt_elem,
        "rt_loop":         rt_loop,
        "rt_hw":           rt_hw,
    }


def _is_dynamic(case_name: str) -> bool:
    """True if the case has symbolic/dynamic dimensions.

    Uses the case name as canonical ground truth, consistent with
    mlir_assertion_stats.py and triton_assertion_stats.py:
    - 'dynamic' in the case name, OR
    - uppercase letters embedded in a shape string adjacent to digits or 'x'
      (e.g. 32xNx512 has symbolic N), excluding param suffixes like _S_P_D.
    """
    if 'dynamic' in case_name.lower():
        return True
    import re
    return bool(re.search(r'(?:\d|x)([A-Z])(?:\d|x)', case_name))


def _run_choreo_job(args_tuple):
    choreo, co_file, timeout_s = args_tuple
    cat = co_file.parent.name
    case = co_file.stem
    is_dyn = _is_dynamic(case)
    try:
        result = run_choreo(choreo, co_file, timeout_s)
    except subprocess.TimeoutExpired:
        result = {"status": "timeout", "generated": 0, "discharged": 0,
                  "static_true": 0, "static_false": 0, "runtime": 0,
                  "resolution_rate": "0.0000",
                  "ut_shape": 0, "ut_elem": 0, "ut_loop": 0, "ut_hw": 0,
                  "rt_shape": 0, "rt_elem": 0, "rt_loop": 0, "rt_hw": 0}
    return dict(category=cat, case_name=case, is_dynamic=int(is_dyn), **result)


def collect(cases_dir: Path, choreo: Path, verbose: bool = False,
            workers: int = 8) -> list[dict]:
    from multiprocessing.pool import ThreadPool
    jobs = []
    for cat_dir in sorted(cases_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name in {"scripts"}:
            continue
        for co_file in sorted(cat_dir.glob("*.co")):
            if co_file.stem.startswith("bench_"):
                continue  # skip performance harness files
            jobs.append((choreo, co_file, 120))

    rows: list[dict] = []
    with ThreadPool(workers) as pool:
        for i, row in enumerate(pool.imap(_run_choreo_job, jobs), 1):
            if verbose:
                print(f"  [{i}/{len(jobs)}] {row['category']}/{row['case_name']}"
                      f" status={row['status']} gen={row['generated']}", flush=True)
            rows.append(row)
    # Sort by (category, case_name) to match original order
    rows.sort(key=lambda r: (r["category"], r["case_name"]))
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--choreo", type=Path, default=None,
                    help="Path to choreo binary (default: auto-detect build-release/debug)")
    ap.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel worker threads (default: 8)")
    args = ap.parse_args(argv)

    choreo = find_choreo_bin(args.choreo)
    print(f"Using choreo: {choreo}")
    print(f"Scanning {args.cases_dir} with {args.workers} workers…")

    rows = collect(args.cases_dir, choreo, verbose=args.verbose,
                   workers=args.workers)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "case_name", "is_dynamic", "status",
                  "generated", "discharged", "static_true", "static_false",
                  "runtime", "resolution_rate",
                  "ut_shape", "ut_elem", "ut_loop", "ut_hw",
                  "rt_shape", "rt_elem", "rt_loop", "rt_hw"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    err = [r for r in rows if r["status"] not in ("ok",)]
    total_gen = sum(r["generated"] for r in ok)
    total_dis = sum(r["discharged"] for r in ok)
    total_rt  = sum(r["runtime"]    for r in ok)
    overall   = total_dis / total_gen if total_gen else 1.0
    print(f"\nWrote {len(rows)} rows → {args.out}")
    print(f"  OK cases    : {len(ok)}")
    print(f"  Error cases : {len(err)}")
    print(f"  Total generated  : {total_gen}")
    print(f"  Total discharged : {total_dis}  ({overall*100:.1f}%)")
    print(f"  Total runtime    : {total_rt}")


if __name__ == "__main__":
    main()
