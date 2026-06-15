"""scripts/choreo_compile_overhead.py
Measure the compile-time overhead of DSVN's symbolic shape analysis using
the concrete-baseline technique:

  For each *dynamic* benchmark program compile twice with `choreo -es`
  (emit target source only — no GPU compilation, eliminating nvcc variability):

    dynamic  : normal mode — symbolic dims are propagated through DSVN's
               value-numbering and assessment-generation passes in full.

    static   : the source is prepended with `#define __STATIC_SHAPE__`, which
               freezes every symbolic dimension to its concrete constant before
               semantic analysis, bypassing all symbolic reasoning.

  Overhead = (median_dynamic / median_static) - 1  per case.

Output: benchmark/results/choreo_compile_overhead.csv
Columns: category, case_name, dynamic_ms, static_ms, overhead_pct, notes

Usage:
  python3 scripts/choreo_compile_overhead.py [--reps N] [--out PATH] [--verbose]
"""

import argparse
import csv
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CHOREO_CANDIDATES = [
    WORKSPACE_ROOT / "choreo" / "build-release" / "choreo",
    WORKSPACE_ROOT / "build-release" / "choreo",
    WORKSPACE_ROOT / "choreo" / "choreo",
    WORKSPACE_ROOT / "choreo" / "build-debug"  / "choreo",
]
CASES_DIR   = WORKSPACE_ROOT / "benchmark" / "choreo"
DEFAULT_OUT = WORKSPACE_ROOT / "benchmark" / "results" / "choreo_compile_overhead.csv"

N_REPS          = 5
COMPILE_TIMEOUT = 60   # seconds per -es invocation


def find_choreo_bin() -> Path:
    for p in CHOREO_CANDIDATES:
        if p.exists():
            return p
    sys.exit("choreo binary not found")


def _is_dynamic(co_file: Path) -> bool:
    text = co_file.read_text(errors="ignore")
    return ("#ifdef __STATIC_SHAPE__" in text and
            "#define __STATIC_SHAPE__" not in text)


def compile_once(choreo: Path, src: Path, out_dir: Path) -> Tuple[Optional[float], bool]:
    """Run `choreo -es src -t cute -o <tmp>` and return (wall-time in ms, ok).
    Returns (None, False) on timeout, (ms, False) on compile error."""
    out = out_dir / (src.stem + ".gen")
    cmd = [str(choreo), str(src), "-es", "-t", "cute", "-o", str(out)]
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                stdin=subprocess.DEVNULL, timeout=COMPILE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, False
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0
    return ms, (result.returncode == 0)


def measure_median(choreo: Path, src: Path, out_dir: Path,
                   reps: int) -> Tuple[Optional[float], str]:
    times = []
    for _ in range(reps):
        ms, ok = compile_once(choreo, src, out_dir)
        if ms is None:
            return None, "timeout"
        if not ok:
            return None, "compile-error"
        times.append(ms)
    return statistics.median(times), ""


def collect(cases_dir: Path, choreo: Path, reps: int,
            verbose: bool = False) -> List[Dict]:
    rows: List[Dict] = []
    with tempfile.TemporaryDirectory(prefix="choreo_co_") as tmpdir:
        td = Path(tmpdir)
        for cat_dir in sorted(cases_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            category = cat_dir.name
            for co_file in sorted(cat_dir.glob("*.co")):
                if not _is_dynamic(co_file):
                    continue
                case_name = co_file.stem
                if verbose:
                    print(f"  {category}/{case_name}", flush=True)

                notes: List[str] = []
                row: Dict = {"category": category, "case_name": case_name}

                # Dynamic mode: compile the file as-is (symbolic DSVN analysis)
                dyn_ms, note = measure_median(choreo, co_file, td, reps)
                if dyn_ms is None:
                    row["dynamic_ms"] = "error"; notes.append(f"dynamic-{note}")
                    if verbose:
                        print(f"    dynamic: {note}")
                else:
                    row["dynamic_ms"] = f"{dyn_ms:.3f}"
                    if verbose:
                        print(f"    dynamic: {dyn_ms:.1f} ms")

                # Static baseline: prepend #define __STATIC_SHAPE__ to freeze all
                # symbolic dimensions → DSVN has nothing symbolic to process.
                static_src_text = "#define __STATIC_SHAPE__\n" + co_file.read_text(
                    errors="ignore")
                with tempfile.NamedTemporaryFile(
                        suffix=".co", mode="w", dir=td, delete=False) as tf:
                    tf.write(static_src_text)
                    static_co = Path(tf.name)

                stat_ms, note = measure_median(choreo, static_co, td, reps)
                static_co.unlink(missing_ok=True)

                if stat_ms is None:
                    row["static_ms"] = "error"; notes.append(f"static-{note}")
                else:
                    row["static_ms"] = f"{stat_ms:.3f}"
                    if verbose:
                        print(f"    static : {stat_ms:.1f} ms")

                # Overhead
                if dyn_ms is not None and stat_ms is not None and stat_ms > 0:
                    overhead = (dyn_ms / stat_ms - 1.0) * 100.0
                    row["overhead_pct"] = f"{overhead:.2f}"
                    if verbose:
                        print(f"    overhead: {overhead:.2f}%")
                else:
                    row["overhead_pct"] = "N/A"

                row["notes"] = "; ".join(notes)
                rows.append(row)
    return rows


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps",      type=int,  default=N_REPS)
    ap.add_argument("--out",       type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cases-dir", type=Path, default=CASES_DIR)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    choreo = find_choreo_bin()
    print(f"choreo   : {choreo}")
    print(f"cases    : {args.cases_dir}")
    print(f"reps     : {args.reps}")
    print(f"method   : choreo -es <dynamic> vs choreo -es <#define __STATIC_SHAPE__>")
    print()

    rows = collect(args.cases_dir, choreo, args.reps, verbose=args.verbose)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "case_name", "dynamic_ms", "static_ms",
                  "overhead_pct", "notes"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    valid = [r for r in rows if r.get("overhead_pct") not in ("N/A", "error", "")]
    if valid:
        overheads = [float(r["overhead_pct"]) for r in valid]
        print(f"\n=== Compile-Time Overhead Summary ({len(overheads)} cases) ===")
        print(f"  mean   : {statistics.mean(overheads):.2f}%")
        print(f"  median : {statistics.median(overheads):.2f}%")
        print(f"  stdev  : {statistics.stdev(overheads):.2f}%"
              if len(overheads) > 1 else "")
        low, high = min(overheads), max(overheads)
        print(f"  range  : {low:.2f}% – {high:.2f}%")

    print(f"\nResults written to {args.out} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
