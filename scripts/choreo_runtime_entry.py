"""scripts/choreo_runtime_entry.py
Measure the runtime overhead of Choreo's assertion levels against the
baseline (no assertions) on all dynamic benchmark cases.

Supported levels (via --levels):
  none        - baseline, no assertions
  entry       - host-side entry-point checks only
  all         - full checks with assertion hoisting
  all-nohoist - full checks without assertion hoisting

Output: benchmark/results/choreo_runtime_entry.csv
Columns: category, case_name, <level>_us per level, overhead_pct (entry vs
none), overhead_<level>_pct per non-none level, notes

Usage:
  python3 scripts/choreo_runtime_entry.py [--reps N] [--out PATH] [--verbose]
  python3 scripts/choreo_runtime_entry.py --levels none,entry,all,all-nohoist
"""

import argparse
import csv
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_local_mem_capacity import capacity_flag  # noqa: E402

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CHOREO_CANDIDATES = [
    WORKSPACE_ROOT / "build-release" / "choreo",
    WORKSPACE_ROOT / "croqtile" / "build-release" / "choreo",
    WORKSPACE_ROOT / "croqtile" / "choreo",
    WORKSPACE_ROOT / "croqtile" / "build-debug"   / "choreo",
]
CASES_DIR   = WORKSPACE_ROOT / "benchmark" / "choreo"
DEFAULT_OUT = WORKSPACE_ROOT / "benchmark" / "results" / "choreo_runtime_entry.csv"

N_REPS         = 5
GEN_TIMEOUT_S  = 30
EXEC_TIMEOUT_S = 600

# Assertion levels supported by --levels (level name -> choreo runtime flags).
# 'none' is the baseline (no assertions); 'entry' is host-side entry checks;
# 'all' is full checks with assertion hoisting; 'all-nohoist' is full checks
# with hoisting disabled.
LEVEL_FLAGS = {
    "none":        ["--runtime-check=none"],
    "entry":       ["--runtime-check=entry"],
    "all":         ["--runtime-check=all"],
    "all-nohoist": ["--runtime-check=all", "--disable-assert-hoist"],
}


def _col(level: str) -> str:
    """Sanitize a level name for use as a CSV column key."""
    return level.replace("-", "_")

# Ensure CUDA and CuTe are discoverable
_CUDA_BIN = "/usr/local/cuda/bin"
if _CUDA_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _CUDA_BIN + ":" + os.environ.get("PATH", "")
if "CUDA_HOME" not in os.environ:
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
if "CUTE_HOME" not in os.environ:
    _cutlass = WORKSPACE_ROOT / "croqtile" / "extern" / "cutlass"
    if _cutlass.is_dir():
        os.environ["CUTE_HOME"] = str(_cutlass)


def find_choreo_bin() -> Path:
    for p in CHOREO_CANDIDATES:
        if p.exists():
            return p
    sys.exit("choreo binary not found in expected locations")


def _is_dynamic(co_file: Path) -> bool:
    text = co_file.read_text(errors="ignore")
    return ("#ifdef __STATIC_SHAPE__" in text and
            "#define __STATIC_SHAPE__" not in text)


def generate_script(choreo: Path, co_file: Path, out_sh: Path,
                    rtc_flags: List[str]) -> bool:
    # The capacity must be derived from the actual GPU: for dynamic shapes it
    # sizes the per-thread local arena, and a value the driver cannot back
    # makes every launch fail.  See gpu_local_mem_capacity.py.
    cmd = [str(choreo), "-gs", "-fc", capacity_flag(), "-t", "cute", str(co_file),
           *rtc_flags, "-o", str(out_sh)]
    try:
        subprocess.run(cmd, capture_output=True, text=True,
                       stdin=subprocess.DEVNULL, timeout=GEN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False
    return out_sh.exists() and out_sh.stat().st_size > 0


def run_once(sh_file: Path) -> Optional[int]:
    """Run shell script --execute; return execution time in microseconds or None."""
    try:
        r = subprocess.run(["bash", str(sh_file), "--execute"],
                           capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=EXEC_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    for line in (r.stdout + r.stderr).splitlines():
        if "Execution time:" in line and ("microseconds" in line or "us" in line):
            try:
                return int(line.split(":")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    return None


def measure_median(sh_file: Path, reps: int) -> Tuple[Optional[float], str]:
    times = []
    for _ in range(reps):
        t = run_once(sh_file)
        if t is None:
            return None, "run-failed"
        times.append(t)
    return statistics.median(times), ""


def collect(cases_dir: Path, choreo: Path, reps: int,
            levels: List[str], verbose: bool = False) -> List[Dict]:
    rows: List[Dict] = []
    with tempfile.TemporaryDirectory(prefix="choreo_rte_") as tmpdir:
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

                row: Dict = {"category": category, "case_name": case_name}
                notes: List[str] = []

                for level in levels:
                    sh = td / f"{case_name}_rtc_{_col(level)}.sh"
                    col = f"{_col(level)}_us"
                    if not generate_script(choreo, co_file, sh, LEVEL_FLAGS[level]):
                        if verbose:
                            print(f"    rtc={level}: gen-failed")
                        row[col] = "error"
                        notes.append(f"gen-{level}-failed")
                        continue
                    med, note = measure_median(sh, reps)
                    if med is None:
                        if verbose:
                            print(f"    rtc={level}: run-failed")
                        row[col] = "error"
                        notes.append(f"run-{level}-failed")
                    else:
                        if verbose:
                            print(f"    rtc={level}: {med:.0f} us")
                        row[col] = f"{med:.0f}"

                # Compute per-level overhead relative to the 'none' baseline.
                if row.get("none_us", "error") != "error":
                    none_t = float(row["none_us"])
                    if none_t > 0:
                        for level in levels:
                            if level == "none":
                                continue
                            col = f"{_col(level)}_us"
                            if row.get(col, "error") != "error":
                                ovhd = (float(row[col]) / none_t - 1.0) * 100.0
                                row[f"overhead_{_col(level)}_pct"] = f"{ovhd:.3f}"
                            else:
                                row[f"overhead_{_col(level)}_pct"] = "N/A"
                        # Backward-compatible column: entry vs none.
                        if row.get("entry_us", "error") != "error":
                            row["overhead_pct"] = f"{(float(row['entry_us']) / none_t - 1.0) * 100.0:.3f}"
                        else:
                            row["overhead_pct"] = "N/A"
                    else:
                        row["overhead_pct"] = "N/A"
                else:
                    row["overhead_pct"] = "N/A"

                row["notes"] = "; ".join(notes)
                rows.append(row)
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps",      type=int,  default=N_REPS)
    ap.add_argument("--out",       type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cases-dir", type=Path, default=CASES_DIR)
    ap.add_argument("--levels",    type=str,
                    default="none,entry",
                    help="comma-separated assertion levels: none,entry,all,all-nohoist")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    for l in levels:
        if l not in LEVEL_FLAGS:
            sys.exit(f"Unknown level '{l}'. Choose from: {', '.join(LEVEL_FLAGS)}")
    if "none" not in levels:
        sys.exit("--levels must include 'none' as the baseline")

    choreo = find_choreo_bin()
    print(f"choreo   : {choreo}")
    print(f"cases    : {args.cases_dir}")
    print(f"reps     : {args.reps}")
    print(f"variants : {'  vs  '.join(levels)}")
    print()

    rows = collect(args.cases_dir, choreo, args.reps, levels,
                   verbose=args.verbose)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "case_name"]
    fieldnames += [f"{_col(l)}_us" for l in levels]
    fieldnames += ["overhead_pct"]
    fieldnames += [f"overhead_{_col(l)}_pct" for l in levels if l != "none"]
    fieldnames += ["notes"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if r.get("none_us", "error") != "error"]
    err = len(rows) - len(ok)

    print(f"\nWrote {len(rows)} rows  ->  {args.out}")
    print(f"Successfully measured baseline: {len(ok)} / {len(rows)}  (errors: {err})")

    if ok:
        # Per-level overhead vs none.
        print("\nOverhead (vs none):")
        level_medians: Dict[str, float] = {}
        for level in levels:
            if level == "none":
                continue
            col = f"overhead_{_col(level)}_pct"
            ovhds = [float(r[col]) for r in ok if r.get(col, "N/A") not in ("N/A", "")]
            if ovhds:
                med = statistics.median(ovhds)
                level_medians[level] = med
                print(f"  {level:12s}: median={med:+.3f}%  mean={statistics.mean(ovhds):+.3f}%"
                      f"  min={min(ovhds):+.3f}%  max={max(ovhds):+.3f}%  n={len(ovhds)}")
            else:
                print(f"  {level:12s}: (no successful measurements)")

        # Hoisting reduction: no-hoist overhead / hoisted overhead.
        if "all" in level_medians and "all-nohoist" in level_medians:
            hoisted = level_medians["all"]
            nohoist = level_medians["all-nohoist"]
            if hoisted > 0:
                print(f"\nHoisting reduction (all-nohoist / all): {nohoist / hoisted:.1f}x")
            elif nohoist > 0:
                print("\nHoisting reduction: infinite (hoisted overhead ~0)")

        import collections
        by_cat: Dict[str, List[float]] = collections.defaultdict(list)
        for r in ok:
            if r.get("overhead_pct", "N/A") != "N/A":
                by_cat[r["category"]].append(float(r["overhead_pct"]))
        if by_cat:
            print("\nPer-category median overhead (entry vs none):")
            for cat, vals in sorted(by_cat.items()):
                print(f"  {cat:25s}: median={statistics.median(vals):+.3f}%  "
                      f"n={len(vals)}")


if __name__ == "__main__":
    main()
