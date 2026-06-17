"""scripts/choreo_runtime_entry.py
Measure the runtime overhead of Choreo's default assertion level (entry) vs
no assertions (none) on all dynamic benchmark cases.

The 'entry' level inserts only host-side runtime_check() calls at function
entry -- a superset of ENTRY-type assessments (host param-shape checks) and
a small number of HOIST-type assessments (pre-kernel-launch checks). These
are the cheapest possible runtime assertions: a handful of integer comparisons
before the GPU kernel is launched, with no device-side overhead.

Output: benchmark/results/choreo_runtime_entry.csv
Columns: category, case_name, none_us, entry_us, overhead_pct, notes

Usage:
  python3 scripts/choreo_runtime_entry.py [--reps N] [--out PATH] [--verbose]
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

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CHOREO_CANDIDATES = [
    WORKSPACE_ROOT / "build-release" / "choreo",
    WORKSPACE_ROOT / "choreo" / "build-release" / "choreo",
    WORKSPACE_ROOT / "choreo" / "choreo",
    WORKSPACE_ROOT / "choreo" / "build-debug"   / "choreo",
]
CASES_DIR   = WORKSPACE_ROOT / "benchmark" / "choreo"
DEFAULT_OUT = WORKSPACE_ROOT / "benchmark" / "results" / "choreo_runtime_entry.csv"

N_REPS         = 5
GEN_TIMEOUT_S  = 30
EXEC_TIMEOUT_S = 600

# Ensure CUDA and CuTe are discoverable
_CUDA_BIN = "/usr/local/cuda/bin"
if _CUDA_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _CUDA_BIN + ":" + os.environ.get("PATH", "")
if "CUDA_HOME" not in os.environ:
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
if "CUTE_HOME" not in os.environ:
    _cutlass = WORKSPACE_ROOT / "choreo" / "extern" / "cutlass"
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
                    rtc_level: str) -> bool:
    cmd = [str(choreo), "-gs", "-fc", "--max-local-mem-capacity=2000000", "-t", "cute", str(co_file),
           f"--runtime-check={rtc_level}", "-o", str(out_sh)]
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
            verbose: bool = False) -> List[Dict]:
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

                for level in ("none", "entry"):
                    sh = td / f"{case_name}_rtc_{level}.sh"
                    col = f"{level}_us"
                    if not generate_script(choreo, co_file, sh, level):
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

                # Compute overhead
                if row.get("none_us", "error") != "error" and \
                   row.get("entry_us", "error") != "error":
                    none_t  = float(row["none_us"])
                    entry_t = float(row["entry_us"])
                    if none_t > 0:
                        ovhd = (entry_t / none_t - 1.0) * 100.0
                        row["overhead_pct"] = f"{ovhd:.3f}"
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
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    choreo = find_choreo_bin()
    print(f"choreo   : {choreo}")
    print(f"cases    : {args.cases_dir}")
    print(f"reps     : {args.reps}")
    print(f"variants : none  vs  entry (default level)")
    print()

    rows = collect(args.cases_dir, choreo, args.reps, verbose=args.verbose)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "case_name", "none_us", "entry_us",
                  "overhead_pct", "notes"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows
          if r.get("none_us", "error") != "error"
          and r.get("entry_us", "error") != "error"]
    err = len(rows) - len(ok)

    print(f"\nWrote {len(rows)} rows  ->  {args.out}")
    print(f"Successfully measured: {len(ok)} / {len(rows)}  (errors: {err})")

    if ok:
        ovhds = [float(r["overhead_pct"]) for r in ok
                 if r.get("overhead_pct", "N/A") != "N/A"]
        if ovhds:
            print(f"\nOverhead (entry vs none):")
            print(f"  median : {statistics.median(ovhds):+.3f}%")
            print(f"  mean   : {statistics.mean(ovhds):+.3f}%")
            print(f"  min    : {min(ovhds):+.3f}%")
            print(f"  max    : {max(ovhds):+.3f}%")

            import collections
            by_cat: Dict[str, List[float]] = collections.defaultdict(list)
            for r in ok:
                if r.get("overhead_pct", "N/A") != "N/A":
                    by_cat[r["category"]].append(float(r["overhead_pct"]))
            print("\nPer-category median overhead (entry vs none):")
            for cat, vals in sorted(by_cat.items()):
                print(f"  {cat:25s}: median={statistics.median(vals):+.3f}%  "
                      f"n={len(vals)}")


if __name__ == "__main__":
    main()
