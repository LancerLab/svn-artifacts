"""scripts/choreo_runtime.py
Measure runtime execution cost of dynamic Choreo cases under five variants:
  - rtc_none   : dynamic shapes, --runtime-check=none  (assertions disabled)
  - rtc_low    : dynamic shapes, --runtime-check=low
  - rtc_medium : dynamic shapes, --runtime-check=medium
  - rtc_high   : dynamic shapes, --runtime-check=high  (all assertions)
  - static     : #define __STATIC_SHAPE__ (fully concrete, no DSVN at runtime)

All GPU runs are SEQUENTIAL (single device).

Output: benchmark/results/choreo_runtime.csv
Columns: category, case_name, rtc_none_us, rtc_low_us, rtc_medium_us,
         rtc_high_us, static_us, notes

Usage:
  python3 scripts/choreo_runtime.py [--reps N] [--out PATH] [--verbose]
"""

import argparse
import csv
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CHOREO_CANDIDATES = [
    WORKSPACE_ROOT / "choreo" / "build-debug"  / "choreo",
    WORKSPACE_ROOT / "choreo" / "build-release" / "choreo",
]
CASES_DIR   = WORKSPACE_ROOT / "benchmark" / "choreo"
DEFAULT_OUT = WORKSPACE_ROOT / "benchmark" / "results" / "choreo_runtime.csv"

N_REPS         = 5
GEN_TIMEOUT_S  = 30    # choreo -gs (frontend only, fast)
EXEC_TIMEOUT_S = 300   # bash --execute (nvcc compile + GPU run)

RTC_LEVELS = ["none", "low", "medium", "high"]


def find_choreo_bin() -> Path:
    for p in CHOREO_CANDIDATES:
        if p.exists():
            return p
    sys.exit("choreo binary not found")


def _is_dynamic(co_file: Path) -> bool:
    text = co_file.read_text(errors="ignore")
    return ("#ifdef __STATIC_SHAPE__" in text and
            "#define __STATIC_SHAPE__" not in text)


def generate_script(choreo: Path, co_file: Path, out_sh: Path,
                    rtc_level: Optional[str] = None,
                    static: bool = False) -> bool:
    """Generate a .sh execute-script via choreo -gs [-t cute] [--runtime-check=].
    Returns True iff the output file was created and is non-empty
    (choreo may exit non-zero on warnings but still produce valid output).
    """
    if static:
        src = "#define __STATIC_SHAPE__\n" + co_file.read_text(errors="ignore")
        with tempfile.NamedTemporaryFile(suffix=".co", mode="w",
                                        delete=False) as tf:
            tf.write(src)
            tmp_co = Path(tf.name)
        input_co = tmp_co
    else:
        tmp_co   = None
        input_co = co_file

    cmd = [str(choreo), str(input_co), "-gs", "-t", "cute", "-o", str(out_sh)]
    if rtc_level is not None:
        cmd.append(f"--runtime-check={rtc_level}")

    try:
        subprocess.run(cmd, capture_output=True, text=True,
                       stdin=subprocess.DEVNULL, timeout=GEN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False
    finally:
        if tmp_co is not None:
            tmp_co.unlink(missing_ok=True)

    return out_sh.exists() and out_sh.stat().st_size > 0


def run_once(sh_file: Path) -> Optional[int]:
    """Run sh --execute; return execution time in microseconds or None."""
    try:
        r = subprocess.run(["bash", str(sh_file), "--execute"],
                           capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=EXEC_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    for line in (r.stdout + r.stderr).splitlines():
        if "Execution time:" in line and "microseconds" in line:
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
    with tempfile.TemporaryDirectory(prefix="choreo_rt_") as tmpdir:
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

                # Dynamic variants
                for level in RTC_LEVELS:
                    sh = td / f"{case_name}_rtc_{level}.sh"
                    col = f"rtc_{level}_us"
                    if not generate_script(choreo, co_file, sh, rtc_level=level):
                        if verbose: print(f"    rtc={level}: gen-failed")
                        row[col] = "error"; notes.append(f"gen-{level}-failed")
                        continue
                    med, note = measure_median(sh, reps)
                    if med is None:
                        if verbose: print(f"    rtc={level}: run-failed")
                        row[col] = "error"; notes.append(f"run-{level}-failed")
                    else:
                        if verbose: print(f"    rtc={level}: {med:.0f} us")
                        row[col] = f"{med:.0f}"

                # Static variant
                sh_stat = td / f"{case_name}_static.sh"
                if not generate_script(choreo, co_file, sh_stat, static=True):
                    if verbose: print(f"    static: gen-failed")
                    row["static_us"] = "error"; notes.append("gen-static-failed")
                else:
                    med, note = measure_median(sh_stat, reps)
                    if med is None:
                        if verbose: print(f"    static: run-failed")
                        row["static_us"] = "error"; notes.append("run-static-failed")
                    else:
                        if verbose: print(f"    static: {med:.0f} us")
                        row["static_us"] = f"{med:.0f}"

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
    print(f"reps     : {args.reps}  (sequential, single GPU)")
    print(f"variants : rtc=none/low/medium/high + static")
    print()

    rows = collect(args.cases_dir, choreo, args.reps, verbose=args.verbose)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "case_name",
                  "rtc_none_us", "rtc_low_us", "rtc_medium_us", "rtc_high_us",
                  "static_us", "notes"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows
          if all(r.get(f"rtc_{l}_us", "error") != "error" for l in RTC_LEVELS)
          and r.get("static_us", "error") != "error"]

    print(f"\nWrote {len(rows)} rows -> {args.out}")
    print(f"Fully measured: {len(ok)} / {len(rows)}")

    if ok:
        def pct(r: Dict, col: str) -> float:
            base = float(r["static_us"])
            return (float(r[col]) - base) / base * 100.0 if base else 0.0
        for level in RTC_LEVELS:
            col = f"rtc_{level}_us"
            ovhds = [pct(r, col) for r in ok]
            print(f"  rtc={level:6s}: overhead vs static  "
                  f"median={statistics.median(ovhds):+.2f}%  "
                  f"mean={statistics.mean(ovhds):+.2f}%")


if __name__ == "__main__":
    main()
